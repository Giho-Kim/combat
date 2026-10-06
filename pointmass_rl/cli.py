import argparse, csv, json, platform
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from .env import (Config, EVALUATION_TASKS, SUCCESS_THRESHOLDS, World,
                  evaluation_task_for_episode, split_strike_action,
                  target_remaining_value)
from .policies import (ApproximateDPPolicy, HeuristicPolicy, NearestTargetPolicy,
                       RandomPolicy, StrikeMAPPOPolicy, TypePriorityPolicy)
from .report import replay, plot

def make_policy(name, config, seed, model=None):
    if name=='random': return RandomPolicy(config,seed)
    if name=='heuristic': return HeuristicPolicy(config,seed)
    if name=='nearest': return NearestTargetPolicy(config,seed)
    if name=='type_priority': return TypePriorityPolicy(config,seed)
    if name=='approximate_dp': return ApproximateDPPolicy(config,seed)
    if name in ('mappo', 'coma', 'compact_coma', 'mat', 'target_mappo', 'target_coma'): return StrikeMAPPOPolicy(model,config)
    raise ValueError(name)

def rollout(config, policy, seed, record=False, success_threshold=None,
            evaluation_task=None):
    if hasattr(policy,'reset'):policy.reset()
    world=World(config);obs=world.reset(seed,success_threshold=success_threshold,
                                       evaluation_task=evaluation_task);frames=[world.snapshot()] if record else [];decisions=[]
    while not world.done:
        action=(policy.predict(obs,world) if isinstance(policy,StrikeMAPPOPolicy)
                else policy.predict(obs))
        if record:
            targets=split_strike_action(action)
            targets=world.committed_targets(targets)
            goals=world.resolve_setpoints(targets)
            for i,(target,goal) in enumerate(zip(targets,goals)):
                decisions.append(dict(step=world.t,drone=i,target_id=int(target),
                    target_type=int(world.mem_type[i,target]),target_life=int(world.mem_life[i,target]),
                    target_score=target_remaining_value(
                        world.mem_type[i,target], world.mem_life[i,target]),
                    estimated_distance=float(np.linalg.norm(goal-world.perceived_pos[i])),
                    goal_x=float(goal[0]),goal_y=float(goal[1]),active=bool(world.agent_active[i])))
        obs,_,_,_,_=world.step(action)
        if record:frames.append(world.snapshot())
    return world.metrics(),frames,decisions

def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def command_config(args):
    config = Config.load(args.config)
    mode = getattr(args, 'mode', None)
    if mode == 'belief' and args.config is None:
        if getattr(args, 'model', None):
            from .belief_training import load_belief_checkpoint
            _, config, _ = load_belief_checkpoint(args.model)
        else:
            config = replace(config, mode='belief', n_agents=5, min_agents=5, horizon=200)
    if mode is not None:
        config = replace(config, mode=mode)
    for option, field in (('gamma', 'discount_gamma'), ('gae_lambda', 'gae_lambda')):
        value = getattr(args, option, None)
        if value is not None:
            config = replace(config, **{field: value})
    interval = getattr(args, 'decision_interval', None)
    if interval is not None:
        config = replace(config, decision_interval=interval)
    if getattr(args, 'sticky_assignment', False):
        config = replace(config, sticky_assignment=True)
    if getattr(args, 'commit_target', False):
        config = replace(config, commit_target=True)
    lawnmower = getattr(args, 'lawnmower', None)
    if lawnmower is not None:
        if config.mode != 'belief':
            raise ValueError('--lawnmower/--no-lawnmower requires --mode belief')
        config = replace(config, belief_lawnmower=lawnmower)
    return config


def train(args):
    c=command_config(args)
    if args.model_based_advantage and (c.mode != 'known' or args.algorithm != 'target_coma'
                                       or args.target_coma_critic != 'graph'
                                       or args.target_coma_actor != 'attention'):
        raise ValueError('--model-based-advantage requires known-mode v15 target_coma')
    if c.mode == 'belief':
        from .belief_training import command_train
        command_train(args, c)
        print(f'Saved {Path(args.out) / "belief_target_coma.pt"}', flush=True)
        return
    try:
        import torch
        from .strike_ppo import train_mappo,save_checkpoint
    except ImportError as e: raise SystemExit(str(e)) from e
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);c.save(out/'config.json')
    print(f'Training centralized Strike PPO: {args.steps:,} agent transitions',flush=True)
    model,episodes,evaluations,actual=train_mappo(
        c,args.steps,args.seed,args.rollout_steps,epochs=args.epochs,progress=True,
        learning_rate=args.learning_rate, minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef, policy_clip=args.policy_clip,
        policy_temperature=args.policy_temperature,
        actor_samples=args.actor_samples,
        guided_training=args.guided_training,
        coma_advantage=args.coma_advantage, critic_epochs=args.critic_epochs,
        target_coma_critic=args.target_coma_critic,
        target_coma_actor=args.target_coma_actor,
        critic_minibatch_size=args.critic_minibatch_size, best_metric=args.best_metric,
        lookahead_credit=args.lookahead_credit,
        deadline_mask=args.deadline_mask,
        eval_interval=args.eval_interval,eval_episodes=args.eval_episodes,eval_seed=args.eval_seed,
        best_path=out/'best.pt',latest_path=out/'latest.pt',checkpoint_dir=out,
        n_envs=args.n_envs,diagnostics_path=out/'training_updates.csv',algorithm=args.algorithm)
    metadata=dict(seed=args.seed,requested_agent_transitions=args.steps,actual_agent_transitions=actual,
                  eval_interval=args.eval_interval,eval_episodes=args.eval_episodes,eval_seed=args.eval_seed,
                  evaluation_tasks=list(EVALUATION_TASKS) if evaluation_task_for_episode(c,0) else None,
                  config=asdict(c),python=platform.python_version(),numpy=str(np.__version__),torch=str(torch.__version__),
                  algorithm=('Model-assisted counterfactual PPO with assignment-damage guidance'
                             if args.lookahead_credit else
                             'Sampled-return counterfactual PPO'
                             if args.coma_advantage == 'sampled_return' else
                             'MAT-style autoregressive PPO with separate team critic'
                             if args.algorithm == 'mat' else
                             'Shared target-scoring actor with COMA advantage and PPO clipped loss'
                             if args.algorithm == 'target_coma' else
                             'Matched-information global MLP actor with COMA advantage and PPO clipped loss'
                             if args.algorithm == 'compact_coma' else
                             'COMA counterfactual advantage with PPO clipped actor loss'
                             if args.algorithm == 'coma' else
                             'Shared target-scoring MAPPO with separate team critic'
                             if args.algorithm == 'target_mappo' else
                             'Feed-forward MAPPO with finite-horizon team returns'),
                  training_settings=model.training_settings)
    model_path = out / f'strike_{args.algorithm}.pt'
    save_checkpoint(model_path,model,metadata);write_csv(out/'training_episodes.csv',episodes)
    save_checkpoint(out/'latest.pt',model,metadata)
    if not evaluations:
        save_checkpoint(out/'best.pt',model,dict(metadata, best_metric='final_model'))
    write_csv(out/'training_evaluations.csv',evaluations)
    (out/'training_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Saved {model_path}',flush=True)

def evaluate(args):
    c=command_config(args)
    if c.mode == 'belief':
        from .belief_training import command_evaluate
        return command_evaluate(args, c)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);c.save(out/'config.json')
    balanced=(c.n_targets==5 and c.min_targets==5 and not c.randomize_counts
              and not c.randomize_target_composition)
    task_suite=evaluation_task_for_episode(c,0) is not None
    if task_suite and args.episodes%len(EVALUATION_TASKS):
        raise SystemExit(f'--episodes must be a multiple of {len(EVALUATION_TASKS)} so every task has the same count')
    if balanced and not task_suite and args.episodes%len(SUCCESS_THRESHOLDS):
        raise SystemExit(f'--episodes must be a multiple of {len(SUCCESS_THRESHOLDS)} so every B case has the same count')
    if args.model:
        import torch
        torch.set_num_threads(1)
    learned = StrikeMAPPOPolicy(args.model, c) if args.model else None
    learned_name = learned.model.algorithm if learned else None
    learned_policies = ({f'{learned_name}_no_resolver': StrikeMAPPOPolicy(args.model, c, resolver=False),
                         f'{learned_name}_resolver': learned} if learned else {})
    names=['nearest','type_priority','approximate_dp']+list(learned_policies);rows=[];runs={};summaries={};task_success={};task_damage={};task_return={};task_discounted_return={};decisions=[]
    record_episode = args.episodes - 1
    progress=tqdm(total=len(names)*args.episodes,desc='Evaluate',unit='episode',dynamic_ncols=True)
    for name in names:
        fixed=learned_policies.get(name)
        for k in range(args.episodes):
            policy=fixed or make_policy(name,c,args.seed+k,args.model)
            task=evaluation_task_for_episode(c,k)
            threshold=SUCCESS_THRESHOLDS[k%len(SUCCESS_THRESHOLDS)] if balanced and task is None else None
            metrics,frames,log=rollout(
                c,policy,args.seed+k,record=k==record_episode,
                success_threshold=threshold,evaluation_task=task)
            rows.append(dict(policy=name,seed=args.seed+k,evaluation_task=task,**metrics))
            if k==record_episode:
                runs[name]=frames
                decisions.extend(dict(policy=name,seed=args.seed+k,**x) for x in log)
            progress.update(1);progress.set_postfix(policy=name,
                                                    return_=f"{metrics['team_return']:.2f}",
                                                    discounted=f"{metrics['discounted_team_return']:.2f}",
                                                    score=f"{metrics['score']}/{metrics['baseline_score']}",
                                                    success=int(metrics['mission_success']))
        local=[r for r in rows if r['policy']==name];summaries[name]={}
        task_success[name]={task:{'mean':float(np.mean(
            [row['mission_success'] for row in local if row['evaluation_task']==task])),
            'n':sum(row['evaluation_task']==task for row in local)}
            for task in EVALUATION_TASKS} if task_suite else {}
        task_damage[name]={task:{'mean':float(np.mean(
            [row['score'] for row in local if row['evaluation_task']==task])),
            'n':sum(row['evaluation_task']==task for row in local)}
            for task in EVALUATION_TASKS} if task_suite else {}
        task_return[name]={task:{'mean':float(np.mean(
            [row['team_return'] for row in local if row['evaluation_task']==task])),
            'n':sum(row['evaluation_task']==task for row in local)}
            for task in EVALUATION_TASKS} if task_suite else {}
        task_discounted_return[name]={task:{'mean':float(np.mean(
            [row['discounted_team_return'] for row in local if row['evaluation_task']==task])),
            'n':sum(row['evaluation_task']==task for row in local)}
            for task in EVALUATION_TASKS} if task_suite else {}
        for key in metrics:
            if key == 'agent_terminated':
                continue
            vals=np.array([r[key] for r in local if r[key] is not None],float)
            sd=float(vals.std(ddof=1)) if len(vals)>1 else 0.
            summaries[name][key]=dict(
                mean=float(vals.mean()) if len(vals) else None,std=sd,
                ci95_halfwidth=1.96*sd/np.sqrt(len(vals)) if len(vals) else None,n=len(vals))
    progress.close()
    (out/'summary.json').write_text(json.dumps(dict(
        episodes=args.episodes,
        replay_episode=record_episode, replay_seed=args.seed+record_episode,
        summary=summaries,task_success=task_success,
        task_damage=task_damage,task_return=task_return,
        task_discounted_return=task_discounted_return),indent=2)+'\n')
    write_csv(out/'episodes.csv',rows);write_csv(out/'decisions.csv',decisions);replay(out/'replay.html',c,runs);plot(out/'comparison.png',summaries)

def main():
    p=argparse.ArgumentParser(description='Strike-priority point-mass reinforcement learning');s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('train');q.add_argument('--steps',type=int,default=1_000_000);q.add_argument('--rollout-steps',type=int,default=200);q.add_argument('--n-envs',type=int,default=32);q.add_argument('--seed',type=int,default=7);q.add_argument('--out',default='runs/train');q.add_argument('--config');q.add_argument('--eval-interval',type=int,default=10_000,help='agent transitions between held-out evaluations; 0 disables evaluation');q.add_argument('--eval-episodes',type=int,default=3);q.add_argument('--eval-seed',type=int,default=10_000);q.set_defaults(func=train)
    q=s.add_parser('evaluate');q.add_argument('--model');q.add_argument('--episodes',type=int,default=30);q.add_argument('--seed',type=int,default=10000);q.add_argument('--out',default='runs/eval');q.add_argument('--config');q.set_defaults(func=evaluate)
    q.add_argument('--lawnmower', action=argparse.BooleanOptionalAction, default=None,
        help='Enable/disable region lawnmower sweeps in belief evaluation; Dubins movement stays enabled (default: config)')
    s.choices['train'].add_argument('--algorithm', choices=('mappo', 'coma', 'compact_coma', 'mat', 'target_mappo', 'target_coma'), default='mappo')
    s.choices['train'].add_argument('--learning-rate', type=float, default=5e-4)
    s.choices['train'].add_argument('--minibatch-size', type=int, default=None)
    s.choices['train'].add_argument('--entropy-coef', type=float, default=.01)
    s.choices['train'].add_argument('--gamma', type=float, default=None)
    s.choices['train'].add_argument('--gae-lambda', type=float, default=None)
    s.choices['train'].add_argument('--coma-advantage', choices=('q', 'sampled_return'), default='q',
        help='COMA actor signal: estimated Q or observed return minus counterfactual baseline')
    s.choices['train'].add_argument('--target-coma-critic', choices=('graph', 'pair'), default='graph',
        help='target_coma Q architecture: variable-slot graph v14 or legacy pair v12')
    s.choices['train'].add_argument('--target-coma-actor', choices=('attention', 'mlp'), default='attention',
        help='With graph Q: variable-slot attention actor v15 or fixed-slot MLP actor v14')
    s.choices['train'].add_argument('--critic-epochs', type=int, default=None,
        help='COMA Q fitting epochs per rollout (default: 4 times --epochs); actor epochs unchanged')
    s.choices['train'].add_argument('--critic-minibatch-size', type=int, default=None)
    s.choices['train'].add_argument('--best-metric', choices=('case1', 'suite_damage'), default='case1')
    model_credit=s.choices['train'].add_mutually_exclusive_group()
    model_credit.add_argument('--lookahead-credit', type=float, default=0.,
        help='COMA auxiliary assignment-damage credit weight; model-assisted training only, no planner at evaluation')
    model_credit.add_argument('--model-based-advantage', action='store_true',
        help='Enable v15 model-assisted counterfactual actor credit with weight 1.0 (training only)')
    s.choices['train'].add_argument('--deadline-mask', action='store_true',
        help='Mask stationary targets that cannot be struck before horizon; saved in checkpoint and also used at evaluation')
    s.choices['train'].add_argument('--policy-clip', type=float, default=.2)
    s.choices['train'].add_argument('--policy-temperature', type=float, default=1.)
    s.choices['train'].add_argument('--guided-training', action='store_true',
        help='Train with value/deadline-aware potential shaping and a .02 reassignment cost; no pretraining')
    s.choices['train'].add_argument('--actor-samples', choices=('all', 'events'), default='all',
        help='Actor loss samples: every decision, or start/pre-strike-completion/final transitions; decisions and GAE stay per-step')
    s.choices['train'].add_argument('--epochs', type=int, default=10,
        help='PPO epochs per collected batch (default: 10)')
    for command in ('train', 'evaluate'):
        s.choices[command].add_argument('--mode', choices=('known', 'belief'), default=None,
            help='known: historical fully observed task; belief: sensor-driven multi-object belief task')
        s.choices[command].add_argument('--commit-target', action='store_true',
            help='Keep the selected target until strike completion or invalidation, protecting approach assignments')
        s.choices[command].add_argument('--sticky-assignment', action='store_true',
            help='Prefer existing approach assignments over newcomers when resolving target conflicts')
        s.choices[command].add_argument('--decision-interval', type=int, default=None,
            help='Target decision interval in simulation steps (default: 1)')
    a=p.parse_args()
    if a.decision_interval is not None and a.decision_interval <= 0:
        p.error('--decision-interval must be positive')
    for k in ('steps','rollout_steps','episodes','n_envs','epochs'):
        if hasattr(a,k) and getattr(a,k)<=0:p.error(f'--{k} must be positive')
    if hasattr(a,'eval_interval') and a.eval_interval<0:p.error('--eval-interval must be nonnegative')
    if hasattr(a,'eval_episodes') and a.eval_episodes<=0:p.error('--eval-episodes must be positive')
    if a.command == 'train':
        if a.model_based_advantage:
            a.lookahead_credit = 1.0
        for key in ('critic_epochs', 'critic_minibatch_size'):
            if getattr(a, key) is not None and getattr(a, key) <= 0:
                p.error(f'--{key.replace("_", "-")} must be positive')
        for key in ('gamma', 'gae_lambda'):
            if getattr(a, key) is not None and not 0 < getattr(a, key) <= 1:
                p.error(f'--{key.replace("_", "-")} must be in (0, 1]')
        if a.learning_rate <= 0: p.error('--learning-rate must be positive')
        if a.minibatch_size is not None and a.minibatch_size <= 0: p.error('--minibatch-size must be positive')
        if a.entropy_coef < 0: p.error('--entropy-coef must be nonnegative')
        if not 0 < a.policy_clip < 1: p.error('--policy-clip must be in (0, 1)')
        if a.policy_temperature <= 0: p.error('--policy-temperature must be positive')
        if a.algorithm not in ('mappo', 'coma', 'compact_coma') and a.policy_temperature != 1:
            p.error('--policy-temperature is only supported for mappo and coma variants')
    a.func(a)
if __name__=='__main__':main()
