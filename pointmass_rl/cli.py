import argparse, csv, json, platform
from dataclasses import asdict
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from .env import Config, World, split_strike_action
from .policies import RandomPolicy, HeuristicPolicy, StrikeMAPPOPolicy
from .report import replay, plot

def make_policy(name, config, seed, model=None):
    if name=='random': return RandomPolicy(config,seed)
    if name=='heuristic': return HeuristicPolicy(config,seed)
    if name=='mappo': return StrikeMAPPOPolicy(model,config)
    raise ValueError(name)

def rollout(config, policy, seed, record=False):
    if hasattr(policy,'reset'):policy.reset()
    world=World(config);obs=world.reset(seed);frames=[world.snapshot()] if record else [];decisions=[]
    while not world.done:
        action=policy.predict(obs)
        if record:
            targets=split_strike_action(action)
            targets=world.committed_targets(targets)
            goals=world.resolve_setpoints(targets)
            for i,(target,goal) in enumerate(zip(targets,goals)):
                decisions.append(dict(step=world.t,drone=i,target_id=int(target),
                    target_type=int(world.mem_type[i,target]),target_life=int(world.mem_life[i,target]),
                    target_score=int(world.target_score[target]),
                    estimated_distance=float(np.linalg.norm(world.mem_pos[i,target]-world.perceived_pos[i])),
                    goal_x=float(goal[0]),goal_y=float(goal[1]),active=bool(world.agent_active[i])))
        obs,_,_,_,_=world.step(action)
        if record:frames.append(world.snapshot())
    return world.metrics(),frames,decisions

def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def interquartile_mean(values):
    values=np.sort(np.asarray(values,dtype=float))
    if not len(values):return None
    lower,upper=.25*len(values),.75*len(values)
    weights=np.array([max(0.,min(i+1,upper)-max(i,lower)) for i in range(len(values))])
    return float(np.sum(values*weights)/np.sum(weights))

def train(args):
    try:
        import torch
        from .strike_ppo import train_mappo,save_checkpoint
    except ImportError as e: raise SystemExit(str(e)) from e
    c=Config.load(args.config);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);c.save(out/'config.json')
    print(f'Training Strike MAPPO: {args.steps:,} agent transitions',flush=True)
    model,episodes,evaluations,actual=train_mappo(
        c,args.steps,args.seed,args.rollout_steps,progress=True,
        eval_interval=args.eval_interval,eval_episodes=args.eval_episodes,eval_seed=args.eval_seed,
        best_path=out/'best.pt',latest_path=out/'latest.pt',checkpoint_dir=out)
    metadata=dict(seed=args.seed,requested_agent_transitions=args.steps,actual_agent_transitions=actual,
                  eval_interval=args.eval_interval,eval_episodes=args.eval_episodes,eval_seed=args.eval_seed,
                  config=asdict(c),python=platform.python_version(),numpy=str(np.__version__),torch=str(torch.__version__),
                  algorithm='CTDE Strike MAPPO: local actor logits + capacity-aware joint resolver + centralized critic')
    save_checkpoint(out/'strike_mappo.pt',model,metadata);write_csv(out/'training_episodes.csv',episodes)
    save_checkpoint(out/'latest.pt',model,metadata)
    if not evaluations:
        save_checkpoint(out/'best.pt',model,dict(metadata, best_metric='final_model'))
    write_csv(out/'training_evaluations.csv',evaluations)
    (out/'training_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Saved {out / "strike_mappo.pt"}',flush=True)

def evaluate(args):
    c=Config.load(args.config);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);c.save(out/'config.json')
    names=['random','heuristic']+(['mappo'] if args.model else []);rows=[];runs={};summaries={};decisions=[]
    progress=tqdm(total=len(names)*args.episodes,desc='Evaluate',unit='episode',dynamic_ncols=True)
    for name in names:
        fixed=make_policy(name,c,args.seed,args.model) if name=='mappo' else None
        for k in range(args.episodes):
            policy=fixed or make_policy(name,c,args.seed+k,args.model)
            metrics,frames,log=rollout(c,policy,args.seed+k,record=k==0)
            rows.append(dict(policy=name,seed=args.seed+k,**metrics))
            if k==0:runs[name]=frames;decisions.extend(dict(policy=name,**x) for x in log)
            progress.update(1);progress.set_postfix(policy=name,return_=f"{metrics['team_return']:.2f}",
                                                    score=f"{metrics['score']}/{metrics['baseline_score']}",
                                                    success=int(metrics['mission_success']))
        local=[r for r in rows if r['policy']==name];summaries[name]={}
        for key in metrics:
            if key == 'agent_terminated':
                continue
            vals=np.array([r[key] for r in local if r[key] is not None],float)
            sd=float(vals.std(ddof=1)) if len(vals)>1 else 0.
            summaries[name][key]=dict(iqm=interquartile_mean(vals),
                mean=float(vals.mean()) if len(vals) else None,std=sd,
                ci95_halfwidth=1.96*sd/np.sqrt(len(vals)) if len(vals) else None,n=len(vals))
    progress.close()
    (out/'summary.json').write_text(json.dumps(dict(episodes=args.episodes,summary=summaries),indent=2)+'\n')
    write_csv(out/'episodes.csv',rows);write_csv(out/'decisions.csv',decisions);replay(out/'replay.html',c,runs);plot(out/'comparison.png',summaries)

def main():
    p=argparse.ArgumentParser(description='Strike-priority point-mass reinforcement learning');s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('train');q.add_argument('--steps',type=int,default=200_000);q.add_argument('--rollout-steps',type=int,default=256);q.add_argument('--seed',type=int,default=7);q.add_argument('--out',default='runs/train');q.add_argument('--config');q.add_argument('--eval-interval',type=int,default=10_000,help='agent transitions between held-out evaluations; 0 disables evaluation');q.add_argument('--eval-episodes',type=int,default=5);q.add_argument('--eval-seed',type=int,default=10_000);q.set_defaults(func=train)
    q=s.add_parser('evaluate');q.add_argument('--model');q.add_argument('--episodes',type=int,default=30);q.add_argument('--seed',type=int,default=10000);q.add_argument('--out',default='runs/eval');q.add_argument('--config');q.set_defaults(func=evaluate)
    a=p.parse_args()
    for k in ('steps','rollout_steps','episodes'):
        if hasattr(a,k) and getattr(a,k)<=0:p.error(f'--{k} must be positive')
    if hasattr(a,'eval_interval') and a.eval_interval<0:p.error('--eval-interval must be nonnegative')
    if hasattr(a,'eval_episodes') and a.eval_episodes<=0:p.error('--eval-episodes must be positive')
    a.func(a)
if __name__=='__main__':main()
