"""On-policy PPO/COMA over persistent episode assignments."""
import csv
from pathlib import Path
import numpy as np
import torch
from .env import World,strike_action
from .strike_target_actor import TargetAllocationActorCritic
from .strike_ppo import (DecisionSchedule,_model_critic_state,_batch,_coma_update,
    _optimizers,model_tensor,resolve_device,model_from_checkpoint,allocated_actions,
    evaluate_model,save_checkpoint)

def train(config,total,seed,*,episodes_per_update=256,n_envs=32,epochs=10,
          critic_epochs=20,minibatch_size=256,critic_minibatch_size=256,
          learning_rate=.0005,entropy_coef=.01,policy_clip=.2,coma_advantage='q',
          eval_interval=100000,eval_episodes=30,eval_seed=10000,out=None,
          initial_checkpoint=None,start_transitions=0,previous_best_key=None,device='auto'):
    if min(total,episodes_per_update,n_envs,epochs,critic_epochs or epochs)<1:
        raise ValueError('training budgets must be positive')
    if config.commit_target or config.decision_interval != 1:
        raise ValueError('Save deployment config with commit_target=False and decision_interval=1')
    rng=np.random.default_rng(seed);torch.manual_seed(seed);device=resolve_device(device)
    if initial_checkpoint:
        model=model_from_checkpoint(initial_checkpoint,config,device)
    else:
        model=TargetAllocationActorCritic(config.n_targets,config.n_agents)
        model.to(device)
    # Factorized sampling preserves the original COMA counterfactual baseline.
    model.training_settings=dict(training='on_policy_persistent_assignment_ppo',
        action_selection='simultaneous',episodes_per_update=episodes_per_update,
        epochs=epochs,critic_epochs=critic_epochs or 2*epochs,
        critic_minibatch_size=critic_minibatch_size,learning_rate=learning_rate,
        policy_clipping=policy_clip,entropy_coef=entropy_coef,coma_advantage=coma_advantage,
        max_grad_norm=5.,target_q_polyak_tau=.01,critic_before_actor=True,
        q_target='observed_complete_episode_return',actor_update='ppo_clip_with_coma_advantage',
        critic_type_self_attention=model.q_body.drone_self is not None,
        deployment='reselect_each_step',device=str(device),n_envs=n_envs,
        initial_checkpoint=str(initial_checkpoint) if initial_checkpoint else None,
        optimizer_reinitialized=initial_checkpoint is not None)
    optimizers=_optimizers(model,learning_rate);completed=start_transitions;end=completed+total
    episode_rows=[];evaluation_rows=[];best=previous_best_key or (-float('inf'),-float('inf'))
    next_eval=(completed//eval_interval+1)*eval_interval if eval_interval>0 else None
    next_checkpoint=(completed//100000+1)*100000
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    def evaluate():
        nonlocal best
        for enabled in [False,True]:
            metrics=evaluate_model(model,config,episodes=eval_episodes,seed=eval_seed,resolver=enabled)
            evaluation_rows.append(dict(agent_transitions=completed,policy='target_allocation_'+('resolver' if enabled else 'no_resolver'),episodes=eval_episodes,seed=eval_seed,**metrics))
            print(f'Eval @ {completed:,} resolver={enabled} return={metrics["team_return"]:.4f}',flush=True)
            if enabled and (metrics['team_return'],metrics['discounted_team_return'])>best:
                best=(metrics['team_return'],metrics['discounted_team_return'])
                save_checkpoint(out/'best.pt',model,dict(agent_transitions=completed,best_metric='mean_team_return_then_discounted_return',best_value=best[0],best_discounted_value=best[1],best_success=metrics['mission_success']))
        save_checkpoint(out/'latest.pt',model,dict(agent_transitions=completed,seed=seed))
    if next_eval is not None and completed==0:evaluate()
    while completed<end:
        obs=[];states=[];actions=[];logps=[];masks=[];active=[];probabilities=[];returns=[]
        for start in range(0,episodes_per_update,n_envs):
            worlds=[World(config) for _ in range(min(n_envs,episodes_per_update-start))]
            for w in worlds:w.reset(int(rng.integers(0,2**31-1)))
            schedules=[DecisionSchedule(config) for _ in worlds]
            state=np.stack([_model_critic_state(model,w,s) for w,s in zip(worlds,schedules)])
            alive=np.stack([w.agent_active.copy() for w in worlds])
            with torch.no_grad():assignment,traces=allocated_actions(model,worlds,schedules,deterministic=False)
            trace=traces[0]
            obs.extend(trace['obs']);states.extend(state);actions.extend(trace['target'])
            logps.extend(trace['logp']);masks.extend(trace['action_mask']);active.extend(alive)
            probabilities.extend(trace['probs'])
            for j,w in enumerate(worlds):
                reward=0.;elapsed=0
                while not w.done:
                    count=int(w.agent_active.sum());_,r,_,_,_=w.step(strike_action(assignment['target'][j]));completed+=count
                    reward+=(config.discount_gamma**elapsed)*float(np.sum(r));elapsed+=1
                returns.append(reward);episode_rows.append(dict(agent_transitions=completed,**w.metrics()))
        # The actor has remained frozen throughout all complete episodes above.
        batch=_batch(obs,states,actions,logps,masks,np.zeros_like(active,dtype=np.float32),returns,active,active,np.zeros(len(states)))
        batch={k:v.to(device) for k,v in batch.items()}
        metrics=_coma_update(model,optimizers,batch,model_tensor(model,np.asarray(probabilities),dtype=torch.float32),epochs,minibatch_size)
        metrics.update(agent_transitions=completed,episodes=len(states),episode_return=float(np.mean(returns)))
        path=out/'training_updates.csv';header=not path.exists()
        with path.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(metrics));
            if header:writer.writeheader()
            writer.writerow(metrics)
        print(f'PPO update @ {completed:,}: {metrics}',flush=True)
        while completed>=next_checkpoint:
            save_checkpoint(out/f'checkpoint_{next_checkpoint}.pt',model,dict(agent_transitions=completed,checkpoint_transition=next_checkpoint,seed=seed));next_checkpoint+=100000
        if next_eval is not None and completed>=next_eval:
            evaluate()
            while next_eval<=completed:next_eval+=eval_interval
        save_checkpoint(out/'latest.pt',model,dict(agent_transitions=completed,seed=seed))
    return model,episode_rows,evaluation_rows,completed
