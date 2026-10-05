"""COMA/PPO for the belief-mode game; a single shared encoder optimizer.

Q warm-up stops gradients at the shared encoder. The subsequent PPO stage
jointly optimizes policy loss and Q loss with one optimizer, so Q warm-up
cannot silently alter the behavior policy through shared representation.
The target copy includes the belief encoder and receives Polyak updates.
"""
import csv
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .belief_env import BeliefWorld
from .belief_policy import BeliefActorCritic, choose_action, stack_observations, subset
from .env import Config, strike_action
from .strike_ppo import _td_lambda_returns


def save_belief_checkpoint(path, model, target, config, metadata, optimizer=None):
    payload = dict(architecture=model.architecture, hidden=model.hidden, mode='belief',
                   config=asdict(config), state_dict=model.state_dict(),
                   target_state_dict=target.state_dict(), metadata=metadata,
                   training_settings=dict(shared_encoder=True, shared_optimizer=True,
                       critic_truth_access=False, update='Q warm-up with detached encoder; joint PPO/Q',
                       belief='histogram unseen intensity + identified Gaussian tracks'))
    if optimizer is not None:
        payload['optimizer_state_dict']=optimizer.state_dict()
    torch.save(payload,Path(path))


def load_belief_checkpoint(path, config=None):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('architecture')!=BeliefActorCritic.architecture:
        raise ValueError('Belief mode requires a belief checkpoint; legacy v15 uses --mode known')
    saved=Config(**payload['config'])
    config=config or saved
    if config.mode!='belief':
        raise ValueError('Belief checkpoint requires --mode belief')
    model=BeliefActorCritic(payload['hidden'])
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model,config,payload


def baseline_action(obs, name, rng):
    mask=obs['mask']
    action=[]
    for i in range(len(mask)):
        valid=np.flatnonzero(mask[i])
        if name=='random':
            choice=rng.choice(valid)
        else:
            distance=np.linalg.norm(obs['beliefs'][valid,12:14]-obs['drones'][i,:2],axis=-1)
            if name=='type_priority':
                value=obs['beliefs'][valid,5:8] @ np.array([5.,2.,1.])
                # Same public belief for all baselines; no oracle access.
                choice=valid[np.lexsort((distance,-value))[0]]
            else:
                choice=valid[np.argmin(distance)]
        action.append(int(choice))
    return np.array(action),np.eye(mask.shape[1],dtype=np.float32)[action]


def evaluate_belief(model, config, episodes=30, seed=10000, resolver='none', baseline=None,
                    replay_frames=None):
    rows=[]
    for episode in range(episodes):
        world=BeliefWorld(config)
        obs=world.reset(seed+episode)
        recording = replay_frames is not None and episode == episodes-1
        if recording:
            from .belief_report import frame
            replay_frames.append(frame(world,obs))
        rng=np.random.default_rng(seed+episode)
        while not world.done:
            action,prob=(baseline_action(obs,baseline,rng) if baseline else
                         choose_action(model,obs,deterministic=True,resolver=resolver))
            obs,_,_,_,metrics=world.step(strike_action(action),prob)
            if recording:
                replay_frames.append(frame(world,obs,action))
        rows.append(dict(seed=seed+episode,**metrics))
    keys=('team_return','discounted_team_return','mission_success','discovery_fraction','discovered_objects')
    return {key:float(np.mean([r[key] for r in rows])) for key in keys},rows


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def update_model(model, target, optimizer, batch, action, old_logp, probs, returns,
                 epochs, critic_epochs, width, entropy_coef, clip, coma_advantage='q'):
    q_losses=[]
    size=len(action)
    for _ in range(critic_epochs):
        for ix in torch.randperm(size).split(width):
            loss=nn.functional.huber_loss(model.joint_q(subset(batch,ix),action[ix],detach_encoder=True),returns[ix],delta=10.)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step()
            target.soft_update(model)
            q_losses.append(float(loss.detach()))
    with torch.no_grad():
        advantage=torch.cat([model.advantages(subset(batch,slice(i,i+32)),action[i:i+32],probs[i:i+32],
                        returns[i:i+32] if coma_advantage=='sampled_return' else None)
                             for i in range(0,size,32)])
        chosen=advantage[batch['decision']]
        if len(chosen):
            advantage=(advantage-chosen.mean())/(chosen.std(unbiased=False)+1e-5)
    actor_losses=[]
    for _ in range(epochs):
        for ix in torch.randperm(size).split(width):
            obs=subset(batch,ix)
            decision=obs['decision']
            if not decision.any():
                continue
            distribution=model.distribution(obs)
            ratio=(distribution.log_prob(action[ix])-old_logp[ix]).exp()
            surrogate=torch.minimum(ratio*advantage[ix],ratio.clamp(1-clip,1+clip)*advantage[ix])
            actor_loss=-surrogate[decision].mean()-entropy_coef*distribution.entropy()[decision].mean()
            q_loss=nn.functional.huber_loss(model.joint_q(obs,action[ix]),returns[ix],delta=10.)
            # One Adam owns the shared parameters; both losses train the same encoder.
            optimizer.zero_grad(set_to_none=True)
            (actor_loss+.5*q_loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step()
            target.soft_update(model)
            actor_losses.append(float(actor_loss.detach()))
    return dict(q_loss=float(np.mean(q_losses)) if q_losses else 0.,
                actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.)


def train_belief(config, total_steps, out, seed=7, n_envs=8, rollout_steps=64,
                 epochs=4, critic_epochs=4, minibatch_size=32, learning_rate=5e-4,
                 entropy_coef=.01, policy_clip=.2, eval_interval=100000,
                 eval_episodes=30, eval_seed=10000, checkpoint_interval=100000,
                 coma_advantage='q'):
    if min(total_steps,n_envs,rollout_steps,epochs,critic_epochs,minibatch_size,checkpoint_interval,eval_episodes)<=0:
        raise ValueError('Training and evaluation sizes must be positive')
    if eval_interval<0 or learning_rate<=0 or entropy_coef<0 or not 0<policy_clip<1:
        raise ValueError('Invalid optimization or evaluation settings')
    if coma_advantage not in ('q','sampled_return'):
        raise ValueError('Invalid COMA advantage')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    out=Path(out)
    if (out/'config.json').exists():
        raise ValueError(f'Existing experiment: {out}; choose a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    config.save(out/'config.json')
    model=BeliefActorCritic()
    target=deepcopy(model).requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,eps=1e-5)
    worlds=[BeliefWorld(config) for _ in range(n_envs)]
    obs=[w.reset(int(rng.integers(2**31))) for w in worlds]
    completed=0
    next_eval=eval_interval if eval_interval else None
    next_checkpoint=checkpoint_interval
    episode_rows,evaluation_rows,update_rows=[],[],[]
    best=(-float('inf'),-float('inf'))
    metadata=dict(seed=seed,mode='belief',agent_transitions=0,
                  training_distribution='world.reset(seed)',n_envs=n_envs,rollout_steps=rollout_steps,
                  epochs=epochs,critic_epochs=critic_epochs,minibatch_size=minibatch_size,
                  learning_rate=learning_rate,entropy_coef=entropy_coef,policy_clip=policy_clip,
                  coma_advantage=coma_advantage,eval_seed=eval_seed,eval_episodes=eval_episodes,
                  checkpoint_kind='weights/target/optimizer; not an exact environment/RNG resume')

    def evaluate_at_current():
        nonlocal best
        for resolver in ('none','score','distance'):
            summary,_=evaluate_belief(model,config,eval_episodes,eval_seed,resolver)
            evaluation_rows.append(dict(agent_transitions=completed,policy=resolver,**summary))
            print(f'Eval {completed} [{resolver}]: {summary}',flush=True)
            if resolver=='none':
                key=(summary['team_return'],summary['mission_success'])
                if key>best:
                    best=key
                    save_belief_checkpoint(out/'best.pt',model,target,config,metadata,optimizer)
        _write_csv(out/'training_evaluations.csv',evaluation_rows)

    if next_eval is not None:
        evaluate_at_current()
    while completed<total_steps:
        stored,actions,logps,probabilities,values,rewards,dones=[],[],[],[],[],[],[]
        for _ in range(rollout_steps):
            batch=stack_observations(obs)
            with torch.no_grad():
                distribution=model.distribution(batch)
                action=distribution.sample()
                values.append(target.joint_q(batch,action).numpy())
            stored.append(batch)
            actions.append(action)
            logps.append(distribution.log_prob(action).detach())
            probabilities.append(distribution.probs.detach())
            step_reward,step_done=[],[]
            active_count=int(batch['active'].sum())
            completed+=active_count
            for i,world in enumerate(worlds):
                obs[i],reward,terminated,truncated,metrics=world.step(strike_action(action[i].numpy()),probabilities[-1][i].numpy())
                done=terminated or truncated
                step_reward.append(reward)
                step_done.append(done)
                if done:
                    episode_rows.append(dict(agent_transitions=completed,**metrics))
                    obs[i]=world.reset(int(rng.integers(2**31)))
            rewards.append(step_reward)
            dones.append(step_done)
            if completed>=total_steps:
                break
        with torch.no_grad():
            boundary=stack_observations(obs)
            boundary_action=model.distribution(boundary).sample()
            boundary_q=target.joint_q(boundary,boundary_action).numpy()
        next_values=np.concatenate((np.asarray(values)[1:],boundary_q[None]),axis=0)
        returns=_td_lambda_returns(np.asarray(rewards,dtype=np.float32),next_values,
                                    np.asarray(dones,dtype=np.float32),config.discount_gamma,config.gae_lambda)
        batch={k:torch.cat([o[k] for o in stored]) for k in stored[0]}
        diagnostics=update_model(model,target,optimizer,batch,torch.cat(actions),torch.cat(logps),
                                 torch.cat(probabilities),torch.from_numpy(returns.flatten()),
                                 epochs,critic_epochs,minibatch_size,entropy_coef,policy_clip,coma_advantage)
        metadata['agent_transitions']=completed
        update_rows.append(dict(agent_transitions=completed,**diagnostics))
        print(f'Update {completed}/{total_steps}: {diagnostics}',flush=True)
        _write_csv(out/'training_updates.csv',update_rows)
        _write_csv(out/'training_episodes.csv',episode_rows)
        if completed>=next_checkpoint:
            # Filename uses actual collected transitions, never a nominal
            # threshold whose weights were collected later.
            save_belief_checkpoint(out/f'checkpoint_{completed}.pt',model,target,config,metadata,optimizer)
            while next_checkpoint<=completed:
                next_checkpoint+=checkpoint_interval
        if next_eval is not None and completed>=next_eval:
            evaluate_at_current()
            while next_eval<=completed:
                next_eval+=eval_interval
        save_belief_checkpoint(out/'latest.pt',model,target,config,metadata,optimizer)
    save_belief_checkpoint(out/'belief_target_coma.pt',model,target,config,metadata,optimizer)
    (out/'training_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return model,episode_rows,evaluation_rows,completed


def command_train(args, config):
    if args.algorithm!='target_coma':
        raise ValueError('Belief mode uses --algorithm target_coma')
    if (args.guided_training or args.lookahead_credit or args.deadline_mask or args.actor_samples!='all'
            or args.target_coma_critic!='graph' or args.target_coma_actor!='attention'
            or args.policy_temperature!=1 or config.sticky_assignment):
        raise ValueError('Belief mode requires graph/attention, all decision samples, and no legacy execution interventions')
    if args.critic_minibatch_size is not None and args.critic_minibatch_size!=args.minibatch_size:
        raise ValueError('Belief mode uses a shared --minibatch-size in environment steps')
    return train_belief(config,args.steps,args.out,seed=args.seed,n_envs=args.n_envs,
            rollout_steps=args.rollout_steps,epochs=args.epochs,critic_epochs=args.critic_epochs or args.epochs,
            minibatch_size=args.minibatch_size or 32,learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,policy_clip=args.policy_clip,eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,eval_seed=args.eval_seed,coma_advantage=args.coma_advantage)


def command_evaluate(args, config):
    torch.set_num_threads(1)
    model,config,_=load_belief_checkpoint(args.model,config) if args.model else (None,config,None)
    summaries,all_rows,runs={},[],{}
    for name in ('random','nearest','type_priority') + (('none','score','distance') if model else ()):
        baseline=name if name in ('random','nearest','type_priority') else None
        frames=[]
        summary,rows=evaluate_belief(model,config,args.episodes,args.seed,
                                    resolver=name if baseline is None else 'none',baseline=baseline,
                                    replay_frames=frames)
        summaries[name]=summary
        runs[name]=frames
        all_rows.extend(dict(policy=name,**r) for r in rows)
        print(f'{name}: {summary}',flush=True)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    config.save(out/'config.json')
    _write_csv(out/'episodes.csv',all_rows)
    from .belief_report import replay
    replay(out/'replay.html',config,runs,args.seed+args.episodes-1)
    (out/'summary.json').write_text(json.dumps(dict(mode='belief',seed=args.seed,episodes=args.episodes,
        movement='dubins',lawnmower=config.belief_lawnmower,
        distribution='world.reset(seed)',summaries=summaries),indent=2)+'\n')
