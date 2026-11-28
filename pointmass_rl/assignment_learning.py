"""Episodic value learning and sequential policy improvement for variable teams."""
from copy import deepcopy
from dataclasses import replace
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .env import Config,World,final_mission_action
from .graph_critic import GraphQBody
from .final_mission_target_actor import TargetAllocationActorCritic
from .final_mission_ppo import (DecisionSchedule,_model_observation,_model_critic_state,
                         resolve_device,model_tensor,save_checkpoint,evaluate_model)

@torch.no_grad()
def sequential_actions(model,worlds,schedules,deterministic=True):
    temporary=deepcopy(schedules);B=len(worlds);N=model.n_agents;T=model.n_targets
    masks=[];decisions=[]
    for w,s in zip(worlds,temporary):
        if w.t==0:s.reset()
        mask,decision=s.mask(w,model.available_actions(w));masks.append(mask);decisions.append(decision)
    masks=np.stack(masks);decisions=np.stack(decisions)
    observations=np.zeros((B,N,model.obs_dim),np.float32)
    actions=np.zeros((B,N),int);logps=np.zeros((B,N));probabilities=np.zeros((B,N,T))
    for rank in range(N):
        indices=[(rank+w.t)%N for w in worlds]
        obs=np.stack([_model_observation(model,w,s)[i] for w,s,i in zip(worlds,temporary,indices)])
        mask=np.stack([masks[j,i] for j,i in enumerate(indices)])
        action,stats=model.act(model_tensor(model,obs),deterministic=deterministic,action_mask=model_tensor(model,mask))
        for j,(w,s,i) in enumerate(zip(worlds,temporary,indices)):
            observations[j,i]=obs[j];actions[j,i]=action['target'][j]
            logps[j,i]=float(stats['logp'][j]);probabilities[j,i]=stats['probs'][j].cpu().numpy()
            if w.agent_active[i] and decisions[j,i]:
                s.targets[i]=actions[j,i];s.previous_action_probabilities[i]=probabilities[j,i]
    for j,(w,s) in enumerate(zip(worlds,schedules)):
        s.record(w,actions[j],decisions[j]&w.agent_active,probabilities[j])
    return final_mission_action(actions),[dict(obs=observations,target=actions,logp=logps,
        probs=probabilities,action_mask=masks,decision=decisions)]

@torch.no_grad()
def improve_assignments(model,state,mask,alive,rng,starts=64,sweeps=2):
    """Bounded multi-start coordinate search over learned Q, training only."""
    N,T=mask.shape
    candidates=np.zeros((starts,N),dtype=np.int64)
    for i in np.flatnonzero(alive):candidates[:,i]=rng.choice(np.flatnonzero(mask[i]),starts)
    def score(a):
        tensor=model_tensor(model,a,dtype=torch.long)
        return model.joint_q(model_tensor(model,state)[None].expand(len(a),-1),tensor).cpu().numpy()
    best=candidates[int(score(candidates).argmax())].copy()
    for _ in range(sweeps):
        for i in rng.permutation(np.flatnonzero(alive)):
            choices=np.flatnonzero(mask[i]);a=np.repeat(best[None],len(choices)+1,axis=0)
            a[1:,i]=choices;best=a[int(score(a).argmax())].copy()
    return best

@torch.no_grad()
def improve_batch(model,states,masks,alive,rng,starts,sweeps):
    B,N,T=masks.shape;device=states.device
    mask=model_tensor(model,masks);live=model_tensor(model,alive)
    candidates=torch.multinomial(mask.reshape(-1,T).float(),starts,replacement=True).reshape(B,N,starts).transpose(1,2)
    candidates=candidates.masked_fill(~live[:,None,:],0)
    def score(rows,actions):
        return torch.cat([model.joint_q(states[r],a) for r,a in zip(rows.split(256),actions.split(256))])
    rows=torch.arange(B,device=device)[:,None].expand(B,starts).reshape(-1)
    values=score(rows,candidates.reshape(-1,N)).reshape(B,starts)
    best=candidates[torch.arange(B,device=device),values.argmax(-1)].clone()
    for _ in range(sweeps):
        for i in rng.permutation(N):
            options=best[:,None,:].expand(B,T+1,N).clone()
            options[:,1:,i]=torch.arange(T,device=device)
            valid=torch.cat((torch.ones(B,1,device=device,dtype=torch.bool),mask[:,i]&live[:,i,None]),dim=1)
            b,k=valid.nonzero(as_tuple=True)
            q=torch.full((B,T+1),-torch.inf,device=device)
            q[b,k]=score(b,options[b,k]);best=options[torch.arange(B,device=device),q.argmax(-1)].clone()
    return best.cpu().numpy()

def interleaved_learning(m,a,out,c,states,actions,returns,seeds,masks,active,counts,rng,initial_observations):
    S=model_tensor(m,np.array(states));A=model_tensor(m,np.array(actions),dtype=torch.long);R=model_tensor(m,np.array(returns),dtype=torch.float32)
    split=rng.permutation(len(states));nvalid=max(1,len(split)//10) if len(split)>1 else 0
    train=split[nvalid:];valid=split[:nvalid]
    critic=list(m.q_body.parameters())+list(m.q_head.parameters());actor=list(m.actor_body.parameters())+list(m.target_head.parameters())
    oq=torch.optim.Adam(critic,lr=.0005,eps=1e-5);op=torch.optim.Adam(actor,lr=.0005,eps=1e-5)
    O=model_tensor(m,np.array(initial_observations));all_masks=np.array(masks);all_active=np.array(active)
    qdone=pdone=0;history=[]
    for round_id in range(a.rounds):
        qend=(round_id+1)*a.critic_updates//a.rounds;pend=(round_id+1)*a.actor_updates//a.rounds
        print(f'Round {round_id+1}/{a.rounds}: critic updates {qdone+1}..{qend}',flush=True)
        while qdone<qend:
            ix=model_tensor(m,rng.choice(train,a.batch_size),dtype=torch.long)
            loss=nn.functional.huber_loss(m.joint_q(S[ix],A[ix]),R[ix],delta=10)
            oq.zero_grad();loss.backward();nn.utils.clip_grad_norm_(critic,5);oq.step();qdone+=1
            if qdone%a.log_every==0 or qdone==qend:print(f'Critic {qdone}/{a.critic_updates}: loss={float(loss.detach()):.6f}',flush=True)
        # The critic is fixed during this actor block. Cache only its current-round
        # labels; actor batches always sample the full training split.
        teacher_cache=np.zeros((len(states),c.n_agents),dtype=np.int64)
        cached=np.zeros(len(states),dtype=bool)
        actor_losses=[]
        print(f'Round {round_id+1}/{a.rounds}: critic finished; actor {pend-pdone} updates from all {len(train)} training episodes',flush=True)
        while pdone<pend:
            episodes=rng.choice(train,a.batch_size)
            missing=np.unique(episodes[~cached[episodes]])
            if pdone%a.log_every==0:
                print(f'Actor update {pdone+1}/{a.actor_updates}: full-data batch, refreshing {len(missing)} assignment targets',flush=True)
            if len(missing):
                teacher_cache[missing]=improve_batch(m,S[missing],all_masks[missing],all_active[missing],rng,a.search_starts,a.search_sweeps)
                cached[missing]=True
            ix=model_tensor(m,episodes,dtype=torch.long)
            Y=model_tensor(m,teacher_cache[episodes],dtype=torch.long)
            X=O[ix].clone();live=model_tensor(m,all_active[episodes]);mask=model_tensor(m,all_masks[episodes])
            # Build exactly the sequential teacher-forced inputs, without resetting worlds.
            base=O[ix].clone();B=len(episodes);N=c.n_agents;T=c.n_targets
            probability_start=3+5*N+6*T;previous_start=probability_start+N*T+N
            probability=base[...,probability_start:probability_start+N*T].reshape(B,N,T,N)
            previous=base[...,previous_start:previous_start+N*T].reshape(B,N,T,N)
            order=model_tensor(m,np.array([rng.permutation(N) for _ in episodes]),dtype=torch.long)
            rows=torch.arange(B,device=S.device)
            for rank in range(N):
                agent=order[:,rank];X[rows,agent]=base[rows,agent]
                exists=live[rows,agent];bidx=rows[exists];drone=agent[exists];target=Y[bidx,drone]
                probability[bidx,:,:,drone]=0;probability[bidx,:,target,drone]=1
                previous[bidx,:,:,drone]=0;previous[bidx,:,target,drone]=1
            logits=m.actor_logits(X).masked_fill(~mask,-torch.inf)
            loss=nn.functional.cross_entropy(logits[live],Y[live])
            op.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor,5);op.step();pdone+=1
            actor_losses.append(float(loss.detach()))
            if pdone%a.log_every==0 or pdone==pend:
                print(f'Round {round_id+1}/{a.rounds}: actor {pdone}/{a.actor_updates}, loss={actor_losses[-1]:.6f}, distinct episodes={int(cached.sum())}',flush=True)
        with torch.no_grad():mae=float((m.joint_q(S[valid],A[valid])-R[valid]).abs().mean()) if nvalid else None
        row=dict(round=round_id+1,critic_updates=qdone,actor_updates=pdone,critic_validation_mae=mae,actor_loss=float(np.mean(actor_losses)),actor_distinct_episodes=int(cached.sum()))
        if a.eval_every and ((round_id+1)%a.eval_every==0 or round_id+1==a.rounds):
            print(f'Evaluation round {round_id+1}: {a.eval_episodes} episodes',flush=True)
            evaluation_model=deepcopy(m).cpu().eval()
            metrics=evaluate_model(evaluation_model,c,episodes=a.eval_episodes,seed=a.eval_seed,resolver=False)
            del evaluation_model
            row.update(eval_team_return=metrics['team_return'],eval_mission_success=metrics['mission_success'])
            cache=out/'evaluations';cache.mkdir(exist_ok=True)
            (cache/f'round_{round_id+1}.json').write_text(json.dumps(dict(checkpoint=f'round_{round_id+1}.pt',episodes=a.eval_episodes,seed=a.eval_seed,**metrics),indent=2))
            print(f"Eval round {round_id+1}: return={metrics['team_return']:.4f}, success={metrics['mission_success']:.3f}",flush=True)
        history.append(row);print(row,flush=True)
        m.target_q_body=deepcopy(m.q_body).requires_grad_(False);m.target_q_head=deepcopy(m.q_head).requires_grad_(False)
        save_checkpoint(out/'latest.pt',m,dict(seed=a.seed,experience_episodes=a.episodes,**row))
        if (round_id+1)%10==0:save_checkpoint(out/f'round_{round_id+1}.pt',m,dict(seed=a.seed,**row))
        (out/'training_history.json').write_text(json.dumps(history,indent=2))
    (out/'training_metadata.json').write_text(json.dumps(dict(settings=m.training_settings,count_pairs=sorted(set(counts)),deployment='sequential actor, per-step reselection',experience='fixed bank of executed persistent assignments',policy_teacher='learned Q search; fresh full-training-set batches, labels cached only within frozen-critic actor block'),indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/target_allocation_variable.json')
    p.add_argument('--out',default='runs/target_allocation_episodic')
    p.add_argument('--episodes',type=int,default=4096)
    p.add_argument('--critic-updates',type=int,default=10000)
    p.add_argument('--actor-updates',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--search-starts',type=int,default=64)
    p.add_argument('--search-sweeps',type=int,default=2)
    p.add_argument('--update-schedule',choices=['staged','interleaved'],default='staged')
    p.add_argument('--rounds',type=int,default=50)
    p.add_argument('--log-every',type=int,default=10,help='Updates between progress prints')
    p.add_argument('--eval-every',type=int,default=10,help='Interleaved rounds between evaluation; 0 disables')
    p.add_argument('--eval-episodes',type=int,default=30)
    p.add_argument('--eval-seed',type=int,default=10000)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--device',default='auto')
    a=p.parse_args()
    if min(a.episodes,a.critic_updates,a.actor_updates,a.batch_size,a.search_starts,a.search_sweeps)<1:p.error('budgets must be positive')
    if a.rounds<1 or a.log_every<1:p.error('rounds and log interval must be positive')
    if a.update_schedule=='interleaved' and min(a.critic_updates,a.actor_updates)<a.rounds:p.error('each round needs critic and actor updates')
    if a.eval_every<0 or a.eval_episodes<1:p.error('invalid evaluation budget')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('output directory must be empty')
    out.mkdir(parents=True,exist_ok=True)
    c=replace(Config.load(a.config),mode='known',commit_target=False,decision_interval=1)
    c.save(out/'config.json');device=resolve_device(a.device);torch.set_num_threads(1)
    torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    m=TargetAllocationActorCritic(c.n_targets,c.n_agents)
    m.q_body=GraphQBody(m.hidden,self_attention=False)
    m.target_q_body=deepcopy(m.q_body).requires_grad_(False)
    states=[];actions=[];returns=[];seeds=[];masks=[];active=[];counts=[];initial_observations=[]
    for ep in range(a.episodes):
        seed=a.seed*1000000+ep;w=World(c);w.reset(seed);schedule=DecisionSchedule(c)
        mask=m.available_actions(w);joint=np.zeros(c.n_agents,dtype=np.int64)
        for i in np.flatnonzero(w.agent_active):joint[i]=rng.choice(np.flatnonzero(mask[i]))
        states.append(_model_critic_state(m,w,schedule));initial_observations.append(_model_observation(m,w,schedule));actions.append(joint.copy())
        masks.append(mask);active.append(w.agent_active.copy());seeds.append(seed)
        counts.append((int(w.agent_active.sum()),int(w.target_exists.sum())))
        before=float(w.score)
        # Keep this actual assignment for the episode, including after a target vanishes.
        while not w.done:w.step(final_mission_action(joint))
        returns.append(float(w.score)-before)
        if (ep+1)%100==0:print(f'Collected {ep+1}/{a.episodes} episodes',flush=True)
    np.savez_compressed(out/'experience.npz',states=states,actions=actions,returns=returns,
        seeds=seeds,masks=masks,active=active,counts=counts,observations=initial_observations)
    m.training_settings={**vars(a),'action_selection':'sequential_current_choices','training':'episodic_value_then_policy_improvement','device':str(device),'max_grad_norm':5.,'critic_type_self_attention':False,'actor_sampling':'fresh_batch_from_full_training_split'}
    if a.update_schedule=='interleaved':
        m.to(device)
        interleaved_learning(m,a,out,c,states,actions,returns,seeds,masks,active,counts,rng,initial_observations)
        return
    m.to(device);S=model_tensor(m,np.array(states));A=model_tensor(m,np.array(actions),dtype=torch.long);R=model_tensor(m,np.array(returns),dtype=torch.float32)
    permutation=rng.permutation(a.episodes);nvalid=max(1,a.episodes//10) if a.episodes>1 else 0
    train=permutation[nvalid:];valid=permutation[:nvalid]
    critic=list(m.q_body.parameters())+list(m.q_head.parameters())
    optimizer=torch.optim.Adam(critic,lr=.0005,eps=1e-5)
    history=[]
    for step in range(a.critic_updates):
        ix=model_tensor(m,rng.choice(train,a.batch_size),dtype=torch.long)
        loss=nn.functional.huber_loss(m.joint_q(S[ix],A[ix]),R[ix],delta=10)
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(critic,5);optimizer.step()
        if (step+1)%100==0 or step+1==a.critic_updates:
            with torch.no_grad():
                mae=float((m.joint_q(S[valid],A[valid])-R[valid]).abs().mean()) if nvalid else None
            history.append(dict(update=step+1,loss=float(loss.detach()),validation_mae=mae))
            print('Critic',history[-1],flush=True)
    (out/'critic_history.json').write_text(json.dumps(history,indent=2))
    m.target_q_body=deepcopy(m.q_body).requires_grad_(False);m.target_q_head=deepcopy(m.q_head).requires_grad_(False)
    save_checkpoint(out/'critic.pt',m,dict(stage='critic',experience_episodes=a.episodes,seed=a.seed))
    teachers=[];observations=[]
    for ep,seed in enumerate(seeds):
        teacher=improve_assignments(m,states[ep],masks[ep],active[ep],rng,a.search_starts,a.search_sweeps)
        w=World(c);w.reset(seed);schedule=DecisionSchedule(c);obs=np.zeros((c.n_agents,m.obs_dim),np.float32)
        for i in rng.permutation(c.n_agents):
            obs[i]=_model_observation(m,w,schedule)[i]
            if active[ep][i]:
                schedule.targets[i]=teacher[i];schedule.previous_action_probabilities[i]=np.eye(c.n_targets)[teacher[i]]
        teachers.append(teacher);observations.append(obs)
        if (ep+1)%100==0:print(f'Policy improvement {ep+1}/{a.episodes}',flush=True)
    np.savez_compressed(out/'policy_targets.npz',actions=teachers)
    X=model_tensor(m,np.array(observations));Y=model_tensor(m,np.array(teachers),dtype=torch.long)
    M=model_tensor(m,np.array(masks));live=model_tensor(m,np.array(active))
    actor=list(m.actor_body.parameters())+list(m.target_head.parameters());optimizer=torch.optim.Adam(actor,lr=.0005,eps=1e-5)
    for step in range(a.actor_updates):
        ix=model_tensor(m,rng.choice(train,a.batch_size),dtype=torch.long)
        logits=m.actor_logits(X[ix]).masked_fill(~M[ix],-torch.inf)
        loss=nn.functional.cross_entropy(logits[live[ix]],Y[ix][live[ix]])
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor,5);optimizer.step()
        if (step+1)%100==0:print(f'Actor {step+1}: {float(loss.detach()):.6f}',flush=True)
        if (step+1)%1000==0:
            save_checkpoint(out/f'actor_{step+1}.pt',m,dict(stage='actor',actor_updates=step+1,seed=a.seed))
    m.target_q_body=deepcopy(m.q_body).requires_grad_(False);m.target_q_head=deepcopy(m.q_head).requires_grad_(False)
    m.training_settings={**vars(a),'action_selection':'sequential_current_choices','training':'episodic_value_then_policy_improvement','device':str(device),'max_grad_norm':5.,'critic_type_self_attention':False,'actor_sampling':'fresh_batch_from_full_training_split'}
    save_checkpoint(out/'latest.pt',m,dict(seed=a.seed,experience_episodes=a.episodes))
    (out/'training_metadata.json').write_text(json.dumps(dict(settings=m.training_settings,count_pairs=sorted(set(counts)),deployment='sequential neural actor; reselect every step',experience='one random persistent assignment executed per episode',policy_teacher='bounded search of learned Q; no oracle environment labels'),indent=2))
    print(f'Saved {out}/latest.pt',flush=True)

if __name__=='__main__':main()
