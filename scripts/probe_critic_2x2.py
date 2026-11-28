"""Controlled supervised critic comparison on exact fixed-assignment 2x2 returns."""
import sys,json,csv,itertools,time
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pointmass_rl.env import Config,World,final_mission_action
from pointmass_rl.final_mission_lookahead import completion_steps
from pointmass_rl.final_mission_ppo import DecisionSchedule,_model_critic_state
from pointmass_rl.final_mission_target_actor import TargetAllocationActorCritic
from pointmass_rl.graph_critic import GraphQBody
out=Path('runs/critic_2x2_probe');out.mkdir(exist_ok=True)
torch.set_num_threads(1)
c=replace(Config.load('configs/target_allocation_variable.json'),n_agents=2,min_agents=2,n_targets=2,min_targets=2,randomize_counts=False,randomize_target_composition=False,min_spawn_distance=0.)
template=TargetAllocationActorCritic(2,2);rng=np.random.default_rng(20261101);actions=np.array(list(itertools.product(range(2),repeat=2)))
states=[];labels=[];categories=[];checks=0
for k in range(4608):
 w=World(c);w.reset(k);w.t=int(rng.integers(0,81));w.targets[:]=rng.uniform(15,85,(2,2));w.pos[:]=rng.uniform(10,90,(2,2));w.target_type[:]=rng.integers(1,4,2);w.target_work_units[:]=[rng.integers(1,3) if t==1 else 1 for t in w.target_type];w.target_exists[:]=True;w.target_completed[:]=False;w.agent_active[:]=True;w.final_mission_participants[:]=False;w.agent_final_mission_progress[:]=0
 eta=completion_steps(w);feas=eta<=c.horizon-w.t;values=np.array([2.5 if t==1 else 2. if t==2 else 1. for t in w.target_type]);y=np.array([sum(values[t]*min(w.target_work_units[t],np.sum((a==t)&feas[:,t])) for t in range(2)) for a in actions],dtype=np.float32)
 if k<24:
  for a,v in zip(actions,y):
   z=deepcopy(w);before=z.score
   while not z.done:z.step(final_mission_action(a))
   assert abs(z.score-before-v)<1e-5,(k,a,v,z.score-before)
   checks+=1
 schedule=DecisionSchedule(c);states.append(_model_critic_state(template,w,schedule));labels.append(y);categories.append(dict(deadline=bool(feas.any() and not feas.all()),competition=bool(np.any(w.target_work_units==1) and feas.all(axis=0).any())))
S=torch.tensor(np.array(states),device='cuda');Y=torch.tensor(np.array(labels),device='cuda');A=torch.tensor(actions,device='cuda');results=[]
for seed in [7,8,9]:
 for added in [False,True]:
  torch.manual_seed(seed);body=GraphQBody(self_attention=added).cuda();head=nn.Linear(64,1).cuda();opt=torch.optim.Adam(list(body.parameters())+list(head.parameters()),lr=.0005);gen=torch.Generator(device='cuda').manual_seed(seed)
  for step in range(1200):
   idx=torch.randint(4096,(128,),device='cuda',generator=gen);s=S[idx,None].expand(-1,4,-1).reshape(-1,S.shape[-1]);a=A[None].expand(128,-1,-1).reshape(-1,2);pred=head(body(s,a)).reshape(-1,4);loss=nn.functional.huber_loss(pred,Y[idx],delta=10.);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(body.parameters())+list(head.parameters()),5.);opt.step()
  with torch.no_grad():
   preds=[]
   for start in range(4096,len(S),128):
    s=S[start:start+128];preds.append(head(body(s[:,None].expand(-1,4,-1).reshape(-1,S.shape[-1]),A[None].expand(len(s),-1,-1).reshape(-1,2))).reshape(-1,4))
   p=torch.cat(preds).cpu().numpy();truth=Y[4096:].cpu().numpy()
  for group in ['all','deadline','competition']:
   mask=np.array([True if group=='all' else r[group] for r in categories[4096:]]);p0=p[mask];y0=truth[mask];pairs=[]
   for i in range(4):
    for j in range(i):
     diff=y0[:,i]-y0[:,j];valid=np.abs(diff)>=.5;pairs.extend(((p0[:,i]-p0[:,j])[valid]*diff[valid]>0).tolist())
   regret=y0.max(1)-y0[np.arange(len(y0)),p0.argmax(1)];results.append(dict(seed=seed,self_attention=added,group=group,n=int(mask.sum()),pair_accuracy=float(np.mean(pairs)),optimal_choice=float(np.mean(regret==0)),regret=float(regret.mean()),mae=float(np.abs(p0-y0).mean())))
  print(seed,added,results[-3:],flush=True)
(out/'results.json').write_text(json.dumps(results,indent=2));np.savez_compressed(out/'dataset.npz',states=np.array(states),returns=np.array(labels),actions=actions)
(out/'protocol.json').write_text(json.dumps(dict(train_states=4096,test_states=512,seeds=[7,8,9],steps=1200,batch_states=128,learning_rate=.0005,grad_norm=5,validated_environment_rollouts=checks,policy='hold both assignments until termination',input='raw current critic state; no derived ETA/capacity features'),indent=2))
