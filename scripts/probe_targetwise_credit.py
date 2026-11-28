"""Observed target-return supervision; no counterfactual simulator labels."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pointmass_rl.final_mission_ppo as trainer
import pointmass_rl.graph_critic as graph
from pointmass_rl.env import World,Config
from pointmass_rl.final_mission_ppo import train_target_allocation,save_checkpoint,evaluate_model
# Reuse identical physical training mixture.
s=Path('scripts/probe_2x2_onpolicy.py').read_text();prefix=s[:s.index('Original=graph.GraphQBody')];exec(prefix.replace("root=Path('runs/critic_2x2_onpolicy_failures')","root=Path('runs/critic_2x2_targetwise_credit')"))
Original=graph.GraphQBody
class Targetwise(Original):
 def __init__(self,*args,**kwargs):
  kwargs['self_attention']=False;super().__init__(*args,**kwargs)
  self.local_readout=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,64))
  self.target_reads[-1].register_forward_hook(self.capture)
 def capture(self,module,inputs,output):self.last_targets=output
 def forward(self,state,action):
  super().forward(state,action);n,t=self.dimensions(state,action);time=self.time_encoder(state[...,:1]);h=self.last_targets;self.local_features=self.local_readout(torch.cat((h,time.unsqueeze(-2).expand_as(h)),-1));return self.local_features.sum(-2)
graph.GraphQBody=Targetwise
records=[];step_original=World.step
def track_step(w,action):
 work_units=w.target_work_units.copy();result=step_original(w,action);progress=work_units-w.target_work_units;values=np.array([2.5 if t==1 else 2. if t==2 else 1. for t in w.target_type]);records.append((progress*values,bool(result[2] or result[3])));return result
World.step=track_step
update_original=trainer._coma_update
def update(m,opts,batch,prob,epochs,width):
 count=len(batch['state']);assert len(records)==count,(len(records),count)
 reward=np.array([a for a,b in records]).reshape(-1,16,2);done=np.array([b for a,b in records]).reshape(-1,16);records.clear();returns=np.zeros_like(reward);carry=np.zeros((16,2))
 # Full episode Monte Carlo where done; bootstrap per-target predictions at cutoffs omitted.
 # Rollout=100 and horizon=100 ensure complete episodes from synchronized resets.
 for step in reversed(range(len(reward))):carry=reward[step]+(1-done[step,:,None])*carry;returns[step]=carry
 complete=np.maximum.accumulate(done[::-1],axis=0)[::-1].reshape(-1)
 labels=torch.tensor(returns.reshape(-1,2),device=batch['state'].device,dtype=torch.float32)
 critic=list(m.q_body.parameters())+list(m.q_head.parameters());losses=[]
 for _ in range(20):
  for idx in torch.tensor(np.flatnonzero(complete),device=labels.device)[torch.randperm(int(complete.sum()),device=labels.device)].split(1024):
   m.joint_q(batch['state'][idx],batch['joint_action'][idx]);q=m.q_head(m.q_body.local_features).squeeze(-1)-m.q_head.bias.squeeze()*.5
   loss=nn.functional.huber_loss(q,labels[idx],delta=10);opts[1].zero_grad();loss.backward();nn.utils.clip_grad_norm_(critic,5);opts[1].step();m.update_target(.01);losses.append(float(loss.detach()))
 with torch.no_grad():
  adv=torch.cat([m.counterfactual_advantages(batch['state'][i:i+256],batch['joint_action'][i:i+256],prob[i:i+256],sampled_returns=batch['returns'][i:i+256]) for i in range(0,count,256)]).flatten()
 b=dict(batch,advantage=adv);metrics=trainer._ppo_update(m,opts,b,epochs,width);metrics['q_loss']=float(np.mean(losses));return metrics
trainer._coma_update=update
torch.set_num_threads(1);out=root/'targetwise_v2_seed7';out.mkdir(exist_ok=True);c.save(out/'config.json')
m,ep,ev,actual=train_target_allocation(c,100000,seed=7,n_envs=16,rollout_steps=100,epochs=10,critic_epochs=20,minibatch_size=1024,critic_minibatch_size=1024,eval_interval=0,device='cuda',diagnostics_path=out/'training_updates.csv')
save_checkpoint(out/'latest.pt',m,dict(agent_transitions=actual,seed=7));m.cpu();World.step=step_original
metrics=evaluate_model(m,c,episodes=100,seed=20000,resolver=False);(out/'result.json').write_text(json.dumps(metrics,indent=2));print('RESULT',metrics,flush=True)
sys.argv=['evaluate',str(root)];exec(compile(Path('scripts/evaluate_2x2_failures.py').read_text(),'evaluation','exec'))
