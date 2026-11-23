"""Learn assignment values from executed episodes, then improve a neural policy."""
import sys,json,itertools
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
s=Path('scripts/probe_matched_sequential.py').read_text();exec(s[:s.index('\nOriginal=graph.GraphQBody')])
from pointmass_rl.strike_target_actor import TargetAllocationActorCritic
from pointmass_rl.graph_critic import GraphQBody
from pointmass_rl.strike_ppo import _model_critic_state,_model_observation,DecisionSchedule
out=Path('runs/critic_2x2_learned_policy_improvement');out.mkdir(exist_ok=True);c.save(out/'config.json');torch.set_num_threads(1);torch.manual_seed(7)
data=np.load('runs/critic_2x2_episode_exploration/experience.npz');actions=torch.tensor(data['actions'],device='cuda');returns=torch.tensor(data['returns'],dtype=torch.float32,device='cuda');m=TargetAllocationActorCritic(2,2);states=[];initial=[];worlds=[]
for i in range(len(actions)):
 w=World(c);w.reset(30000+i);sched=DecisionSchedule(c);states.append(_model_critic_state(m,w,sched));initial.append(_model_observation(m,w,sched));worlds.append(w)
states=torch.tensor(np.array(states),device='cuda');m.q_body=GraphQBody(self_attention=False);m.cuda();critic=list(m.q_body.parameters())+list(m.q_head.parameters());opt=torch.optim.Adam(critic,lr=.0005);gen=torch.Generator(device='cuda').manual_seed(7)
for step in range(5000):
 idx=torch.randint(len(states),(256,),device='cuda',generator=gen);q=m.joint_q(states[idx],actions[idx]);loss=nn.functional.huber_loss(q,returns[idx],delta=10);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(critic,5);opt.step()
 if (step+1)%1000==0:print('Critic',step+1,float(loss.detach()),flush=True)
alternatives=torch.tensor(list(itertools.product(range(2),repeat=2)),device='cuda');teachers=[]
with torch.no_grad():
 for batch in states.split(128):
  q=m.joint_q(batch[:,None].expand(-1,4,-1).reshape(-1,states.shape[-1]),alternatives[None].expand(len(batch),-1,-1).reshape(-1,2)).reshape(-1,4);teachers.extend(alternatives[q.argmax(-1)].cpu().numpy())
O=[]
for i,(w,choice) in enumerate(zip(worlds,teachers)):
 schedule=DecisionSchedule(c);obs=np.zeros((2,m.obs_dim),np.float32)
 for j in ([0,1] if i%2 else [1,0]):
  obs[j]=_model_observation(m,w,schedule)[j];schedule.targets[j]=choice[j];schedule.previous_action_probabilities[j]=np.eye(2)[choice[j]]
 O.append(obs)
x=torch.tensor(np.array(O),device='cuda');teacher=torch.tensor(np.array(teachers),device='cuda');actor=list(m.actor_body.parameters())+list(m.target_head.parameters());opt=torch.optim.Adam(actor,lr=.0005)
for step in range(4000):
 idx=torch.randint(len(x),(256,),device='cuda',generator=gen);logits=m.actor_logits(x[idx]);loss=nn.functional.cross_entropy(logits.reshape(-1,2),teacher[idx].flatten());opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor,5);opt.step()
 if (step+1)%1000==0:print('Actor',step+1,float(loss.detach()),flush=True)
m.target_q_body=deepcopy(m.q_body).requires_grad_(False);m.target_q_head=deepcopy(m.q_head).requires_grad_(False)
m.cpu();save_checkpoint(out/'latest.pt',m,dict(seed=7,experience_episodes=4096,training_labels='executed episode return only; actor labels from learned critic',oracle_labels=False));(out/'result.json').write_text('{}')
# Reconstruct the saved critic topology when evaluating.
import pointmass_rl.graph_critic as graph
graph.GraphQBody=lambda *a,**kw:GraphQBody(*a,**dict(kw,self_attention=False))
s=Path('scripts/probe_sequential_assignment.py').read_text();a=s.index('calls=0');b=s.index('trainer.allocated_actions=sequential');exec(s[a:b]+'trainer.allocated_actions=sequential\n')
sys.argv=['evaluation',str(out)];src=Path('scripts/evaluate_2x2_failures.py').read_text().replace("for out in sorted(root.glob('*_seed*')):","for out in [root]:");exec(compile(src,'evaluation','exec'))
