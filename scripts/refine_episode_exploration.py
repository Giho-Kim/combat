"""Model-free assignment-policy learning from one random executed assignment per episode."""
import sys,json
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
# Reuse physical families and resets only, not the previous training loops.
s=Path('scripts/probe_matched_sequential.py').read_text();exec(s[:s.index('\nOriginal=graph.GraphQBody')])
from pointmass_rl.final_mission_target_actor import TargetAllocationActorCritic
from pointmass_rl.final_mission_ppo import DecisionSchedule,_model_observation,model_from_checkpoint
out=Path('runs/critic_2x2_episode_exploration');out.mkdir(exist_ok=True);c.save(out/'config.json');torch.set_num_threads(1);torch.manual_seed(7);rng=np.random.default_rng(7);m=TargetAllocationActorCritic(2,2)
data=np.load(out/'experience.npz');O=data['observations'];A=data['actions'];R=data['returns']
m=model_from_checkpoint(out/'latest.pt',c)
x=torch.tensor(np.array(O),device='cuda');a=torch.tensor(np.array(A),device='cuda');r=torch.tensor(R,dtype=torch.float32,device='cuda');m.cuda();params=list(m.actor_body.parameters())+list(m.target_head.parameters());optim=torch.optim.Adam(params,lr=.0005);gen=torch.Generator(device='cuda').manual_seed(7)
for step in range(15000):
 idx=torch.randint(len(x),(256,),device='cuda',generator=gen);logits=m.actor_logits(x[idx]);logp=logits.log_softmax(-1);chosen=logp.gather(-1,a[idx,...,None]).squeeze(-1);weight=4*chosen.sum(-1).exp()
 # Importance-weighted return of the executed episode; behavior P(joint)=1/4.
 loss=-(weight*r[idx]).mean();optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,5);optim.step()
 if (step+1)%3000==0:print('Update',step+1,float(loss.detach()),flush=True)
m.cpu();save_checkpoint(out/'latest.pt',m,dict(episodes=4096,seed=7,method='importance_weighted_episode_return',behavior_joint_probability=.25,unselected_action_labels=False));(out/'result.json').write_text('{}')
# Same sequential neural execution used in the on-policy probe; it has no allocation rules.
s=Path('scripts/probe_sequential_assignment.py').read_text();a0=s.index('calls=0');a1=s.index('trainer.allocated_actions=sequential');exec(s[a0:a1]+'trainer.allocated_actions=sequential\n')
sys.argv=['evaluation',str(out)];src=Path('scripts/evaluate_2x2_failures.py').read_text().replace("for out in sorted(root.glob('*_seed*')):","for out in [root]:")
exec(compile(src,'evaluation','exec'))
