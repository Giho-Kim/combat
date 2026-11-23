"""2x2 factorial ablation within the successful learned-Q distillation experiment."""
import sys,json
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
s=Path('scripts/probe_matched_sequential.py').read_text();exec(s[:s.index('\nOriginal=graph.GraphQBody')])
from pointmass_rl.strike_target_actor import TargetAllocationActorCritic
from pointmass_rl.graph_critic import GraphQBody
from pointmass_rl.strike_ppo import allocated_actions as simultaneous,_model_critic_state,_model_observation,save_checkpoint
from pointmass_rl.assignment_learning import sequential_actions
root=Path('runs/critic_2x2_factorial');root.mkdir(exist_ok=True);c=replace(c,commit_target=False,decision_interval=1);torch.set_num_threads(1)
data=np.load('runs/critic_2x2_episode_exploration/experience.npz');A=np.array(data['actions']);Rfixed=np.array(data['returns'],np.float32)
torch.manual_seed(7);template=TargetAllocationActorCritic(2,2);states=[];observations=[]
for i in range(len(A)):
 w=World(c);w.reset(30000+i);schedule=DecisionSchedule(c);states.append(_model_critic_state(template,w,schedule));observations.append(_model_observation(template,w,schedule))
np.savez_compressed(root/'initial_states.npz',states=states,observations=observations,actions=A,held_returns=Rfixed)
# Frozen initial actor is the common continuation policy in the dynamic-data arm.
# Each initial state and initial executed assignment matches the held-data arm.
if (root/'dynamic_returns.npy').exists():Rdynamic=np.load(root/'dynamic_returns.npy')
else:
 behavior=template.cuda().eval();torch.manual_seed(7001);Rdynamic=[]
 for start in range(0,len(A),64):
  worlds=[];schedules=[]
  for i in range(start,min(start+64,len(A))):
   w=World(c);w.reset(30000+i);schedule=DecisionSchedule(c);schedule.record(w,A[i],w.agent_active,np.eye(2)[A[i]]);w.step(strike_action(A[i]));worlds.append(w);schedules.append(schedule)
  while not all(w.done for w in worlds):
   live=[i for i,w in enumerate(worlds) if not w.done]
   with torch.no_grad():action,_=simultaneous(behavior,[worlds[i] for i in live],[schedules[i] for i in live],deterministic=False)
   for j,i in enumerate(live):worlds[i].step(strike_action(action['target'][j]))
  Rdynamic.extend(float(w.score) for w in worlds)
  if (start+64)%512==0:print('Dynamic episodes',min(start+64,len(A)),flush=True)
 Rdynamic=np.array(Rdynamic,np.float32);np.save(root/'dynamic_returns.npy',Rdynamic)
S=torch.tensor(np.array(states),device='cuda');O=torch.tensor(np.array(observations),device='cuda');AA=torch.tensor(A,device='cuda');N=len(S)
qix=torch.randint(N,(5000,256),generator=torch.Generator().manual_seed(17)).cuda();pix=torch.randint(N,(4000,256),generator=torch.Generator().manual_seed(27)).cuda();alternatives=torch.tensor([[0,0],[0,1],[1,0],[1,1]],device='cuda')
for mode,R in [('held',Rfixed),('dynamic',Rdynamic)]:
 for schedule in ['staged','interleaved']:
  out=root/f'{mode}_{schedule}_seed7';out.mkdir(exist_ok=True);c.save(out/'config.json')
  if (out/'latest.pt').exists():continue
  torch.manual_seed(7);m=TargetAllocationActorCritic(2,2);m.q_body=GraphQBody(self_attention=False);m.target_q_body=deepcopy(m.q_body).requires_grad_(False);m.cuda();YY=torch.tensor(R,device='cuda');critic=list(m.q_body.parameters())+list(m.q_head.parameters());actor=list(m.actor_body.parameters())+list(m.target_head.parameters());oq=torch.optim.Adam(critic,lr=.0005);op=torch.optim.Adam(actor,lr=.0005)
  def qstep(step):
   ix=qix[step];loss=nn.functional.huber_loss(m.joint_q(S[ix],AA[ix]),YY[ix],delta=10);oq.zero_grad();loss.backward();nn.utils.clip_grad_norm_(critic,5);oq.step()
  def pstep(step):
   ix=pix[step];batch=S[ix];obs=O[ix].clone()
   with torch.no_grad():
    q=m.joint_q(batch[:,None].expand(-1,4,-1).reshape(-1,S.shape[-1]),alternatives[None].expand(len(batch),-1,-1).reshape(-1,2)).reshape(-1,4);teacher=alternatives[q.argmax(-1)]
    first=torch.where(ix%2==1,0,1);second=1-first;row=torch.arange(len(ix),device='cuda');target=teacher[row,first]
    # Probability/history columns of the earlier current choice, exactly as in teacher forcing.
    obs[row,second,25+first]=0;obs[row,second,27+first]=0
    obs[row,second,25+2*target+first]=1;obs[row,second,31+2*target+first]=1
   logits=m.actor_logits(obs);loss=nn.functional.cross_entropy(logits.reshape(-1,2),teacher.flatten());op.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor,5);op.step()
  if schedule=='staged':
   for k in range(5000):qstep(k)
   for k in range(4000):pstep(k)
  else:
   for block in range(50):
    for k in range(block*100,(block+1)*100):qstep(k)
    for k in range(block*80,(block+1)*80):pstep(k)
  m.target_q_body=deepcopy(m.q_body).requires_grad_(False);m.target_q_head=deepcopy(m.q_head).requires_grad_(False);m.training_settings={'action_selection':'sequential_current_choices','ablation':f'{mode}_{schedule}'};save_checkpoint(out/'latest.pt',m,dict(seed=7,data=mode,update_schedule=schedule));(out/'result.json').write_text('{}');print('Trained',mode,schedule,flush=True)
# Evaluate from fresh worlds, not the custom training reset, with per-step reselection.
World.reset=reset_original
DecisionSchedule.mask=old_mask
sys.argv=['evaluation',str(root)];exec(compile(Path('scripts/evaluate_2x2_failures.py').read_text(),'evaluation','exec'))
(root/'protocol.json').write_text(json.dumps(dict(training_seed=7,initial_states=4096,initial_actions='identical uniform executed assignments in both data arms',dynamic_continuation='fixed untrained actor, fresh stochastic choices every step',held_continuation='hold initial assignment until terminal',critic_updates=5000,actor_updates=4000,batch_size=256,schedules={'staged':'5000 critic then 4000 distillation','interleaved':'50 rounds of 100 critic then 80 distillation'},evaluation='greedy sequential neural actor, per-step reselection, same 50 cases per failure family',scope='offline learned-Q distillation ablation, not PPO'),indent=2))
