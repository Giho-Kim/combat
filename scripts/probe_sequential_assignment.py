"""Sequential neural assignments with native rewards and Monte Carlo actor updates."""
import sys,json,itertools
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pointmass_rl.strike_ppo as trainer
from pointmass_rl.env import strike_action
s=Path('scripts/probe_2x2_event_credit.py').read_text();exec(s[:s.index('Original=graph.GraphQBody')].replace("root=Path('runs/critic_2x2_event_credit')","root=Path('runs/critic_2x2_sequential_assignment')"))
Original=graph.GraphQBody
graph.GraphQBody=lambda *a,**kw:Original(*a,**dict(kw,self_attention=False))
calls=0
def sequential(model,worlds,schedules,deterministic=False):
 global calls
 B=len(worlds);N=model.n_agents;T=model.n_targets
 temporary=deepcopy(schedules)
 masks=[];decisions=[]
 for w,s in zip(worlds,temporary):
  if w.t==0:s.reset()
  mask,decision=s.mask(w,model.available_actions(w));masks.append(mask);decisions.append(decision)
 masks=np.stack(masks);decisions=np.stack(decisions);obs=np.zeros((B,N,model.obs_dim),np.float32);actions=np.zeros((B,N),int);logps=np.zeros((B,N));probs=np.zeros((B,N,T))
 order=list(range(N))
 if calls%2:order.reverse()
 calls+=1
 for i in order:
  o=np.stack([trainer._model_observation(model,w,s)[i] for w,s in zip(worlds,temporary)]);obs[:,i]=o
  x,stats=model.act(trainer.model_tensor(model,o),deterministic=deterministic,action_mask=trainer.model_tensor(model,masks[:,i]));a=x['target'];actions[:,i]=a;logps[:,i]=stats['logp'].detach().cpu().numpy();probs[:,i]=stats['probs'].detach().cpu().numpy()
  for j,(w,s) in enumerate(zip(worlds,temporary)):
   if decisions[j,i]:s.targets[i]=a[j];s.previous_action_probabilities[i]=probs[j,i]
 for j,(w,s) in enumerate(zip(worlds,schedules)):
  s.record(w,actions[j],decisions[j]&w.agent_active,probs[j])
 trace=dict(obs=obs,target=actions,logp=logps,probs=probs,action_mask=masks,decision=decisions)
 return strike_action(actions),[trace]
trainer.allocated_actions=sequential
# An action-independent state baseline keeps autoregressive policy gradients valid.
def update(m,optimizers,batch,probabilities,epochs,width):
 size=len(batch['state']);critic=trainer._critic_parameters(m);losses=[]
 for _ in range(20):
  for idx in torch.randperm(size,device=batch['state'].device).split(1024):
   pred=m.joint_q(batch['state'][idx],batch['joint_action'][idx]);loss=torch.nn.functional.huber_loss(pred,batch['returns'][idx],delta=10);optimizers[1].zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(critic,5);optimizers[1].step();m.update_target(.01);losses.append(float(loss.detach()))
 with torch.no_grad():
  alternatives=torch.tensor(list(itertools.product(range(2),repeat=2)),device=batch['state'].device);baselines=[]
  for states in batch['state'].split(256):baselines.append(m.joint_q(states[:,None].expand(-1,4,-1).reshape(-1,states.shape[-1]),alternatives[None].expand(len(states),-1,-1).reshape(-1,2)).reshape(-1,4).mean(-1))
  advantage=(batch['returns']-torch.cat(baselines)).repeat_interleave(2)
 b=dict(batch,advantage=advantage);metrics=trainer._ppo_update(m,optimizers,b,epochs,width);metrics['q_loss']=float(np.mean(losses));return metrics
trainer._coma_update=update
torch.set_num_threads(1);out=root/'sequential_seed7';out.mkdir(exist_ok=True);c.save(out/'config.json')
m,ep,ev,actual=train_target_allocation(c,100000,seed=7,n_envs=16,rollout_steps=100,epochs=10,critic_epochs=20,minibatch_size=1024,critic_minibatch_size=1024,eval_interval=0,device='cuda',diagnostics_path=out/'training_updates.csv')
save_checkpoint(out/'latest.pt',m,dict(agent_transitions=actual,seed=7));m.cpu();metrics=evaluate_model(m,c,episodes=100,seed=20000,resolver=False);(out/'result.json').write_text(json.dumps(metrics,indent=2));print('RESULT',metrics,flush=True)
sys.argv=['evaluation',str(root)];exec(compile(Path('scripts/evaluate_2x2_failures.py').read_text(),'evaluation','exec'))
