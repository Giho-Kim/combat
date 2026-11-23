"""Controlled one-decision COMA/PPO experiment; execution holds assignment to end."""
import sys,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pointmass_rl.strike_target_actor import TargetAllocationActorCritic
from pointmass_rl.graph_critic import GraphQBody
out=Path('runs/critic_2x2_learning_probe');d=np.load(out/'dataset.npz');torch.set_num_threads(1)
S=torch.tensor(d['states'],device='cuda');O=torch.tensor(d['observations'],device='cuda');Y=torch.tensor(d['returns'],device='cuda');A=torch.tensor(d['actions'],device='cuda');results=[]
variants=['critic_warmup']
for seed in [7,8,9]:
 for variant in variants:
  torch.manual_seed(seed);m=TargetAllocationActorCritic(2,2).cuda();m.q_body=GraphQBody(self_attention=variant=='self_attention').cuda();actor=list(m.actor_body.parameters())+list(m.target_head.parameters());critic=list(m.q_body.parameters())+list(m.q_head.parameters());oa=torch.optim.Adam(actor,lr=.0005,eps=1e-5);oq=torch.optim.Adam(critic,lr=.0005,eps=1e-5);gen=torch.Generator(device='cuda').manual_seed(seed)
  for pre in range(1500):
   idx=torch.randint(4096,(128,),device='cuda',generator=gen);ss=S[idx];prediction=m.joint_q(ss[:,None].expand(-1,4,-1).reshape(-1,ss.shape[-1]),A[None].expand(128,-1,-1).reshape(-1,2));loss=nn.functional.huber_loss(prediction,Y[idx].flatten(),delta=10);oq.zero_grad();loss.backward();nn.utils.clip_grad_norm_(critic,5);oq.step()
  for step in range(600):
   idx=torch.randint(4096,(128,),device='cuda',generator=gen);s,o,y=S[idx],O[idx],Y[idx]
   with torch.no_grad():
    old_logits=m.actor_logits(o);prob=old_logits.softmax(-1);a=torch.multinomial(prob.reshape(-1,2),1,generator=gen).reshape(-1,2);old_log=old_logits.log_softmax(-1).gather(-1,a[...,None]).squeeze(-1);reward=y.gather(1,(a[:,0]*2+a[:,1])[:,None]).squeeze(1)
   for _ in range(5 if variant=='critic_more' else 1):
    if variant in ['full_coverage','critic_warmup']:pred=m.joint_q(s[:,None].expand(-1,4,-1).reshape(-1,s.shape[-1]),A[None].expand(128,-1,-1).reshape(-1,2));target=y.flatten()
    else:pred=m.joint_q(s,a);target=reward
    qloss=nn.functional.huber_loss(pred,target,delta=10);oq.zero_grad();qloss.backward();nn.utils.clip_grad_norm_(critic,5);oq.step()
   with torch.no_grad():
    qs=m.joint_q(s[:,None].expand(-1,4,-1).reshape(-1,s.shape[-1]),A[None].expand(128,-1,-1).reshape(-1,2)).reshape(-1,2,2)
    actual=qs[torch.arange(128,device='cuda'),a[:,0],a[:,1]];adv=[]
    for agent in range(2):
     alternatives=qs[torch.arange(128,device='cuda'),:,a[:,1]] if agent==0 else qs[torch.arange(128,device='cuda'),a[:,0],:]
     if variant=='oracle_advantage':
      yy=y.reshape(-1,2,2);alternatives=yy[torch.arange(128,device='cuda'),:,a[:,1]] if agent==0 else yy[torch.arange(128,device='cuda'),a[:,0],:]
     numerator=reward if variant in ['sampled_return','oracle_advantage'] else actual
     adv.append(numerator-(prob[:,agent]*alternatives).sum(-1))
    adv=torch.stack(adv,-1);adv=(adv-adv.mean())/adv.std().clamp_min(1e-6)
   for _ in range(2):
    logits=m.actor_logits(o);logp=logits.log_softmax(-1);lp=logp.gather(-1,a[...,None]).squeeze(-1);ratio=(lp-old_log).exp();entropy=-(logp.exp()*logp).sum(-1).mean();loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()-.01*entropy;oa.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor,5);oa.step()
  with torch.no_grad():
   ps=[];qs=[]
   for start in range(4096,len(S),128):
    s=S[start:start+128];ps.append(m.actor_logits(O[start:start+128]).softmax(-1).cpu().numpy());qs.append(m.joint_q(s[:,None].expand(-1,4,-1).reshape(-1,s.shape[-1]),A[None].expand(len(s),-1,-1).reshape(-1,2)).reshape(-1,4).cpu().numpy())
   p=np.concatenate(ps);q=np.concatenate(qs);y=Y[4096:].cpu().numpy();action=p.argmax(-1);chosen=action[:,0]*2+action[:,1]
  for group in ['all','deadline','competition']:
   mask=np.ones(len(y),bool) if group=='all' else d[group][4096:];yy=y[mask];qq=q[mask];cc=chosen[mask];regret=yy.max(-1)-yy[np.arange(len(yy)),cc];qp=[]
   for i in range(4):
    for j in range(i):
     diff=yy[:,i]-yy[:,j];valid=np.abs(diff)>=.5;qp.extend(((qq[:,i]-qq[:,j])[valid]*diff[valid]>0).tolist())
   results.append(dict(seed=seed,variant=variant,group=group,n=int(mask.sum()),actor_optimal=float(np.mean(regret==0)),actor_regret=float(regret.mean()),critic_pair_accuracy=float(np.mean(qp))))
  (out/'warmup_results.json').write_text(json.dumps(results,indent=2));print(seed,variant,results[-3:],flush=True)
(out/'warmup_protocol.json').write_text(json.dumps(dict(train_states=4096,test_states=512,seeds=[7,8,9],steps=600,batch_states=128,actor_epochs=2,critic_updates='1 except critic_more=5',learning_rate=.0005,policy_clip=.2,entropy=.01,grad_clip=5,execution='one joint assignment held until terminal; no resolver',supervision='exact fixed-assignment return',variants=variants),indent=2))
