"""Learned keep/reselect mixture using existing attention features and native COMA."""
import sys,json
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pointmass_rl.final_mission_target_actor import TargetAllocationActorCritic as OriginalActor
from pointmass_rl.graph_critic import GraphQBody
import pointmass_rl.final_mission_target_actor as actor_module
class PersistentActor(OriginalActor):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.actor_body.keep_head=nn.Sequential(nn.Linear(self.hidden,1))
  nn.init.zeros_(self.actor_body.keep_head[0].weight);nn.init.zeros_(self.actor_body.keep_head[0].bias)
  self.actor_body.register_forward_hook(self.capture)
 def capture(self,module,inputs,output):self.features=output
 def actor_logits(self,obs,**kwargs):
  scores=super().actor_logits(obs,**kwargs);features=self.features;self.features=None
  n=self.n_agents;t=self.n_targets;start=3+5*n+6*t+n*t;identity=obs[...,start:start+n];previous=obs[...,start+n:start+n+n*t].reshape(*obs.shape[:-1],t,n)
  own=(previous*identity.unsqueeze(-2)).sum(-1);valid=self.target_mask(obs);own=own*valid
  keep=torch.sigmoid(self.actor_body.keep_head((features*own.unsqueeze(-1)).sum(-2)))
  fresh=scores.masked_fill(~valid,-torch.inf).softmax(-1)
  keep=keep*(own.sum(-1,keepdim=True)>0)
  mixture=(1-keep)*fresh+keep*own
  return mixture.clamp_min(1e-20).log()
actor_module.TargetAllocationActorCritic=PersistentActor
# Execute only the shared physical-scenario reset code from the existing probe.
s=Path('scripts/probe_2x2_onpolicy.py').read_text();prefix=s[:s.index('Original=graph.GraphQBody')];prefix=prefix.replace("root=Path('runs/critic_2x2_onpolicy_failures')","root=Path('runs/critic_2x2_learned_persistence')");exec(prefix)
graph.GraphQBody=lambda *args,**kwargs:GraphQBody(*args,**dict(kwargs,self_attention=False))
out=root/'learned_keep_seed7';out.mkdir(exist_ok=True);c.save(out/'config.json')
m,ep,ev,actual=train_target_allocation(c,30000,seed=7,n_envs=16,rollout_steps=100,epochs=10,critic_epochs=20,minibatch_size=1024,critic_minibatch_size=1024,eval_interval=0,device='cuda',diagnostics_path=out/'training_updates.csv')
save_checkpoint(out/'latest.pt',m,dict(agent_transitions=actual,seed=7));m.cpu()
# Evaluation uses normal held-out episodes and the same two diagnostic scenarios.
metrics=evaluate_model(m,c,episodes=100,seed=20000,resolver=False);(out/'result.json').write_text(json.dumps(metrics,indent=2));print('RESULT',metrics,flush=True)
# Loading this custom class is supported by the temporary actor factory above.
src=Path('scripts/evaluate_2x2_failures.py').read_text().replace("root=Path(sys.argv[1] if len(sys.argv)>1 else 'runs/critic_2x2_onpolicy_failures')","root=Path('runs/critic_2x2_learned_persistence')")
exec(compile(src,'evaluate_failure_cases','exec'))
