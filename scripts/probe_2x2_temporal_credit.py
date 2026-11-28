"""Native repeated-decision 2x2 PPO/COMA probe, with no oracle training labels."""
import sys,json
from pathlib import Path
from dataclasses import replace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pointmass_rl.env import Config
from pointmass_rl.final_mission_ppo import train_target_allocation,evaluate_model,save_checkpoint,model_from_checkpoint,checkpoint_payload
import pointmass_rl.graph_critic as graph
root=Path('runs/critic_2x2_temporal_credit');root.mkdir(exist_ok=True)
c=replace(Config.load('configs/target_allocation_variable.json'),n_agents=2,min_agents=2,n_targets=2,min_targets=2,randomize_counts=False,randomize_target_composition=False)
# Sample physical scenarios only; no derived values enter observations or losses.
from pointmass_rl.env import World
import numpy as np
reset_original=World.reset
def reset_probe(self,*args,**kwargs):
 result=reset_original(self,*args,**kwargs)
 if self.rng.random()<.5:
  deadline=self.rng.random()<.5
  self.pos[:]=np.array([[10,49],[10,51]]) if deadline else np.array([[10,50],[20,50]])
  self.targets[:]=np.array([[60,50],[20,50]]) if deadline else np.array([[15,50],[55,50]])
  self.pos+=self.rng.uniform(-2,2,(2,2));self.targets+=self.rng.uniform(-2,2,(2,2))
  self.target_type[:]=[1,2];self.target_work_units[:]=[1,1]
  if self.rng.random()<.5:self.pos[:]=self.pos[::-1].copy()
  if self.rng.random()<.5:
   self.targets[:]=self.targets[::-1].copy();self.target_type[:]=self.target_type[::-1].copy();self.target_work_units[:]=self.target_work_units[::-1].copy()
  self.mem_type[:]=self.target_type;self.mem_work_units[:]=self.target_work_units
 return result
World.reset=reset_probe
import pointmass_rl.final_mission_ppo as trainer
original_returns=trainer._td_lambda_returns
def complete_returns(rewards,next_q,done,gamma,trace_lambda):
 return original_returns(rewards,next_q,done,gamma,1.0)
trainer._td_lambda_returns=complete_returns
Original=graph.GraphQBody
results=[]
variants=[('mc_q',False,20,'q'),('mc_sampled_return',False,20,'sampled_return')]
torch.set_num_threads(1)
for seed in [7]:
 for name,added,critic_epochs,adv in variants:
  out=root/f'{name}_seed{seed}';out.mkdir(exist_ok=True);c.save(out/'config.json')
  class SelectedGraph(Original):
   def __init__(self,*args,**kwargs):kwargs.setdefault('self_attention',added);super().__init__(*args,**kwargs)
  graph.GraphQBody=SelectedGraph
  print('START',name,seed,flush=True)
  if (out/'latest.pt').exists():
   m=model_from_checkpoint(out/'latest.pt',c);actual=checkpoint_payload(out/'latest.pt')['metadata']['agent_transitions']
  else:
   m,episodes,ev,actual=train_target_allocation(c,30000,seed=seed,n_envs=16,rollout_steps=100,epochs=10,critic_epochs=critic_epochs,minibatch_size=1024,critic_minibatch_size=1024,eval_interval=0,device='cuda',coma_advantage=adv,diagnostics_path=out/'training_updates.csv')
   save_checkpoint(out/'latest.pt',m,dict(agent_transitions=actual,seed=seed))
  m=m.cpu()
  metrics=evaluate_model(m,c,episodes=100,seed=20000,resolver=False)
  row=dict(variant=name,seed=seed,transitions=actual,**metrics);results.append(row);(out/'result.json').write_text(json.dumps(row,indent=2));(root/'results.json').write_text(json.dumps(results,indent=2));print('RESULT',row,flush=True)
graph.GraphQBody=Original
(root/'protocol.json').write_text(json.dumps(dict(steps_per_run=30000,seeds=[7,8],n_envs=16,rollout_steps=100,actor_epochs=10,minibatch_size=1024,gamma=1.,gae_lambda=.97,grad_norm=5,eval_episodes=100,eval_seed=20000,resolver=False,model_based_labels=False,derived_features=False,decision_interval=1),indent=2))
