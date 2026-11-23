"""Controlled supervised critic comparison on exact fixed-assignment 2x2 returns."""
import sys,json,csv,itertools,time
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pointmass_rl.env import Config,World,strike_action
from pointmass_rl.strike_lookahead import completion_steps
from pointmass_rl.strike_ppo import DecisionSchedule,_model_critic_state
from pointmass_rl.strike_target_actor import TargetAllocationActorCritic
from pointmass_rl.graph_critic import GraphQBody
out=Path('runs/critic_2x2_learning_probe');out.mkdir(exist_ok=True)
torch.set_num_threads(1)
c=replace(Config.load('configs/target_allocation_variable.json'),n_agents=2,min_agents=2,n_targets=2,min_targets=2,randomize_counts=False,randomize_target_composition=False,min_spawn_distance=0.)
template=TargetAllocationActorCritic(2,2);rng=np.random.default_rng(20261101);actions=np.array(list(itertools.product(range(2),repeat=2)))
states=[];labels=[];categories=[];observations=[];checks=0
for k in range(4608):
 w=World(c);w.reset(k);w.t=int(rng.integers(0,81));w.targets[:]=rng.uniform(15,85,(2,2));w.pos[:]=rng.uniform(10,90,(2,2));w.target_type[:]=rng.integers(1,4,2);w.target_life[:]=[rng.integers(1,3) if t==1 else 1 for t in w.target_type];w.target_exists[:]=True;w.destroyed[:]=False;w.agent_active[:]=True;w.strike_participants[:]=False;w.agent_strike_progress[:]=0
 eta=completion_steps(w);feas=eta<=c.horizon-w.t;values=np.array([2.5 if t==1 else 2. if t==2 else 1. for t in w.target_type]);y=np.array([sum(values[t]*min(w.target_life[t],np.sum((a==t)&feas[:,t])) for t in range(2)) for a in actions],dtype=np.float32)
 if k<24:
  for a,v in zip(actions,y):
   z=deepcopy(w);before=z.score
   while not z.done:z.step(strike_action(a))
   assert abs(z.score-before-v)<1e-5,(k,a,v,z.score-before)
   checks+=1
 schedule=DecisionSchedule(c);states.append(_model_critic_state(template,w,schedule));observations.append(template.observation(w,schedule.previous_action_probabilities,schedule.targets));labels.append(y);categories.append(dict(deadline=bool(feas.any() and not feas.all()),competition=bool(np.any(w.target_life==1) and feas.all(axis=0).any())))

np.savez_compressed(out/'dataset.npz',states=np.array(states),returns=np.array(labels),observations=np.array(observations),actions=actions,deadline=[r['deadline'] for r in categories],competition=[r['competition'] for r in categories])
print('dataset saved',flush=True)
