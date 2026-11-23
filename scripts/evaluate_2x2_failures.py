"""Controlled physical failure scenarios; reference values used for evaluation only."""
import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pointmass_rl.env import Config,World
from pointmass_rl.strike_ppo import model_from_checkpoint,DecisionSchedule,scheduled_action
root=Path(sys.argv[1] if len(sys.argv)>1 else 'runs/critic_2x2_onpolicy_failures');torch.set_num_threads(1);results=[]
for out in sorted(root.glob('*_seed*')):
 if not (out/'result.json').exists():continue
 c=Config.load(out/'config.json');model=model_from_checkpoint(out/'latest.pt',c);model.eval()
 for case in ['deadline','yield_last_slot']:
  scores=[];initial=[]
  for seed in range(20000,20050):
   w=World(c);w.reset(seed);rng=np.random.default_rng(seed);w.t=50 if case=='deadline' else 30
   w.pos[:]=np.array([[24,49],[24,51]]) if case=='deadline' else np.array([[24,50],[32,50]])
   w.targets[:]=np.array([[50,50],[30,50]]) if case=='deadline' else np.array([[30,50],[55,50]])
   w.pos+=rng.uniform(-.6,.6,(2,2));w.targets+=rng.uniform(-.6,.6,(2,2));w.target_type[:]=[1,2];w.target_life[:]=[1,1];w.target_exists[:]=True;w.destroyed[:]=False;w.agent_active[:]=True;w.strike_participants[:]=False;w.agent_strike_progress[:]=0;w.mem_type[:]=w.target_type;w.mem_life[:]=w.target_life
   schedule=DecisionSchedule(c);first=None
   with torch.no_grad():
    while not w.done:
     action,_=scheduled_action(model,w,schedule,deterministic=True,resolver=False)
     if first is None:first=action['target'].tolist()
     w.step(action)
   scores.append(float(w.score));initial.append(first)
  row=dict(run=out.name,case=case,mean_return=float(np.mean(scores)),optimal_fraction=float(np.mean(np.isclose(scores,2. if case=='deadline' else 4.5))),initial_choices=initial);results.append(row)
(root/'failure_cases.json').write_text(json.dumps(results,indent=2));print(json.dumps([{k:v for k,v in x.items() if k!='initial_choices'} for x in results],indent=2))
