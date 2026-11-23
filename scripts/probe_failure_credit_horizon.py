"""Evaluate how long an alternative choice must persist to change failure outcome."""
import sys,json,itertools
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pointmass_rl.env import Config,World,strike_action
from pointmass_rl.strike_ppo import model_from_checkpoint,DecisionSchedule,scheduled_action,_model_critic_state
root=Path('runs/critic_2x2_onpolicy_failures');out=root/'credit_horizon';out.mkdir(exist_ok=True);torch.set_num_threads(1);rows=[]
for run in ['cross_20_seed7','cross_sampled_return_seed7']:
 c=Config.load(root/run/'config.json');m=model_from_checkpoint(root/run/'latest.pt',c);m.eval()
 for case in ['deadline','yield_last_slot']:
  w=World(c);w.reset(20000);w.t=50 if case=='deadline' else 30;w.pos[:]=[[24,49],[24,51]] if case=='deadline' else [[24,50],[32,50]];w.targets[:]=[[50,50],[30,50]] if case=='deadline' else [[30,50],[55,50]];w.target_type[:]=[1,2];w.target_life[:]=[1,1];w.target_exists[:]=True;w.destroyed[:]=False;w.agent_active[:]=True;w.strike_participants[:]=False;w.agent_strike_progress[:]=0;w.mem_type[:]=w.target_type;w.mem_life[:]=w.target_life
  s=DecisionSchedule(c)
  with torch.no_grad():
   a,_=scheduled_action(m,deepcopy(w),deepcopy(s),deterministic=True,resolver=False)
   state=torch.tensor(_model_critic_state(m,w,s))[None]
   for joint in itertools.product(range(2),repeat=2):
    q=float(m.joint_q(state,torch.tensor([joint]))[0])
    for duration in [1,10,100]:
     z=deepcopy(w);sched=deepcopy(s);t0=z.t
     while not z.done:
      if z.t-t0<duration:
       action=strike_action(joint);sched.record(z,np.array(joint),z.agent_active)
      else:action,_=scheduled_action(m,z,sched,deterministic=True,resolver=False)
      z.step(action)
     rows.append(dict(run=run,case=case,actor_choice=a['target'].tolist(),joint=list(joint),q=q,duration=duration,return_=float(z.score)))
(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
