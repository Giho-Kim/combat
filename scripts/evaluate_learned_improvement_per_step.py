import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pointmass_rl.final_mission_ppo as trainer
from pointmass_rl.env import final_mission_action
import numpy as np
from copy import deepcopy
from dataclasses import replace
p=Path('runs/critic_2x2_learned_policy_improvement/latest.pt')
s=Path('scripts/probe_sequential_assignment.py').read_text();a=s.index('calls=0');b=s.index('trainer.allocated_actions=sequential');exec(s[a:b]+'trainer.allocated_actions=sequential\n')
sys.argv=['evaluation',str(p.parent)];src=Path('scripts/evaluate_2x2_failures.py').read_text().replace("for out in sorted(root.glob('*_seed*')):","for out in [root]:");src=src.replace('range(20000,20050)','range(21000,21200)')
src=src.replace('schedule=DecisionSchedule(c);first=None',"""shift=rng.uniform(-10,10,2);w.pos+=shift;w.targets+=shift
   if rng.random()<.5:w.pos[:]=w.pos[::-1].copy()
   if rng.random()<.5:
    w.targets[:]=w.targets[::-1].copy();w.target_type[:]=w.target_type[::-1].copy();w.target_work_units[:]=w.target_work_units[::-1].copy()
   w.mem_type[:]=w.target_type;w.mem_work_units[:]=w.target_work_units
   schedule=DecisionSchedule(c);first=None""")
src=src.replace("root/'failure_cases.json'","root/'per_step_failure_cases.json'")
src=src.replace("c=Config.load(out/'config.json');", "c=replace(Config.load(out/'config.json'),commit_target=False,decision_interval=1);")
exec(compile(src,'evaluation','exec'))
