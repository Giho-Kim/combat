import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pointmass_rl.final_mission_ppo as trainer
from pointmass_rl.env import final_mission_action
import numpy as np
from copy import deepcopy
p=Path('runs/critic_2x2_learned_policy_improvement/latest.pt');payload=torch.load(p,weights_only=False,map_location='cpu');state=payload['state_dict'];state={k:v for k,v in state.items() if not k.startswith('target_q_body.')};state.update({'target_q_body.'+k[len('q_body.'):]:v.clone() for k,v in list(state.items()) if k.startswith('q_body.')});payload['state_dict']=state;torch.save(payload,p)
s=Path('scripts/probe_sequential_assignment.py').read_text();a=s.index('calls=0');b=s.index('trainer.allocated_actions=sequential');exec(s[a:b]+'trainer.allocated_actions=sequential\n')
sys.argv=['evaluation',str(p.parent)];src=Path('scripts/evaluate_2x2_failures.py').read_text().replace("for out in sorted(root.glob('*_seed*')):","for out in [root]:");exec(compile(src,'evaluation','exec'))
