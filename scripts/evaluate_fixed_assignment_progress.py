"""Evaluate saved interleaved checkpoints without running more training."""
import sys,json,csv,argparse
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pointmass_rl.env import Config
from pointmass_rl.final_mission_ppo import model_from_checkpoint,evaluate_model
p=argparse.ArgumentParser();p.add_argument('--run',default='runs/target_allocation_fixed_assignment_variable');p.add_argument('--checkpoint',default='latest.pt');p.add_argument('--episodes',type=int,default=30);p.add_argument('--seed',type=int,default=10000);a=p.parse_args();torch.set_num_threads(1)
run=Path(a.run);c=Config.load(run/'config.json');cache=run/'evaluations';cache.mkdir(exist_ok=True)
dest=cache/(Path(a.checkpoint).stem+'.json')
if dest.exists():result=json.load(open(dest))
else:
 m=model_from_checkpoint(run/a.checkpoint,c,device='cpu');m.eval()
 result=dict(checkpoint=a.checkpoint,episodes=a.episodes,seed=a.seed,**evaluate_model(m,c,episodes=a.episodes,seed=a.seed,resolver=False));dest.write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
