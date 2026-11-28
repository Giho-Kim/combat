"""Evaluate one immutable checkpoint on the fixed curve scenarios."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pointmass_rl.env import Config
from pointmass_rl.final_mission_ppo import checkpoint_payload,model_from_checkpoint,evaluate_model
torch.set_num_threads(1)
resolver='--no-resolver' not in sys.argv[2:]
path=Path(sys.argv[1]);config=Config.load(path.parent/'config.json')
model=model_from_checkpoint(path,config,device='cpu');model.eval()
# Resolver reuses the same observation with different masks. Cache only the
# deterministic unmasked logits, leaving every mask and decision unchanged.
original_logits=model.actor_logits
cached={}
def cached_logits(obs, **kwargs):
    if not kwargs and 'obs' in cached and torch.equal(obs,cached['obs']):
        return cached['logits']
    logits=original_logits(obs,**kwargs)
    if not kwargs:cached.update(obs=obs.clone(),logits=logits)
    return logits
model.actor_logits=cached_logits

r=dict(agent_transitions=checkpoint_payload(path)['metadata']['agent_transitions'],checkpoint=path.name,
    episodes=30,seed=10000,**evaluate_model(model,config,episodes=30,seed=10000,resolver=resolver))
(path.parent/'curve_evaluations'/f'{path.stem}{"" if resolver else "_no_resolver"}.json').write_text(json.dumps(r,indent=2))
print(f'{r["agent_transitions"]/1e6:.3f}M resolver={resolver} return={r["team_return"]:.4f}',flush=True)
