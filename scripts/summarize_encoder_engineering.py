"""Compare matched encoder experiments using saved evaluation logs and held-out seeds."""
import csv,json,sys
from pathlib import Path
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pointmass_rl.env import Config,World
from pointmass_rl.strike_ppo import model_from_checkpoint,evaluate_model,DecisionSchedule,_model_observation
OUT=ROOT/'runs/encoder_engineering_500k'
torch.set_num_threads(1)
rows=[];fig,axes=plt.subplots(1,2,figsize=(12,4.6))
with torch.no_grad():
 for name in ['baseline','residual_small','token_norm']:
    p=OUT/name;meta=json.loads((p/'training_metadata.json').read_text());c=Config.load(p/'config.json');model=model_from_checkpoint(p/'latest.pt',c);model.eval()
    assert model.training_settings['encoder_engineering']==name
    ev=[r for r in csv.DictReader((p/'training_evaluations.csv').open()) if r['policy']=='target_coma_no_resolver'];u=list(csv.DictReader((p/'training_updates.csv').open()))
    held=evaluate_model(model,c,episodes=30,seed=20000,resolver=False)
    (p/'heldout_no_resolver.json').write_text(json.dumps(held,indent=2))
    cos=[];same=[]
    for seed in range(20000,20030):
      w=World(c);w.reset(seed);schedule=DecisionSchedule(c);obs=torch.as_tensor(_model_observation(model,w,schedule),dtype=torch.float32);st=model.state_from_observation(obs[0],model.n_agents,model.n_targets)[None];d,t,alive,exists=model.q_body.encode(st,model.n_agents,model.n_targets,encoder=model.actor_encoder);z=torch.nn.functional.normalize(d[0,alive[0]],dim=-1);s=z@z.T
      if len(z)>1:cos.append(float(s[~torch.eye(len(z),dtype=torch.bool)].mean()))
      logits=model.actor_logits(obs);mask=model.target_mask(obs);ch=logits.masked_fill(~mask,-torch.inf).argmax(-1)[alive[0]];same.append(int(len(ch.unique())==1))
    row=dict(variant=name,steps=meta['actual_agent_transitions'],fixed_return=float(ev[-1]['team_return']),fixed_success=float(ev[-1]['mission_success']),heldout_return=held['team_return'],heldout_success=held['mission_success'],last5_kl=float(np.mean([float(r['approx_kl']) for r in u[-5:]])),last_entropy=float(u[-1]['entropy']),initial_drone_cosine=float(np.mean(cos)),initial_all_same_fraction=float(np.mean(same)))
    rows.append(row);print(row,flush=True)
    for ax,key in zip(axes,['team_return','mission_success']):ax.plot([float(r['agent_transitions'])/1e6 for r in ev],[float(r[key]) for r in ev],marker='o',label=name);ax.set_xlabel('Agent transitions (million)');ax.set_ylabel(key);ax.grid(alpha=.2)
axes[0].legend();fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/f'comparison.{ext}',dpi=180)
(OUT/'summary.json').write_text(json.dumps(rows,indent=2))
with (OUT/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
