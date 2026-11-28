"""Plot saved actor input interventions for drone rescue-site allocation."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.dissect_actor_attention import scores, probabilities
from pointmass_rl.env import Config
from pointmass_rl.final_mission_ppo import model_from_checkpoint
out=ROOT/'runs/target_allocation_attention_anatomy'
run=ROOT/'runs/target_allocation_variable_aligned_basic_warmstart'
torch.set_num_threads(1)
data=np.load(out/'inputs.npz')
q,p,mask,available=[data[k] for k in ('query','peers','peer_mask','available')]
base=np.load(out/'attention.npz')['baseline_logits']
changes=np.load(out/'intervention_logits.npz')
contexts=json.loads((out/'contexts.json').read_text())
cases=json.loads((out/'case_studies.json').read_text())
model=model_from_checkpoint(run/'latest.pt',Config.load(run/'config.json'));model.eval()
fig,axes=plt.subplots(1,3,figsize=(15,5.2))
titles=['Target-type preference','Peer-intention avoidance','Own-intention persistence']
interventions=['Target type neutralized','One peer’s intention reset','Own previous probabilities reset']
with torch.no_grad():
    for ax,case,title,label in zip(axes,cases,titles,interventions):
        i=case['context'];c=contexts[i];legal=np.flatnonzero(available[i])
        factual=probabilities(base[i:i+1],available[i:i+1])[0]
        if case['case']==1:
            logits=changes['target_type'][i]
        elif case['case']==3:
            logits=changes['own_previous_probability'][i]
        else:
            peers=p[i].copy();peers[:,case['peer'],5]=1/q.shape[1];peers[:,case['peer'],6]=0
            logits=scores(model,q[i:i+1],peers[None],mask[i:i+1])[0]
        alternative=probabilities(logits[None],available[i:i+1])[0]
        y=np.arange(len(legal))
        ax.barh(y-.18,factual[legal],.34,color='#2563eb')
        ax.barh(y+.18,alternative[legal],.34,color='#d97706')
        ax.set_yticks(y,[f'T{k} · Type {c["types"][k]} · {np.linalg.norm(q[i,k,:2])*10:.2f} km' for k in legal],fontsize=9)
        ax.set_xlim(0,1.04)
        ax.set_xlabel('Actor choice probability')
        ax.set_title(title,fontsize=13,pad=24)
        ax.text(.5,1.015,label,transform=ax.transAxes,ha='center',fontsize=9,color='#475569')
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
fig.legend(handles=[Patch(color='#2563eb',label='Observed inputs'),Patch(color='#d97706',label='Changed inputs')],loc='lower center',ncol=2,frameon=False)
fig.tight_layout(rect=(0,.09,1,1),w_pad=2)
for ext in ['pdf','png']:fig.savefig(out/f'decision_cases_compact.{ext}',dpi=200)
plt.close(fig)
