"""Compare initial and pre-consumption input sensitivity from saved contexts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runs/target_allocation_attention_anatomy'
contexts=json.loads((out/'contexts.json').read_text())
trajectories=json.loads((out/'trajectories.json').read_text())
initial_life={r['seed']:r['life'] for r in trajectories if r['step']==0}
available=np.load(out/'inputs.npz')['available']
base=np.load(out/'attention.npz')['baseline_logits']
changed=np.load(out/'intervention_logits.npz')
choice=np.where(available,base,-np.inf).argmax(-1)
groups=[np.array([c['step']==0 for c in contexts]),
        np.array([c['step']>0 and c['life']==initial_life[c['seed']] for c in contexts])]
features={'target_position':'Target position','target_type':'Target type','target_life':'Target life',
'own_previous_probability':'Own previous probabilities','own_previous_choice':'Own previous choice',
'peer_previous_intent':'Peers’ previous intentions','peer_position':'Peer positions',
'peer_records_initial':'All peer details reset','peer_records_mean':'Only average peer details kept'}
values=np.array([[100*np.mean((np.where(available,changed[key],-np.inf).argmax(-1)!=choice)[g]) for g in groups] for key in features])
fig,ax=plt.subplots(figsize=(7.8,5.7))
ax.imshow(values,cmap='YlOrRd',vmin=0,vmax=70,aspect='auto')
ax.set_yticks(range(len(features)),list(features.values()),fontsize=11)
ax.set_xticks([0,1],['Initial','In progress'],fontsize=12)
ax.tick_params(length=0,pad=9)
for i in range(len(features)):
    for j in range(2):ax.text(j,i,f'{values[i,j]:.1f}%',ha='center',va='center',fontsize=12,color='white' if values[i,j]>45 else '#111827')
ax.set_title('Choice changes after input intervention',pad=16,fontsize=14)
for spine in ax.spines.values():spine.set_visible(False)
fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(out/f'input_effects_compact.{ext}',dpi=200)
(out/'input_effects_compact_data.json').write_text(json.dumps({'groups':['Initial','In progress'],'samples':[int(g.sum()) for g in groups],'features':list(features),'choice_change_percent':values.tolist()},indent=2))
print('Samples:',[int(g.sum()) for g in groups])
