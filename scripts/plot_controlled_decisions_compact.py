"""Render controlled comparisons with the positive-side condition in bold."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runs/target_allocation_attention_anatomy'
with (out/'controlled_probes.csv').open() as f:rows=list(csv.DictReader(f))
groups={}
for r in rows:groups.setdefault((r['context'],r['target']),{})[r['intervention']]=r
pairs=[('target_nearer','target_farther',r'Target distance: $\mathbf{nearer}$ vs farther'),
('type_2','type_3',r'Target type: $\mathbf{Type\ 2}$ vs Type 3'),
('own_stay','own_other',r'Own previous choice: $\mathbf{this\ target}$ vs another'),
('peer_same','peer_other',r'Peer’s previous choice: $\mathbf{this\ target}$ vs another'),
('claiming_peer_nearer','claiming_peer_farther',r'Peer choosing this target: $\mathbf{nearer}$ vs farther')]
values=[100*np.mean([float(g[a]['probability'])-float(g[b]['probability']) for g in groups.values() if a in g and b in g]) for a,b,_ in pairs]
fig,ax=plt.subplots(figsize=(12,4.5))
ax.barh([p[2] for p in pairs][::-1],values[::-1],color=['#d97706' if v<0 else '#2563eb' for v in values[::-1]])
ax.axvline(0,color='#475569',lw=1)
ax.set_xlabel('Probability difference (percentage points)')
ax.set_title('Effect on target choice probability',pad=14)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(out/f'controlled_decisions.{ext}',dpi=200)
plt.close(fig)
