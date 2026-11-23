import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
folders=[('Existing policy','critic_2x2_reference_100k'),('Target-wise critic','critic_2x2_targetwise_credit'),('Sequential policy','critic_2x2_sequential_assignment')];rows=[]
for label,folder in folders:
 p=Path('runs')/folder/'failure_cases.json'
 if not p.exists():continue
 for x in json.load(open(p)):
  if 'v2' not in x['run'] and folder=='critic_2x2_targetwise_credit':continue
  rows.append(dict(method=label,**{k:v for k,v in x.items() if k!='initial_choices'}))
out=Path('runs/critic_2x2_sequential_assignment');(out/'comparison.json').write_text(json.dumps(rows,indent=2));fig,axes=plt.subplots(1,2,figsize=(10,3.5))
for ax,case in zip(axes,['deadline','yield_last_slot']):
 a=[r for r in rows if r['case']==case];ax.barh([r['method'] for r in a],[100*r['optimal_fraction'] for r in a]);ax.set(xlim=(0,100),xlabel='Optimal completion outcome (%)',title=case.replace('_',' ').title());ax.invert_yaxis()
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'comparison.{ext}',dpi=180)
print(json.dumps(rows,indent=2))
