import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('runs/critic_2x2_factorial');rows=json.load(open(p/'failure_cases.json'))
order=['held_staged','held_interleaved','dynamic_staged','dynamic_interleaved'];labels=['Held / staged','Held / interleaved','Re-evaluated / staged','Re-evaluated / interleaved']
fig,axes=plt.subplots(1,2,figsize=(11,3.5))
for ax,case,title in zip(axes,['deadline','yield_last_slot'],['Deadline','Yielding']):
 values=[next(r['optimal_fraction']*100 for r in rows if r['run']==name+'_seed7' and r['case']==case) for name in order]
 ax.barh(labels,values,color=['#2469a5','#5b99c8','#d38a43','#e7b77a']);ax.set(xlim=(0,110),title=title,xlabel='Optimal outcome (%)');ax.invert_yaxis()
 for i,v in enumerate(values):ax.text(v+1,i,f'{v:.0f}%',va='center')
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(p/f'comparison.{ext}',dpi=180)
print(json.dumps([{k:v for k,v in r.items() if k!='initial_choices'} for r in rows],indent=2))
