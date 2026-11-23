import argparse,json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',default='runs/target_allocation_fixed_assignment_variable');a=p.parse_args();run=Path(a.run)
rows=[]
for path in sorted((run/'evaluations').glob('round_*.json'),key=lambda x:int(x.stem.split('_')[-1])):
 r=json.load(open(path));rows.append(dict(round=int(path.stem.split('_')[-1]),**r))
base=json.load(open('runs/target_allocation_variable_aligned_basic/curve_evaluations/baselines.json'))
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,key,title in zip(axes,['team_return','mission_success'],['Mean team return','Mission success / oracle']):
 scale=1 if key=='team_return' else base['oracle']['mission_success'];ax.plot([r['round'] for r in rows],[r[key]/scale for r in rows],'-o',label='Fixed-assignment learning',color='#2676b7')
 for name,color in [('oracle','#555555'),('nearest','#299b62'),('type_priority','#d29326')]:ax.axhline(base[name][key]/scale,color=color,ls='--',label=name.replace('_',' ').title())
 ax.set(title=title,xlabel='Training round');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(run/f'learning_curve.{ext}',dpi=180)
with (run/'evaluation_summary.csv').open('w',newline='') as f:
 fields=['round','team_return','mission_success'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
print([(r['round'],r['team_return'],r['mission_success']) for r in rows])
