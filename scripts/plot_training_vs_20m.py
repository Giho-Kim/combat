"""Compare saved live training evaluations with the existing 20M run."""
import argparse,csv,json,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pointmass_rl.env import Config

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',default='runs/target_allocation_temporal_scratch');p.add_argument('--label',default='Temporal critic');a=p.parse_args()
 run=Path(a.run);base=Path('runs/target_allocation_variable_aligned_basic_warmstart');original=Path('runs/target_allocation_variable_aligned_basic')
 assert json.loads(json.dumps(asdict(Config.load(run/'config.json'))))==json.loads(json.dumps(asdict(Config.load(original/'config.json')))),'Evaluation configurations differ'
 baseline=json.load(open(original/'curve_evaluations/baselines.json'))
 old=list(csv.DictReader(open(base/'checkpoint_evaluation_continued.csv')))
 rows=list(csv.DictReader(open(run/'training_evaluations.csv')));group={}
 for row in rows:
  policy=row['policy']
  if policy.endswith('_no_resolver'):mode='no_resolver'
  elif policy.endswith('_resolver'):mode='resolver'
  else:continue
  assert int(row['episodes'])==30 and int(row['seed'])==10000,'Evaluation scenario set differs'
  group.setdefault(int(row['agent_transitions']),{})[mode]=row
 steps=sorted(k for k,v in group.items() if len(v)==2)
 fig,axes=plt.subplots(1,2,figsize=(13,4.4))
 def ma(values):return [np.mean(values[max(0,i-4):i+1]) for i in range(len(values))]
 for ax,key,title in zip(axes,['team_return','mission_success'],['Mean team return','Mission success / oracle']):
  scale=1 if key=='team_return' else baseline['oracle']['mission_success']
  x=[float(r['agent_transitions'])/1e6 for r in old];y=[float(r['team_return' if key=='team_return' else 'mission_success_over_oracle']) for r in old]
  ax.plot(x,y,color='#2474ba',alpha=.25,lw=1);ax.plot(x,ma(y),color='#2474ba',label='20M: score resolver (MA 5)')
  old_raw=[]
  for folder,offset in [(original,0),(base,10000000)]:
   old_raw.extend((float(r['agent_transitions'])/1e6+offset/1e6,float(r[key])/scale) for r in csv.DictReader(open(folder/'training_evaluations.csv')) if r['policy'].endswith('_no_resolver') and not(offset and int(r['agent_transitions'])==0))
  ax.plot([r[0] for r in old_raw],[r[1] for r in old_raw],color='#748a9e',alpha=.2,lw=1)
  ax.plot([r[0] for r in old_raw],ma([r[1] for r in old_raw]),color='#586e85',label='20M: no resolver (MA 5)')
  for mode,color in [('resolver','#e37d25'),('no_resolver','#a03bb4')]:
   ax.plot(np.array(steps)/1e6,[float(group[k][mode][key])/scale for k in steps],'-o',lw=1.5,ms=3,color=color,label=f'{a.label}: '+('score resolver' if mode=='resolver' else 'no resolver'))
  for name,color in [('oracle','#555555'),('nearest','#299b62'),('type_priority','#cfab27')]:ax.axhline(baseline[name][key]/scale,color=color,ls='--',lw=1,label=name.replace('_',' ').title())
  ax.set(title=title,xlabel='Agent transitions (millions)');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(run/f'learning_curve_vs_20m.{ext}',dpi=180,bbox_inches='tight')
 print('latest',steps[-1],{mode:{key:group[steps[-1]][mode][key] for key in ['team_return','mission_success']} for mode in ['resolver','no_resolver']})

if __name__=='__main__':main()
