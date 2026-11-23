import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('runs/critic_2x2_learning_probe');r=json.load(open(p/'results.json'))
for extra in ['warmup_results.json','exact_expected_results.json']:
 if (p/extra).exists():r+=json.load(open(p/extra))
variants=['coma','critic_more','sampled_return','full_coverage','self_attention','oracle_advantage','critic_warmup','exact_expected'];labels=['COMA','More critic updates','Sampled return','Full action coverage','Self + cross','Exact advantage','Critic warmup + full coverage','Exact expected reward']
fig,axes=plt.subplots(1,2,figsize=(13,5));summary=[]
for ax,group,title in zip(axes,['deadline','competition'],['Deadline decisions','Competition / yielding']):
 means=[];std=[]
 for v in variants:
  rows=[x for x in r if x['variant']==v and x['group']==group];vals=np.array([x['actor_optimal']*100 for x in rows]);means.append(vals.mean());std.append(vals.std());summary.append(dict(group=group,variant=v,seeds=len(rows),actor_optimal_mean=float(vals.mean()),actor_optimal_std=float(vals.std()),regret=float(np.mean([x['actor_regret'] for x in rows])),critic_pair_accuracy=float(np.mean([x['critic_pair_accuracy'] for x in rows]))))
 ax.barh(np.arange(len(variants)),means,xerr=std,capsize=3);ax.set_yticks(np.arange(len(variants)),labels);ax.invert_yaxis();ax.set(xlim=(0,100),xlabel='Optimal joint assignment (%)',title=title);ax.grid(axis='x',alpha=.2)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(p/f'comparison.{ext}',dpi=180)
(p/'aggregate.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
