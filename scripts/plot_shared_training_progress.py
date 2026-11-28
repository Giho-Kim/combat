"""Evaluate immutable saved checkpoints and plot current training progress."""
import argparse,csv,json,sys,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,PercentFormatter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pointmass_rl.env import Config
from pointmass_rl.final_mission_ppo import model_from_checkpoint,checkpoint_payload,evaluate_model
parser=argparse.ArgumentParser()
parser.add_argument('--run',default='runs/target_allocation_shared_attention')
parser.add_argument('--title', default='Target Allocation | shared attention | 30 fixed evaluation scenarios')
parser.add_argument('--evaluate-missing',action='store_true',help='Explicitly evaluate checkpoints with missing cached results')
parser.add_argument('--workers',type=int,default=1,help='Number of concurrent checkpoint evaluations')
args=parser.parse_args()
if args.workers < 1:parser.error('--workers must be at least 1')
run=ROOT/args.run
cache=run/'curve_evaluations';cache.mkdir(exist_ok=True)
c=Config.load(run/'config.json');torch.set_num_threads(1)
baseline_run=ROOT/'runs/target_allocation_variable_aligned_basic'
assert c==Config.load(baseline_run/'config.json'),'Baseline configuration differs'
baselines=json.loads((baseline_run/'curve_evaluations/baselines.json').read_text())
paths=sorted(run.glob('checkpoint_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
training_eval=run/'training_evaluations.csv'
if training_eval.exists():
    with training_eval.open(newline='') as f:
        logged={(int(r['agent_transitions']),r['policy']):r for r in csv.DictReader(f)}
    for path in paths:
        transitions=int(checkpoint_payload(path)['metadata']['agent_transitions'])
        for resolver,policy in ((True,'target_coma_resolver'),(False,'target_coma_no_resolver')):
            dest=cache/f'{path.stem}{"" if resolver else "_no_resolver"}.json'
            if dest.exists() or (transitions,policy) not in logged:continue
            row=logged[transitions,policy]
            metrics={k:(int(v) if k in ('episodes','seed') or k.startswith('episodes_b') else float(v))
                     for k,v in row.items() if k not in ('agent_transitions','policy') and v!=''}
            dest.write_text(json.dumps(dict(agent_transitions=transitions,checkpoint=path.name,**metrics),indent=2))
def evaluate_path(task):
    path, resolver = task
    suffix='' if resolver else '_no_resolver'
    dest=cache/f'{path.stem}{suffix}.json'
    if dest.exists():return
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_shared_curve_point.py'),str(path)]
        +([] if resolver else ['--no-resolver']),check=True)
if args.evaluate_missing:
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(evaluate_path,[(path,resolver) for path in paths for resolver in (True,False)]))
paths=[path for path in paths if (cache/f'{path.stem}.json').exists() and (cache/f'{path.stem}_no_resolver.json').exists()]
if not paths:raise SystemExit('No paired saved evaluations; no new evaluation was performed.')
rows=[]
for path in paths:
    dest=cache/f'{path.stem}.json'
    if dest.exists():r=json.loads(dest.read_text())
    else:
        payload=checkpoint_payload(path)
        model=model_from_checkpoint(path,c,device='cpu');model.eval()
        metrics=evaluate_model(model,c,episodes=30,seed=10000,resolver=True)
        r=dict(agent_transitions=payload['metadata']['agent_transitions'],checkpoint=path.name,episodes=30,seed=10000,**metrics)
        dest.write_text(json.dumps(r,indent=2))
    raw=json.loads((cache/f'{path.stem}_no_resolver.json').read_text())
    r.update(no_resolver_return=raw['team_return'],no_resolver_success=raw['mission_success'])
    rows.append(r)
    print(f'{r["agent_transitions"]/1e6:.3f}M return={r["team_return"]:.4f} success={r["mission_success"]:.3f}',flush=True)
with (run/'learning_curve_evaluations.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
x=np.array([v['agent_transitions']/1e6 for v in rows])
for ax,key,title in zip(axes,['team_return','mission_success'],['Mean team return','Mission success / oracle']):
    scale=1 if key=='team_return' else baselines['oracle']['mission_success']
    y=np.array([v[key]/scale for v in rows])
    ax.plot(x,y,color='#2563eb',lw=1.6,marker='o',ms=3)
    if len(y)>=3:ax.plot(x[2:],np.convolve(y,np.ones(3)/3,'valid'),color='#93c5fd',lw=2.5)
    raw_y=np.array([v['no_resolver_return' if key=='team_return' else 'no_resolver_success']/scale for v in rows])
    ax.plot(x,raw_y,color='#ea580c',lw=1.6,marker='o',ms=3)
    if len(raw_y)>=3:ax.plot(x[2:],np.convolve(raw_y,np.ones(3)/3,'valid'),color='#fdba74',lw=2.5)
    levels=[baselines[name][key]/scale for name in ['oracle','nearest','type_priority']]
    for value,color in zip(levels,['#111827','#059669','#9333ea']):ax.axhline(value,color=color,ls='--',lw=1.4)
    if np.isclose(levels[1],levels[2]):ax.plot([0,max(x)/2],[levels[1]]*2,color='#059669',ls='--',lw=2)
    ax.set_title(title);ax.set_xlabel('Training agent transitions');ax.grid(alpha=.2);ax.set_xlim(left=0)
    ax.spines[['top','right']].set_visible(False);ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}M'))
axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_ylim(0,1.08)
fig.suptitle(args.title)
handles=[Line2D([0],[0],color=co,lw=2,ls=sty) for co,sty in [('#2563eb','-'),('#93c5fd','-'),('#ea580c','-'),('#fdba74','-'),('#111827','--'),('#059669','--'),('#9333ea','--')]]
fig.legend(handles,['Score resolver','Score resolver MA 3','No resolver','No resolver MA 3','Oracle','Nearest','Type priority'],loc='lower center',ncol=4,frameon=False)
fig.tight_layout(rect=(0,.14,1,.93))
for ext in ['pdf','png']:fig.savefig(run/f'learning_curve.{ext}',dpi=180)
plt.close(fig)
print('Saved learning_curve.pdf/png',flush=True)
