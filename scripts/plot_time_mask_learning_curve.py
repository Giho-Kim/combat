"""Evaluate saved 2.5M checkpoints with the time mask; overlay saved curves."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,PercentFormatter
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pointmass_rl.env import Config
from pointmass_rl.strike_ppo import checkpoint_payload,model_from_checkpoint,evaluate_model,save_checkpoint
from pointmass_rl.strike_target_actor import TargetAllocationActorCritic


def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def main():
    base=ROOT/'runs/target_allocation_variable_aligned_basic'
    warm=ROOT/'runs/target_allocation_variable_aligned_basic_warmstart'
    cache=warm/'time_mask_curve_evaluations';cache.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    base_rows=[r for r in read(base/'training_evaluations.csv') if r['policy']=='target_coma_resolver']
    offset=max(int(r['agent_transitions']) for r in base_rows)
    oracle=json.loads((base/'curve_evaluations/baselines.json').read_text())
    original=read(warm/'checkpoint_evaluation_continued.csv')
    initial_output=cache/'basic_0.json'
    if initial_output.exists():
        initial=json.loads(initial_output.read_text())
    else:
        metadata=json.loads((base/'training_metadata.json').read_text())
        seed=int(metadata['seed'])
        if metadata['training_settings'].get('initial_checkpoint') is not None:
            raise ValueError('Initial run was warm-started; cannot reconstruct from seed alone')
        torch.manual_seed(seed);np.random.seed(seed)
        config=Config.load(base/'config.json')
        model=TargetAllocationActorCritic(config.n_targets,config.n_agents);model.eval()
        model.training_settings={'policy_temperature':1.,'deadline_mask':False}
        regular=evaluate_model(model,config,episodes=30,seed=10000,resolver=True)
        recorded=next(r for r in base_rows if int(r['agent_transitions'])==0)
        for key in ('team_return','mission_success'):
            if not np.isclose(regular[key],float(recorded[key]),atol=1e-8,rtol=0):
                raise ValueError(f'Reconstructed initial policy differs from saved Eval@0: {key}')
        path=cache/f'initial_seed_{seed}.pt'
        save_checkpoint(path,model,dict(seed=seed,agent_transitions=0,reconstructed_initialization=True,
            verified_against_saved_eval_at_zero=True))
        model.training_settings['deadline_mask']=True
        timed=evaluate_model(model,config,episodes=30,seed=10000,resolver=True)
        initial=dict(run='basic',checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            checkpoint_interval=0,local_agent_transitions=0,agent_transitions=0,episodes=30,seed=10000,
            score_return=regular['team_return'],score_success=regular['mission_success'],
            time_mask_return=timed['team_return'],time_mask_success=timed['mission_success'],
            initialization_seed=seed,reconstructed_initialization=True,verified_against_saved_eval_at_zero=True)
        initial_output.write_text(json.dumps(initial,indent=2))
    original=([dict(run='basic',agent_transitions=0,team_return=initial['score_return'],
        mission_success_over_oracle=initial['score_success']/oracle['oracle']['mission_success'])]
        +[r for r in original if int(r['agent_transitions'])>0])
    points=[initial]
    print(f'0M: {initial["score_return"]:.4f} → {initial["time_mask_return"]:.4f}; initial-policy verification passed',flush=True)
    for label,folder,shift in [('basic',base,0),('warmstart',warm,offset)]:
        config=Config.load(folder/'config.json')
        saved={int(r['agent_transitions']):r for r in read(folder/'training_evaluations.csv') if r['policy']=='target_coma_resolver'}
        for nominal in [2500000,5000000,7500000,10000000]:
            path=folder/f'checkpoint_{nominal}.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest()
            output=cache/f'{label}_{nominal}.json'
            payload=checkpoint_payload(path);meta=payload['metadata']
            actual=int(meta.get('agent_transitions',meta.get('actual_agent_transitions',nominal)))
            if output.exists() and json.loads(output.read_text())['checkpoint_sha256']==digest:
                point=json.loads(output.read_text())
            else:
                model=model_from_checkpoint(path,config);model.eval()
                if actual in saved:
                    regular={key:float(saved[actual][key]) for key in ['team_return','mission_success']}
                else:
                    model.training_settings['deadline_mask']=False
                    regular=evaluate_model(model,config,episodes=30,seed=10000,resolver=True)
                model.training_settings['deadline_mask']=True
                timed=evaluate_model(model,config,episodes=30,seed=10000,resolver=True)
                point=dict(run=label,checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=digest,
                    checkpoint_interval=nominal,local_agent_transitions=actual,agent_transitions=shift+actual,
                    episodes=30,seed=10000,score_return=regular['team_return'],score_success=regular['mission_success'],
                    time_mask_return=timed['team_return'],time_mask_success=timed['mission_success'])
                output.write_text(json.dumps(point,indent=2))
            points.append(point)
            print(f'{point["agent_transitions"]/1e6:.3f}M: {point["score_return"]:.4f} → '
                  f'{point["time_mask_return"]:.4f}; success {point["score_success"]:.3f} → {point["time_mask_success"]:.3f}',flush=True)
    with (warm/'time_mask_checkpoint_evaluations.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(key for point in points for key in point))
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(points)
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.2),facecolor='white')
    success_oracle=oracle['oracle']['mission_success']
    masked=json.loads((cache/'heuristics_time_mask.json').read_text())
    for ax,key,title in zip(axes,['team_return','mission_success_over_oracle'],['Mean team return','Mission success / oracle']):
        x=np.array([int(r['agent_transitions']) for r in original]);y=np.array([float(r[key]) for r in original])
        px=np.array([r['agent_transitions'] for r in points])
        py=np.array([r['time_mask_return'] if key=='team_return' else r['time_mask_success']/success_oracle for r in points])
        # Requested shared visual origin; the measured 0M time-mask result
        # remains unchanged in the evaluation CSV/JSON and is not drawn here.
        py[0]=initial['score_return'] if key=='team_return' else initial['score_success']/success_oracle
        ax.plot(px,py,color='#ea580c',lw=2.5,zorder=5)
        ax.scatter(px[1:],py[1:],color='#ea580c',s=25,zorder=6)
        ax.scatter(px[:1],py[:1],facecolors='white',edgecolors='#ea580c',s=30,zorder=6)
        for xx,yy in zip(px,py):ax.annotate(f'{yy:.2f}' if key=='team_return' else f'{yy:.0%}',(xx,yy),xytext=(0,-14) if key=='team_return' and xx>0 else (14,-3) if xx==0 else (0,8),textcoords='offset points',ha='center' if xx>0 else 'left',fontsize=8,color='#c2410c')
        lines=[oracle['oracle']['team_return'],oracle['nearest']['team_return'],oracle['type_priority']['team_return']] if key=='team_return' else [1.,oracle['nearest']['mission_success']/success_oracle,oracle['type_priority']['mission_success']/success_oracle]
        for value,color in zip(lines,['#111827','#059669','#9333ea']):ax.axhline(value,color=color,ls='--',lw=1.5,zorder=3)
        # Heuristics given the same time mask as the learned policy.
        for name,color in [('nearest','#059669'),('type_priority','#9333ea')]:
            value=masked[name]['team_return'] if key=='team_return' else masked[name]['mission_success']/success_oracle
            ax.axhline(value,color=color,ls='-',lw=3,alpha=.45,zorder=1)
        # Shared baseline success values are drawn in two colors for readability.
        if np.isclose(lines[1],lines[2]):
            ax.plot([0,x[-1]/2],[lines[1]]*2,color='#059669',ls='--',lw=2,zorder=4)
        ax.set_title(title);ax.set_xlabel('Training agent transitions');ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False);ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v/1e6:g}M'))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_ylim(0,1.08)
    handles=[Line2D([0],[0],color=color,lw=2,ls=style,marker=marker) for color,style,marker in
        [('#ea580c','-','o'),('#111827','--',None),
         ('#059669','--',None),('#059669','-',None),('#9333ea','--',None),('#9333ea','-',None)]]
    for h in (handles[3],handles[5]):h.set_alpha(.45);h.set_linewidth(3)
    fig.legend(handles,['Learned (score resolver + time mask)','Oracle',
                        'Nearest','Nearest + time mask','Type priority','Type priority + time mask'],
               loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.045))
    fig.suptitle('Target Allocation | continued training | same 30 evaluation scenarios')
    fig.text(.5,.015,'0M: shared baseline reference (hollow marker); filled orange markers are time-mask evaluations.',
             ha='center',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.2,1,.94))
    for ext in ['pdf','png']:fig.savefig(warm/f'learning_curve_with_time_mask.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
