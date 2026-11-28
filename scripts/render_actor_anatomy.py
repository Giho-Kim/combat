"""Render drone rescue-site choices and saved actor interventions."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.dissect_actor_attention import scores,probabilities,csv_file
from pointmass_rl.env import Config
from pointmass_rl.final_mission_ppo import model_from_checkpoint

OUT=ROOT/'runs/target_allocation_attention_anatomy'


def load_csv(name):return list(csv.DictReader((OUT/name).open()))


def save(fig,name):
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(OUT/f'{name}.{ext}',dpi=180)
    plt.close(fig)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    run=ROOT/'runs/target_allocation_variable_aligned_basic_warmstart'
    config=Config.load(run/'config.json')
    model=model_from_checkpoint(run/'latest.pt',config);model.eval()
    data=np.load(OUT/'inputs.npz');contexts=json.loads((OUT/'contexts.json').read_text())
    q,p,mask,available=[data[k] for k in ('query','peers','peer_mask','available')]
    attention=np.load(OUT/'attention.npz');base=attention['baseline_logits']
    changed=np.load(OUT/'intervention_logits.npz');prob=probabilities(base,available)
    choices=prob.argmax(-1)
    world_groups={}
    for index,c in enumerate(contexts):world_groups.setdefault((c['seed'],c['step']),[]).append(index)
    priority_pairs=[]
    for indices in world_groups.values():
        for position,i in enumerate(indices):
            for j in indices[:position]:
                if choices[i]!=choices[j]:continue
                k=int(choices[i]);gap=float(base[i,k]-base[j,k])
                distance=float(np.linalg.norm(q[i,k,:2])-np.linalg.norm(q[j,k,:2]))
                if abs(gap)<1e-7 or abs(distance)<1e-7:continue
                priority_pairs.append(dict(first=i,second=j,target=k,stage=contexts[i]['stage'],
                    score_gap=gap,distance_gap=distance,nearer_has_higher_score=gap*distance<0))
    csv_file(OUT/'resolver_priority_pairs.csv',priority_pairs)
    priority_summary={}
    for feature in changed.files:
        values=[(changed[feature][r['first'],r['target']]-changed[feature][r['second'],r['target']])*r['score_gap']<0
                for r in priority_pairs]
        priority_summary[feature]=float(np.mean(values)) if values else 0.
    # Recover the observed previous choices, not the choices recorded after act().
    for i,c in enumerate(contexts):
        history=[]
        for peer in range(p.shape[-2]):
            value=q[i,:,8] if peer==c['agent'] else p[i,:,peer,6]
            history.append(int(value.argmax()) if value.max()>.5 else -1)
        c['previous_choices']=history
    (OUT/'contexts.json').write_text(json.dumps(contexts))
    features=load_csv('interventions.csv');heads=load_csv('heads.csv')
    phases=['initial','negotiation','travel','after_work_progress']
    feature_names={'target_position':'Target position','target_type':'Target type','target_work_units':'Target work_units',
        'own_previous_probability':'Own previous probabilities','own_previous_choice':'Own previous choice',
        'peer_previous_intent':'Peers’ previous intentions','peer_position':'Peer positions',
        'peer_records_initial':'All peer details reset','peer_records_mean':'Only average peer details kept'}
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),gridspec_kw={'width_ratios':[1.4,1]})
    names=list(feature_names)
    matrix=np.array([[next(float(r['decision_flip_rate'])*100 for r in features if r['feature']==feature and r['stage']==stage)
        for stage in phases] for feature in names])
    im=axes[0].imshow(matrix,cmap='YlOrRd',vmin=0,vmax=70,aspect='auto')
    axes[0].set_yticks(range(len(names)),[feature_names[n] for n in names])
    axes[0].set_xticks(range(4),['Initial','Steps 1–10','Travel','After work progress'],rotation=15)
    for i in range(len(names)):
        for j in range(4):axes[0].text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',fontsize=9)
    axes[0].set_title('Choices changed after input intervention')
    labels=[f'Layer {r["layer"]} / head {r["head"]}' for r in heads]
    values=[next(float(r['decision_flip_rate'])*100 for r in features if r['stage']=='all'
        and r['feature']==f'layer_{head["layer"]}_head_{head["head"]}') for head in heads]
    axes[1].barh(labels[::-1],values[::-1],color='#2563eb')
    axes[1].set_xlabel('Choice changes when this head message is removed (%)')
    axes[1].set_title('Causal head-message ablation')
    axes[1].spines[['top','right']].set_visible(False);axes[1].grid(axis='x',alpha=.2)
    fig.suptitle('Saved actor | 640 balanced decision contexts | raw actor before resolver')
    save(fig,'input_and_head_effects')

    rows=load_csv('controlled_probes.csv');groups={}
    for row in rows:groups.setdefault((row['context'],row['target']),{})[row['intervention']]=row
    pairs=[('target_nearer','target_farther','Target 20% nearer vs 20% farther'),
        ('type_2','type_3','Type 2 vs Type 3, same work_units'),
        ('own_stay','own_other','Own previous intention: same vs another target'),
        ('peer_same','peer_other','Peer intention: same vs another target'),
        ('claiming_peer_nearer','claiming_peer_farther','Claiming peer: nearer vs farther')]
    deltas=[];probe_summary={}
    for left,right,label in pairs:
        values=[float(g[left]['probability'])-float(g[right]['probability']) for g in groups.values() if left in g and right in g]
        deltas.append(np.mean(values)*100)
        probe_summary[label]=dict(samples=len(values),mean_probability_change=float(np.mean(values)))
    fig,ax=plt.subplots(figsize=(11,4.2))
    ax.barh([pair[2] for pair in pairs][::-1],deltas[::-1],color=['#d97706' if v<0 else '#2563eb' for v in deltas[::-1]])
    ax.axvline(0,color='#475569',lw=1);ax.set_xlabel('Mean change in the candidate’s choice probability (percentage points)')
    ax.set_title('Controlled input changes: geometry, type and prior intentions')
    ax.spines[['top','right']].set_visible(False);save(fig,'controlled_decisions')

    episodes=load_csv('rollout_ablations.csv')+load_csv('peer_information_rollouts.csv')
    variants=['baseline','no_peer_messages','peer_records_initial','peer_records_mean']
    labels=['Original actor','No peer messages','Peer details reset','Only average peer details']
    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    rollout_summary={}
    for ax,resolver in zip(axes,['False','True']):
        values=[]
        for variant in variants:
            selected=[float(r['team_return']) for r in episodes if r['resolver']==resolver and r['ablation']==variant]
            values.append(np.mean(selected));rollout_summary[f'{resolver}/{variant}']=float(np.mean(selected))
        bars=ax.bar(range(len(labels)),values,color=['#2563eb','#d97706','#dc2626','#059669'])
        ax.bar_label(bars,fmt='%.2f');ax.set_xticks(range(len(labels)),labels,rotation=15,ha='right',fontsize=9)
        ax.set_title('Score resolver' if resolver=='True' else 'No resolver')
        ax.set_ylabel('Mean return');ax.set_ylim(0,15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Full trajectories | same 30 held-out scenarios | no retraining')
    save(fig,'peer_information_rollouts')

    # Find a factual state whose choice changes when ONE actual claiming peer's
    # history is reset; retain the positions, own history and valid-action mask.
    peer_cases=[]
    for i,c in enumerate(contexts):
        if c['stage'] not in ('negotiation','travel'):continue
        for peer in np.flatnonzero(mask[i].any(0)):
            old=c['previous_choices'][peer]
            if old<0 or not available[i,old] or choices[i]==old:continue
            b=p[i].copy();b[:,peer,5]=1/q.shape[1];b[:,peer,6]=0
            candidate=scores(model,q[i:i+1],b[None],mask[i:i+1])[0]
            newprob=probabilities(candidate[None],available[i:i+1])[0]
            if int(newprob.argmax())==old:
                peer_cases.append((float(newprob[old]-prob[i,old]),i,int(peer),old,candidate))
    peer_cases.sort(key=lambda r:-r[0])
    initial=next(i for i,c in enumerate(contexts) if c['stage']=='initial' and choices[i]!=c['nearest']
        and c['types'][choices[i]]<c['types'][c['nearest']] and choices[i]==c['executed_choice']
        and c['types'][int(np.where(available[i],changed['target_type'][i],-np.inf).argmax())]>c['types'][choices[i]])
    history_changed=changed['own_previous_probability']
    candidates=[i for i,c in enumerate(contexts) if c['stage'] in ('travel','after_work_progress')
        and int(np.where(available[i],history_changed[i],-np.inf).argmax())!=choices[i]]
    early=[]
    for i in candidates:
        if contexts[i]['stage']!='travel' or contexts[i]['step']>40:continue
        alternative=int(np.where(available[i],history_changed[i],-np.inf).argmax())
        distances=np.linalg.norm(q[i,[choices[i],alternative],:2],axis=-1)*config.size
        eta=np.maximum(1,np.ceil(np.maximum(distances-config.final_mission_range,0)/(config.final_mission_speed*config.dt)))+config.service_steps_per_unit-1
        if (eta<=config.horizon-contexts[i]['step']).all():early.append(i)
    history=max(early or candidates,key=lambda i:prob[i,choices[i]])
    cases=[dict(name='Initial: type and geometry',index=initial,
        alternative=changed['target_type'][initial],intervention='Type information neutralized')]
    if peer_cases:
        _,i,peer,target,alternative=peer_cases[0]
        cases.append(dict(name='Peer competition',index=i,alternative=alternative,
            intervention=f'Only peer {peer} prior intention reset',peer=peer,claimed_target=target))
    cases.append(dict(name='Preference persistence',index=history,alternative=history_changed[history],
        intervention='Only own previous probabilities reset'))
    case_rows=[]
    with PdfPages(OUT/'decision_cases.pdf') as pdf:
        for number,case in enumerate(cases,1):
            i=case['index'];c=contexts[i];chosen=int(choices[i]);legal=np.flatnonzero(available[i])
            alternative_prob=probabilities(case['alternative'][None],available[i:i+1])[0]
            peer_ids=np.flatnonzero(mask[i,chosen])
            intervened=int(alternative_prob.argmax())
            competitor=intervened if intervened!=chosen else int(legal[np.argsort(base[i,legal])[-2]])
            margins=[]
            for peer in peer_ids:
                drop=mask[i:i+1].copy();drop[:,:,peer]=False
                without=scores(model,q[i:i+1],p[i:i+1],drop)[0]
                margins.append(float((base[i,chosen]-base[i,competitor])-(without[chosen]-without[competitor])))
            fig,axes=plt.subplots(1,3,figsize=(15,4.8),gridspec_kw={'width_ratios':[1.25,1,1]})
            x=np.arange(len(legal));axes[0].barh(x-.17,prob[i,legal],.32,label='Observed inputs',color='#2563eb')
            axes[0].barh(x+.17,alternative_prob[legal],.32,label='Changed inputs',color='#d97706')
            labels=[f'Target {k}: Type {c["types"][k]}, {np.linalg.norm(q[i,k,:2])*10:.2f} km' for k in legal]
            axes[0].set_yticks(x,labels,fontsize=9);axes[0].set_xlabel('Actor choice probability');axes[0].legend(fontsize=8)
            axes[0].set_title(case['intervention'],fontsize=10)
            read_target=competitor if case['name']=='Peer competition' else chosen
            grid=np.stack([attention[f'layer_{layer}_gate'][i,read_target,peer_ids,head] for layer in range(2) for head in range(4)])
            image=axes[1].imshow(grid,vmin=0,vmax=1,aspect='auto',cmap='Blues')
            axes[1].set_yticks(range(8),[f'Layer {l+1}, head {h+1}' for l in range(2) for h in range(4)],fontsize=9)
            axes[1].set_xticks(range(len(peer_ids)),[f'P{peer}' for peer in peer_ids]);axes[1].set_title(f'Gates while reading Target {read_target}',fontsize=10)
            axes[1].set_xlabel('Peers · white = 0, blue = 1')
            axes[2].barh([f'Peer {peer}' for peer in peer_ids],margins,color=['#2563eb' if v>=0 else '#d97706' for v in margins])
            axes[2].axvline(0,color='#475569',lw=1);axes[2].set_title('Effect of removing each peer message',fontsize=10)
            axes[2].set_xlabel(f'Support for Target {chosen} over Target {competitor}')
            fig.suptitle(f'Case {number}: {case["name"]} | seed {c["seed"]}, step {c["step"]}, drone {c["agent"]}\n'
                         f'Raw actor: Target {chosen}; deployed after resolver: Target {c["executed_choice"]}',fontsize=12)
            fig.tight_layout();pdf.savefig(fig);fig.savefig(OUT/f'case_{number}.png',dpi=150);plt.close(fig)
            row=dict(case=number,name=case['name'],context=i,seed=c['seed'],step=c['step'],agent=c['agent'],
                original_target=chosen,intervened_target=int(alternative_prob.argmax()),executed_target=c['executed_choice'],
                original_probability=float(prob[i,chosen]),changed_probability=float(alternative_prob[chosen]),
                intervention=case['intervention'])
            if 'peer' in case:row.update(peer=case['peer'],claimed_target=case['claimed_target'])
            case_rows.append(row)
    (OUT/'case_studies.json').write_text(json.dumps(case_rows,indent=2))
    phase_summary={}
    for stage in phases:
        ix=[i for i,c in enumerate(contexts) if c['stage']==stage]
        phase_summary[stage]=dict(samples=len(ix),nearest_choice_rate=float(np.mean([choices[i]==contexts[i]['nearest'] for i in ix])),
            previous_choice_retention=float(np.mean([choices[i]==contexts[i]['previous_choices'][contexts[i]['agent']] for i in ix])),
            resolver_changes_choice=float(np.mean([choices[i]!=contexts[i]['executed_choice'] for i in ix])))
    (OUT/'summary.json').write_text(json.dumps(dict(phases=phase_summary,controlled_probes=probe_summary,
        rollout_ablations=rollout_summary,head_statistics=heads,case_studies=case_rows,
        priority_pairs=len(priority_pairs),priority_nearer_has_higher_score=float(np.mean([r['nearer_has_higher_score'] for r in priority_pairs])),
        priority_order_change_by_intervention=priority_summary),indent=2))
    print(json.dumps(case_rows,indent=2),flush=True)


if __name__=='__main__':main()
