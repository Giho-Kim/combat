"""Audit frozen rescue-site allocation policies for decision timing and action values.

Changes no training/environment code. Event variants only alter the schedule;
counterfactual tests change one executed action, then resume the original policy.
"""
import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pointmass_rl.env import Config, World, final_mission_action
from pointmass_rl.final_mission_ppo import (DecisionSchedule, _model_observation,
    _model_critic_state, model_from_checkpoint, scheduled_action)


def assignment_upper_bound(world):
    """Exact initial assignment value: stationary targets, unit-use drones."""
    if world.c.target_motion_scale != 0 or world.c.final_mission_probability != 1:
        raise ValueError('Assignment bound requires stationary deterministic targets')
    from pointmass_rl.final_mission_lookahead import completion_steps
    agents=np.flatnonzero(world.agent_active)
    masks=np.arange(1 << len(agents));dp=np.full(len(masks),-np.inf);dp[0]=0.
    feasible=completion_steps(world)[agents] <= world.c.horizon-world.t
    for target in np.flatnonzero(world.target_exists & ~world.target_completed):
        value={1:2.5,2:2.,3:1.}[int(world.target_type[target])]
        for _ in range(int(world.target_work_units[target])):
            nxt=dp.copy()
            for agent in np.flatnonzero(feasible[:,target]):
                bit=1 << int(agent);source=masks[(masks & bit)==0]
                np.maximum.at(nxt,source | bit,dp[source]+value)
            dp=nxt
    return float(dp.max())


class EventSchedule(DecisionSchedule):
    def __init__(self, config, team_interrupt=False):
        super().__init__(config)
        self.interval = config.horizon
        self.team_interrupt = team_interrupt
        self.last_work_units = None

    def mask(self, world, available):
        if self.team_interrupt and self.last_work_units is not None and np.any(self.last_work_units != world.target_work_units):
            self.next_decision[world.agent_active & (world.locked_targets() < 0)] = world.t
        self.last_work_units = world.target_work_units.copy()
        return super().mask(world, available)


def write_csv(path, rows):
    if not rows: return
    with path.open('w', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def episode(model, config, seed, mode, resolver, stochastic=False, collect=False):
    w=World(config); w.reset(seed)
    s=DecisionSchedule(config) if mode=='per_step' else EventSchedule(config, mode=='event_team')
    torch.manual_seed(seed+900000)
    switches=decision_count=agent_steps=over=0
    previous=np.full(w.n,-1); snapshots=[]; q_rows=[]; entropy=[]
    stepset={0,25,50,75}
    while not w.done:
        mask,decision=s.mask(w,model.available_actions(w))
        free=decision & w.agent_active
        before_score=float(w.score); t=w.t
        state=torch.as_tensor(_model_critic_state(model,w,s))[None] if collect else None
        saved=(deepcopy(w),deepcopy(s)) if collect and t in stepset and free.any() else None
        action,stats=scheduled_action(model,w,s,deterministic=not stochastic,resolver=resolver)
        probabilities=stats['allocation_traces'][0]['probs'][0]
        agent_entropy=-(probabilities*np.log(np.maximum(probabilities,1e-30))).sum(-1)
        entropy.extend(agent_entropy[free].tolist())
        joint=action['target'].copy()
        if saved is not None:
            eligible=np.flatnonzero(free & (mask.sum(-1)>1))
            if len(eligible):
                # A high-entropy free agent provides several testable alternatives.
                agent=int(max(eligible,key=lambda i:float(agent_entropy[i])))
                snapshots.append(dict(world=saved[0],schedule=saved[1],joint=joint.copy(),
                    agent=agent,mask=mask[agent].copy(),prob=probabilities.copy(),
                    decisions=decision.copy(),seed=seed,step=t))
        if collect and w.agent_active.any():
            critic_joint=np.where(w.agent_active,joint,0)
            q=float(model.joint_q(state,torch.as_tensor(critic_joint)[None])[0])
            q_rows.append(dict(seed=seed,step=t,q=q,score_before=before_score))
        switches+=int((w.agent_active & (previous>=0) & (previous!=joint)).sum())
        decision_count+=int(free.sum()); agent_steps+=int(w.agent_active.sum())
        for target in np.unique(joint[w.agent_active]):
            capacity=min(2 if w.target_type[target]==1 else 1,int(w.target_work_units[target]))
            over+=max(0,int((w.agent_active & (joint==target)).sum())-capacity)
        previous=joint
        w.step(action)
    for r in q_rows:r['remaining_return']=float(w.score)-r.pop('score_before')
    metrics=w.metrics()
    row=dict(seed=seed,mode=mode,resolver=resolver,stochastic=stochastic,
        team_return=float(metrics['team_return']),success=float(metrics['mission_success']),
        removed=int(w.target_completed[w.target_exists].sum()),switches=switches,
        switch_rate=switches/max(1,agent_steps),decision_rate=decision_count/max(1,agent_steps),
        over_capacity_rate=over/max(1,agent_steps),entropy=float(np.mean(entropy)) if entropy else 0.)
    return row,snapshots,q_rows


@torch.no_grad()
def counterfactuals(model,snapshots,repeats):
    rows=[]
    for index,snapshot in enumerate(snapshots):
        w0=snapshot['world']; agent=snapshot['agent']
        state=torch.as_tensor(_model_critic_state(model,w0,snapshot['schedule']))
        options=np.flatnonzero(snapshot['mask'])
        joints=np.tile(snapshot['joint'],(len(options),1))
        joints[:,agent]=options
        critic_joints=np.where(w0.agent_active[None],joints,0)
        q=model.joint_q(state[None].expand(len(options),-1),torch.as_tensor(critic_joints)).numpy()
        returns=[]
        for option,joint in zip(options,joints):
            outcomes=[]
            for repeat in range(repeats):
                # Common random numbers across alternative first actions.
                torch.manual_seed(snapshot['seed']*10000+snapshot['step']*10+repeat)
                w=deepcopy(w0);s=deepcopy(snapshot['schedule']);score=float(w.score)
                s.record(w,joint,snapshot['decisions'],snapshot['prob'])
                w.step(final_mission_action(joint))
                while not w.done:
                    action,_=scheduled_action(model,w,s,deterministic=False,resolver=False)
                    w.step(action)
                outcomes.append(float(w.score)-score)
            returns.append(outcomes)
        mean=np.mean(returns,axis=-1)
        prob=snapshot['prob'][agent,options];prob=prob/prob.sum()
        for k,(option,pred,observed) in enumerate(zip(options,q,mean)):
            rows.append(dict(state=index,seed=snapshot['seed'],step=snapshot['step'],agent=agent,
                target=int(option),chosen=int(option)==int(snapshot['joint'][agent]),prob=float(prob[k]),
                predicted_q=float(pred),rollout_return=float(observed),
                rollout_std=float(np.std(returns[k])),critic_adv=float(pred-np.dot(prob,q)),
                measured_adv=float(observed-np.dot(prob,mean)),rollout_repeats=repeats))
        if (index+1)%5==0:print(f'Counterfactual states: {index+1}/{len(snapshots)}',flush=True)
    return rows


def bootstrap_delta(values):
    values=np.asarray(values);rng=np.random.default_rng(7)
    samples=values[rng.integers(len(values),size=(5000,len(values)))].mean(-1)
    return dict(mean=float(values.mean()),ci95=np.quantile(samples,[.025,.975]).tolist())


def summarize(rows,q_rows,cf):
    groups={}
    for checkpoint in sorted({r['checkpoint'] for r in rows}):
        for cohort in ('evaluation','holdout'):
            local=[r for r in rows if r['checkpoint']==checkpoint and r['cohort']==cohort]
            for resolver in (False,True):
                base={r['seed']:r for r in local if r['resolver']==resolver and r['mode']=='per_step' and not r['stochastic']}
                for mode in ('per_step','event_own','event_team'):
                    selected=[r for r in local if r['resolver']==resolver and r['mode']==mode and not r['stochastic']]
                    name=f'{checkpoint}/{cohort}/{"score" if resolver else "none"}/{mode}'
                    groups[name]={key:float(np.mean([r[key] for r in selected])) for key in
                        ('team_return','success','removed','switches','switch_rate','decision_rate','over_capacity_rate','entropy')}
                    groups[name]['paired_return_change']=bootstrap_delta([r['team_return']-base[r['seed']]['team_return'] for r in selected])
    calibration={}
    for checkpoint in sorted({r['checkpoint'] for r in rows}):
        selected=[r for r in q_rows if r['checkpoint']==checkpoint]
        pred=np.array([r['q'] for r in selected]);actual=np.array([r['remaining_return'] for r in selected])
        calibration[checkpoint]=dict(samples=len(selected),mae=float(np.abs(pred-actual).mean()),
            bias=float((pred-actual).mean()),explained_variance=float(1-np.var(actual-pred)/np.var(actual)))
    cf_summary={}
    for checkpoint in sorted({r['checkpoint'] for r in rows}):
        subset=[r for r in cf if r['checkpoint']==checkpoint];pairs=[];confident=[];regrets=[];ranges=[];effects=[]
        for state in sorted({r['state'] for r in subset}):
            selected=[r for r in subset if r['state']==state]
            q=np.array([r['predicted_q'] for r in selected]);ret=np.array([r['rollout_return'] for r in selected])
            for i in range(len(q)):
                for j in range(i):
                    correct=bool((q[i]-q[j])*(ret[i]-ret[j])>0)
                    if abs(ret[i]-ret[j])>=.5:pairs.append(correct)
                    repeats=selected[i].get('rollout_repeats',3)
                    stderr=np.sqrt((selected[i]['rollout_std']**2+selected[j]['rollout_std']**2)/max(1,repeats-1))
                    if abs(ret[i]-ret[j])>=max(.5,2.5*stderr):confident.append(correct)
            regrets.append(float(ret.max()-ret[q.argmax()]));ranges.append(float(np.ptp(q)));effects.append(float(np.ptp(ret)))
        cf_summary[checkpoint]=dict(states=len(ranges),meaningful_pairs=len(pairs),
            pairwise_accuracy=float(np.mean(pairs)) if pairs else None,
            conservative_pairs=len(confident),conservative_accuracy=float(np.mean(confident)) if confident else None,
            q_best_rollout_regret=float(np.mean(regrets)),mean_q_range=float(np.mean(ranges)),
            mean_measured_action_effect=float(np.mean(effects)))
    return dict(performance=groups,stochastic_critic_calibration=calibration,counterfactual=cf_summary)


def render_report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((out/'episodes.csv').open()))
    q_rows=list(csv.DictReader((out/'critic_calibration.csv').open()))
    cf=list(csv.DictReader((out/'counterfactual_actions.csv').open()))
    colors=['#2563eb','#d97706','#059669']
    checkpoints=[name for name in ('latest','best') if any(r['checkpoint']==name for r in rows)]
    protocol=json.loads((out/'protocol.json').read_text())
    modes=['per_step','event_own','event_team']
    labels=['Every step','Hold until invalid / complete','Also re-decide on team progress']
    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    for ax,resolver in zip(axes,['False','True']):
        for i,(mode,label,color) in enumerate(zip(modes,labels,colors)):
            values=[np.mean([float(r['team_return']) for r in rows if r['checkpoint']==checkpoint
                and r['mode']==mode and r['resolver']==resolver and r['stochastic']=='False'])
                for checkpoint in checkpoints]
            bars=ax.bar(np.arange(len(checkpoints))+(i-1)*.24,values,.23,label=label,color=color)
            ax.bar_label(bars,fmt='%.2f',fontsize=9)
        ax.set_xticks(np.arange(len(checkpoints)),['Last checkpoint' if c=='latest' else 'Best checkpoint' for c in checkpoints])
        ax.set_title('Score resolver' if resolver=='True' else 'No resolver')
        ax.set_ylabel('Mean team return');ax.set_ylim(0,15.5)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Frozen policies | 30 original + 30 held-out scenarios per condition')
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.1,1,.95))
    for ext in ('pdf','png'):fig.savefig(out/f'decision_comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4))
    for checkpoint,color in zip(checkpoints,colors):
        selected=[r for r in q_rows if r['checkpoint']==checkpoint]
        actual=np.array([float(r['remaining_return']) for r in selected]);q=np.array([float(r['q']) for r in selected])
        axes[0].scatter(actual,q,s=6,alpha=.12,color=color,label=checkpoint)
        selected=[r for r in cf if r['checkpoint']==checkpoint];actual=[];q=[]
        for state in sorted({r['state'] for r in selected}):
            group=[r for r in selected if r['state']==state]
            x=np.array([float(r['rollout_return']) for r in group]);y=np.array([float(r['predicted_q']) for r in group])
            actual.extend(x-x.mean());q.extend(y-y.mean())
        axes[1].scatter(actual,q,s=22,alpha=.7,color=color,label=checkpoint)
    axes[0].plot([0,25],[0,25],'k--',lw=1);axes[0].set_title('Actual policy choices')
    axes[0].set_xlabel('Observed remaining return');axes[0].set_ylabel('Critic Q')
    axes[1].plot([-3,3],[-3,3],'k--',lw=1);axes[1].set_title('One-agent alternative choices')
    axes[1].set_xlabel('Rollout return minus state mean');axes[1].set_ylabel('Critic Q minus state mean')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.suptitle(f'Counterfactual audit: {protocol["cf_states"]} states per checkpoint, '
                 f'{protocol["cf_repeats"]} continuations per action')
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(out/f'credit_assignment.{ext}',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/target_allocation_variable_aligned_basic_warmstart')
    p.add_argument('--out',default='runs/target_allocation_credit_audit')
    p.add_argument('--episodes',type=int,default=30);p.add_argument('--cf-states',type=int,default=20)
    p.add_argument('--cf-repeats',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--checkpoints',nargs='+',choices=('latest','best'),default=['latest','best'])
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args();run=ROOT/args.run;out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        render_report(out);return
    config=Config.load(run/'config.json');torch.set_num_threads(1)
    all_rows=[];all_q=[];all_cf=[]
    if args.resume and (out/'episodes.csv').exists():
        all_rows=list(csv.DictReader((out/'episodes.csv').open()))
        for row in all_rows:
            for key in ('seed','removed','switches'):row[key]=int(row[key])
            for key in ('resolver','stochastic'):row[key]=row[key]=='True'
            for key in ('team_return','success','switch_rate','decision_rate','over_capacity_rate','entropy'):row[key]=float(row[key])
    for checkpoint in args.checkpoints:
        model=model_from_checkpoint(run/f'{checkpoint}.pt',config);model.eval()
        for cohort,first_seed in (('evaluation',10000),('holdout',20000)):
            for resolver in (False,True):
                for mode in ('per_step','event_own','event_team'):
                    cached=[r for r in all_rows if r['checkpoint']==checkpoint and r['cohort']==cohort
                            and r['resolver']==resolver and r['mode']==mode and not r['stochastic']]
                    if len(cached)==args.episodes:continue
                    local=[]
                    for seed in range(first_seed,first_seed+args.episodes):
                        row,_,_=episode(model,config,seed,mode,resolver)
                        row.update(checkpoint=checkpoint,cohort=cohort);local.append(row);all_rows.append(row)
                    print(checkpoint,cohort,resolver,mode,'return',np.mean([r['team_return'] for r in local]),flush=True)
                    write_csv(out/'episodes.csv',all_rows)
        snapshots=[]
        for seed in range(20000,20000+args.episodes):
            row,saved,q=episode(model,config,seed,'per_step',False,stochastic=True,collect=True)
            row.update(checkpoint=checkpoint,cohort='holdout');all_rows.append(row)
            for r in q:r.update(checkpoint=checkpoint)
            all_q.extend(q)
            snapshots.extend(saved)
        # Spread across seeds and mission times rather than taking the first few states.
        indices=np.linspace(0,len(snapshots)-1,min(args.cf_states,len(snapshots)),dtype=int)
        cf=counterfactuals(model,[snapshots[i] for i in indices],args.cf_repeats)
        for r in cf:r.update(checkpoint=checkpoint)
        all_cf.extend(cf)
        write_csv(out/'episodes.csv',all_rows)
        write_csv(out/'counterfactual_actions.csv',all_cf)
        write_csv(out/'critic_calibration.csv',all_q)
    write_csv(out/'episodes.csv',all_rows)
    summary=summarize(all_rows,all_q,all_cf)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'protocol.json').write_text(json.dumps(dict(run=str(run),config=config.__dict__,episodes=args.episodes,
        cf_states=args.cf_states,cf_repeats=args.cf_repeats,
        event_own='Hold assignment until own target invalidates or service completes; physical service locks unchanged.',
        event_team='As event_own, also re-decide for unlocked agents whenever any target work_units changes.',
        counterfactual='One agent action changed for one step; then original stochastic per-step policy without resolver resumes. Common RNG seeds across alternatives.'),indent=2))
    print(json.dumps(summary['counterfactual'],indent=2),flush=True)
    render_report(out)


if __name__=='__main__':main()
