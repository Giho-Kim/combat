"""Analyze drone assignments to rescue sites and decisions that reduced the achievable rescue score."""
import argparse
import csv
import hashlib
import json
from copy import deepcopy
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
from pointmass_rl.env import Config,World,final_mission_action
from pointmass_rl.final_mission_lookahead import completion_steps
from pointmass_rl.final_mission_ppo import DecisionSchedule,model_from_checkpoint,scheduled_action,_model_observation
from scripts.audit_credit_and_decisions import assignment_upper_bound


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def optimal_plan(world,preference=None):
    """Lexicographic: value, fewest changed agent destinations, earliest finish.

    Actual final_mission participants are fixed. Unfinished locked work reserves a
    work_units/slot even when it can no longer finish by the deadline.
    """
    if world.c.target_motion_scale!=0 or world.c.final_mission_probability!=1:
        raise ValueError('This oracle requires stationary deterministic targets')
    n=world.n;preference=np.full(n,-1) if preference is None else np.asarray(preference)
    locks=world.locked_targets();plan=np.full(n,-1,dtype=int)
    agents=np.flatnonzero(world.agent_active & (locks<0));remaining=world.c.horizon-world.t
    eta=completion_steps(world);reserved=np.zeros(world.c.n_targets,dtype=int);fixed_value=0.
    for i in np.flatnonzero(world.agent_active & (locks>=0)):
        j=int(locks[i]);plan[i]=j;reserved[j]+=1
        if eta[i,j]<=remaining:fixed_value+={1:2.5,2:2.,3:1.}[int(world.target_type[j])]
    slots=[j for j in np.flatnonzero(world.target_exists & ~world.target_completed)
           for _ in range(max(0,int(world.target_work_units[j])-reserved[j]))]
    states=np.arange(1 << len(agents));dp=np.full(len(states),-10**18,dtype=np.int64);dp[0]=0
    parents=[]
    for target in slots:
        nxt=dp.copy();parent=np.full(len(states),-1,dtype=np.int16)
        units={1:5,2:4,3:2}[int(world.target_type[target])]
        for local,agent in enumerate(agents):
            if eta[agent,target]>remaining:continue
            bit=1 << local;source=states[((states & bit)==0)&(dp>-10**17)]
            bonus=1 if preference[agent]==target else -1 if preference[agent]<0 else 0
            candidate=dp[source]+units*10**7+bonus*10**4-int(eta[agent,target])
            destination=source|bit;better=candidate>nxt[destination]
            nxt[destination[better]]=candidate[better];parent[destination[better]]=local
        dp=nxt;parents.append(parent)
    state=int(np.argmax(dp))
    for target,parent in reversed(list(zip(slots,parents))):
        local=int(parent[state])
        if local>=0:plan[agents[local]]=target;state^=1 << local
    value=fixed_value+sum({1:2.5,2:2.,3:1.}[int(world.target_type[plan[i]])] for i in agents if plan[i]>=0)
    return float(value),plan


@torch.no_grad()
def rollout(model,config,seed):
    w=World(config);w.reset(seed);initial=deepcopy(w);schedule=DecisionSchedule(config)
    actual=np.full(w.n,-1,dtype=int);finish=np.full(w.n,-1,dtype=int);first=np.full(w.n,-1,dtype=int)
    previous=np.full(w.n,-1,dtype=int);switches=np.zeros(w.n,dtype=int)
    travelled=np.zeros(w.n);trace=[];worlds=[]
    start_bound,_=optimal_plan(w);last_bound=start_bound;first_loss=None
    while not w.done:
        before=deepcopy(w);worlds.append(before)
        logits=model.actor_logits(torch.as_tensor(_model_observation(model,w,schedule))).numpy()
        available=model.available_actions(w);raw=np.where(available,logits,-np.inf).argmax(-1)
        action,_=scheduled_action(model,w,schedule,resolver=True);chosen=action['target'].copy()
        if w.t==0:first=chosen.copy()
        switches+=w.agent_active&(previous>=0)&(previous!=chosen);previous=chosen.copy()
        active=w.agent_active.copy();participants=w.final_mission_participants.copy();pos=w.pos.copy()
        w.step(action);travelled+=np.linalg.norm(w.pos-pos,axis=-1)
        completed_agents=np.flatnonzero(active & ~w.agent_active)
        for i in completed_agents:actual[i]=chosen[i];finish[i]=w.t
        value,_=optimal_plan(w);bound=float(w.score)+value
        if bound>last_bound+1e-7:raise AssertionError(('Bound increased',seed,w.t,last_bound,bound))
        row=dict(step=w.t-1,score_after=float(w.score),upper_bound_after=bound,
            pos_before=pos.tolist(),chosen=chosen.tolist(),raw=raw.tolist(),
            logits=logits.tolist(),retired_agents=completed_agents.tolist(),
            started=np.argwhere(w.final_mission_participants & ~participants).tolist(),
            bound_loss=last_bound-bound)
        if first_loss is None and bound<start_bound-1e-7:first_loss=(len(trace),before,chosen.copy())
        trace.append(row);last_bound=bound
    assert abs(last_bound-w.score)<1e-7
    return dict(world=w,initial=initial,actual=actual,finish=finish,first=first,
                switches=switches,travelled=travelled,trace=trace,worlds=worlds,first_loss=first_loss)


def one_step_rescue(model,world,chosen,initial_bound):
    rows=[];baseline_after=deepcopy(world);baseline_after.step(final_mission_action(chosen))
    actual_bound=baseline_after.score+optimal_plan(baseline_after)[0]
    available=model.available_actions(world);locked=world.locked_targets()
    for i in np.flatnonzero(world.agent_active & (locked<0)):
        for j in np.flatnonzero(available[i]):
            if j==chosen[i]:continue
            variant=chosen.copy();variant[i]=j;branch=deepcopy(world);branch.step(final_mission_action(variant))
            bound=branch.score+optimal_plan(branch)[0]
            if bound>actual_bound+1e-7:
                rows.append(dict(drone=int(i),actual_target=int(chosen[i]),alternative_target=int(j),
                    actual_bound=float(actual_bound),alternative_bound=float(bound),
                    prevents_first_loss=bool(bound>=initial_bound-1e-7)))
    rows.sort(key=lambda r:(-r['alternative_bound'],r['drone'],r['alternative_target']))
    _,plan=optimal_plan(world,chosen)
    command=np.where(plan>=0,plan,chosen);branch=deepcopy(world);branch.step(final_mission_action(command))
    joint_bound=branch.score+optimal_plan(branch)[0]
    assert joint_bound>=initial_bound-1e-7,(world.t,joint_bound,initial_bound)
    return rows,command.tolist(),float(joint_bound)


def simulate_fixed(config,seed,plan,return_assignment=False):
    w=World(config);w.reset(seed)
    selected=plan.copy();eta=completion_steps(w);absent=np.flatnonzero(~w.target_exists)
    for i in np.flatnonzero(w.agent_active & (plan<0)):
        selected[i]=absent[0] if len(absent) else int(np.argmax(eta[i]))
    selected[~w.agent_active]=0;actual=np.full(w.n,-1)
    while not w.done:
        active=w.agent_active.copy();w.step(final_mission_action(selected))
        completed_agents=active & ~w.agent_active;actual[completed_agents]=selected[completed_agents]
    return (float(w.score),actual) if return_assignment else float(w.score)


def plot_case(pdf,seed,result,plan,oracle,rows,rescue,out):
    initial=result['initial'];w=result['world'];actual=result['actual'];first_loss=result['first_loss']
    fig,axes=plt.subplots(1,2,figsize=(12,5.6));colors=plt.cm.tab10.colors
    changed={r['drone'] for r in rows if r['changed']}
    for ax,label,assignment in [(axes[0],'Executed policy paths',actual),(axes[1],'Optimal assignment (fewest changes)',plan)]:
        for j in np.flatnonzero(initial.target_exists):
            x,y=initial.targets[j]/10
            ax.scatter(x,y,s=70,c=[colors[j%10]],edgecolors='black',zorder=4)
            ax.annotate(f'T{j}',(x,y),xytext=(6,5 if j%2==0 else -10),textcoords='offset points',fontsize=9)
        for i in np.flatnonzero(initial.agent_active):
            start=initial.pos[i]/10;target=int(assignment[i])
            ax.scatter(*start,s=14,c='black')
            if ax is axes[0]:
                path=np.array([r['pos_before'][i] for r in result['trace']]+[w.pos[i].tolist()])/10
                ax.plot(path[:,0],path[:,1],color=colors[i%10] if i in changed else '#cbd5e1',lw=2 if i in changed else .7,alpha=.9)
                end=path[-1]
                if i in changed:ax.annotate(f'D{i}',end,xytext=(5,10 if i%2==0 else -14),textcoords='offset points',fontsize=9,color=colors[i%10])
            elif target>=0:
                end=initial.targets[target]/10;ax.plot([start[0],end[0]],[start[1],end[1]],color=colors[i%10] if i in changed else '#cbd5e1',lw=2 if i in changed else .7,alpha=.9)
                if i in changed:ax.annotate(f'D{i}',end,xytext=(-20,13 if i%2==0 else -18),textcoords='offset points',fontsize=9,color=colors[i%10])
        points=np.concatenate((initial.pos[initial.agent_active],initial.targets[initial.target_exists]))/10
        lower=np.maximum(0,points.min(0)-.55);upper=np.minimum(10,points.max(0)+.55)
        span=max(upper-lower);middle=(upper+lower)/2
        ax.set(xlim=(max(0,middle[0]-span/2),middle[0]+span/2),
               ylim=(max(0,middle[1]-span/2),middle[1]+span/2),xlabel='x (km)',ylabel='y (km)',title=label)
        ax.set_aspect('equal');ax.grid(alpha=.15)
    table=' | '.join(f'D{r["drone"]}: {"unfinished" if r["actual_target"]<0 else "T"+str(r["actual_target"])} → '
                     f'{"unused" if r["oracle_target"]<0 else "T"+str(r["oracle_target"])}' for r in rows if r['changed'])
    index,before,chosen=first_loss
    cause=f'First irrecoverable loss follows the decision at step {before.t}; the case report lists alternative rescue-site assignments.'
    fig.suptitle(f'Seed {seed} | actual {w.score:g}, optimum {oracle:g}, lost value {oracle-w.score:g}',fontsize=15)
    fig.text(.5,.055,table,ha='center',fontsize=9,wrap=True)
    fig.text(.5,.015,cause,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.94));pdf.savefig(fig)
    fig.savefig(out/f'seed_{seed}.png',dpi=160)
    plt.close(fig)


def render_saved(out,config):
    import html
    cases=[json.loads(path.read_text()) for path in sorted(out.glob('seed_*.json'))]
    assignment_rows=list(csv.DictReader((out/'agent_assignments.csv').open()));initials={}
    for row in assignment_rows:
        seed=int(row['seed'])
        if seed not in initials:
            w=World(config);w.reset(seed);initials[seed]=(w,completion_steps(w))
        w,eta=initials[seed];i=int(row['drone']);target=int(row['initial_target'])
        row['initial_type']=int(w.target_type[target]);row['initial_earliest_finish']=int(eta[i,target])
        row['initial_deadline_feasible']=bool(eta[i,target]<=config.horizon)
        case=next((case for case in cases if case['summary']['seed']==seed),None)
        if case is not None:
            row['initial_raw_target']=case['trace'][0]['raw'][i]
            row['resolver_changed_initial']=row['initial_raw_target']!=target
    write_csv(out/'agent_assignments.csv',assignment_rows)
    parts=[];compact=[]
    with PdfPages(out/'failure_cases.pdf') as pdf:
        for case in cases:
            seed=case['summary']['seed'];w=World(config);w.reset(seed);initial=deepcopy(w)
            before=None;index=case['first_loss_trace']['step']
            for row in case['trace']:
                if w.t==index:before=deepcopy(w)
                w.step(final_mission_action(row['chosen']))
            assert abs(w.score-case['summary']['actual_return'])<1e-7
            r=dict(world=w,initial=initial,actual=np.array(case['actual_successful_assignment']),
                trace=case['trace'],first_loss=(index,before,np.array(case['first_loss_trace']['chosen'])))
            plot_case(pdf,seed,r,np.array(case['optimal_assignment']),case['summary']['oracle_return'],
                      case['agents'],case['one_step_rescues'],out)
            rows=[row for row in assignment_rows if int(row['seed'])==seed]
            changed=[row for row in rows if row['changed']=='True']
            def target_label(target):return '미완료' if int(target)<0 else f'T{target}'
            table=[]
            for row in rows:
                style=' class="changed"' if row['changed']=='True' else ''
                raw=row.get('initial_raw_target','')
                prefix=f'T{raw} → ' if str(raw)!=row['initial_target'] else ''
                table.append(f'<tr{style}><td>D{row["drone"]}</td><td>{prefix}T{row["initial_target"]} / {row["initial_earliest_finish"]}</td>'
                    f'<td>{target_label(row["actual_target"])} / {row["actual_finish_step"]}</td><td>{target_label(row["oracle_target"])} / {row["oracle_earliest_finish"]}</td>'
                    f'<td>{float(row["actual_value"]):g} → {float(row["oracle_value"]):g}</td></tr>')
            rescue=case['one_step_rescues']
            rescue_text=''.join(f'<li>D{r["drone"]}: T{r["actual_target"]} → T{r["alternative_target"]}; '
                f'다음 상태에서 달성 가능한 총점 {r["actual_bound"]:g} → {r["alternative_bound"]:g}</li>' for r in rescue[:5])
            if not rescue_text:rescue_text='<li>한 대의 선택 변경으로는 개선되지 않으며 공동 재배정이 필요합니다.</li>'
            targets=[row for row in csv.DictReader((out/'target_outcomes.csv').open()) if int(row['seed'])==seed]
            target_text=' · '.join(f'T{t["target"]}: Type {t["type"]}, work_units {t["initial_work_units"]}, 정책/최적 처리 work_units {t["actual_work_units_removed"]}/{t["oracle_work_units_removed"]}' for t in targets)
            explanation=''
            if seed==20010:
                explanation=('<p><b>역할 교환이 필요한 사례:</b> D2가 T5에 더 가깝지만, D1은 T6를 102스텝에 완료하므로 마감을 넘깁니다. '
                    'D1→T5(91), D2→T6(99)로 나누면 둘 다 완료합니다. 가까운 드론에게 무조건 맡기는 것이 최적은 아닙니다.</p>')
            parts.append(f'<section id="seed-{seed}"><h2>Seed {seed}: {w.score:g} → {case["summary"]["oracle_return"]:g}</h2>'
                f'<img src="seed_{seed}.png"><p>{html.escape(target_text)}</p>'+explanation+
                '<table><tr><th>드론</th><th>초기 actor→배정 / 직행 완료 step</th><th>정책 완료 작업 지점 / 완료 step</th>'
                '<th>최적 배정 / 직행 완료 step</th><th>얻는 가치</th></tr>'+''.join(table)+'</table>'
                f'<details><summary>처음 상한을 잃는 결정: step {index}</summary><ul>{rescue_text}</ul>'
                '<p>이 검사는 해당 한 스텝 뒤에도 최적 후속 배정으로 상한을 달성할 여지가 남는지 확인합니다.</p>'
                f'<a href="seed_{seed}.json">전체 결정·점수·원점수 기록</a></details></section>')
            changes='; '.join(f'D{row["drone"]}: 최초 T{row["initial_target"]}(완료 {row["initial_earliest_finish"]}) / '
                f'정책 {target_label(row["actual_target"])} → 최적 {target_label(row["oracle_target"])}(완료 {row["oracle_earliest_finish"]})' for row in changed)
            compact.append(dict(seed=seed,actual=case['summary']['actual_return'],oracle=case['summary']['oracle_return'],
                gap=case['summary']['gap'],differences=changes))
    write_csv(out/'failure_summary.csv',compact)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>정책과 최적 배정 비교</title>'
        '<style>body{font:16px system-ui;max-width:1250px;margin:28px auto;padding:0 20px;line-height:1.5}'
        'img{width:100%}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{padding:8px;border:1px solid #cbd5e1;text-align:left}'
        '.changed{background:#fff2d9}section{margin:42px 0;border-top:2px solid #94a3b8}details{padding:12px;background:#f1f5f9}</style>'
        '<h1>같은 30개 시나리오의 정책 배정과 물리 상한</h1>'
        '<p>Seed 20000–20029, 마지막 체크포인트, argmax + score resolver. 22개는 상한 달성, 아래 8개는 점수 손실이 있습니다.</p>'
        '<p>최적해가 여러 개이면 실제 완료 배정을 최대한 유지하는 최적해를 고릅니다. 노란 행이 바꿔야 하는 드론입니다. '
        '완료 step은 이동과 처리 시간을 모두 포함하며 제한은 100입니다. -1은 미완료 또는 최적해의 미사용 드론입니다.</p>'
        '<p>각 최적 배정의 점수는 실제 World.step 재생으로 확인했습니다.</p>'+''.join(parts))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=20000)
    parser.add_argument('--episodes',type=int,default=30)
    parser.add_argument('--out',default='runs/target_allocation_optimality_cases')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    run=ROOT/'runs/target_allocation_variable_aligned_basic_warmstart'
    config=Config.load(run/'config.json');torch.set_num_threads(1)
    if args.report_only:
        render_saved(out,config);return
    model=model_from_checkpoint(run/'latest.pt',config);model.eval()
    summaries=[];agent_rows=[];target_rows=[];cases=[];rescue_rows=[]
    with PdfPages(out/'failure_cases.pdf') as pdf:
        for seed in range(args.seed,args.seed+args.episodes):
            r=rollout(model,config,seed);initial=r['initial'];actual=r['actual']
            oracle,plan=optimal_plan(initial,actual)
            assert abs(oracle-assignment_upper_bound(initial))<1e-7
            replay_score,replay_assignment=simulate_fixed(config,seed,plan,return_assignment=True)
            assert abs(replay_score-oracle)<1e-7
            assert np.array_equal(replay_assignment[initial.agent_active],plan[initial.agent_active])
            value={1:2.5,2:2.,3:1.};eta=completion_steps(initial);agents=[]
            for i in np.flatnonzero(initial.agent_active):
                old=int(actual[i]);new=int(plan[i]);row=dict(seed=seed,drone=int(i),
                    initial_target=int(r['first'][i]),initial_raw_target=int(r['trace'][0]['raw'][i]),actual_target=old,oracle_target=new,
                    actual_type=0 if old<0 else int(initial.target_type[old]),
                    oracle_type=0 if new<0 else int(initial.target_type[new]),
                    actual_value=0. if old<0 else value[int(initial.target_type[old])],
                    oracle_value=0. if new<0 else value[int(initial.target_type[new])],
                    actual_finish_step=int(r['finish'][i]),oracle_earliest_finish=-1 if new<0 else int(eta[i,new]),
                    switches=int(r['switches'][i]),actual_travel_km=float(r['travelled'][i]/10),
                    changed=old!=new)
                agents.append(row);agent_rows.append(row)
            for j in np.flatnonzero(initial.target_exists):
                target_rows.append(dict(seed=seed,target=int(j),type=int(initial.target_type[j]),
                    initial_work_units=int(initial.target_work_units[j]),actual_work_units_removed=int(initial.target_work_units[j]-r['world'].target_work_units[j]),
                    oracle_work_units_removed=int((plan==j).sum()),x_km=float(initial.targets[j,0]/10),y_km=float(initial.targets[j,1]/10)))
            gap=oracle-r['world'].score
            assert abs(sum(a['oracle_value']-a['actual_value'] for a in agents)-gap)<1e-7
            summary=dict(seed=seed,actual_return=float(r['world'].score),oracle_return=oracle,gap=float(gap),
                changed_drones=sum(a['changed'] for a in agents),first_irrecoverable_decision=-1)
            if gap>1e-7:
                index,before,chosen=r['first_loss'];summary['first_irrecoverable_decision']=before.t
                rescues,joint,joint_bound=one_step_rescue(model,before,chosen,oracle)
                rescue_rows.extend(dict(seed=seed,step=before.t,**rescue) for rescue in rescues)
                case=dict(summary=summary,agents=agents,optimal_assignment=plan.tolist(),
                    actual_successful_assignment=actual.tolist(),first_action=r['first'].tolist(),
                    first_loss_trace=r['trace'][index],one_step_rescues=rescues,
                    bound_preserving_joint_action=joint,joint_rescue_bound=joint_bound,
                    trace=r['trace'],initial_positions=initial.pos.tolist(),initial_targets=initial.targets.tolist(),
                    oracle_replay=dict(score=replay_score,realized_assignment=replay_assignment.tolist(),exact_assignment_verified=True))
                cases.append(case);(out/f'seed_{seed}.json').write_text(json.dumps(case,indent=2))
                plot_case(pdf,seed,r,plan,oracle,agents,rescues,out)
            else:
                assert np.array_equal(plan[initial.agent_active],actual[initial.agent_active]),seed
            summaries.append(summary);print(summary,flush=True)
    write_csv(out/'all_30.csv',summaries);write_csv(out/'agent_assignments.csv',agent_rows)
    write_csv(out/'target_outcomes.csv',target_rows);write_csv(out/'one_step_rescues.csv',rescue_rows)
    (out/'summary.json').write_text(json.dumps(dict(checkpoint=str(run/'latest.pt'),
        sha256=hashlib.sha256((run/'latest.pt').read_bytes()).hexdigest(),
        seed_start=args.seed,episodes=args.episodes,suboptimal=sum(s['gap']>1e-7 for s in summaries),
        optimal=sum(s['gap']<=1e-7 for s in summaries),mean_return=float(np.mean([s['actual_return'] for s in summaries])),
        mean_oracle=float(np.mean([s['oracle_return'] for s in summaries])),
        method='Exact matching DP: maximize value, minimize changes relative to actual successful assignments, minimize finish-time sum. Every recovered oracle plan replayed through World.step. At the first irrecoverable step, test every legal one-agent action change with the next-state exact bound.'),indent=2))
    render_saved(out,config)


if __name__=='__main__':main()
