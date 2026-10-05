"""Checkpoint curves on held-out samples of the unconditioned training reset."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from pointmass_rl.env import Config, World, DAMAGE_REWARD_RATE, SUCCESS_BONUS
from pointmass_rl.policies import NearestTargetPolicy, TypePriorityPolicy
import pointmass_rl.strike_ppo as ppo
from scripts.verify_solve123 import physical_optimum


def score_resolver():
    # Isolate this audit from the training implementation. Only the conflict
    # ranking changes; locks, capacities, rejection and reselection stay intact.
    source = inspect.getsource(ppo.evaluation_resolved_action)
    needle = 'float(np.linalg.norm(world.pos[agent] - world.targets[target])), int(agent)))'
    if source.count(needle) != 1:
        raise RuntimeError('Resolver changed; review score-priority audit adapter')
    source = source.replace(needle, '-float(priority[agent, target]), int(agent)))')
    source = source.replace('    initial = targets.copy()\n',
        "    initial = targets.copy()\n"
        "    priority = model.actor_logits(torch.as_tensor(trace['obs'], dtype=torch.float32))[0].detach().numpy()\n")
    namespace = dict(vars(ppo))
    exec(compile(source, '<score-priority-audit>', 'exec'), namespace)
    return namespace['evaluation_resolved_action']


def evaluate_job(run, checkpoint, mode, episodes, seed):
    torch.set_num_threads(1)
    config = Config.load(Path(run) / 'config.json')
    model = ppo.model_from_checkpoint(checkpoint, config).eval() if checkpoint else None
    resolve = score_resolver() if mode == 'score' else ppo.evaluation_resolved_action
    rows = []
    with torch.no_grad():
        for episode in range(episodes):
            world = World(config)
            obs = world.reset(seed + episode)  # No evaluation_task or forced B.
            threshold = world.formation_one_initial_score
            if mode == 'oracle':
                if SUCCESS_BONUS != 0:
                    raise ValueError('Oracle return conversion needs review for nonzero bonus')
                damage = physical_optimum(world)
                metrics = dict(team_return=DAMAGE_REWARD_RATE * damage,
                               mission_success=bool(damage >= threshold), score=damage)
            else:
                schedule = ppo.DecisionSchedule(config)
                policy = (NearestTargetPolicy(config) if mode == 'nearest' else
                          TypePriorityPolicy(config) if mode == 'type_priority' else None)
                while not world.done:
                    if policy is not None:
                        action = policy.predict(obs)
                    elif mode == 'none':
                        action, _ = ppo.scheduled_action(model, world, schedule, resolver=False)
                    else:
                        action, _ = resolve(model, world, schedule, deterministic=True)
                    obs, _, _, _, _ = world.step(action)
                metrics = world.metrics()
            rows.append(dict(seed=seed + episode, B=float(threshold),
                             team_return=float(metrics['team_return']),
                             mission_success=bool(metrics['mission_success']),
                             damage=float(metrics['score'])))
    return rows


def summarize(rows):
    result = {}
    for key in ('team_return', 'mission_success'):
        x = np.asarray([r[key] for r in rows], dtype=float)
        result[key] = float(x.mean())
        result[key + '_se'] = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x)>1 else 0.
    return result


def interquartile_mean(values):
    """Mean of the middle half, with exact quartiles for 100 episodes."""
    ordered = np.sort(np.fromiter(values, dtype=float))
    if len(ordered) == 0 or len(ordered) % 4:
        raise ValueError('IQM requires a nonempty episode count divisible by four')
    quarter = len(ordered) // 4
    return float(ordered[quarter:-quarter].mean())


def plot(out, results, smooth_window=1, statistic='mean'):
    if statistic not in ('mean', 'iqm'):
        raise ValueError('statistic must be mean or iqm')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, FuncFormatter
    styles = [('score', 'Learned policy: score ranking', '#2563eb'),
              ('distance', 'Learned policy: distance ranking', '#d97706'),
              ('oracle', 'Oracle (physical upper bound)', '#111827'),
              ('nearest', 'Nearest first', '#059669'),
              ('type_priority', 'Type priority', '#9333ea')]
    oracle_success = next(r['summary']['mission_success'] for r in results if r['mode'] == 'oracle')
    if oracle_success <= 0:
        raise ValueError('Relative success needs at least one Oracle success')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, key, title in zip(axes, ('team_return', 'mission_success'),
                              ('Episode return IQM' if statistic == 'iqm' else 'Episode return',
                               'Success rate / Oracle success rate')):
        for mode, label, color in styles:
            entries = sorted([r for r in results if r['mode']==mode], key=lambda r:r['steps'])
            if not entries: continue
            metric = lambda r: (r['summary'][key] / oracle_success if key == 'mission_success'
                                else interquartile_mean(row['team_return'] for row in r['episodes'])
                                if statistic == 'iqm' else r['summary'][key])
            if mode in ('oracle', 'nearest', 'type_priority'):
                ax.axhline(metric(entries[0]), color=color, ls='--', lw=1.6, label=label)
            else:
                if len(entries) < smooth_window:
                    raise ValueError('Moving-average window exceeds available checkpoints')
                steps = np.asarray([r['steps'] for r in entries])
                values = np.asarray([metric(r) for r in entries])
                averages = np.convolve(values, np.ones(smooth_window) / smooth_window, mode='valid')
                ax.plot(steps[smooth_window - 1:], averages,
                        color=color, marker='o', ms=3.5, lw=2, label=label)
        ax.set_title(title)
        ax.set_xlabel('Training agent transitions')
        ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1e6:g}M'))
        if key == 'mission_success':
            ax.set_ylim(.45,1.05)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.set_ylim(bottom=5)
            ax.set_ylabel('Undiscounted team return')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    episodes = len(results[0]['episodes'])
    suffix = ('_iqm' if statistic == 'iqm' else '') + (f'_ma{smooth_window}' if smooth_window > 1 else '')
    smoothing = f' | trailing {smooth_window}-checkpoint average' if smooth_window > 1 else ''
    summary = ' | return IQM across episodes' if statistic == 'iqm' else ''
    fig.suptitle(f'Training-distribution evaluation | {episodes} fixed held-out scenarios | deterministic actions{summary}{smoothing}')
    fig.tight_layout(rect=(0,.13,1,.94))
    fig.savefig(out/f'learning_curves{suffix}.png',dpi=180)
    fig.savefig(out/f'learning_curves{suffix}.pdf')
    plt.close(fig)
    return out/f'learning_curves{suffix}.png'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/target_coma_v15'))
    parser.add_argument('--out',type=Path,default=Path('runs/target_coma_v15_train_dist_audit'))
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--seed',type=int,default=30000)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--max-steps',type=int,default=None)
    parser.add_argument('--smooth-window',type=int,default=1,
                        help='Trailing checkpoint moving average for learned-policy curves')
    parser.add_argument('--statistic',choices=('mean','iqm'),default='mean',
                        help='Aggregation of episode returns within each checkpoint; success stays a rate')
    args=parser.parse_args()
    if args.smooth_window < 1:
        parser.error('--smooth-window must be positive')
    args.out.mkdir(parents=True,exist_ok=True)
    checkpoints=sorted(args.run.glob('checkpoint_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    if args.max_steps is not None:
        checkpoints=[p for p in checkpoints if int(p.stem.split('_')[-1])<=args.max_steps]
    manifest=dict(run=str(args.run.resolve()),episodes=args.episodes,seed=args.seed,
                  reset='world.reset(seed): unconditioned training distribution',
                  score_priority='raw actor logit, fixed within each resolver call',
                  oracle='physical damage upper bound converted to undiscounted return',
                  checkpoints=[str(p) for p in checkpoints],config=json.loads((args.run/'config.json').read_text()))
    manifest_path=args.out/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        old_checkpoints=previous.pop('checkpoints')
        settings={k:v for k,v in manifest.items() if k!='checkpoints'}
        if previous!=settings or not set(old_checkpoints).issubset(manifest['checkpoints']):
            raise ValueError('Audit settings differ or checkpoints were removed; use a fresh output directory')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    tasks=[(0,None,m) for m in ('oracle','nearest','type_priority')]
    tasks += [(int(p.stem.split('_')[-1]),str(p),m) for p in checkpoints for m in ('score','distance','none')]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={}
        for steps,path,mode in tasks:
            target=args.out/f'{mode}_{steps}.json'
            if target.exists():
                results.append(json.loads(target.read_text()))
            else:
                future=pool.submit(evaluate_job,str(args.run),path,mode,args.episodes,args.seed)
                futures[future]=(steps,mode,target)
        for future in as_completed(futures):
            steps,mode,target=futures[future]
            rows=future.result()
            result=dict(steps=steps,mode=mode,summary=summarize(rows),episodes=rows)
            target.write_text(json.dumps(result,indent=2)+'\n')
            results.append(result)
            print(json.dumps(dict(completed=len(results),total=len(tasks),steps=steps,mode=mode,**result['summary'])),flush=True)
    with (args.out/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['steps','mode','team_return','team_return_se','mission_success','mission_success_se'])
        writer.writeheader()
        for result in sorted(results,key=lambda r:(r['steps'],r['mode'])):
            writer.writerow(dict(steps=result['steps'],mode=result['mode'],**result['summary']))
    if args.statistic == 'iqm':
        with (args.out/'summary_iqm.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['steps','mode','team_return_iqm','mission_success'])
            writer.writeheader()
            for result in sorted(results,key=lambda r:(r['steps'],r['mode'])):
                writer.writerow(dict(steps=result['steps'],mode=result['mode'],
                    team_return_iqm=interquartile_mean(row['team_return'] for row in result['episodes']),
                    mission_success=result['summary']['mission_success']))
    print(str(plot(args.out,results,args.smooth_window,args.statistic)),flush=True)


if __name__=='__main__':
    main()
