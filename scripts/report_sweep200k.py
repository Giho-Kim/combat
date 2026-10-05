"""Aggregate the short experiments without selecting intermediate checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    records = []
    for root in args.roots:
        for path in Path(root).glob('*/result.json'):
            record = json.loads(path.read_text())
            record['artifact'] = str(path.parent)
            records.append(record)
    records.sort(key=lambda row: (row['name'], row['seed']))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = ['variant', 'seed', 'transitions', 'seconds', 'screen_case1', 'screen_case2',
              'screen_case3', 'heldout_case1', 'heldout_case2', 'heldout_case3', 'artifact']
    with (out/'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(dict(variant=r['name'], seed=r['seed'], transitions=r['completed'],
                seconds=r['seconds'], artifact=r['artifact'],
                **{f'{split}_case{i+1}': value for split in ('screen', 'heldout')
                   for i, value in enumerate(r[split]['cases'])}))
    lines = ['# 200k-transition experiments', '',
        f'{len(records)} completed training runs; {len(set(r["name"] for r in records))} configurations.', '',
        'All runs start from scratch. Default: 5 drones, gamma=1, lambda=.97, 8 envs, '
        '200 rollout steps, 5 PPO epochs, learning rate=.0005, entropy=.01, clip=.2. '
        'Half case layouts, half uniform-B random layouts. Decisions remain every physical step.', '',
        'Each final checkpoint is evaluated on 30 scenarios at seeds 10000–10029 (screen) '
        'and 20000–20029 (held-out), 10 episodes per case. No best-intermediate-checkpoint selection. '
        'All reported scores use original Damage: Type 1 hits are 2.5 + 2.5, including reward14 experiments.', '',
        'The one_four training reward changes partial-hit valuation (1 + 4); it is not '
        'claimed to preserve the training objective. Shared-target variants change the actor architecture '
        'and explicitly add Euclidean distance, but do not mask unreachable targets or impose target priorities.', '',
        '| Variant | Seed | Screen cases 1 / 2 / 3 | Held-out cases 1 / 2 / 3 |',
        '|---|---:|---|---|']
    for r in records:
        fmt = lambda split: ' / '.join(f'{v:.2f}' for v in r[split]['cases'])
        lines.append(f'| {r["name"]} | {r["seed"]} | {fmt("screen")} | {fmt("heldout")} |')
    lines += ['', '## Repeated-seed comparisons', '',
        'Means and sample standard deviations below are across training seeds, '
        'not confidence intervals. Three seeds do not establish statistical significance.', '']
    for name in sorted(set(r['name'] for r in records)):
        group = [r for r in records if r['name'] == name]
        if len(group) < 2:
            continue
        for split in ('screen', 'heldout'):
            values = [[r[split]['cases'][i] for r in group] for i in range(3)]
            summary = ' / '.join(f'{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}' for v in values)
            lines.append(f'- {name}, {split}, seeds {[r["seed"] for r in group]}: {summary}')
    lines += ['', '## Exact settings', '']
    for r in records:
        lines += [f'### {r["name"]}, seed {r["seed"]}', '', '```json',
                  json.dumps(r['settings'], indent=2), '```', '', f'Artifacts: `{r["artifact"]}`', '']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(out/'REPORT.md')
    print(out/'results.csv')


if __name__ == '__main__':
    main()
