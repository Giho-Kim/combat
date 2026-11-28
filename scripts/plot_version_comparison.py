"""Plot saved checkpoint comparisons for drone rescue-site allocation."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'runs/target_allocation_variable_aligned_basic_warmstart'
with (folder / 'time_mask_checkpoint_evaluations.csv').open() as f:
    rows = list(csv.DictReader(f))
oracle = json.loads((ROOT / 'runs/target_allocation_variable_aligned_basic/curve_evaluations/baselines.json').read_text())
x = np.array([float(r['agent_transitions']) for r in rows])
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), facecolor='white')
for ax, metric, title in zip(axes, ['return', 'success'], ['Mean team return', 'Mission success / oracle']):
    scale = 1 if metric == 'return' else oracle['oracle']['mission_success']
    blue = np.array([float(r[f'score_{metric}']) / scale for r in rows])
    orange = np.array([float(r[f'time_mask_{metric}']) / scale for r in rows])
    orange[0] = blue[0]
    ax.plot(x, blue, color='#2563eb', lw=2.3, marker='o', ms=5, zorder=4)
    ax.plot(x, orange, color='#ea580c', lw=2.3, marker='o', ms=5, zorder=5)
    key = 'team_return' if metric == 'return' else 'mission_success'
    levels = [oracle[name][key] / scale for name in ['oracle', 'nearest', 'type_priority']]
    for value, color in zip(levels, ['#111827', '#059669', '#9333ea']):
        ax.axhline(value, color=color, ls='--', lw=1.5)
    if np.isclose(levels[1], levels[2]):
        ax.plot([0, x[-1] / 2], [levels[1]] * 2, color='#059669', ls='--', lw=1.8, zorder=3)
    ax.set_title(title)
    ax.set_xlabel('Training agent transitions')
    ax.set_xticks(np.arange(0, 20_000_001, 2_500_000))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1e6:g}M'))
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
axes[1].yaxis.set_major_formatter(PercentFormatter(1))
axes[1].set_ylim(0, 1.08)
handles = [Line2D([0], [0], color=c, lw=2, ls=ls, marker=m) for c, ls, m in [
    ('#2563eb', '-', 'o'), ('#ea580c', '-', 'o'), ('#111827', '--', None),
    ('#059669', '--', None), ('#9333ea', '--', None)]]
fig.legend(handles, ['Version 0', 'Version 1', 'Oracle', 'Nearest', 'Type priority'],
           loc='lower center', ncol=5, frameon=False)
fig.suptitle('Target Allocation')
fig.tight_layout(rect=(0, .09, 1, .94))
for ext in ['pdf', 'png']:
    fig.savefig(folder / f'learning_curve_versions.{ext}', dpi=180)
plt.close(fig)
