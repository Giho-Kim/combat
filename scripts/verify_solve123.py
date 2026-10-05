"""Audit fixed and independent evaluation seeds, including physical optima."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from pointmass_rl.env import Config, World, evaluation_task_for_episode
from pointmass_rl.strike_lookahead import completion_steps
from pointmass_rl.strike_ppo import (DecisionSchedule, model_from_checkpoint,
                                    save_checkpoint, scheduled_action)


def physical_optimum(world):
    """Exact maximum D for stationary, deterministic, expendable-drone tasks.

    Independent straight-line travel is fastest; each drone can spend one life.
    Enumerating assignments thus gives the attainable damage upper bound.
    This is for validation only and never supplies actions to the actor.
    """
    if world.c.target_motion_scale != 0 or world.c.strike_probability != 1:
        raise ValueError('Physical oracle requires stationary deterministic targets')
    if world.c.n_targets ** world.n > 1_000_000:
        raise ValueError('Exact audit enumeration is limited to 1,000,000 joint assignments')
    assignments = np.asarray(list(itertools.product(range(world.c.n_targets), repeat=world.n)))
    feasible = completion_steps(world) <= world.c.horizon - world.t
    values = np.zeros(len(assignments))
    for target in np.flatnonzero(world.target_exists & ~world.destroyed):
        count = ((assignments == target) & feasible[:, target] & world.agent_active).sum(-1)
        count = np.minimum(count, world.target_life[target])
        values += count * {1: 2.5, 2: 2., 3: 1.}[int(world.target_type[target])]
    return float(values.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--episodes', type=int, default=90)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--out', required=True)
    parser.add_argument('--deadline-mask', action='store_true',
        help='Explicit post-training execution intervention; recorded in report and saved snapshot')
    args = parser.parse_args()
    if args.episodes <= 0 or args.episodes % 3:
        parser.error('episodes must be a positive multiple of 3')
    torch.set_num_threads(1)
    config = Config.load(args.config)
    model = model_from_checkpoint(args.model, config)
    training_mask = model.training_settings.get('deadline_mask', False)
    if args.deadline_mask:
        model.training_settings['deadline_mask'] = True
    model.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config.save(out / 'config.json')
    metadata = dict(source_checkpoint=args.model,
        training_deadline_mask=training_mask,
        evaluation_deadline_mask=model.training_settings.get('deadline_mask', False),
        post_training_deadline_mask=bool(args.deadline_mask and not training_mask))
    save_checkpoint(out / 'model.pt', model, metadata)
    rows, summaries = [], {}
    with torch.no_grad():
        for resolver in (False, True):
            for episode in range(args.episodes):
                world = World(config)
                world.reset(args.seed + episode,
                    evaluation_task=evaluation_task_for_episode(config, episode))
                optimum = physical_optimum(world)
                schedule = DecisionSchedule(config)
                while not world.done:
                    action, _ = scheduled_action(model, world, schedule, resolver=resolver)
                    world.step(action)
                rows.append(dict(seed=args.seed + episode, case=episode % 3 + 1,
                    resolver=resolver, optimum=optimum, damage=world.score,
                    return_value=world.rewards_total))
            summary = []
            for case in (1, 2, 3):
                group = [row for row in rows if row['resolver'] == resolver and row['case'] == case]
                summary.append(dict(case=case, episodes=len(group),
                    damage=float(np.mean([row['damage'] for row in group])),
                    optimum=float(np.mean([row['optimum'] for row in group])),
                    optimum_matches=sum(abs(row['damage'] - row['optimum']) < 1e-6 for row in group)))
            print(json.dumps(dict(resolver=resolver, summary=summary)), flush=True)
            summaries['resolver' if resolver else 'no_resolver'] = summary
            report = dict(metadata, seed=args.seed, episodes_per_mode=args.episodes,
                          summaries=summaries, rows=rows)
            (out / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
