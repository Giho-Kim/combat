"""Evaluate exploration PPO with the revisit guard against the original heuristic."""

import argparse
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from pointmass_rl.belief_training import (GUARDED_EXPLORATION_RESOLVER, evaluate_belief,
                                          load_belief_checkpoint, summarize_belief_rows)
from pointmass_rl.env import strike_action
from pointmass_rl.exploration import PolygonExplorationWorld


def initialize(checkpoint):
    global model, config
    torch.set_num_threads(1)
    model, config, _ = load_belief_checkpoint(checkpoint)
    if not config.belief_exploration_only or config.belief_representation != 'gaussian_uniform':
        raise ValueError('The revisit guard requires a Gaussian belief exploration checkpoint')


@torch.no_grad()
def evaluate(task):
    policy, seed = task
    if policy == 'ppo_revisit_guard':
        _, rows = evaluate_belief(model, config, episodes=1, seed=seed,
                                  resolver=GUARDED_EXPLORATION_RESOLVER)
        return dict(policy=policy, **rows[0])

    world = PolygonExplorationWorld(config)
    world.reset(seed)
    centers = world.belief.bounds.mean(axis=1)
    one_hot = np.eye(world.action_count, dtype=np.float32)

    while not world.done:
        mask, decision = world.action_mask()
        action = mask.argmax(axis=1)
        deciding = np.flatnonzero(decision)
        state = world._world
        if deciding.size:
            mass = world.belief.weights.sum(axis=1)
            distance_km = np.linalg.norm(state.pos[:, None] - centers[None], axis=-1) / 10
            heuristic = mass[None, :] / (1 + distance_km)
            occupied = set(action[state.agent_active & ~decision])
            for agent in deciding:
                legal = np.flatnonzero(mask[agent])
                candidates = [region for region in legal if region not in occupied]
                if not candidates:
                    candidates = legal
                action[agent] = candidates[np.argmax(heuristic[agent, candidates])]
                occupied.add(action[agent])

        _, _, _, _, metrics = world.step(strike_action(action), one_hot[action], decisions_only=True)

    return dict(policy=policy, seed=seed, interventions=0, **metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', type=Path, help='Optional path for per-episode CSV')
    args = parser.parse_args()
    if min(args.episodes, args.workers) < 1:
        parser.error('--episodes and --workers must be positive')

    policies = ('belief_heuristic', 'ppo_revisit_guard')
    tasks = [(policy, seed) for seed in range(args.seed, args.seed + args.episodes)
             for policy in policies]
    if args.workers == 1:
        initialize(args.checkpoint)
        rows = [evaluate(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)),
                                 mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize, initargs=(args.checkpoint,)) as pool:
            rows = list(pool.map(evaluate, tasks))

    for policy in policies:
        records = [row for row in rows if row['policy'] == policy]
        summary = summarize_belief_rows(records)
        print(f"{policy}: {summary['team_return']:.3f} points across {len(records)} episodes")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
