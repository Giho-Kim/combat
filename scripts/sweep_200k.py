"""Reproducible short experiments; training reward variants never change evaluation Damage.

Run from the repository root with python -m scripts.sweep_200k.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from pointmass_rl.env import Config, World, EVALUATION_TASKS, evaluation_task_for_episode, strike_action
from pointmass_rl import strike_ppo as ppo


BASE = dict(n_envs=8, rollout_steps=200, epochs=5, learning_rate=5e-4,
            gae_lambda=.97, entropy_coef=.01, policy_clip=.2, reward='original')
VARIANTS = {
    'baseline': {},
    'batch32_epoch10': dict(n_envs=32, epochs=10),
    'lambda1': dict(gae_lambda=1.),
    'lambda1_noentropy': dict(gae_lambda=1., entropy_coef=0.),
    'lambda1_lowentropy': dict(gae_lambda=1., entropy_coef=.001),
    'lambda1_lr1e3': dict(gae_lambda=1., learning_rate=.001),
    'lambda1_lr3e3': dict(gae_lambda=1., learning_rate=.003),
    'lambda1_minibatch512': dict(gae_lambda=1., minibatch_size=512),
    'lambda1_env4': dict(gae_lambda=1., n_envs=4, rollout_steps=100),
    'clip03_lr1e3': dict(policy_clip=.3, learning_rate=.001),
    'reward14': dict(reward='one_four'),
    'reward14_lambda1': dict(reward='one_four', gae_lambda=1.),
    'temperature02': dict(policy_temperature=.2, gae_lambda=1.),
    'temperature005': dict(policy_temperature=.05, gae_lambda=1.),
    'shared_target': dict(algorithm='target_mappo'),
    'shared_target_lambda1': dict(algorithm='target_mappo', gae_lambda=1.),
    'shared_target_lambda1_lr1e3': dict(algorithm='target_mappo', gae_lambda=1., learning_rate=.001),
    'shared_target_lambda1_lowentropy': dict(algorithm='target_mappo', gae_lambda=1., entropy_coef=.001),
}


@contextmanager
def training_reward(mode):
    """Change only reward, leaving positions, damage accounting and observations intact."""
    if mode == 'original':
        yield
        return
    if mode != 'one_four':
        raise ValueError(mode)
    original = World.step

    def step(world, action):
        before = world.target_life.copy()
        active = world.agent_active.copy()
        obs, reward, terminated, truncated, metrics = original(world, action)
        # Type 1: remaining reward value 5 -> 4 -> 0, damage stays 5 -> 2.5 -> 0.
        mask = world.target_type == 1
        old_value = np.where(before >= 2, 5., np.where(before == 1, 4., 0.))
        new_value = np.where(world.target_life >= 2, 5.,
                             np.where(world.target_life == 1, 4., 0.))
        correction = .1 * ((old_value - new_value - 2.5 * (before - world.target_life))[mask]).sum()
        if active.any():
            reward[active] += correction / active.sum()
        return obs, reward, terminated, truncated, metrics

    with patch.object(World, 'step', step):
        yield


@torch.no_grad()
def evaluate(model, config, episodes, seed):
    worlds = [World(config) for _ in range(episodes)]
    schedules = [ppo.DecisionSchedule(config) for _ in worlds]
    for index, world in enumerate(worlds):
        world.reset(seed + index, evaluation_task=evaluation_task_for_episode(config, index))
    previous = np.full((episodes, config.n_agents), -1)
    switches = np.zeros(episodes, dtype=int)
    for _ in range(config.horizon):
        action, _ = ppo.allocated_actions(model, worlds, schedules, deterministic=True)
        targets = action['target']
        active = np.stack([world.agent_active for world in worlds])
        switches += (active & (previous >= 0) & (previous != targets)).sum(axis=1)
        previous = targets.copy()
        for index, world in enumerate(worlds):
            world.step(strike_action(targets[index]))
    rows = [dict(seed=seed+i, case=i % 3 + 1, switches=int(switches[i]), **world.metrics())
            for i, world in enumerate(worlds)]
    means = [float(np.mean([row['score'] for row in rows if row['case'] == case]))
             for case in (1, 2, 3)]
    return dict(cases=means, mean=float(np.mean(means)), episodes=rows)


def run_one(name, seed, root, steps, eval_episodes):
    torch.set_num_threads(1)
    spec = dict(BASE, **VARIANTS[name])
    directory = Path(root) / f'{name}_seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    if (directory / 'settings.json').exists():
        raise RuntimeError(f'Incomplete experiment exists: {directory}; use a new output directory')
    config = replace(Config.load('configs/default.json'), gae_lambda=spec['gae_lambda'])
    config.save(directory / 'config.json')
    settings = dict(spec, seed=seed, requested_transitions=steps,
                    eval_episodes=eval_episodes, screen_seed=10000, heldout_seed=20000)
    (directory / 'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    started = time.monotonic()
    kwargs = {key: value for key, value in spec.items() if key not in ('reward', 'gae_lambda')}
    with training_reward(spec['reward']):
        model, episodes, _, completed = ppo.train_mappo(config, steps, seed=seed,
            eval_interval=0, diagnostics_path=directory/'updates.csv', **kwargs)
    # Always evaluate the final model, without selecting a lucky intermediate checkpoint.
    ppo.save_checkpoint(directory/'final.pt', model, settings)
    screen = evaluate(model, config, eval_episodes, 10000)
    heldout = evaluate(model, config, eval_episodes, 20000)
    result = dict(name=name, seed=seed, settings=settings, completed=completed,
        seconds=round(time.monotonic()-started, 2), screen=screen, heldout=heldout,
        training_episode_count=len(episodes),
        training_last100_damage=float(np.mean([row['score'] for row in episodes[-100:]])))
    result_path.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[7])
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--eval-episodes', type=int, default=30)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.eval_episodes % 3 or args.eval_episodes <= 0:
        parser.error('--eval-episodes must be a positive multiple of 3')
    Path(args.out).mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_one, name, seed, args.out, args.steps, args.eval_episodes):
                (name, seed) for name in args.variants for seed in args.seeds}
        for job in as_completed(jobs):
            result = job.result()
            results.append(result)
            print(json.dumps({key:result[key] for key in ('name','seed','seconds','completed')} |
                  dict(screen=result['screen']['cases'], heldout=result['heldout']['cases'])), flush=True)
    results.sort(key=lambda row: (row['name'], row['seed']))
    (Path(args.out)/'summary.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
