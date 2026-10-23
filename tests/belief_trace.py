"""Deterministic belief-mode rollouts reduced to per-step digests.

The golden file pins the environment's observable behavior so refactors of the
decision/commitment logic can be checked for exact equivalence. Regenerate it
only for an intended behavior change:

    python -m tests.belief_trace
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from pointmass_rl.belief_env import BeliefWorld
from pointmass_rl.env import Config, strike_action
from pointmass_rl.task_automaton import TaskPhase


GOLDEN = Path(__file__).with_name('data') / 'belief_trace_digests.json'
STEPS = 400
PHASE_CODES = {TaskPhase.INACTIVE: 'X', TaskPhase.ENGAGED: 'E', TaskPhase.IDLE: '-',
               TaskPhase.TRANSIT: 'T', TaskPhase.SWEEPING: 'S', TaskPhase.ARRIVED: 'A'}


def scenarios():
    base = Config.load(Path(__file__).parents[1] / 'configs' / 'belief.json')
    short = dict(horizon=STEPS)
    return {
        'gaussian_intel': replace(base, **short),
        'histogram_intel': replace(base, belief_representation='histogram', **short),
        'histogram_plain': replace(base, belief_representation='histogram',
                                   belief_initial_intel=False, belief_soft_detection=False, **short),
        'no_lawnmower': replace(base, belief_lawnmower=False, **short),
        'commit_interval': replace(base, commit_target=True, decision_interval=5, **short),
    }


def _choose(rng, mask, cells, last):
    """Sticky masked random policy, biased toward tracks so that transit,
    sweeps, arrivals and locks all occur."""
    action = np.zeros(len(mask), dtype=int)
    for i, row in enumerate(mask):
        if row[last[i]] and rng.random() < .97:
            action[i] = last[i]
            continue
        options = np.flatnonzero(row)
        tracks = options[options >= cells]
        action[i] = rng.choice(tracks if len(tracks) and rng.random() < .5 else options)
    return action


def phase_codes(world):
    """One letter per agent: X inactive, E engaged, - idle, T transit,
    S sweeping, A arrived."""
    return ''.join(PHASE_CODES[phase] for phase in world.tasks.phases())


def rollout(config, seed):
    world = BeliefWorld(config)
    obs = world.reset(seed)
    rng = np.random.default_rng(seed + 1)
    digests, phases, counts = [], [], dict(sweeping=0, arrived=0, locked=0, held=0)
    last = np.zeros(world.n, dtype=int)
    while True:
        h = hashlib.sha256()
        for key in sorted(obs):
            h.update(key.encode())
            h.update(np.ascontiguousarray(obs[key]).tobytes())
        h.update(np.ascontiguousarray(world._world.pos).tobytes())
        h.update(np.ascontiguousarray(world._world.heading).tobytes())
        mask, decision = obs['mask'], obs['decision']
        codes = phase_codes(world)
        phases.append(codes)
        counts['sweeping'] += codes.count('S')
        counts['arrived'] += codes.count('A')
        counts['locked'] += codes.count('E')
        counts['held'] += int((obs['active'] & ~decision & (mask.sum(1) == 1)).sum())
        if world.done:
            digests.append(h.hexdigest()[:16])
            break
        action = last = _choose(rng, mask, world.belief.cells, last)
        obs, reward, _, _, _ = world.step(strike_action(action))
        h.update(np.float64(reward).tobytes())
        digests.append(h.hexdigest()[:16])
    return dict(digests=digests, phases=phases, counts=counts, return_=float(world._world.rewards_total))


def compute():
    return {name: {str(seed): rollout(config, seed) for seed in (3, 11)}
            for name, config in scenarios().items()}


EVENT_GOLDEN = GOLDEN.with_name('belief_event_trace_digests.json')


def event_scenarios():
    """Event decision rule without a hold cap, with and without a report."""
    base = scenarios()
    return {name: replace(base[source], belief_decision_rule='event', decision_interval=1800)
            for name, source in (('gaussian_event', 'gaussian_intel'), ('histogram_event', 'histogram_plain'))}


def compute_event():
    return {name: {str(seed): rollout(config, seed) for seed in (3, 11)}
            for name, config in event_scenarios().items()}


if __name__ == '__main__':
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(compute()) + '\n')
    EVENT_GOLDEN.write_text(json.dumps(compute_event()) + '\n')
    print(GOLDEN, EVENT_GOLDEN)
