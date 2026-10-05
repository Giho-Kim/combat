"""Trace a frozen policy without changing training or environment settings."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pointmass_rl.env import Config, World, evaluation_task_for_episode
from pointmass_rl.policies import TypePriorityPolicy
from pointmass_rl.strike_ppo import (
    DecisionSchedule, StrikeActorCritic, actor_observation, load_checkpoint,
)


@torch.no_grad()
def rollout(model, config, seed, task, mode, sampling_seed=0, trace=False):
    torch.manual_seed(sampling_seed)
    world = World(config)
    obs = world.reset(seed, evaluation_task=task)
    schedule = DecisionSchedule(config)
    baseline = TypePriorityPolicy(config)
    initial = world.snapshot()
    previous = np.full(world.n, -1)
    switches = np.zeros(world.n, dtype=int)
    events = []
    frames = []
    while not world.done:
        active = world.agent_active.copy()
        participants = world.strike_participants.copy()
        life = world.target_life.copy()
        if mode == 'type_priority':
            action = baseline.predict(obs)
            decision = active & (world.locked_targets() < 0)
            probabilities = None
        else:
            actor_obs = torch.as_tensor(actor_observation(world), dtype=torch.float32)
            available = model.target_mask(actor_obs).numpy()
            mask, decision = schedule.mask(world, available)
            action_mask = torch.as_tensor(mask)
            probabilities = model.distributions(actor_obs, action_mask).probs.tolist()
            action, _ = model.act(actor_obs, deterministic=mode == 'deterministic',
                                  action_mask=action_mask)
            schedule.record(world, action['target'], decision)
        targets = action['target']
        changed = active & (previous >= 0) & (targets != previous)
        switches += changed
        if trace:
            frames.append(dict(state=world.snapshot(), targets=targets.tolist(),
                               decision=decision.tolist(), probabilities=probabilities))
        obs, _, _, _, _ = world.step(action)
        started = np.argwhere(world.strike_participants & ~participants).tolist()
        expended = np.flatnonzero(active & ~world.agent_active).tolist()
        if started or expended or np.any(life != world.target_life):
            events.append(dict(step=world.t, started_target_agent=started,
                               expended_agents=expended, life=world.target_life.tolist(),
                               score=world.score))
        previous = targets.copy()
    return dict(seed=seed, task=task, mode=mode, sampling_seed=sampling_seed,
                score=world.score, switches=switches.tolist(), events=events,
                initial=initial, final=world.snapshot(), frames=frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=30)
    args = parser.parse_args()
    metadata = json.loads((args.run / 'training_metadata.json').read_text())
    config = Config.load(args.run / 'config.json')
    torch.set_num_threads(1)
    model = StrikeActorCritic(config.n_targets, config.n_agents)
    checkpoint_metadata = load_checkpoint(args.run / 'latest.pt', model)
    model.eval()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for episode in range(metadata['eval_episodes']):
        seed = metadata['eval_seed'] + episode
        task = evaluation_task_for_episode(config, episode)
        for mode in ('deterministic', 'type_priority'):
            result = rollout(model, config, seed, task, mode, trace=True)
            (args.out / f'{seed}_{mode}.json').write_text(json.dumps(result))
            summary = {key: result[key] for key in ('seed', 'task', 'mode', 'score', 'switches', 'events')}
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        if episode % 3 == 1:
            scores = []
            for trial in range(args.samples):
                result = rollout(model, config, seed, task, 'stochastic', trial)
                scores.append(result['score'])
                if result['score'] >= 12 and not any(score >= 12 for score in scores[:-1]):
                    traced = rollout(model, config, seed, task, 'stochastic', trial, trace=True)
                    (args.out / f'{seed}_stochastic_success.json').write_text(json.dumps(traced))
            summary = dict(seed=seed, mode='stochastic', samples=args.samples,
                           mean=float(np.mean(scores)), maximum=max(scores),
                           successes=sum(score >= 12 for score in scores), scores=scores)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (args.out / 'summary.json').write_text(json.dumps(
        dict(run=str(args.run.resolve()), checkpoint_metadata=checkpoint_metadata,
             summaries=summaries), indent=2))


if __name__ == '__main__':
    main()
