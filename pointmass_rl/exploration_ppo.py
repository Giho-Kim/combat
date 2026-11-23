"""On-policy PPO over asynchronous, complete polygon decisions.

Physics and sensing still run every second. Each drone has its own sequence of
options, ending at sweep completion or episode end. Full episodes are collected
with frozen weights, so Monte Carlo targets need no learned Q or bootstrap.
"""
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .belief_policy import stack_observations, subset
from .belief_training import (GUARDED_EXPLORATION_RESOLVER, _write_csv, evaluate_belief,
                              save_belief_checkpoint, summarize_belief_rows)
from .env import strike_action
from .exploration import ExplorationPPOActorCritic, PolygonExplorationWorld


_evaluation_model = None
_evaluation_config = None
EVALUATION_RESOLVERS = ('none', 'score_time_limit')


def score_key(summary):
    return summary['team_return'], summary['mission_success']


def _initialize_evaluation(config, hidden, weights, device):
    """Spawn independent environments with one frozen policy copy per process."""
    global _evaluation_model, _evaluation_config
    torch.set_num_threads(1)
    _evaluation_config = config
    _evaluation_model = ExplorationPPOActorCritic(hidden)
    _evaluation_model.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})
    _evaluation_model.to(device).eval()


def _evaluate_episode(task):
    resolver, episode, seed = task
    _, rows = evaluate_belief(_evaluation_model, _evaluation_config, episodes=1,
                              seed=seed, resolver=resolver)
    return resolver, episode, rows[0]


def evaluate_exploration(model, config, episodes=10, seed=10000, workers=4,
                         resolvers=EVALUATION_RESOLVERS):
    """Parallelize all resolver/episode pairs, retaining serial seeds and order.

    Use spawn rather than forking the CUDA training process. Worker actors use
    the training model's device; each worker runs physics and belief on its CPU.
    NumPy snapshots avoid passing live CUDA tensors or shared Torch storage.
    """
    if min(episodes, workers) <= 0:
        raise ValueError('Evaluation episodes and workers must be positive')
    if workers == 1:
        return {resolver: evaluate_belief(model, config, episodes, seed, resolver)
                for resolver in resolvers}
    weights = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    tasks = [(resolver, episode, seed+episode)
             for resolver in resolvers for episode in range(episodes)]
    rows = {resolver: [None]*episodes for resolver in resolvers}
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
            mp_context=multiprocessing.get_context('spawn'), initializer=_initialize_evaluation,
            initargs=(config, model.hidden, weights, str(next(model.parameters()).device))) as pool:
        for resolver, episode, row in pool.map(_evaluate_episode, tasks, chunksize=1):
            rows[resolver][episode] = row
    return {resolver: (summarize_belief_rows(records), records) for resolver, records in rows.items()}


def finish_trajectory(trajectory, gamma):
    """Discount in physical steps, not in the number of polygon selections."""
    future = 0.
    for sample in reversed(trajectory):
        future = sample['reward'] + gamma**sample['duration'] * future
        sample['return'] = future


@torch.no_grad()
def collect_decisions(model, worlds, rng, minimum_samples):
    """Finish every started episode before changing the behavior policy.

minimum_samples is a lower bound; collecting whole waves of episodes also
retains decisions whose sweeps are interrupted by the finite mission horizon.
Every option receives the shared team rewards during its own time interval.
"""
    device = next(model.parameters()).device
    samples, episodes, transitions = [], [], 0
    while len(samples) < minimum_samples:
        observations = [w.reset(int(rng.integers(2**31))) for w in worlds]
        pending = [[None] * w.n for w in worlds]
        trajectories = [[[] for _ in range(w.n)] for w in worlds]
        running = set(range(len(worlds)))
        while running:
            ids = sorted(i for i in running if observations[i]['decision'].any())
            selected = {}
            if ids:
                batch = stack_observations([observations[i] for i in ids], device=device)
                distribution = model.distribution(batch)
                actions = distribution.sample()
                logps = distribution.log_prob(actions).cpu()
                values = model.state_value(batch).cpu()
                probabilities = distribution.probs.cpu().numpy()
                actions = actions.cpu()
                for row, i in enumerate(ids):
                    action = actions[row].numpy()
                    selected[i] = action, probabilities[row]
                    observation = {k: v.copy() for k, v in observations[i].items()}
                    for agent in np.flatnonzero(observation['decision']):
                        pending[i][agent] = dict(obs=observation, agent=int(agent),
                            action=action.copy(), old_logp=float(logps[row, agent]),
                            value=float(values[row]), reward=0., duration=0)
            for i in sorted(running):
                world, obs = worlds[i], observations[i]
                if i in selected:
                    action, probabilities = selected[i]
                else:
                    # Locked options have one legal action; no network call.
                    action = obs['mask'].argmax(-1)
                    probabilities = np.eye(world.action_count, dtype=np.float32)[action]
                transitions += int(obs['active'].sum())
                new_obs, reward, terminated, truncated, metrics = world.step(strike_action(action), probabilities)
                done = terminated or truncated
                for agent, sample in enumerate(pending[i]):
                    if sample is None:
                        continue
                    sample['reward'] += world.c.discount_gamma**sample['duration'] * reward
                    sample['duration'] += 1
                    if done or new_obs['decision'][agent]:
                        trajectories[i][agent].append(sample)
                        pending[i][agent] = None
                observations[i] = new_obs
                if done:
                    running.remove(i)
                    episodes.append(dict(**metrics))
                    for trajectory in trajectories[i]:
                        finish_trajectory(trajectory, world.c.discount_gamma)
                        samples.extend(trajectory)
    return samples, episodes, transitions


def update_decisions(model, optimizer, samples, epochs, critic_epochs, width, entropy_coef, clip):
    device = next(model.parameters()).device
    batch = stack_observations([s['obs'] for s in samples], device=device)
    agents = torch.tensor([s['agent'] for s in samples], device=device)
    actions = torch.as_tensor(np.stack([s['action'] for s in samples]), device=device)
    old_logps = torch.tensor([s['old_logp'] for s in samples], device=device)
    returns = torch.tensor([s['return'] for s in samples], dtype=torch.float32, device=device)
    # Freeze the baseline recorded under the behavior policy before critic fitting.
    advantages = returns - torch.tensor([s['value'] for s in samples], device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-5)
    value_losses, policy_losses, entropies = [], [], []
    for _ in range(critic_epochs):
        for ix in torch.randperm(len(samples), device=device).split(width):
            loss = nn.functional.huber_loss(model.state_value(subset(batch, ix)), returns[ix], delta=10.)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            value_losses.append(float(loss.detach()))
    for _ in range(epochs):
        for ix in torch.randperm(len(samples), device=device).split(width):
            distribution = model.distribution(subset(batch, ix))
            rows = torch.arange(len(ix), device=device)
            logps = distribution.log_prob(actions[ix])[rows, agents[ix]]
            entropy = distribution.entropy()[rows, agents[ix]].mean()
            ratio = (logps-old_logps[ix]).exp()
            objective = torch.minimum(ratio*advantages[ix], ratio.clamp(1-clip, 1+clip)*advantages[ix])
            loss = -objective.mean()-entropy_coef*entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            policy_losses.append(float(loss.detach()))
            entropies.append(float(entropy.detach()))
    return dict(value_loss=float(np.mean(value_losses)), actor_loss=float(np.mean(policy_losses)),
                entropy=float(np.mean(entropies)), decision_samples=len(samples),
                mean_option_steps=float(np.mean([s['duration'] for s in samples])),
                mean_return=float(returns.mean()))


def train_exploration_ppo(config, total_steps, out, seed=7, n_envs=8, rollout_decisions=64,
                          epochs=4, critic_epochs=4, minibatch_size=32, learning_rate=5e-4,
                          entropy_coef=.01, policy_clip=.2, eval_interval=100000,
                          eval_episodes=10, eval_seed=10000, checkpoint_interval=100000, device='auto',
                          eval_workers=4, eval_screen_episodes=0, eval_revisit_guard=False):
    if not config.belief_exploration_only or config.mode != 'belief':
        raise ValueError('Decision PPO requires belief_exploration_only in belief mode')
    if eval_revisit_guard and config.belief_representation != 'gaussian_uniform':
        raise ValueError('The revisit guard requires Gaussian belief exploration')
    eval_resolvers = (GUARDED_EXPLORATION_RESOLVER,) if eval_revisit_guard else EVALUATION_RESOLVERS
    best_policy = eval_resolvers[-1]
    if min(total_steps, n_envs, rollout_decisions, epochs, critic_epochs, minibatch_size,
           eval_episodes, checkpoint_interval, eval_workers) <= 0:
        raise ValueError('Training sizes must be positive')
    if eval_interval < 0 or eval_screen_episodes < 0 or learning_rate <= 0 or entropy_coef < 0 or not 0 < policy_clip < 1:
        raise ValueError('Invalid optimization or evaluation settings')
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Device must be auto, cpu, or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use a CUDA-enabled PyTorch environment')
    device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else
                          'cpu' if device == 'auto' else device)
    out = Path(out)
    if (out/'config.json').exists():
        raise ValueError(f'Existing experiment: {out}; choose a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    config.save(out/'config.json')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ExplorationPPOActorCritic().to(device)
    print(f'Device: {device}' + (f' ({torch.cuda.get_device_name(device)})' if device.type == 'cuda' else ''), flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    worlds = [PolygonExplorationWorld(config) for _ in range(n_envs)]
    metadata = dict(seed=seed, mode='belief', algorithm='polygon_decision_ppo', agent_transitions=0,
        decision_samples=0, device=str(device), n_envs=n_envs, rollout_decisions=rollout_decisions, epochs=epochs,
        critic_epochs=critic_epochs, minibatch_size=minibatch_size, learning_rate=learning_rate,
        entropy_coef=entropy_coef, policy_clip=policy_clip, eval_seed=eval_seed, eval_episodes=eval_episodes,
        eval_workers=eval_workers, eval_screen_episodes=min(eval_screen_episodes, eval_episodes),
        evaluation_policies=list(eval_resolvers), best_policy=best_policy,
        deadline_filter='first physical entry along Dubins/lawnmower route; no sweep-completion requirement',
        return_target='full-episode Monte Carlo; physical-step discount; no GAE or Q bootstrap',
        actor_advantage='observed return minus frozen behavior state value',
        collection='frozen policy; complete episode waves; asynchronous per-drone polygon options',
        budget='agent physics transitions; finish episode wave before updating/stopping',
        checkpoint_kind='weights/optimizer; not an exact environment/RNG resume')
    episode_rows, evaluation_rows, update_rows = [], [], []
    screen_rows = []
    best = (-float('inf'), -float('inf'))
    best_screen = best
    next_eval = eval_interval if eval_interval else None
    next_checkpoint = checkpoint_interval

    def save(name):
        save_belief_checkpoint(out/name, model, model, config, metadata.copy(), optimizer)

    def evaluate(final=False):
        nonlocal best, best_screen
        screen_count = metadata['eval_screen_episodes']
        steps = metadata['agent_transitions']
        if not final and 0 < screen_count < eval_episodes:
            print(f'Screening {steps} on {screen_count} paired seeds', flush=True)
            screened = evaluate_exploration(model, config, screen_count, eval_seed, eval_workers,
                                            resolvers=eval_resolvers)
            for resolver, (summary, _) in screened.items():
                screen_rows.append(dict(agent_transitions=steps, policy=resolver,
                                        episodes=screen_count, **summary))
            key = score_key(screened[best_policy][0])
            promoted = key > best_screen
            for row in screen_rows[-len(eval_resolvers):]:
                row['promoted'] = promoted
            _write_csv(out/'training_eval_screening.csv', screen_rows)
            if not promoted:
                print(f'Screen {steps}: {key[0]:.3f} vs best checkpoint {best_screen[0]:.3f}; full evaluation skipped',
                      flush=True)
                return
            rest = evaluate_exploration(model, config, eval_episodes-screen_count,
                                        eval_seed+screen_count, eval_workers, resolvers=eval_resolvers)
            rows = {name: screened[name][1]+rest[name][1] for name in eval_resolvers}
            results = {name: (summarize_belief_rows(records), records) for name, records in rows.items()}
        else:
            print(f'Evaluating {steps} on all {eval_episodes} paired seeds with up to {eval_workers} workers',
                  flush=True)
            results = evaluate_exploration(model, config, eval_episodes, eval_seed, eval_workers,
                                           resolvers=eval_resolvers)
        for resolver in eval_resolvers:
            summary, rows = results[resolver]
            evaluation_rows.append(dict(agent_transitions=metadata['agent_transitions'], policy=resolver, **summary))
            print(f"Eval {metadata['agent_transitions']} [{resolver}]: {summary}", flush=True)
            if resolver == best_policy:
                key = score_key(summary)
                if key > best:
                    best = key
                    if screen_count:
                        best_screen = score_key(summarize_belief_rows(rows[:screen_count]))
                    save('best.pt')
        _write_csv(out/'training_evaluations.csv', evaluation_rows)

    save('latest.pt')
    while metadata['agent_transitions'] < total_steps:
        samples, episodes, steps = collect_decisions(model, worlds, rng, rollout_decisions)
        diagnostics = update_decisions(model, optimizer, samples, epochs, critic_epochs,
                                       minibatch_size, entropy_coef, policy_clip)
        metadata['agent_transitions'] += steps
        metadata['decision_samples'] += len(samples)
        completed = metadata['agent_transitions']
        episode_rows.extend(dict(agent_transitions=completed, **row) for row in episodes)
        update_rows.append(dict(agent_transitions=completed, **diagnostics))
        print(f'Update {completed}/{total_steps}: {diagnostics}', flush=True)
        _write_csv(out/'training_updates.csv', update_rows)
        _write_csv(out/'training_episodes.csv', episode_rows)
        save('latest.pt')
        if completed >= next_checkpoint:
            save(f'checkpoint_{completed}.pt')
            while next_checkpoint <= completed:
                next_checkpoint += checkpoint_interval
        if next_eval is not None and completed >= next_eval and completed < total_steps:
            evaluate()
            while next_eval <= completed:
                next_eval += eval_interval
        (out/'training_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    # Always evaluate the final policy when evaluation is enabled.
    if next_eval is not None and (not evaluation_rows or evaluation_rows[-1]['agent_transitions'] != completed):
        evaluate(final=True)
    save('exploration_ppo.pt')
    return model, episode_rows, evaluation_rows, metadata['agent_transitions']
