"""CTDE MAPPO for cooperative strike-target prioritization.

Every active drone chooses one known, live target on every simulation step.
The environment resolves that choice into a strike approach, so the policy's
only job is target priority and multi-drone allocation.
"""
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
except ImportError as exc:
    raise ImportError("Install the rl extra: pip install -e '.[rl]'") from exc

from .env import SELF_FEATURES, TARGET_FEATURES, strike_action


class StrikeActorCritic(nn.Module):
    """Decentralized shared actor and centralized joint-observation critic."""
    def __init__(self, obs_dim, n_targets, n_agents, hidden=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_targets = n_targets
        self.n_agents = n_agents
        self.state_dim = obs_dim * n_agents + n_targets + 1
        self.actor_body = nn.Sequential(nn.Linear(obs_dim + n_agents, hidden), nn.Tanh(),
                                        nn.Linear(hidden, hidden), nn.Tanh())
        self.target_head = nn.Linear(hidden, n_targets)
        self.critic_body = nn.Sequential(nn.Linear(self.state_dim + n_agents, hidden), nn.Tanh(),
                                         nn.Linear(hidden, hidden), nn.Tanh())
        self.value_head = nn.Linear(hidden, 1)

    def target_mask(self, obs):
        ids = torch.arange(self.n_targets, device=obs.device)
        selectable = SELF_FEATURES + ids * TARGET_FEATURES + 3
        return obs[..., selectable] > 0

    def distributions(self, obs, agent_id=None, locked_target=None):
        if agent_id is None:
            agent_id = torch.arange(len(obs), device=obs.device)
        identity = torch.nn.functional.one_hot(agent_id.long(), self.n_agents).to(obs.dtype)
        h = self.actor_body(torch.cat([obs, identity], dim=-1))
        mask = self.target_mask(obs)
        # After all targets are destroyed, keep a harmless fallback action so
        # the fixed-horizon episode can continue without an invalid Categorical.
        empty = ~mask.any(dim=-1)
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True
        if locked_target is not None:
            locked_target = torch.as_tensor(locked_target, device=obs.device, dtype=torch.long)
            safe_target = locked_target.clamp(min=0)
            lock_valid = (locked_target >= 0) & mask.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
            forced = torch.nn.functional.one_hot(safe_target, self.n_targets).bool()
            mask = torch.where(lock_valid.unsqueeze(-1), forced, mask)
        logits = self.target_head(h).masked_fill(~mask, -1e9)
        return Categorical(logits=logits)

    def values(self, state, agent_id):
        # A single team value must also be meaningful after an individual
        # drone disappears. Agent identity is relevant only to the actor.
        identity = state.new_zeros((*state.shape[:-1], self.n_agents))
        h = self.critic_body(torch.cat([state, identity], dim=-1))
        return self.value_head(h).squeeze(-1)

    def act(self, obs, deterministic=False, locked_target=None, agent_id=None):
        dist = self.distributions(obs, agent_id, locked_target)
        target = dist.probs.argmax(-1) if deterministic else dist.sample()
        return strike_action(target.cpu().numpy()), {"logp": dist.log_prob(target)}

    def evaluate_actions(self, obs, agent_id, target, locked_target):
        dist = self.distributions(obs, agent_id, locked_target)
        return dist.log_prob(target), dist.entropy()


def save_checkpoint(path, model, metadata):
    torch.save({"state_dict": model.state_dict(), "obs_dim": model.obs_dim,
                "n_targets": model.n_targets, "n_agents": model.n_agents,
                "architecture": "strike_mappo_v11",
                "metadata": metadata}, Path(path))


def load_checkpoint(path, model):
    from torch.torch_version import TorchVersion
    safe_context = getattr(torch.serialization, "safe_globals", None)
    if safe_context is not None:
        with safe_context([TorchVersion]):
            payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    else:
        torch.serialization.add_safe_globals([TorchVersion])
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("architecture") != "strike_mappo_v11":
        raise ValueError("Checkpoint is not a strike MAPPO model; retrain it")
    if (payload["obs_dim"] != model.obs_dim or payload["n_targets"] != model.n_targets
            or payload["n_agents"] != model.n_agents):
        raise ValueError("Model shape does not match config")
    model.load_state_dict(payload["state_dict"])
    return payload.get("metadata", {})


def critic_state(world, obs):
    """Joint observations, strike progress, and the normalized reward margin."""
    progress = world.strike_progress / world.c.strike_steps_per_life
    reward_margin = ((world.formation_one_initial_score - world.score)
                     / world.formation_one_initial_score)
    state = np.concatenate([obs.reshape(-1), progress, [reward_margin]]).astype(np.float32)
    return np.repeat(state[None], world.c.n_agents, axis=0)


def _advantages(reward, value, next_value, done, gamma=.99, gae_lambda=.99):
    advantage = np.zeros_like(reward)
    carry = np.zeros(reward.shape[1], dtype=np.float32)
    for t in reversed(range(len(reward))):
        delta = reward[t] + gamma * next_value[t] * (1 - done[t]) - value[t]
        carry = delta + gamma * gae_lambda * (1 - done[t]) * carry
        advantage[t] = carry
    return advantage, advantage + value


def _batch(obs, state, agent_id, target, locked_target, logp, advantage, returns,
           active, decision):
    mask = np.asarray(active, dtype=bool).reshape(-1)
    obs_array = np.asarray(obs)
    return (torch.as_tensor(obs_array.reshape(-1, obs_array.shape[-1])[mask], dtype=torch.float32),
            torch.as_tensor(np.asarray(state).reshape(-1, np.asarray(state).shape[-1])[mask], dtype=torch.float32),
            torch.as_tensor(np.asarray(agent_id).reshape(-1)[mask], dtype=torch.long),
            torch.as_tensor(np.asarray(target).reshape(-1)[mask], dtype=torch.long),
            torch.as_tensor(np.asarray(locked_target).reshape(-1)[mask], dtype=torch.long),
            torch.as_tensor(np.asarray(logp).reshape(-1)[mask], dtype=torch.float32),
            torch.as_tensor(np.asarray(advantage).reshape(-1)[mask], dtype=torch.float32),
            torch.as_tensor(np.asarray(returns).reshape(-1)[mask], dtype=torch.float32),
            torch.as_tensor(np.asarray(decision).reshape(-1)[mask], dtype=torch.bool))


def _scale_actor_advantage(advantage, decision):
    # Simultaneous choices share a team advantage. Centering a rollout with
    # only one decision time erases its entire policy-gradient signal.
    chosen = advantage[decision]
    if not len(chosen):
        return advantage
    return advantage / chosen.square().mean().sqrt().clamp(min=1.0)


def _ppo_update(model, optimizer, tensors, epochs, minibatch_size):
    obs, state, agent_id, target, locked_target, old_logp, advantage, returns, decision = tensors
    advantage = _scale_actor_advantage(advantage, decision)
    for _ in range(epochs):
        order = torch.randperm(len(obs))
        for start in range(0, len(obs), minibatch_size):
            idx = order[start:start + minibatch_size]
            logp, entropy = model.evaluate_actions(
                obs[idx], agent_id[idx], target[idx], locked_target[idx])
            value = model.values(state[idx], agent_id[idx])
            ratio = (logp - old_logp[idx]).exp()
            clipped = ratio.clamp(.8, 1.2)
            choose = decision[idx]
            if choose.any():
                actor_loss = -torch.minimum(
                    ratio[choose] * advantage[idx][choose],
                    clipped[choose] * advantage[idx][choose]).mean()
                entropy_bonus = entropy[choose].mean()
            else:
                actor_loss = value.sum() * 0.0
                entropy_bonus = value.sum() * 0.0
            loss = (actor_loss + .5 * (value - returns[idx]).pow(2).mean()
                    - .01 * entropy_bonus)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.actor_body.parameters())
                                     + list(model.target_head.parameters()), .5)
            nn.utils.clip_grad_norm_(list(model.critic_body.parameters())
                                     + list(model.value_head.parameters()), .5)
            optimizer.step()


def _interquartile_mean(values):
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values):
        return float("nan")
    lower, upper = 0.25 * len(values), 0.75 * len(values)
    weights = np.array([
        max(0.0, min(i + 1, upper) - max(i, lower))
        for i in range(len(values))])
    return float(np.sum(values * weights) / np.sum(weights))


def _summarize(rows):
    keys = ("team_return", "mission_success", "score", "baseline_score", "destroyed_fraction",
            "score_auc", "steps")
    return {key: _interquartile_mean([row[key] for row in rows]) for key in keys}


def evaluate_model(model, config, episodes=5, seed=10000):
    """Evaluate the current decentralized actors on held-out scenario seeds."""
    from .env import World
    rows = []
    with torch.no_grad():
        for episode in range(episodes):
            world = World(config)
            obs = world.reset(seed + episode)
            locked_target = np.full(config.n_agents, -1, dtype=int)
            while not world.done:
                locked_target = world.locked_targets()
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                mask = model.target_mask(obs_t).numpy()
                lock_valid = ((locked_target >= 0)
                              & mask[np.arange(config.n_agents), np.maximum(locked_target, 0)])
                locked_target[~lock_valid] = -1
                action, _ = model.act(obs_t, deterministic=True, locked_target=locked_target)
                locked_target = action["target"].copy()
                obs, _, _, _, _ = world.step(action)
            rows.append(world.metrics())
    return _summarize(rows)


def evaluate_baseline(policy_class, config, episodes=5, seed=10000):
    """Evaluate a baseline on exactly the same held-out scenario seeds."""
    from .env import World
    rows = []
    for episode in range(episodes):
        episode_seed = seed + episode
        world = World(config)
        obs = world.reset(episode_seed)
        policy = policy_class(config, episode_seed)
        while not world.done:
            action = policy.predict(obs)
            obs, _, _, _, _ = world.step(action)
        rows.append(world.metrics())
    return _summarize(rows)


def train_mappo(config, total_agent_transitions, seed=7, rollout_steps=256, epochs=6,
              minibatch_size=256, learning_rate=3e-4, progress=False,
              eval_interval=10000, eval_episodes=5, eval_seed=10000,
              best_path=None, latest_path=None, checkpoint_dir=None,
              checkpoint_interval=100000):
    from .env import World
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    world = World(config)
    obs = world.reset(int(rng.integers(2**31)))
    locked_target = np.full(config.n_agents, -1, dtype=int)
    model = StrikeActorCritic(world.obs_dim, config.n_targets, config.n_agents)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    completed, episode_rows, evaluation_rows = 0, [], []
    best_return = -float("inf")
    next_eval = eval_interval if eval_interval > 0 else None
    next_checkpoint = checkpoint_interval if checkpoint_dir is not None else None
    progress_bar = tqdm(total=total_agent_transitions, desc="Train", unit="transition",
                        dynamic_ncols=True, disable=not progress)

    while completed < total_agent_transitions:
        steps = min(rollout_steps, max(1, int(np.ceil(
            (total_agent_transitions - completed) / config.n_agents))))
        observations, states, agent_ids, targets, locks, logps, values = ([] for _ in range(7))
        next_values, rewards, dones, active, decisions = [], [], [], [], []
        for _ in range(steps):
            locked_target = world.locked_targets()
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            mask = model.target_mask(obs_t).numpy()
            lock_valid = ((locked_target >= 0)
                          & mask[np.arange(config.n_agents), np.maximum(locked_target, 0)])
            locked_target[~lock_valid] = -1
            lock_before_action = locked_target.copy()
            ids_t = torch.arange(config.n_agents)
            state = critic_state(world, obs)
            with torch.no_grad():
                action, stats = model.act(obs_t, locked_target=locked_target, agent_id=ids_t)
                value = model.values(torch.as_tensor(state, dtype=torch.float32), ids_t)
            locked_target = action["target"].copy()
            active_before = world.agent_active.copy()
            next_obs, reward, terminated, truncated, metrics = world.step(action)
            env_done = terminated or truncated
            with torch.no_grad():
                next_state = critic_state(world, next_obs)
                next_value = model.values(torch.as_tensor(next_state, dtype=torch.float32), ids_t)
            observations.append(obs.copy())
            states.append(state)
            agent_ids.append(np.arange(config.n_agents))
            targets.append(action["target"].copy())
            locks.append(lock_before_action)
            logps.append(stats["logp"].numpy())
            values.append(value.numpy())
            next_values.append(next_value.numpy())
            # Credit earlier choices with the entire cooperative outcome,
            # including rewards/penalties after that drone has disappeared.
            rewards.append(np.full(config.n_agents, reward.sum(), dtype=np.float32))
            dones.append(np.full(config.n_agents, env_done, dtype=float))
            active.append(active_before)
            decisions.append(lock_before_action < 0)
            obs = next_obs
            if env_done:
                episode_rows.append(dict(agent_transitions=completed + int(np.asarray(active).sum()),
                                         **metrics))
                obs = world.reset(int(rng.integers(2**31)))
                locked_target = np.full(config.n_agents, -1, dtype=int)

        advantage, returns = _advantages(np.asarray(rewards), np.asarray(values),
                                         np.asarray(next_values), np.asarray(dones),
                                         gamma=config.discount_gamma)
        tensors = _batch(observations, states, agent_ids, targets, locks, logps,
                         advantage, returns, active, decisions)
        _ppo_update(model, optimizer, tensors, epochs, minibatch_size)
        count = int(np.asarray(active).sum())
        completed += count
        progress_bar.update(min(count, total_agent_transitions - progress_bar.n))
        while next_checkpoint is not None and completed >= next_checkpoint:
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{next_checkpoint}.pt"
            save_checkpoint(checkpoint_path, model, {
                "agent_transitions": completed, "checkpoint_transition": next_checkpoint,
                "seed": seed})
            progress_bar.write(f"Saved periodic checkpoint: {checkpoint_path}")
            next_checkpoint += checkpoint_interval
        if next_eval is not None and completed >= next_eval:
            if latest_path is not None:
                save_checkpoint(latest_path, model, {"agent_transitions": completed,
                                                     "seed": seed})
            from .policies import HeuristicPolicy, RandomPolicy
            evaluations = {
                "random": evaluate_baseline(RandomPolicy, config, eval_episodes, eval_seed),
                "heuristic": evaluate_baseline(HeuristicPolicy, config, eval_episodes, eval_seed),
                "mappo": evaluate_model(model, config, eval_episodes, eval_seed),
            }
            mappo_return = evaluations["mappo"]["team_return"]
            if best_path is not None and mappo_return > best_return:
                best_return = mappo_return
                save_checkpoint(best_path, model, {
                    "best_metric": "interquartile_mean_team_return",
                    "best_value": best_return,
                    "agent_transitions": completed,
                    "eval_episodes": eval_episodes,
                    "eval_seed": eval_seed,
                })
                progress_bar.write(
                    f"Saved best checkpoint: {best_path} (iqm_return={best_return:.3f})")
            for policy_name, evaluation in evaluations.items():
                evaluation_rows.append(dict(agent_transitions=completed, policy=policy_name,
                                            episodes=eval_episodes, seed=eval_seed, **evaluation))
                score_ratio = evaluation["score"] / max(1.0, evaluation["baseline_score"])
                progress_bar.write(
                    f"Eval @ {completed:,} [{policy_name}]: "
                    f"iqm_return={evaluation['team_return']:.3f} "
                    f"success={evaluation['mission_success']:.3f} "
                    f"score={score_ratio:.3f} destroyed={evaluation['destroyed_fraction']:.3f} "
                    f"auc={evaluation['score_auc']:.3f} steps={evaluation['steps']:.1f}")
            while next_eval <= completed:
                next_eval += eval_interval
        if episode_rows:
            latest = episode_rows[-1]
            progress_bar.set_postfix(episodes=len(episode_rows), score=latest["score"],
                                     success=int(latest["mission_success"]))
    progress_bar.close()
    if latest_path is not None:
        save_checkpoint(latest_path, model, {"agent_transitions": completed,
                                             "seed": seed})
    return model, episode_rows, evaluation_rows, completed
