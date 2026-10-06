"""Centralized PPO for cooperative strike-target prioritization.

Each drone's shared actor sees the global state and its one-hot identity.
The actor masks destroyed targets and targets whose strike slots are full.
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import csv

import numpy as np
from tqdm.auto import tqdm

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
except ImportError as exc:
    raise ImportError("Install the rl extra: pip install -e '.[rl]'") from exc

from .env import (DAMAGE_REWARD_RATE, EVALUATION_TASKS,
                  SUCCESS_THRESHOLDS,
                  evaluation_task_for_episode, strike_action)


def _uses_balanced_thresholds(config):
    return (config.n_targets == 5 and config.min_targets == 5
            and not config.randomize_counts
            and not config.randomize_target_composition)


def _evaluation_threshold(config, episode):
    return SUCCESS_THRESHOLDS[episode % len(SUCCESS_THRESHOLDS)] \
        if (_uses_balanced_thresholds(config)
            and evaluation_task_for_episode(config, episode) is None) else None


def _validate_evaluation_episodes(config, episodes):
    if evaluation_task_for_episode(config, 0) is not None:
        if episodes % len(EVALUATION_TASKS) != 0:
            raise ValueError(
                f"evaluation episodes must be a multiple of {len(EVALUATION_TASKS)} "
                "to represent every task equally")
    elif (_uses_balanced_thresholds(config)
          and episodes % len(SUCCESS_THRESHOLDS) != 0):
        raise ValueError(
            f"evaluation episodes must be a multiple of {len(SUCCESS_THRESHOLDS)} "
            "to represent every B case equally")


class ValueNorm(nn.Module):
    def __init__(self, beta=0.99999, epsilon=1e-5):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.register_buffer('running_mean', torch.zeros(()))
        self.register_buffer('running_mean_sq', torch.zeros(()))
        self.register_buffer('debiasing_term', torch.zeros(()))

    def moments(self):
        denominator = self.debiasing_term.clamp(min=self.epsilon)
        mean = self.running_mean / denominator
        variance = (self.running_mean_sq / denominator - mean.square()).clamp(min=1e-2)
        return mean, variance

    @torch.no_grad()
    def update(self, returns):
        self.running_mean.mul_(self.beta).add_(returns.mean() * (1 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(returns.square().mean() * (1 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1 - self.beta)

    def normalize(self, returns):
        mean, variance = self.moments()
        return (returns - mean) / variance.sqrt()

    def denormalize(self, values):
        mean, variance = self.moments()
        return values * variance.sqrt() + mean


def _mappo_body(input_dim, hidden):
    body = nn.Sequential(nn.LayerNorm(input_dim),
                         nn.Linear(input_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden))
    for layer in body.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(layer.bias)
    return body


class StrikeActorCritic(nn.Module):
    """Shared global-state actor with agent ID and a team-state critic."""
    architecture = 'strike_mappo_v24'
    one_hot_target_type = True
    relative_actor_coordinates = True
    algorithm = 'mappo'
    autoregressive = False
    uses_previous_action_probabilities = True

    def __init__(self, n_targets, n_agents, hidden=64):
        super().__init__()
        self.n_targets = n_targets
        self.n_agents = n_agents
        self.target_features = 6 if self.one_hot_target_type else 4
        self.state_dim = 3 + 5 * n_agents + self.target_features * n_targets + n_agents * n_targets
        self.obs_dim = self.state_dim + n_agents
        self.hidden = hidden
        self.actor_body = _mappo_body(self.obs_dim, hidden)
        self.target_head = nn.Linear(hidden, n_targets)
        self.critic_body = _mappo_body(self.state_dim, hidden)
        self.value_head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.target_head.weight, gain=0.01)
        nn.init.zeros_(self.target_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
        self.value_normalizer = ValueNorm()

    def actor_logits(self, obs):
        h = self.actor_body(obs)
        return self.target_head(h)

    def observation(self, world, previous_action_probabilities=None):
        return actor_observation(world, relative=self.relative_actor_coordinates,
                                 one_hot=self.one_hot_target_type,
                                 previous_action_probabilities=previous_action_probabilities)

    def available_actions(self, world):
        global_obs = torch.as_tensor(actor_observation(world, one_hot=self.one_hot_target_type), dtype=torch.float32)
        available = self.target_mask(global_obs).cpu().numpy()
        if getattr(self, 'training_settings', {}).get('deadline_mask', False):
            from .strike_lookahead import deadline_action_mask
            available = deadline_action_mask(world, available)
        return available

    def target_mask(self, obs):
        # State stores target life after position and type, followed by
        # target-major assignments for each drone.
        target_start = 3 + 5 * self.n_agents
        width = self.target_features
        target_records = obs[..., target_start:target_start + width * self.n_targets].reshape(
            *obs.shape[:-1], self.n_targets, width)
        life = target_records[..., -1]
        type_one = (target_records[..., 2] > .5 if self.one_hot_target_type
                    else target_records[..., 2] < .5)
        drone_start = 3
        drone_records = obs[..., drone_start:drone_start + 5 * self.n_agents]
        participating = drone_records[..., 3::5] > 0.5
        assignment_start = target_start + width * self.n_targets
        assignment_end = assignment_start + self.n_targets * self.n_agents
        assignments = obs[..., assignment_start:assignment_end].reshape(
            *obs.shape[:-1], self.n_targets, self.n_agents) > 0.5
        occupied = (assignments & participating.unsqueeze(-2)).sum(dim=-1)
        capacity = torch.where(type_one, 2, 1)
        remaining_life = torch.round(life * 2).long()
        capacity = torch.minimum(capacity, remaining_life)
        identities = obs[..., -self.n_agents:].argmax(dim=-1)
        own_participation = participating.gather(
            -1, identities.unsqueeze(-1)).squeeze(-1)
        own_assignment = assignments.gather(
            -1, identities[..., None, None].expand(
                *identities.shape, self.n_targets, 1)).squeeze(-1)
        own_slot = own_participation.unsqueeze(-1) & own_assignment
        mask = (life > 0) & ((occupied < capacity) | own_slot)
        # Preserve the existing valid-Categorical fallback when no target can
        # be selected. This does not add an action or change the action space.
        mask = mask.clone()
        mask[..., 0] |= ~mask.any(dim=-1)
        return mask

    def distributions(self, obs, action_mask=None):
        if action_mask is None:
            action_mask = self.target_mask(obs)
        temperature = getattr(self, 'training_settings', {}).get('policy_temperature', 1.0)
        logits = (self.actor_logits(obs) / temperature).masked_fill(~action_mask, -torch.inf)
        return Categorical(logits=logits)

    def values(self, state):
        # Keep one common baseline so local damage-credit differences remain
        # in the actor advantage instead of being explained away by identity.
        h = self.critic_body(state)
        return self.value_head(h).squeeze(-1)

    def act(self, obs, deterministic=False, action_mask=None):
        if action_mask is None:
            action_mask = self.target_mask(obs)
        dist = self.distributions(obs, action_mask)
        target = dist.probs.argmax(-1) if deterministic else dist.sample()
        return strike_action(target.cpu().numpy()), {
            "logp": dist.log_prob(target), "action_mask": action_mask,
            "probs": dist.probs}

    def evaluate_actions(self, obs, target, action_mask):
        dist = self.distributions(obs, action_mask)
        return dist.log_prob(target), dist.entropy()


class LegacyStrikeCOMAActorCritic(StrikeActorCritic):
    """MAPPO actor trained with COMA counterfactual advantages."""
    architecture = 'strike_coma_ppo_v1'
    algorithm = 'coma'

    def __init__(self, n_targets, n_agents, hidden=64):
        super().__init__(n_targets, n_agents, hidden)
        joint_action_dim = n_agents * n_targets
        self.q_body = _mappo_body(self.state_dim + joint_action_dim, hidden)
        self.q_head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.q_head.weight, gain=1.0)
        nn.init.zeros_(self.q_head.bias)

    def joint_q(self, state, joint_action):
        encoded_action = torch.nn.functional.one_hot(
            joint_action.long(), self.n_targets).to(state.dtype).flatten(-2)
        features = torch.cat((state, encoded_action), dim=-1)
        return self.q_head(self.q_body(features)).squeeze(-1)

    def counterfactual_advantages(self, state, joint_action, action_probabilities,
                                 sampled_returns=None):
        """Hold other actions fixed and marginalize only each agent's action."""
        batch = state.shape[0]
        alternatives = joint_action[:, None, None, :].expand(
            batch, self.n_agents, self.n_targets, self.n_agents).clone()
        agent = torch.arange(self.n_agents, device=state.device)
        action = torch.arange(self.n_targets, device=state.device)
        alternatives[:, agent[:, None], action[None, :], agent[:, None]] = action
        expanded_state = state[:, None, None, :].expand(
            batch, self.n_agents, self.n_targets, self.state_dim)
        q_values = self.joint_q(
            expanded_state.reshape(-1, self.state_dim),
            alternatives.reshape(-1, self.n_agents)).reshape(
                batch, self.n_agents, self.n_targets)
        actual_q = q_values.gather(-1, joint_action.unsqueeze(-1)).squeeze(-1)
        baseline = (action_probabilities * q_values).sum(dim=-1)
        # With sampled returns, Q is only an action-independent control variate.
        # Poorly fitted Q differences no longer replace the observed outcome.
        outcome = actual_q if sampled_returns is None else sampled_returns[:, None]
        return outcome - baseline


class StrikeCOMAActorCritic(LegacyStrikeCOMAActorCritic):
    """Q-only counterfactual PPO; target Q and returns use raw reward units."""
    architecture = 'strike_coma_ppo_v3'
    critic_previous_probabilities = True
    critic_mask_inactive_intent = False

    def __init__(self, n_targets, n_agents, hidden=64):
        super().__init__(n_targets, n_agents, hidden)
        del self.critic_body, self.value_head, self.value_normalizer
        if self.critic_previous_probabilities:
            self.state_dim += n_agents * n_targets
            self.q_body = _mappo_body(self.state_dim + n_agents * n_targets, hidden)
        self.target_q_body = deepcopy(self.q_body).requires_grad_(False)
        self.target_q_head = deepcopy(self.q_head).requires_grad_(False)

    def values(self, state):
        raise RuntimeError('COMA uses joint_q(state, action), not a separate V critic')

    def target_joint_q(self, state, joint_action):
        action = torch.nn.functional.one_hot(
            joint_action.long(), self.n_targets).to(state.dtype).flatten(-2)
        return self.target_q_head(self.target_q_body(
            torch.cat((state, action), -1))).squeeze(-1)

    @torch.no_grad()
    def update_target(self, tau=.01):
        for online, target in ((self.q_body, self.target_q_body),
                               (self.q_head, self.target_q_head)):
            for source, destination in zip(online.parameters(), target.parameters()):
                destination.lerp_(source, tau)


class CompactStrikeCOMAActorCritic(StrikeCOMAActorCritic):
    """Global MLP actor with the same information as the target COMA scorer."""
    architecture = 'strike_compact_coma_v1'
    algorithm = 'compact_coma'
    compact_inputs = True
    critic_strike_participants = False
    critic_mask_inactive_intent = True

    def __init__(self, n_targets, n_agents, hidden=64):
        super().__init__(n_targets, n_agents, hidden)
        # Union of scorer inputs across targets: time, all target records,
        # own previous probabilities, and each peer's geometry/status/intent.
        self.obs_dim = (1 + (self.target_features + 1) * n_targets
                        + (3 + n_targets) * (n_agents - 1))
        self.actor_body = _mappo_body(self.obs_dim, hidden)
        self.target_head = nn.Linear(hidden, n_targets)
        nn.init.orthogonal_(self.target_head.weight, gain=.01)
        nn.init.zeros_(self.target_head.bias)
        # Match target_coma's physical state, previous intent and joint action.
        self.state_dim = (1 + 4 * n_agents + self.target_features * n_targets
                          + n_agents * n_targets)
        self.q_body = _mappo_body(self.state_dim + n_agents * n_targets, hidden)
        self.target_q_body = deepcopy(self.q_body).requires_grad_(False)

    def observation(self, world, previous_action_probabilities=None):
        if previous_action_probabilities is None:
            previous_action_probabilities = np.full(
                (self.n_agents, self.n_targets), 1.0 / self.n_targets,
                dtype=np.float32)
        full = super().observation(world, previous_action_probabilities)
        target_start = 3 + 5 * self.n_agents
        target_end = target_start + self.target_features * self.n_targets
        probabilities = full[0, target_end:-self.n_agents].reshape(
            self.n_targets, self.n_agents).T
        observations = []
        for agent in range(self.n_agents):
            drones = full[agent, 3:target_start].reshape(self.n_agents, 5)
            alive = drones[:, 2:3]
            peer_features = np.concatenate((drones[:, :2] * alive,
                                            alive * (1 - drones[:, 3:4]),
                                            probabilities * alive), axis=-1)
            others = np.arange(self.n_agents) != agent
            observations.append(np.concatenate((
                full[agent, :1], full[agent, target_start:target_end],
                probabilities[agent] * drones[agent, 2],
                peer_features[others].reshape(-1))))
        return np.asarray(observations, dtype=np.float32)


class LegacyQOnlyStrikeCOMAActorCritic(StrikeCOMAActorCritic):
    architecture = 'strike_coma_ppo_v2'
    critic_previous_probabilities = False


class ScalarTypeStrikeActorCritic(StrikeActorCritic):
    """Read pre-one-hot MAPPO checkpoints without changing their inputs."""
    architecture = 'strike_mappo_v22'
    one_hot_target_type = False
    uses_previous_action_probabilities = False


class HardAssignmentStrikeActorCritic(StrikeActorCritic):
    """Read v23 checkpoints whose actor observed hard previous assignments."""
    architecture = 'strike_mappo_v23'
    uses_previous_action_probabilities = False


class DecisionSchedule:
    def __init__(self, config):
        self.interval = config.decision_interval
        self.commit_target = config.commit_target
        self.targets = np.full(config.n_agents, -1, dtype=int)
        self.next_decision = np.zeros(config.n_agents, dtype=int)
        self.previous_action_probabilities = np.full(
            (config.n_agents, config.n_targets), 1.0 / config.n_targets,
            dtype=np.float32)

    def reset(self):
        self.targets.fill(-1)
        self.next_decision.fill(0)
        self.previous_action_probabilities.fill(
            1.0 / self.previous_action_probabilities.shape[-1])

    def mask(self, world, available):
        if world.t == 0:
            self.reset()
        mask = available.copy()
        decision = np.zeros(world.n, dtype=bool)
        locks = world.locked_targets()
        for agent in range(world.n):
            if not world.agent_active[agent]:
                continue
            target = self.targets[agent]
            if locks[agent] >= 0:
                target = int(locks[agent])
            elif (target < 0 or (not self.commit_target and world.t >= self.next_decision[agent])
                  or not available[agent, target]):
                decision[agent] = True
                continue
            mask[agent] = False
            mask[agent, target] = True
        return mask, decision

    def record(self, world, targets, decision, probabilities=None):
        self.targets[decision] = targets[decision]
        self.next_decision[decision] = world.t + self.interval
        if probabilities is not None:
            self.previous_action_probabilities[:] = probabilities


def _normalized_drone_velocities(world):
    """Measured displacement/dt, in world axes, normalized by movement speed."""
    return ((world.vel / world.c.strike_speed)
            * world.agent_active[:, None]).astype(np.float32)


def _model_observation(model, world, schedule):
    if getattr(model, 'actor_previous_actions', False):
        return model.observation(world, schedule.previous_action_probabilities,
                                 schedule.targets)
    if model.uses_previous_action_probabilities:
        return model.observation(world, schedule.previous_action_probabilities)
    return model.observation(world)


def _model_critic_state(model, world, schedule):
    state = critic_state(world, one_hot=model.one_hot_target_type)[0]
    if getattr(model, 'compact_inputs', False):
        target_start = 3 + 5 * model.n_agents
        drones = state[3:target_start].reshape(model.n_agents, 5)
        target_end = target_start + model.target_features * model.n_targets
        parts = [state[:1], drones[:, [0, 1, 2, 4]].reshape(-1),
                 state[target_start:target_end]]
        if model.critic_strike_participants:
            parts.append(world.strike_participants.astype(np.float32).reshape(-1))
        state = np.concatenate(parts)
    if getattr(model, 'critic_previous_probabilities', False):
        # Reset before reading at episode starts, even before action collection.
        probabilities = (np.full_like(schedule.previous_action_probabilities,
                                     1.0 / model.n_targets) if world.t == 0
                         else schedule.previous_action_probabilities)
        if getattr(model, 'critic_mask_inactive_intent', False):
            probabilities = probabilities * world.agent_active[:, None]
        state = np.concatenate((state, probabilities.T.reshape(-1)))
    if getattr(model, 'critic_previous_actions', False):
        previous = np.zeros((model.n_agents, model.n_targets), dtype=np.float32)
        if world.t > 0:
            valid = world.agent_active & (schedule.targets >= 0)
            previous[np.flatnonzero(valid), schedule.targets[valid]] = 1.0
        state = np.concatenate((state, previous.T.reshape(-1)))
    if getattr(model, 'drone_velocity', False):
        state = np.concatenate((state, _normalized_drone_velocities(world).reshape(-1)))
    return state


def allocated_actions(model, worlds, schedules, deterministic=False):
    """Sample every drone once and pass the joint action through unchanged."""
    for world, schedule in zip(worlds, schedules):
        if world.t == 0:
            schedule.reset()
    observations = np.stack([_model_observation(model, world, schedule)
                             for world, schedule in zip(worlds, schedules)])
    available = np.stack([model.available_actions(world) for world in worlds])
    scheduled = [schedule.mask(world, available[index])
                 for index, (world, schedule) in enumerate(zip(worlds, schedules))]
    masks = np.stack([entry[0] for entry in scheduled])
    decisions = np.stack([entry[1] for entry in scheduled])
    active = np.stack([world.agent_active for world in worlds])
    tensor_obs = torch.as_tensor(observations, dtype=torch.float32)
    action, stats = model.act(tensor_obs, deterministic=deterministic,
                              action_mask=torch.as_tensor(masks))
    targets = action['target']
    probabilities = (stats['probs'].detach().cpu().numpy().copy()
                     if 'probs' in stats else None)
    traces = [dict(obs=observations, target=targets.copy(),
                   logp=stats['logp'].detach().cpu().numpy(),
                   probs=probabilities,
                   action_mask=stats['action_mask'].detach().cpu().numpy().copy(),
                   decision=decisions.copy())]
    for index, (world, schedule) in enumerate(zip(worlds, schedules)):
        changed = active[index] & decisions[index]
        schedule.record(world, targets[index], changed,
                        None if probabilities is None else probabilities[index])
    return strike_action(targets), traces


def evaluation_resolved_action(model, world, schedule, deterministic=True):
    """Distance-priority rejection/reselection used only for evaluation."""
    action, traces = allocated_actions(model, [world], [schedule], deterministic)
    trace = traces[0]
    targets = action['target'][0].copy()
    initial = targets.copy()
    available = model.available_actions(world)
    masks = trace['action_mask'][0].copy()
    locked = world.locked_targets()
    rejected_before = np.zeros_like(masks)
    probabilities = trace['probs'][0].copy() if trace['probs'] is not None else None
    for _ in range(world.n * world.c.n_targets + 1):
        rejected = []
        for target in range(world.c.n_targets):
            candidates = np.flatnonzero(world.agent_active & (targets == target))
            capacity = min(2 if world.target_type[target] == 1 else 1,
                           int(world.target_life[target]))
            ranked = sorted(candidates, key=lambda agent: (
                locked[agent] != target,
                float(np.linalg.norm(world.pos[agent] - world.targets[target])), int(agent)))
            rejected.extend(agent for agent in ranked[capacity:] if locked[agent] < 0)
        if not rejected:
            break
        movable = []
        for agent in rejected:
            rejected_before[agent, targets[agent]] = True
            alternatives = available[agent] & ~rejected_before[agent]
            if alternatives.any():
                masks[agent] = alternatives
                movable.append(agent)
        if not movable:
            break  # Fewer slots than drones: keep legal proposals; no invented idle action.
        sampling_mask = masks.copy()
        for agent in range(world.n):
            if agent not in movable:
                sampling_mask[agent] = False
                sampling_mask[agent, targets[agent]] = True
        proposed, stats = model.act(torch.as_tensor(trace['obs']),
            deterministic=deterministic, action_mask=torch.as_tensor(sampling_mask[None]))
        targets[movable] = proposed['target'][0, movable]
        if probabilities is not None:
            probabilities[movable] = stats['probs'][0, movable].detach().cpu().numpy()
    changed = world.agent_active & (trace['decision'][0] | (targets != initial))
    schedule.record(world, targets, changed, probabilities)
    return strike_action(targets), {'allocation_traces': traces,
                                    'resolver_changed': targets != initial}


def scheduled_action(model, world, schedule, deterministic=True, resolver=False):
    if resolver:
        return evaluation_resolved_action(model, world, schedule, deterministic)
    action, traces = allocated_actions(model, [world], [schedule], deterministic)
    return strike_action(action['target'][0]), {'allocation_traces': traces}


def save_checkpoint(path, model, metadata):
    torch.save({"state_dict": model.state_dict(), "obs_dim": model.obs_dim,
                "n_targets": model.n_targets, "n_agents": model.n_agents,
                "architecture": model.architecture, "hidden": model.hidden,
                "training_settings": getattr(model, 'training_settings', {}),
                "metadata": metadata}, Path(path))


def checkpoint_payload(path):
    from torch.torch_version import TorchVersion
    safe_context = getattr(torch.serialization, "safe_globals", None)
    if safe_context is not None:
        with safe_context([TorchVersion]):
            payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    else:
        torch.serialization.add_safe_globals([TorchVersion])
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return payload


def model_from_checkpoint(path, config):
    payload = checkpoint_payload(path)
    architecture = payload.get('architecture')
    if architecture in ('strike_mat_v1', 'strike_mat_v2', 'strike_mat_v3', 'strike_mat_v4'):
        from .strike_mat import (StrikeMATActorCritic, LegacyStrikeMATActorCritic,
                                 LocalStrikeMATActorCritic, ScalarTypeStrikeMATActorCritic)
        model_class = {'strike_mat_v1': LegacyStrikeMATActorCritic,
                       'strike_mat_v2': LocalStrikeMATActorCritic,
                       'strike_mat_v3': ScalarTypeStrikeMATActorCritic,
                       'strike_mat_v4': StrikeMATActorCritic}[architecture]
    elif architecture in ('strike_target_mappo_v1', 'strike_target_mappo_v2',
                          'strike_target_mappo_v3', 'strike_target_mappo_v4', 'strike_target_mappo_v5',
                          'strike_target_mappo_v6', 'strike_target_mappo_v7', 'strike_target_mappo_v8'):
        from .strike_target_actor import (HardAssignmentTargetScoringActorCritic,
                                          ScalarTypeTargetScoringActorCritic,
                                          LegacyTargetScoringActorCritic,
                                          UnmaskedTargetScoringActorCritic,
                                          FullInputTargetScoringActorCritic,
                                          ProgressTargetScoringActorCritic,
                                          AliveStatusTargetScoringActorCritic,
                                          TargetScoringActorCritic)
        model_class = {'strike_target_mappo_v1': ScalarTypeTargetScoringActorCritic,
                       'strike_target_mappo_v2': HardAssignmentTargetScoringActorCritic,
                       'strike_target_mappo_v3': LegacyTargetScoringActorCritic,
                       'strike_target_mappo_v4': UnmaskedTargetScoringActorCritic,
                       'strike_target_mappo_v5': FullInputTargetScoringActorCritic,
                       'strike_target_mappo_v6': ProgressTargetScoringActorCritic,
                       'strike_target_mappo_v7': AliveStatusTargetScoringActorCritic,
                       'strike_target_mappo_v8': TargetScoringActorCritic}[architecture]
    elif architecture in ('strike_mappo_v21', 'strike_mappo_v22'):
        model_class = ScalarTypeStrikeActorCritic
    elif architecture == 'strike_mappo_v23':
        model_class = HardAssignmentStrikeActorCritic
    elif architecture == 'strike_mappo_v24':
        model_class = StrikeActorCritic
    elif architecture == 'strike_coma_ppo_v1':
        model_class = LegacyStrikeCOMAActorCritic
    elif architecture == 'strike_coma_ppo_v2':
        model_class = LegacyQOnlyStrikeCOMAActorCritic
    elif architecture == 'strike_coma_ppo_v3':
        model_class = StrikeCOMAActorCritic
    elif architecture == 'strike_compact_coma_v1':
        model_class = CompactStrikeCOMAActorCritic
    elif architecture == 'strike_target_coma_v1':
        from .strike_target_actor import LegacyTargetScoringCOMAActorCritic
        model_class = LegacyTargetScoringCOMAActorCritic
    elif architecture == 'strike_target_coma_v2':
        from .strike_target_actor import FullInputTargetScoringCOMAActorCritic
        model_class = FullInputTargetScoringCOMAActorCritic
    elif architecture == 'strike_target_coma_v3':
        from .strike_target_actor import StrikeMatrixTargetScoringCOMAActorCritic
        model_class = StrikeMatrixTargetScoringCOMAActorCritic
    elif architecture == 'strike_target_coma_v4':
        from .strike_target_actor import ProgressTargetScoringCOMAActorCritic
        model_class = ProgressTargetScoringCOMAActorCritic
    elif architecture == 'strike_target_coma_v5':
        from .strike_target_actor import AliveStatusTargetScoringCOMAActorCritic
        model_class = AliveStatusTargetScoringCOMAActorCritic
    elif architecture == 'strike_target_coma_v6':
        from .strike_target_actor import PreviousProbabilityTargetScoringCOMAActorCritic
        model_class = PreviousProbabilityTargetScoringCOMAActorCritic
    elif architecture == 'strike_target_coma_v7':
        from .strike_target_actor import PreviousChoiceActorTargetScoringCOMAActorCritic
        model_class = PreviousChoiceActorTargetScoringCOMAActorCritic
    elif architecture == 'strike_target_coma_v8':
        from .strike_target_actor import NoVelocityTargetScoringCOMAActorCritic
        model_class = NoVelocityTargetScoringCOMAActorCritic
    elif architecture == 'strike_target_coma_v9':
        from .strike_target_actor import LegacyV9TargetScoringCOMAActorCritic
        model_class = LegacyV9TargetScoringCOMAActorCritic
    elif architecture == 'strike_target_coma_v10':
        from .strike_target_actor import LegacyV10TargetScoringCOMAActorCritic
        model_class = LegacyV10TargetScoringCOMAActorCritic
    elif architecture == 'strike_target_coma_v11':
        from .strike_target_actor import LegacyV11TargetScoringCOMAActorCritic
        model_class = LegacyV11TargetScoringCOMAActorCritic
    elif architecture == 'strike_target_coma_v12':
        from .strike_target_actor import TargetScoringCOMAActorCritic
        model_class = TargetScoringCOMAActorCritic
    elif architecture == 'strike_target_coma_v13':
        from .strike_target_actor import DistanceTargetScoringCOMAActorCritic
        model_class = DistanceTargetScoringCOMAActorCritic
    elif architecture == 'strike_target_coma_v14':
        from .strike_target_actor import GraphTargetScoringCOMAActorCritic
        model_class = GraphTargetScoringCOMAActorCritic
    elif architecture == 'strike_target_coma_v15':
        from .strike_target_actor import AttentionTargetScoringCOMAActorCritic
        model_class = AttentionTargetScoringCOMAActorCritic
    else:
        raise ValueError(f'Unsupported checkpoint architecture: {architecture}')
    model = model_class(config.n_targets, config.n_agents, payload.get('hidden', 64))
    if architecture == 'strike_mappo_v21':
        model.architecture = architecture
        model.relative_actor_coordinates = False
    load_checkpoint(path, model)
    return model


def load_checkpoint(path, model):
    payload = checkpoint_payload(path)
    if payload.get('architecture') != model.architecture:
        raise ValueError('Checkpoint architecture does not match the requested policy')
    if model.architecture != 'strike_target_coma_v15' and (
            payload["obs_dim"] != model.obs_dim or payload["n_targets"] != model.n_targets
            or payload["n_agents"] != model.n_agents):
        raise ValueError("Model shape does not match config")
    model.load_state_dict(payload["state_dict"])
    model.training_settings = payload.get('training_settings', {})
    return payload.get("metadata", {})


def critic_state(world, one_hot=True):
    """Global mission, drone, target, and current-assignment state."""
    c = world.c
    value_scale = max(1.0, 5.0 * c.n_targets)
    mission = np.array([(c.horizon - world.t) / c.horizon,
                        world.formation_one_initial_score / value_scale,
                        world.score / value_scale])
    participation = world.strike_participants.any(axis=0)
    drones = np.column_stack((world.pos / c.size,
                              world.agent_active.astype(float),
                              participation.astype(float),
                              world.agent_strike_progress / c.strike_steps_per_life))
    target_type = ((world.target_type[:, None] == np.arange(1, 4)).astype(float)
                   * world.target_exists[:, None] if one_hot else world.target_type / 3.0)
    targets = np.column_stack((np.nan_to_num(world.targets, nan=0.0) / c.size,
                               target_type,
                               world.target_life / 2.0))
    state = np.concatenate((mission, drones.ravel(), targets.ravel(),
                            world.target_assignment.astype(float).ravel())).astype(np.float32)
    return np.repeat(state[None], world.c.n_agents, axis=0)


def actor_observation(world, relative=True, one_hot=True,
                      previous_action_probabilities=None):
    """Team state plus either hard assignments or previous policy probabilities."""
    state = critic_state(world, one_hot=one_hot)
    width = 6 if one_hot else 4
    if previous_action_probabilities is not None:
        probabilities = np.asarray(previous_action_probabilities, dtype=np.float32)
        expected = (world.n, world.c.n_targets)
        if probabilities.shape != expected:
            raise ValueError(
                f'Expected previous action probabilities with shape {expected}, '
                f'got {probabilities.shape}')
        assignment_start = 3 + 5 * world.n + width * world.c.n_targets
        # The state layout is target-major, whereas policy probabilities are
        # naturally agent-major.
        state[:, assignment_start:assignment_start + probabilities.size] = \
            probabilities.T.reshape(-1)
    if relative:
        target_start = 3 + 5 * world.n
        for agent in range(world.n):
            origin = world.pos[agent] / world.c.size
            drones = state[agent, 3:target_start].reshape(world.n, 5)
            drones[:, :2] -= origin
            targets = state[agent, target_start:target_start + width * world.c.n_targets].reshape(-1, width)
            targets[:, :2] -= origin
            targets[~world.target_exists, :2] = 0
    identity = np.eye(world.n, dtype=np.float32)
    return np.concatenate((state, identity), axis=1)


def _advantages(reward, value, next_value, done, gamma=.99, gae_lambda=.95):
    advantage = np.zeros_like(reward)
    carry = np.zeros(reward.shape[1], dtype=np.float32)
    for t in reversed(range(len(reward))):
        delta = reward[t] + gamma * next_value[t] * (1 - done[t]) - value[t]
        carry = delta + gamma * gae_lambda * (1 - done[t]) * carry
        advantage[t] = carry
    return advantage, advantage + value


def _batch(obs, state, target, logp, action_mask, advantage, returns, active, decision,
           old_values, critic_active=None):
    observations = np.asarray(obs)
    states = np.asarray(state)
    target_array = np.asarray(target)
    joint_target_array = np.where(np.asarray(active), target_array, 0)
    return dict(
        obs=torch.as_tensor(observations.reshape(-1, observations.shape[-1]), dtype=torch.float32),
        state=torch.as_tensor(states.reshape(-1, states.shape[-1]), dtype=torch.float32),
        target=torch.as_tensor(target_array.reshape(-1), dtype=torch.long),
        joint_action=torch.as_tensor(
            joint_target_array.reshape(-1, target_array.shape[-1]), dtype=torch.long),
        logp=torch.as_tensor(np.asarray(logp).reshape(-1), dtype=torch.float32),
        action_mask=torch.as_tensor(np.asarray(action_mask).reshape(
            -1, np.asarray(action_mask).shape[-1]), dtype=torch.bool),
        advantage=torch.as_tensor(np.asarray(advantage).reshape(-1), dtype=torch.float32),
        returns=torch.as_tensor(np.asarray(returns).reshape(-1), dtype=torch.float32),
        old_values=torch.as_tensor(np.asarray(old_values).reshape(-1), dtype=torch.float32),
        active=torch.as_tensor(np.asarray(active).reshape(-1), dtype=torch.bool),
        critic_active=torch.as_tensor(
            (np.asarray(active).any(axis=-1) if critic_active is None
             else np.asarray(critic_active)).reshape(-1), dtype=torch.bool),
        decision=torch.as_tensor((np.asarray(active) & np.asarray(decision)).reshape(-1), dtype=torch.bool))


def _scale_actor_advantage(advantage, decision):
    """Apply the conventional PPO mean/std normalization to decisions."""
    chosen = advantage[decision]
    if not len(chosen):
        return advantage
    return (advantage - chosen.mean()) / (chosen.std(unbiased=False) + 1e-5)


def _value_loss(model, values, old_values, returns, active):
    model.value_normalizer.update(returns)
    normalized = model.value_normalizer.normalize(returns)
    clipped = old_values + (values - old_values).clamp(-.2, .2)
    def huber(error):
        magnitude = error.abs()
        quadratic = magnitude.clamp(max=10.0)
        return .5 * quadratic.square() + 10.0 * (magnitude - quadratic)
    loss = torch.maximum(huber(normalized - values), huber(normalized - clipped))
    return loss[active].mean()


def _critic_parameters(model):
    if isinstance(model, StrikeCOMAActorCritic):
        return list(model.q_body.parameters()) + list(model.q_head.parameters())
    parameters = list(model.critic_body.parameters()) + list(model.value_head.parameters())
    if isinstance(model, StrikeCOMAActorCritic):
        parameters += list(model.q_body.parameters()) + list(model.q_head.parameters())
    return parameters


def _td_lambda_returns(rewards, next_q, done, gamma, trace_lambda):
    """Terminal cuts the trace; rollout cutoff bootstraps from target Q."""
    returns = np.empty_like(rewards)
    carry = next_q[-1].copy()
    for t in reversed(range(len(rewards))):
        carry = rewards[t] + gamma * (1 - done[t]) * (
            (1 - trace_lambda) * next_q[t] + trace_lambda * carry)
        returns[t] = carry
    return returns


def _coma_update(model, optimizers, batch, probabilities, epochs, minibatch_size):
    """Fit Q first, then freeze counterfactual advantages for all PPO epochs."""
    optimizer = optimizers[1]
    losses = []
    size = len(batch['state'])
    settings = getattr(model, 'training_settings', {})
    width = settings.get('critic_minibatch_size') or minibatch_size or size
    for _ in range(settings.get('critic_epochs') or epochs):
        order = torch.randperm(size)
        for start in range(0, size, width):
            indices = order[start:start + width]
            indices = indices[batch['critic_active'][indices]]
            if not len(indices):
                continue
            prediction = model.joint_q(batch['state'][indices], batch['joint_action'][indices])
            loss = torch.nn.functional.huber_loss(prediction, batch['returns'][indices], delta=10.)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(_critic_parameters(model), 10.)
            optimizer.step()
            # Track every completed Q optimizer step, including minibatches.
            # The rollout's precomputed TD-lambda targets remain fixed.
            model.update_target(tau=settings.get('target_q_polyak_tau', .01))
            losses.append(loss.item())
    batch = dict(batch)
    with torch.no_grad():
        # Chunk the counterfactual expansion (agents x actions) to bound memory.
        advantages = [model.counterfactual_advantages(
            batch['state'][start:start + 256], batch['joint_action'][start:start + 256],
            probabilities[start:start + 256],
            sampled_returns=(batch['returns'][start:start + 256]
                             if settings.get('coma_advantage') == 'sampled_return' else None))
            for start in range(0, size, 256)]
        batch['advantage'] = torch.cat(advantages).flatten()
        if 'lookahead_advantage' in batch:
            batch['advantage'] += settings.get('lookahead_credit', 0.) * batch['lookahead_advantage']
    metrics = _ppo_update(model, optimizers, batch, epochs, minibatch_size)
    metrics['q_loss'] = float(np.mean(losses)) if losses else 0.
    return metrics


def _optimizers(model, learning_rate=5e-4):
    actor = list(model.actor_body.parameters()) + list(model.target_head.parameters())
    critic = _critic_parameters(model)
    return (torch.optim.Adam(actor, lr=learning_rate, eps=1e-5),
            torch.optim.Adam(critic, lr=learning_rate, eps=1e-5))


def _ppo_update(model, optimizers, batch, epochs, minibatch_size=None):
    actor_optimizer, critic_optimizer = optimizers
    settings = getattr(model, 'training_settings', {})
    policy_clip = settings.get('policy_clipping', .2)
    entropy_coef = settings.get('entropy_coef', .01)
    if model.autoregressive:
        batch = dict(batch)
        for key in ('obs', 'target', 'logp', 'action_mask', 'advantage', 'active', 'decision'):
            value = batch[key]
            batch[key] = value.reshape(-1, model.n_agents, *value.shape[1:])
    advantage = _scale_actor_advantage(batch['advantage'], batch['decision'])
    batch_size = len(batch['obs'])
    actor_minibatch_size = minibatch_size or batch_size
    # Default to the original v12 rollout minibatches. Decision-only batching
    # remains opt-in for reproducing the separate commitment experiments.
    if settings.get('actor_minibatches') == 'decision_rows_only_joint_prefix_preserved_for_mat':
        actor_rows = (batch['decision'].any(dim=-1) if model.autoregressive
                      else batch['decision'])
        actor_indices = torch.nonzero(actor_rows, as_tuple=False).flatten()
    else:
        actor_indices = torch.arange(batch_size)
    records = []
    for _ in range(epochs):
        order = actor_indices[torch.randperm(len(actor_indices))]
        for start in range(0, len(order), actor_minibatch_size):
            indices = order[start:start + actor_minibatch_size]
            sample = {key: batch[key][indices] for key in
                      ('obs', 'target', 'logp', 'action_mask', 'decision', 'active')}
            decision = sample['decision']
            if not sample['active'].any():
                continue
            metrics = dict(policy_loss=0.0, entropy=0.0, approx_kl=0.0,
                           clip_fraction=0.0, actor_grad_norm=0.0)
            if decision.any():
                logp, entropy = model.evaluate_actions(
                    sample['obs'], sample['target'], sample['action_mask'])
                log_ratio = logp - sample['logp']
                ratio = log_ratio.exp()
                surrogate = torch.minimum(ratio * advantage[indices],
                                          ratio.clamp(1 - policy_clip, 1 + policy_clip) * advantage[indices])
                actor_loss = -surrogate[decision].mean()
                actor_entropy = entropy[decision].mean()
                actor_optimizer.zero_grad()
                (actor_loss - entropy_coef * actor_entropy).backward()
                actor_grad = nn.utils.clip_grad_norm_(
                    list(model.actor_body.parameters()) + list(model.target_head.parameters()), 10.0)
                actor_optimizer.step()
                metrics.update(policy_loss=actor_loss.item(), entropy=actor_entropy.item(),
                               approx_kl=((ratio - 1) - log_ratio)[decision].mean().item(),
                               clip_fraction=((ratio - 1).abs() > policy_clip)[decision].float().mean().item(),
                               actor_grad_norm=float(actor_grad))
            if decision.any():
                records.append(metrics)
        critic_batch_size = len(batch['state'])
        if isinstance(model, StrikeCOMAActorCritic):
            continue  # Q was fitted before the fixed-advantage actor update.
        critic_minibatch_size = minibatch_size or critic_batch_size
        critic_order = torch.randperm(critic_batch_size)
        for start in range(0, critic_batch_size, critic_minibatch_size):
            indices = critic_order[start:start + critic_minibatch_size]
            sample = {key: batch[key][indices] for key in
                      ('state', 'old_values', 'returns', 'critic_active', 'joint_action')}
            if not sample['critic_active'].any():
                continue
            values = model.values(sample['state'])
            value_loss = _value_loss(model, values, sample['old_values'],
                                     sample['returns'], sample['critic_active'])
            critic_loss = value_loss
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad = nn.utils.clip_grad_norm_(_critic_parameters(model), 10.0)
            critic_optimizer.step()
            record = dict(value_loss=value_loss.item(),
                          critic_grad_norm=float(critic_grad))
            records.append(record)
    keys = ('policy_loss', 'entropy', 'approx_kl', 'clip_fraction',
            'actor_grad_norm', 'value_loss', 'q_loss', 'critic_grad_norm')
    return {key: float(np.mean([row[key] for row in records if key in row]))
            if any(key in row for row in records) else 0.0
            for key in keys}


def _interquartile_mean(values):
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values):
        return float('nan')
    lower, upper = 0.25 * len(values), 0.75 * len(values)
    weights = np.array([
        max(0.0, min(i + 1, upper) - max(i, lower))
        for i in range(len(values))])
    return float(np.sum(values * weights) / np.sum(weights))


def _summarize(rows):
    keys = ("team_return", "discounted_team_return", "mission_success",
            "score", "baseline_score", "destroyed_fraction", "score_auc", "steps")
    summary = {
        key: (float(np.mean([row[key] for row in rows]))
              if key == 'mission_success'
              else _interquartile_mean([row[key] for row in rows]))
        for key in keys}
    for baseline in sorted({int(row["baseline_score"]) for row in rows}):
        group = [row for row in rows if int(row["baseline_score"]) == baseline]
        summary[f"success_b{baseline}"] = float(np.mean(
            [row["mission_success"] for row in group]))
        summary[f"episodes_b{baseline}"] = len(group)
    for case_index, task in enumerate(EVALUATION_TASKS, start=1):
        group = [row for row in rows if row.get("evaluation_task") == task]
        if group:
            task_metrics = {
                'damage': float(np.mean([row['score'] for row in group])),
                'success': float(np.mean([row['mission_success'] for row in group])),
                'team_return': float(np.mean([row['team_return'] for row in group])),
                'discounted_return': float(np.mean(
                    [row['discounted_team_return'] for row in group])),
            }
            for metric, value in task_metrics.items():
                summary[f'{metric}_{task}'] = value
                summary[f'case{case_index}_{metric}'] = value
            summary[f"episodes_{task}"] = len(group)
            summary[f"case{case_index}_episodes"] = len(group)
    return summary


def _suite_checkpoint_key(resolved, unresolved):
    """Maximize the evaluation suite; break ties with unaided policy quality."""
    return (float(np.mean([resolved[f'case{i}_damage'] for i in (1, 2, 3)])),
            float(np.mean([unresolved[f'case{i}_damage'] for i in (1, 2, 3)])),
            resolved['discounted_team_return'])


def evaluate_model(model, config, episodes=3, seed=10000, resolver=True):
    """Evaluate the current actor/resolver on held-out scenario seeds."""
    from .env import World
    _validate_evaluation_episodes(config, episodes)
    rows = []
    with torch.no_grad():
        for episode in range(episodes):
            world = World(config)
            task = evaluation_task_for_episode(config, episode)
            obs = world.reset(
                seed + episode,
                success_threshold=_evaluation_threshold(config, episode),
                evaluation_task=task)
            schedule = DecisionSchedule(config)
            while not world.done:
                action, _ = scheduled_action(model, world, schedule, resolver=resolver)
                obs, _, _, _, _ = world.step(action)
            rows.append(dict(world.metrics(), evaluation_task=task))
    return _summarize(rows)


def evaluate_baseline(policy_class, config, episodes=3, seed=10000):
    """Evaluate a baseline on exactly the same held-out scenario seeds."""
    from .env import World
    _validate_evaluation_episodes(config, episodes)
    rows = []
    for episode in range(episodes):
        episode_seed = seed + episode
        world = World(config)
        task = evaluation_task_for_episode(config, episode)
        obs = world.reset(
            episode_seed,
            success_threshold=_evaluation_threshold(config, episode),
            evaluation_task=task)
        policy = policy_class(config, episode_seed)
        while not world.done:
            action = policy.predict(obs)
            obs, _, _, _, _ = world.step(action)
        rows.append(dict(world.metrics(), evaluation_task=task))
    return _summarize(rows)


def _reset_training_world(world, rng, b12_probability=0.):
    """Sample the original random-B training task and near/far layout."""
    seed = int(rng.integers(2**31))
    if b12_probability and rng.random() < b12_probability:
        # Curriculum changes only reset sampling, never physical dynamics or eval.
        original = world.c
        world.c = replace(original, formation_separation=25., formation_distance_spread=33.)
        try:
            world.reset(seed, success_threshold=12)
        finally:
            world.c = original
    else:
        world.reset(seed)
    return None


def _training_team_reward(world, previous_life, previous_score):
    """Training target values: type 3/2/1 totals are 1/2/5."""
    life_lost = np.maximum(np.asarray(previous_life) - world.target_life, 0)
    value_per_life = np.select(
        [world.target_type == 1, world.target_type == 2, world.target_type == 3],
        [2.5, 2.0, 1.0], default=0.0)
    return float(DAMAGE_REWARD_RATE * np.dot(life_lost, value_per_life))


def _actor_event(world):
    """Select pre-action states using only information already observed."""
    finishing = (world.agent_active & world.strike_participants.any(axis=0)
                 & (world.agent_strike_progress >= world.c.strike_steps_per_life - 1))
    return bool(world.t == 0 or world.t == world.c.horizon - 1 or finishing.any())


def train_mappo(config, total_agent_transitions, seed=7, rollout_steps=200, epochs=10,
              minibatch_size=None, learning_rate=5e-4, progress=False,
              eval_interval=10000, eval_episodes=3, eval_seed=10000,
              best_path=None, latest_path=None, checkpoint_dir=None,
              checkpoint_interval=100000, n_envs=32, diagnostics_path=None,
              algorithm='mappo', entropy_coef=.01, policy_clip=.2,
              policy_temperature=1.0, actor_samples='all', guided_training=False,
              coma_advantage='q', critic_epochs=None, critic_minibatch_size=None,
              best_metric='case1', lookahead_credit=0., deadline_mask=False,
              initial_checkpoint=None, actor_target_distances=False,
              training_b12_probability=0., actor_decision_minibatches=False,
              target_coma_critic='graph', target_coma_actor='attention'):
    from .env import World
    from .strike_guidance import progress_potential, avoidable_switches
    from .strike_lookahead import counterfactual_credit
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    if coma_advantage not in ('q', 'sampled_return'):
        raise ValueError('coma_advantage must be q or sampled_return')
    if actor_target_distances and algorithm != 'target_coma':
        raise ValueError('Explicit actor target distances require target_coma')
    if target_coma_critic not in ('graph', 'pair'):
        raise ValueError('target_coma_critic must be graph or pair')
    if target_coma_actor not in ('attention', 'mlp'):
        raise ValueError('target_coma_actor must be attention or mlp')
    if algorithm == 'target_coma' and target_coma_critic == 'graph' and actor_target_distances:
        raise ValueError('Graph critic currently preserves the v12 actor; use pair for the v13 experiment')
    if not 0 <= training_b12_probability <= 1:
        raise ValueError('training_b12_probability must be between zero and one')
    if training_b12_probability and not _uses_balanced_thresholds(config):
        raise ValueError('B12 curriculum requires the fixed five-target training distribution')
    if best_metric not in ('case1', 'suite_damage'):
        raise ValueError('best_metric must be case1 or suite_damage')
    if not np.isfinite(lookahead_credit) or lookahead_credit < 0:
        raise ValueError('lookahead_credit must be finite and nonnegative')
    if deadline_mask and config.target_motion_scale != 0:
        raise ValueError('deadline_mask requires stationary targets')
    if any(value is not None and value <= 0
           for value in (critic_epochs, critic_minibatch_size)):
        raise ValueError('Critic training sizes must be positive')
    if algorithm not in ('coma', 'target_coma', 'compact_coma') and (
            coma_advantage != 'q' or critic_epochs is not None or critic_minibatch_size is not None
            or lookahead_credit):
        raise ValueError('Counterfactual advantage and Q options require COMA')
    if actor_samples not in ('all', 'events'):
        raise ValueError('actor_samples must be all or events')
    if guided_training and actor_samples != 'all':
        raise ValueError('guided_training requires actor_samples=all to learn approach decisions')
    if config.commit_target and actor_samples != 'all':
        raise ValueError('commit_target requires actor_samples=all to retain every actual reselection')
    if min(total_agent_transitions, rollout_steps, epochs, n_envs) <= 0:
        raise ValueError('Training sizes must be positive')
    if entropy_coef < 0 or not 0 < policy_clip < 1:
        raise ValueError('entropy_coef must be nonnegative and policy_clip must be in (0, 1)')
    if policy_temperature <= 0 or (algorithm not in ('mappo', 'coma', 'compact_coma') and policy_temperature != 1.0):
        raise ValueError('Non-default policy_temperature requires MAPPO/COMA and must be positive')
    worlds = [World(config) for _ in range(n_envs)]
    schedules = [DecisionSchedule(config) for _ in worlds]
    for world in worlds:
        _reset_training_world(world, rng, training_b12_probability)
    if algorithm == 'mat':
        from .strike_mat import StrikeMATActorCritic
        model_class = StrikeMATActorCritic
    elif algorithm == 'target_mappo':
        from .strike_target_actor import TargetScoringActorCritic
        model_class = TargetScoringActorCritic
    elif algorithm == 'mappo':
        model_class = StrikeActorCritic
    elif algorithm == 'coma':
        model_class = StrikeCOMAActorCritic
    elif algorithm == 'compact_coma':
        model_class = CompactStrikeCOMAActorCritic
    elif algorithm == 'target_coma':
        from .strike_target_actor import TargetScoringCOMAActorCritic
        model_class = TargetScoringCOMAActorCritic
        if target_coma_critic == 'graph':
            from .strike_target_actor import GraphTargetScoringCOMAActorCritic
            model_class = GraphTargetScoringCOMAActorCritic
            if target_coma_actor == 'attention':
                from .strike_target_actor import AttentionTargetScoringCOMAActorCritic
                model_class = AttentionTargetScoringCOMAActorCritic
        if actor_target_distances:
            from .strike_target_actor import DistanceTargetScoringCOMAActorCritic
            model_class = DistanceTargetScoringCOMAActorCritic
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    model = model_class(config.n_targets, config.n_agents)
    if initial_checkpoint is not None:
        # Weight warm-start, not an optimizer/RNG-exact training resume.
        load_checkpoint(initial_checkpoint, model)
    model.training_settings = dict(n_envs=n_envs, rollout_steps=rollout_steps,
        epochs=epochs, minibatch_size=minibatch_size, learning_rate=learning_rate,
        critic_learning_rate=learning_rate, adam_epsilon=1e-5, hidden=64,
        initial_checkpoint=str(initial_checkpoint) if initial_checkpoint is not None else None,
        optimizer_reinitialized=initial_checkpoint is not None,
        actor_target_distances=actor_target_distances,
        training_b12_probability=training_b12_probability,
        activation='relu', feature_normalization=True, orthogonal=True,
        actor_gain=.01, value_normalization=True, huber_delta=10.0,
        value_clipping=.2, policy_clipping=policy_clip, entropy_coef=entropy_coef,
        policy_temperature=policy_temperature,
        target_type_encoding='one_hot',
        actor_sample_mode=actor_samples,
        actor_minibatches=('decision_rows_only_joint_prefix_preserved_for_mat'
                          if actor_decision_minibatches else 'all_rollout_rows'),
        guided_training=guided_training,
        lookahead_credit=lookahead_credit,
        deadline_mask=deadline_mask,
        lookahead_credit_description=('auxiliary_counterfactual_assignment_damage_forecast'
                                      if lookahead_credit else None),
        progress_shaping_scale=1.0 if guided_training else 0.0,
        switch_cost=.02 if guided_training else 0.0,
        training_reward='target_value_type3_1_type2_2_type1_5',
        success_bonus=0.0,
        damage_reward_rate=DAMAGE_REWARD_RATE,
        value_loss_coef=1.0, max_grad_norm=10.0, recurrent=False,
        gamma=config.discount_gamma, gae_lambda=config.gae_lambda,
        finite_horizon_terminal=True, rollout_execution='synchronous_batched',
        decision_interval=config.decision_interval,
        commit_target=config.commit_target,
        training_distribution=('B12_mixture_with_uniform_B_distance_25_58'
            if training_b12_probability else 'uniform_B_5_8_9_11_12_distance_25_58'
            if evaluation_task_for_episode(config, 0) is not None else
            'uniform_agent_target_counts_random_majority_F1_iid_types'
            if config.randomize_counts and config.randomize_target_composition else 'random'),
        selection_allocation='direct_joint_action_no_rejection_or_reselection')
    model.training_settings['evaluation_allocation'] = 'distance_priority_rejection_reselection'
    model.training_settings['evaluation_modes'] = ['no_resolver', 'resolver']
    model.training_settings['best_evaluation_mode'] = 'resolver'
    model.training_settings['best_metric'] = best_metric
    if getattr(model, 'drone_velocity', False):
        model.training_settings.update(
            actor_drone_velocity='self_and_peers_world_vx_vy_over_strike_speed',
            critic_drone_velocity='all_drones_world_vx_vy_over_strike_speed')
    if not model.autoregressive:
        model.training_settings.update(actor_coordinates='self_relative',
                                       critic_coordinates='absolute',
                                       previous_joint_action='softmax_probabilities')
    if algorithm in ('target_mappo', 'target_coma'):
        model.training_settings['actor_architecture'] = 'shared_target_scorer_with_teammates'
        if model.architecture == 'strike_target_coma_v15':
            model.training_settings.update(
                actor_architecture='own_target_query_peer_cross_attention',
                actor_slot_independent_weights=True,
                actor_attention='unnormalized_sigmoid_positive_messages',
                actor_activation='silu', actor_feature_normalization=False,
                actor_body_initialization='pytorch_default',
                actor_heads=4, actor_layers=2,
                actor_relation_inputs='previous_probabilities_and_choices_only')
        model.training_settings['actor_intent_active_only'] = True
        if getattr(model, 'actor_previous_actions', False):
            model.training_settings['actor_previous_actual_choice'] = 'target_major_one_hot'
    elif algorithm == 'compact_coma':
        model.training_settings['actor_architecture'] = 'global_mlp_matched_target_coma_information'
        model.training_settings['actor_intent_active_only'] = True
    if getattr(model, 'critic_previous_probabilities', False):
        model.training_settings['critic_previous_joint_action'] = (
            'active_previous_probabilities_appended'
            if getattr(model, 'critic_mask_inactive_intent', False)
            else 'previous_probabilities_appended')
    if getattr(model, 'critic_previous_actions', False):
        model.training_settings['critic_previous_actual_choice'] = 'active_target_major_one_hot'
    if isinstance(model, StrikeCOMAActorCritic):
        model.training_settings.update(
            algorithm='coma_counterfactual_advantage_with_ppo_clip',
            critic='centralized_joint_action_Q_with_target_Q',
            actor_advantage='Q(s,a)-sum_a_i pi_old(a_i|o_i)Q(s,(a_-i,a_i))',
            # Keep the original Q-based actor objective by default; give Q
            # four fitting passes for each actor pass on the same rollout.
            coma_advantage=coma_advantage,
            critic_epochs=critic_epochs if critic_epochs is not None else 4 * epochs,
            critic_minibatch_size=critic_minibatch_size,
            q_target='sampled_SARSA_TD_lambda_return',
            value_normalization=False, value_clipping=None,
            target_q_polyak_tau=.01,
            target_q_update_frequency='every_critic_optimizer_step',
            critic_before_actor=True)
        if algorithm == 'target_coma':
            model.training_settings['critic'] = 'pair_embedding_target_concat_joint_Q_with_target_Q'
            model.training_settings['critic_action_encoding'] = 'encode_selected_drone_target_pairs_then_pool_per_target'
            model.training_settings['critic_team_input'] = 'raw_state_concat_ordered_target_pools_and_counts'
            if target_coma_critic == 'graph':
                model.training_settings.update(
                    critic='variable_slot_relational_graph_Q_with_target_Q',
                    critic_action_encoding='current_assignment_relation_on_bipartite_edges',
                    critic_team_input='time_query_gated_readout_of_drone_and_target_tokens',
                    critic_attention='unnormalized_sigmoid_gates_positive_messages',
                    critic_hidden=64, critic_heads=4, critic_rounds=2,
                    critic_slot_independent_weights=True)
        if coma_advantage == 'sampled_return':
            model.training_settings['algorithm'] = 'sampled_return_counterfactual_ppo'
            model.training_settings['actor_advantage'] = (
                'sampled_return-sum_a_i pi_old(a_i|o_i)Q(s,(a_-i,a_i))')
        if lookahead_credit:
            model.training_settings['algorithm'] = 'model_assisted_counterfactual_ppo'
            model.training_settings['actor_advantage'] += (
                f'+{lookahead_credit:g}*counterfactual_assignment_damage_forecast')
    if model.autoregressive:
        model.training_settings.update(algorithm='mat', actor_activation='gelu',
            attention_heads=1, transformer_blocks=1, agent_order='fixed',
            actor_minibatch_unit='environment_steps', critic='separate_team_mlp',
            actor_observation='global_team_state', actor_agent_id='decoder_embedding',
            transformer_encoder_blocks=0, transformer_decoder_blocks=1)
    optimizers = _optimizers(model, learning_rate)
    completed, episode_rows, evaluation_rows = 0, [], []
    best_key = (-float("inf"), -float("inf"))
    baseline_evaluations = None
    next_eval = eval_interval if eval_interval > 0 else None
    next_checkpoint = checkpoint_interval if checkpoint_dir is not None else None
    progress_bar = tqdm(total=total_agent_transitions, desc="Train", unit="transition",
                        dynamic_ncols=True, disable=not progress)

    def run_evaluation(agent_transitions):
        nonlocal best_key, baseline_evaluations
        if latest_path is not None:
            save_checkpoint(latest_path, model, {"agent_transitions": agent_transitions,
                                                 "seed": seed})
        from .policies import NearestTargetPolicy, RandomPolicy, TypePriorityPolicy
        if baseline_evaluations is None:
            baseline_evaluations = {
                "random": evaluate_baseline(
                    RandomPolicy, config, eval_episodes, eval_seed),
                "nearest": evaluate_baseline(
                    NearestTargetPolicy, config, eval_episodes, eval_seed),
                "type_priority": evaluate_baseline(
                    TypePriorityPolicy, config, eval_episodes, eval_seed),
            }
        evaluations = dict(baseline_evaluations)
        for mode, enabled in (('no_resolver', False), ('resolver', True)):
            evaluations[f'{algorithm}_{mode}'] = evaluate_model(
                model, config, eval_episodes, eval_seed, resolver=enabled)
        task_evaluation = evaluations[f'{algorithm}_resolver']
        mappo_success = task_evaluation["mission_success"]
        mappo_return = task_evaluation["team_return"]
        mappo_discounted_return = task_evaluation["discounted_team_return"]
        if best_metric == 'suite_damage' and 'case1_damage' in task_evaluation:
            candidate_key = _suite_checkpoint_key(
                task_evaluation, evaluations[f'{algorithm}_no_resolver'])
            best_metadata = {
                'best_metric': 'mean_case_damage_then_no_resolver_damage_then_discounted_return',
                'best_case_damage': [task_evaluation[f'case{i}_damage'] for i in (1, 2, 3)],
                'best_mean_damage': candidate_key[0],
                'best_mean_no_resolver_damage': candidate_key[1],
            }
        elif 'case1_damage' in task_evaluation:
            candidate_key = (task_evaluation['case1_damage'],
                             task_evaluation['case1_discounted_return'])
            best_metadata = {
                'best_metric': 'case1_damage_then_case1_discounted_return',
                'best_case1_damage': task_evaluation['case1_damage'],
                'best_case1_success': task_evaluation['case1_success'],
                'best_case1_team_return': task_evaluation['case1_team_return'],
                'best_case1_discounted_return':
                    task_evaluation['case1_discounted_return'],
            }
        else:
            candidate_key = (mappo_return, mappo_success)
            best_metadata = {
                'best_metric': 'interquartile_mean_team_return',
                'best_success': mappo_success,
                'best_value': mappo_return,
                'best_discounted_value': mappo_discounted_return,
            }
        if best_path is not None and candidate_key > best_key:
            best_key = candidate_key
            save_checkpoint(best_path, model, dict(best_metadata, **{
                "agent_transitions": agent_transitions,
                "eval_episodes": eval_episodes,
                "eval_seed": eval_seed,
                "evaluation_allocation": "distance_priority_rejection_reselection",
            }))
            if best_metric == 'suite_damage' and 'case1_damage' in task_evaluation:
                progress_bar.write(f"Saved best checkpoint: {best_path} "
                                   f"(mean_case_D={candidate_key[0]:.3f})")
            elif 'case1_damage' in task_evaluation:
                progress_bar.write(
                    f"Saved best checkpoint: {best_path} "
                    f"(case1_D={task_evaluation['case1_damage']:.3f}, "
                    f"case1_discounted_return="
                    f"{task_evaluation['case1_discounted_return']:.3f})")
            else:
                progress_bar.write(
                    f"Saved best checkpoint: {best_path} "
                    f"(success={mappo_success:.3f}, "
                    f"undiscounted_return={mappo_return:.3f}, "
                    f"discounted_return={mappo_discounted_return:.3f})")
        for policy_name, evaluation in evaluations.items():
            evaluation_rows.append(dict(agent_transitions=agent_transitions,
                                        policy=policy_name, episodes=eval_episodes,
                                        seed=eval_seed, **evaluation))
            damage_by_task = " ".join(
                f"case{index}={evaluation['damage_' + task]:.3f}"
                for index, task in enumerate(EVALUATION_TASKS, start=1)
                if 'damage_' + task in evaluation)
            damage_display = (f"D_by_case=[{damage_by_task}]" if damage_by_task
                              else f"D={evaluation['score']:.3f}")
            case_one_display = (
                f" case1_return={evaluation['case1_team_return']:.3f}"
                f" case1_success={evaluation['case1_success']:.3f}"
                if 'case1_team_return' in evaluation else '')
            progress_bar.write(
                f"Eval @ {agent_transitions:,} [{policy_name}]: "
                f"{damage_display} "
                f"discounted_return={evaluation['discounted_team_return']:.3f}"
                f"{case_one_display}")

    if next_eval is not None:
        run_evaluation(0)

    while completed < total_agent_transitions:
        steps = min(rollout_steps, max(1, int(np.ceil(
            (total_agent_transitions - completed) / (config.n_agents * n_envs)))))
        observations, states, targets, logps, action_masks, values = ([] for _ in range(6))
        next_values, rewards, dones, active, decisions = ([] for _ in range(5))
        allocation_records = []
        lookahead_advantages = []
        shaping_total, switch_total = 0.0, 0
        collected = 0
        for _ in range(steps):
            event_mask = np.array([_actor_event(world) for world in worlds])
            state = np.stack([_model_critic_state(model, world, schedule)
                              for world, schedule in zip(worlds, schedules)])
            with torch.no_grad():
                action, traces = allocated_actions(model, worlds, schedules)
                if isinstance(model, StrikeCOMAActorCritic):
                    joint = np.where(np.stack([w.agent_active for w in worlds]),
                                     action['target'], 0)
                    value = model.target_joint_q(torch.as_tensor(state), torch.as_tensor(joint))
                else:
                    value = model.values(torch.as_tensor(state, dtype=torch.float32))
            # Allocation/scheduling is complete before filtering the loss.
            # GAE and critic retain every physical transition and its reward.
            if actor_samples == 'events':
                for trace in traces:
                    trace['decision'] &= event_mask[:, None]
            allocation_records.append(traces)
            if lookahead_credit:
                lookahead_advantages.append(np.stack([
                    counterfactual_credit(world, action['target'][index], traces[0]['probs'][index])
                    for index, world in enumerate(worlds)]))
            active_before = np.stack([world.agent_active.copy() for world in worlds])
            step_rewards, step_dones, completed_episodes = [], [], []
            for env_index, world in enumerate(worlds):
                potential = progress_potential(world) if guided_training else 0.0
                switches = (avoidable_switches(world, action['target'][env_index])
                            if guided_training else 0)
                _, reward, terminated, truncated, metrics = world.step(
                    strike_action(action['target'][env_index]))
                shaping = (config.discount_gamma * progress_potential(world) - potential
                           if guided_training else 0.0)
                step_rewards.append(float(np.sum(reward)) + shaping - .02 * switches)
                shaping_total += shaping
                switch_total += switches
                step_dones.append(terminated or truncated)
                if terminated or truncated:
                    completed_episodes.append((world, metrics))
            with torch.no_grad():
                next_state = np.stack([_model_critic_state(model, world, schedule)
                                       for world, schedule in zip(worlds, schedules)])
                next_value = (torch.zeros(n_envs) if isinstance(model, StrikeCOMAActorCritic)
                              else model.values(torch.as_tensor(next_state, dtype=torch.float32)))
            states.append(state)
            values.append(value.numpy())
            next_values.append(next_value.numpy())
            rewards.append(step_rewards)
            dones.append(step_dones)
            active.append(active_before)
            collected += int(active_before.sum())
            for world, metrics in completed_episodes:
                episode_rows.append(dict(agent_transitions=completed + collected, **metrics))
                _reset_training_world(world, rng, training_b12_probability)

        values_array = np.asarray(values)
        next_values_array = np.asarray(next_values)
        if isinstance(model, StrikeCOMAActorCritic):
            # Interior transitions use the actual next rollout action. Only the
            # final nonterminal boundary needs a fresh policy sample; copying
            # schedules prevents that sample from changing execution history.
            with torch.no_grad():
                boundary_state = np.stack([_model_critic_state(model, w, schedule)
                                           for w, schedule in zip(worlds, schedules)])
                boundary_action, _ = allocated_actions(model, worlds, deepcopy(schedules))
                boundary_joint = np.where(np.stack([w.agent_active for w in worlds]),
                                          boundary_action['target'], 0)
                boundary_q = model.target_joint_q(torch.as_tensor(boundary_state),
                                                   torch.as_tensor(boundary_joint)).numpy()
            raw_values = values_array
            raw_next_values = np.concatenate((values_array[1:], boundary_q[None]), axis=0)
            team_returns = _td_lambda_returns(
                np.asarray(rewards, dtype=np.float32), raw_next_values,
                np.asarray(dones, dtype=np.float32), config.discount_gamma, config.gae_lambda)
            team_advantage = np.zeros_like(team_returns)  # replaced after Q fitting
            old_probabilities = np.asarray([
                traces[0]['probs'] for traces in allocation_records])
        else:
            with torch.no_grad():
                raw_values = model.value_normalizer.denormalize(torch.as_tensor(values_array)).numpy()
                raw_next_values = model.value_normalizer.denormalize(torch.as_tensor(next_values_array)).numpy()
            team_advantage, team_returns = _advantages(
                np.asarray(rewards, dtype=np.float32), raw_values, raw_next_values,
                np.asarray(dones, dtype=np.float32),
                gamma=config.discount_gamma, gae_lambda=config.gae_lambda)
        actor_advantages, actor_active = [], []
        for step, traces in enumerate(allocation_records):
            for trace in traces:
                observations.append(trace['obs'])
                targets.append(trace['target'])
                logps.append(trace['logp'])
                action_masks.append(trace['action_mask'])
                decisions.append(trace['decision'])
                actor_active.append(active[step])
                actor_advantages.append(np.repeat(team_advantage[step, :, None], config.n_agents, axis=-1))
        tensors = _batch(observations, states, targets, logps, action_masks,
                         actor_advantages, team_returns, actor_active, decisions, values_array,
                         critic_active=np.asarray(active).any(axis=-1))
        if lookahead_credit:
            tensors['lookahead_advantage'] = torch.as_tensor(
                np.asarray(lookahead_advantages).reshape(-1), dtype=torch.float32)
        if isinstance(model, StrikeCOMAActorCritic):
            probabilities = torch.as_tensor(old_probabilities.reshape(
                -1, config.n_agents, config.n_targets), dtype=torch.float32)
            diagnostics = _coma_update(model, optimizers, tensors, probabilities, epochs, minibatch_size)
        else:
            diagnostics = _ppo_update(model, optimizers, tensors, epochs, minibatch_size)
        count = int(np.asarray(active).sum())
        completed += count
        variance = float(np.var(team_returns))
        diagnostics.update(agent_transitions=completed,
                           explained_variance=1 - float(np.var(team_returns - raw_values)) / variance
                           if variance > 1e-12 else float('nan'),
                           actor_samples=int(tensors['decision'].sum()),
                           progress_shaping_sum=shaping_total,
                           penalized_switches=switch_total,
                           critic_samples=int(tensors['critic_active'].sum()))
        if diagnostics_path is not None:
            path = Path(diagnostics_path)
            has_header = path.exists() and path.stat().st_size > 0
            with path.open('a', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(diagnostics))
                if not has_header:
                    writer.writeheader()
                writer.writerow(diagnostics)
        progress_bar.update(min(count, total_agent_transitions - progress_bar.n))
        while next_checkpoint is not None and completed >= next_checkpoint:
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{next_checkpoint}.pt"
            save_checkpoint(checkpoint_path, model, {
                "agent_transitions": completed, "checkpoint_transition": next_checkpoint,
                "seed": seed})
            progress_bar.write(f"Saved periodic checkpoint: {checkpoint_path}")
            next_checkpoint += checkpoint_interval
        if next_eval is not None and completed >= next_eval:
            run_evaluation(completed)
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
