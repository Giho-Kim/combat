"""Target Allocation: peer-attention actor and relational counterfactual critic."""
import numpy as np
import torch
from copy import deepcopy
from torch import nn

from .strike_ppo import (_ActorCriticBase, _TargetQBase, _initialization_mlp,
                         _normalized_drone_velocities)


class TargetScoringBase(_ActorCriticBase):
    teammate_features = True
    mask_inactive_intent = True
    compact_inputs = True
    critic_strike_participants = True
    actor_teammate_progress = False
    actor_idle_status = True
    actor_previous_actions = False
    drone_velocity = False
    actor_target_distances = False

    def __init__(self, n_targets, n_agents, hidden=64):
        super().__init__(n_targets, n_agents, hidden)
        teammate_width = 4 * (n_agents - 1) if self.teammate_features else 0
        compact_teammate_width = 5 if self.actor_teammate_progress else 4
        scorer_width = (self.target_features + 2 + compact_teammate_width * (n_agents - 1)
                        if self.compact_inputs else self.target_features + 9 + teammate_width)
        if self.actor_previous_actions:
            self.obs_dim += n_agents * n_targets
            scorer_width += n_agents
        if self.drone_velocity:
            self.obs_dim += 2 * n_agents
            scorer_width += 2 * n_agents
        if self.actor_target_distances:
            scorer_width += n_agents
        self.actor_body = _initialization_mlp(scorer_width, hidden)
        self.target_head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.target_head.weight, gain=.01)
        nn.init.zeros_(self.target_head.bias)
        if self.compact_inputs:
            strike_matrix_width = (n_agents * n_targets
                                   if self.critic_strike_participants else 0)
            self.state_dim = 1 + 4 * n_agents + self.target_features * n_targets + strike_matrix_width
            if self.drone_velocity:
                self.state_dim += 2 * n_agents
            if isinstance(self, _TargetQBase):
                self.state_dim += n_agents * n_targets
                if getattr(self, 'critic_previous_actions', False):
                    self.state_dim += n_agents * n_targets
                self.q_body = _initialization_mlp(self.state_dim + n_agents * n_targets, hidden)
                self.target_q_body = deepcopy(self.q_body).requires_grad_(False)
            else:
                self.critic_body = _initialization_mlp(self.state_dim, hidden)

    def actor_logits(self, obs):
        velocities = None
        if self.drone_velocity:
            velocities = obs[..., -2 * self.n_agents:].reshape(
                *obs.shape[:-1], self.n_agents, 2)
            obs = obs[..., :-2 * self.n_agents]
        previous_actions = None
        if self.actor_previous_actions:
            previous_actions = obs[..., -self.n_targets * self.n_agents:].reshape(
                *obs.shape[:-1], self.n_targets, self.n_agents)
            obs = obs[..., :-self.n_targets * self.n_agents]
        start = 3 + 5 * self.n_agents
        targets = obs[..., start:start + self.target_features * self.n_targets].reshape(
            *obs.shape[:-1], self.n_targets, self.target_features)
        drones = obs[..., 3:start].reshape(*obs.shape[:-1], self.n_agents, 5)
        assignments = obs[..., start + self.target_features * self.n_targets:-self.n_agents].reshape(
            *obs.shape[:-1], self.n_targets, self.n_agents)
        identity = obs[..., -self.n_agents:]
        if previous_actions is not None:
            previous_actions = previous_actions * drones[..., 2].unsqueeze(-2)
        if self.mask_inactive_intent:
            assignments = assignments * drones[..., 2].unsqueeze(-2)
        own_assignment = (assignments * identity.unsqueeze(-2)).sum(-1, keepdim=True)
        if self.compact_inputs:
            alive = drones[..., 2:3]
            status = (alive * (1 - drones[..., 3:4])
                      if self.actor_idle_status else alive)
            geometry = torch.cat((drones[..., :2] * alive, status), dim=-1)
            if self.actor_teammate_progress:
                geometry = torch.cat((geometry, drones[..., 4:5] * alive), dim=-1)
            geometry_width = geometry.shape[-1]
            geometry = geometry.unsqueeze(-3).expand(
                *targets.shape[:-1], self.n_agents, geometry_width)
            teammates = torch.cat((geometry, assignments.unsqueeze(-1)), dim=-1)
            own_previous_action = None
            if previous_actions is not None:
                own_previous_action = (previous_actions * identity.unsqueeze(-2)).sum(
                    -1, keepdim=True)
                teammates = torch.cat((teammates, previous_actions.unsqueeze(-1)), dim=-1)
            others = (identity < .5).unsqueeze(-2).expand(
                *targets.shape[:-1], self.n_agents)
            teammates = teammates[others].reshape(
                *targets.shape[:-1], (geometry_width + 1 + int(previous_actions is not None))
                * (self.n_agents - 1))
            time = obs[..., :1].unsqueeze(-2).expand(*targets.shape[:-1], 1)
            pieces = (targets, time, own_assignment)
            if own_previous_action is not None:
                pieces += (own_previous_action,)
            features = torch.cat((*pieces, teammates), -1)
            if velocities is not None:
                velocities = velocities * alive
                own_velocity = (velocities * identity.unsqueeze(-1)).sum(-2)
                peer_velocities = velocities[identity < .5].reshape(
                    *identity.shape[:-1], 2 * (self.n_agents - 1))
                motion = torch.cat((own_velocity, peer_velocities), -1)
                # Actual world-frame velocity: self first, then peers in the
                # same stable order as their relative-position records.
                motion = motion.unsqueeze(-2).expand(*targets.shape[:-1], 2 * self.n_agents)
                features = torch.cat((features, motion), -1)
            if self.actor_target_distances:
                # Observed geometry only: no speed-based arrival prediction or mask.
                distances = (targets[..., :2].unsqueeze(-2)
                             - drones[..., :2].unsqueeze(-3)).norm(dim=-1)
                distances = distances * drones[..., 2].unsqueeze(-2)
                own_distance = (distances * identity.unsqueeze(-2)).sum(-1, keepdim=True)
                peer_distances = distances[others].reshape(
                    *targets.shape[:-1], self.n_agents - 1)
                features = torch.cat((features, own_distance, peer_distances), -1)
            return self.target_head(self.actor_body(features)).squeeze(-1)
        participants = (assignments * drones[..., 3].unsqueeze(-2)).sum(-1, keepdim=True)
        assigned = assignments.sum(-1, keepdim=True)
        own = (drones * identity.unsqueeze(-1)).sum(-2)
        # All inputs are observed state or geometric summaries. No target priority,
        # reward estimate, deadline mask, planner or preferred action is supplied.
        distance = targets[..., :2].norm(dim=-1, keepdim=True)
        active_fraction = drones[..., 2].mean(-1, keepdim=True)
        context = torch.cat((obs[..., :3], active_fraction, own[..., 4:5]), dim=-1)
        context = context.unsqueeze(-2).expand(*targets.shape[:-1], 5)
        features = torch.cat((targets, distance, own_assignment,
                              participants / self.n_agents, assigned / self.n_agents, context), dim=-1)
        if self.teammate_features:
            # For target j, retain each other drone's (relative x, y, alive,
            # previous probability of j). Stable drone-ID order, excluding self.
            alive = drones[..., 2:3]
            geometry = torch.cat((drones[..., :2] * alive, alive), dim=-1)
            geometry = geometry.unsqueeze(-3).expand(
                *targets.shape[:-1], self.n_agents, 3)
            intent = assignments * alive.squeeze(-1).unsqueeze(-2)
            teammates = torch.cat((geometry, intent.unsqueeze(-1)), dim=-1)
            others = (identity < .5).unsqueeze(-2).expand(
                *targets.shape[:-1], self.n_agents)
            teammates = teammates[others].reshape(
                *targets.shape[:-1], 4 * (self.n_agents - 1))
            features = torch.cat((features, teammates), dim=-1)
        return self.target_head(self.actor_body(features)).squeeze(-1)


class TargetHistoryBase(TargetScoringBase, _TargetQBase):
    """Shared target scorer with teammate inputs and the Q-only COMA critic."""
    critic_previous_probabilities = True
    critic_mask_inactive_intent = True
    critic_strike_participants = False
    actor_previous_actions = True
    critic_previous_actions = True
    drone_velocity = True

    def observation(self, world, previous_action_probabilities=None, previous_actions=None):
        base = super().observation(world, previous_action_probabilities)
        if previous_actions is None:
            previous_actions = np.full(self.n_agents, -1, dtype=int)
        previous_actions = np.asarray(previous_actions)
        if (previous_actions.shape != (self.n_agents,)
                or np.any((previous_actions < -1) | (previous_actions >= self.n_targets))):
            raise ValueError('previous_actions must contain one target ID or -1 per agent')
        chosen = np.zeros((self.n_agents, self.n_targets), dtype=np.float32)
        valid = previous_actions >= 0
        chosen[np.flatnonzero(valid), previous_actions[valid]] = 1.0
        flattened = chosen.T.reshape(-1)
        parts = [base, np.broadcast_to(flattened, (self.n_agents, len(flattened)))]
        if self.drone_velocity:
            velocities = _normalized_drone_velocities(world).reshape(-1)
            parts.append(np.broadcast_to(velocities, (self.n_agents, len(velocities))))
        return np.concatenate(parts, axis=-1)

    def target_mask(self, obs):
        if self.actor_previous_actions and obs.shape[-1] == self.obs_dim:
            suffix = self.n_targets * self.n_agents + (2 * self.n_agents if self.drone_velocity else 0)
            obs = obs[..., :-suffix]
        return super().target_mask(obs)




class AssignmentPooling(nn.Module):
    """Sum live drone embeddings into stable target slots."""

    def aggregate_assignments(self, embeddings, alive, joint_action):
        # Action IDs select edges; no action one-hot is concatenated to features.
        if joint_action.shape != alive.shape:
            raise ValueError('joint_action must contain one target per drone')
        if torch.any((joint_action < 0) | (joint_action >= self.n_targets)):
            raise ValueError('joint_action contains an invalid target')
        targets = torch.arange(self.n_targets, device=joint_action.device)
        membership = (joint_action.unsqueeze(-2) == targets[:, None])
        weights = membership.to(embeddings.dtype) * alive.unsqueeze(-2)
        return weights @ embeddings, weights.sum(-1, keepdim=True)


class AssignmentQBody(AssignmentPooling):
    """Legacy v11 context-conditioned target pooling."""

    def __init__(self, state_dim, n_agents, n_targets, hidden):
        super().__init__()
        self.n_agents, self.n_targets = n_agents, n_targets
        self.context = _initialization_mlp(state_dim, hidden)
        self.drone_encoder = _initialization_mlp(6 + 2 * n_targets, hidden)
        self.target_encoder = _initialization_mlp(6 + 2 * hidden + 1, hidden)
        self.team_encoder = _initialization_mlp(2 * hidden, hidden)

    def forward(self, state, joint_action):
        n, t = self.n_agents, self.n_targets
        drone_end = 1 + 4 * n
        target_end = drone_end + 6 * t
        drones = state[..., 1:drone_end].reshape(*state.shape[:-1], n, 4)
        targets = state[..., drone_end:target_end].reshape(*state.shape[:-1], t, 6)
        probabilities = state[..., target_end:target_end + n * t].reshape(
            *state.shape[:-1], t, n).transpose(-1, -2)
        choices = state[..., target_end + n * t:target_end + 2 * n * t].reshape(
            *state.shape[:-1], t, n).transpose(-1, -2)
        velocities = state[..., -2 * n:].reshape(*state.shape[:-1], n, 2)
        embeddings = self.drone_encoder(torch.cat(
            (drones, velocities, probabilities, choices), dim=-1))
        pooled, counts = self.aggregate_assignments(embeddings, drones[..., 2], joint_action)
        context = self.context(state)
        target_features = self.target_encoder(torch.cat((
            targets, pooled, counts,
            context.unsqueeze(-2).expand(*state.shape[:-1], t, context.shape[-1])), -1))
        # Nonexistent padded targets contribute nothing; destroyed targets still
        # carry their type and remain part of the global physical state.
        exists = targets[..., 2:5].sum(-1, keepdim=True)
        summary = (target_features * exists).sum(-2)
        return self.team_encoder(torch.cat((context, summary), -1))


class AssignmentCriticBase(TargetHistoryBase):
    """Shared target-wise assignment critic producing a single joint-action Q."""

    def __init__(self, n_targets, n_agents, hidden=64):
        super().__init__(n_targets, n_agents, hidden)
        self.q_body = AssignmentQBody(self.state_dim, n_agents, n_targets, hidden)
        self.target_q_body = deepcopy(self.q_body).requires_grad_(False)

    def joint_q(self, state, joint_action):
        return self.q_head(self.q_body(state, joint_action)).squeeze(-1)

    def target_joint_q(self, state, joint_action):
        return self.target_q_head(self.target_q_body(state, joint_action)).squeeze(-1)








class RelationalCriticBase(AssignmentCriticBase):
    """Shared initialization and slot-independent relational online/target Q."""

    def __init__(self, n_targets, n_agents, hidden=64):
        super().__init__(n_targets, n_agents, hidden)
        from .graph_critic import GraphQBody
        self.q_body = GraphQBody(hidden)
        self.target_q_body = deepcopy(self.q_body).requires_grad_(False)

    def load_critic_checkpoint(self, path):
        """Transfer online and target Q across slot counts; leave actor untouched."""
        from .strike_ppo import checkpoint_payload
        payload = checkpoint_payload(path)
        if payload.get('architecture') != 'target_allocation' or payload.get('hidden', 64) != self.hidden:
            raise ValueError('Critic transfer requires a Target Allocation checkpoint with matching hidden width')
        for name in ('q_body', 'q_head', 'target_q_body', 'target_q_head'):
            prefix = name + '.'
            weights = {key[len(prefix):]: value for key, value in payload['state_dict'].items()
                       if key.startswith(prefix)}
            getattr(self, name).load_state_dict(weights)
        return payload.get('metadata', {})

    def counterfactual_advantages(self, state, joint_action, action_probabilities,
                                 sampled_returns=None):
        n, t = self.q_body.dimensions(state, joint_action)
        batch = state.shape[0]
        if action_probabilities.shape != (batch, n, t):
            raise ValueError('action_probabilities must match runtime drone/target counts')
        alternatives = joint_action[:, None, None, :].expand(batch, n, t, n).clone()
        agent = torch.arange(n, device=state.device)
        action = torch.arange(t, device=state.device)
        alternatives[:, agent[:, None], action[None, :], agent[:, None]] = action
        expanded = state[:, None, None, :].expand(batch, n, t, state.shape[-1])
        # Bound edge-attention expansion memory for larger joint action spaces.
        states, actions = expanded.reshape(-1, state.shape[-1]), alternatives.reshape(-1, n)
        q = torch.cat([self.joint_q(states[start:start + 64], actions[start:start + 64])
                       for start in range(0, len(states), 64)]).reshape(batch, n, t)
        actual = q.gather(-1, joint_action.unsqueeze(-1)).squeeze(-1)
        outcome = actual if sampled_returns is None else sampled_returns[:, None]
        return outcome - (action_probabilities * q).sum(-1)


class TargetAllocationActorCritic(RelationalCriticBase):
    """Slot-independent peer cross-attention actor and relational graph Q."""
    architecture = 'target_allocation'
    algorithm = 'target_coma'
    display_name = 'Target Allocation'

    def __init__(self, n_targets, n_agents, hidden=64):
        super().__init__(n_targets, n_agents, hidden)
        from .attention_actor import AttentionActorBody
        self.actor_body = AttentionActorBody(hidden)

    def actor_logits(self, obs, *, n_agents=None, n_targets=None):
        n = self.n_agents if n_agents is None else n_agents
        t = self.n_targets if n_targets is None else n_targets
        if n < 1 or t < 1 or obs.shape[-1] != 3 + 8 * n + 6 * t + 2 * n * t:
            raise ValueError('Actor observation width does not match drone/target slots')
        drone_end, target_end = 3 + 5 * n, 3 + 5 * n + 6 * t
        drones = obs[..., 3:drone_end].reshape(*obs.shape[:-1], n, 5)
        targets = obs[..., drone_end:target_end].reshape(*obs.shape[:-1], t, 6)
        probability = obs[..., target_end:target_end + n * t].reshape(*obs.shape[:-1], t, n)
        identity = obs[..., target_end + n * t:target_end + n * t + n]
        start = target_end + n * t + n
        previous = obs[..., start:start + n * t].reshape(*obs.shape[:-1], t, n)
        velocity = obs[..., -2 * n:].reshape(*obs.shape[:-1], n, 2)
        alive = drones[..., 2] > .5
        exists = targets[..., 2:5].sum(-1) > .5
        targets = torch.where(exists.unsqueeze(-1), targets, torch.zeros_like(targets))
        drones = torch.where(alive.unsqueeze(-1), drones, torch.zeros_like(drones))
        velocity = torch.where(alive.unsqueeze(-1), velocity, torch.zeros_like(velocity))
        intent_mask = alive.unsqueeze(-2) & exists.unsqueeze(-1)
        probability = torch.where(intent_mask, probability, torch.zeros_like(probability))
        previous = torch.where(intent_mask, previous, torch.zeros_like(previous))
        own_probability = (probability * identity.unsqueeze(-2)).sum(-1, keepdim=True)
        own_previous = (previous * identity.unsqueeze(-2)).sum(-1, keepdim=True)
        idle = drones[..., 2:3] * (1 - drones[..., 3:4])
        own = (torch.cat((drones[..., 2:3], idle, velocity), -1) * identity.unsqueeze(-1)).sum(-2)
        time = obs[..., :1].unsqueeze(-2).expand(*obs.shape[:-1], t, 1)
        own = own.unsqueeze(-2).expand(*obs.shape[:-1], t, 4)
        query = torch.cat((targets, time, own_probability, own_previous, own), -1)
        peer = torch.cat((drones[..., :2], idle, velocity), -1).unsqueeze(-3).expand(
            *obs.shape[:-1], t, n, 5)
        peer = torch.cat((peer, probability.unsqueeze(-1), previous.unsqueeze(-1)), -1)
        mask = intent_mask & (identity < .5).unsqueeze(-2)
        features = self.actor_body(query, peer, mask)
        return self.target_head(features).squeeze(-1)
