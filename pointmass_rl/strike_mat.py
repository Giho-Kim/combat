"""MAT-style actor following PKU-MARL's encoder/decoder and shifted actions.

Reference: https://github.com/PKU-MARL/Multi-Agent-Transformer
The centralized critic and PPO training loop are retained from strike_ppo.
"""
import math

import torch
from torch import nn
from torch.nn import functional as functional
from torch.distributions import Categorical

from .env import SELF_FEATURES, TARGET_FEATURES, strike_action
from .strike_ppo import StrikeActorCritic, critic_state


def linear(inputs, outputs, gain=.01, bias=True):
    layer = nn.Linear(inputs, outputs, bias=bias)
    nn.init.orthogonal_(layer.weight, gain=gain)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


class Attention(nn.Module):
    def __init__(self, hidden, causal=False):
        super().__init__()
        self.query = linear(hidden, hidden)
        self.key = linear(hidden, hidden)
        self.value = linear(hidden, hidden)
        self.output = linear(hidden, hidden)
        self.causal = causal

    def forward(self, query, source):
        scores = self.query(query) @ self.key(source).transpose(-2, -1)
        scores = scores / math.sqrt(query.shape[-1])
        if self.causal:
            future = torch.ones(scores.shape[-2:], device=scores.device,
                                dtype=torch.bool).triu(1)
            scores = scores.masked_fill(future, -torch.inf)
        return self.output(scores.softmax(-1) @ self.value(source))


class MATActor(nn.Module):
    def __init__(self, obs_dim, n_targets, hidden, n_agents=None):
        super().__init__()
        self.n_agents = n_agents
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
            linear(obs_dim, hidden, math.sqrt(2)), nn.GELU(), nn.LayerNorm(hidden))
        if n_agents is None:
            self.encoder_attention = Attention(hidden)
            self.encoder_norm = nn.LayerNorm(hidden)
            self.encoder_mlp = nn.Sequential(linear(hidden, hidden, math.sqrt(2)),
                                             nn.GELU(), linear(hidden, hidden))
            self.encoder_output_norm = nn.LayerNorm(hidden)
        else:
            self.agent_embedding = nn.Embedding(n_agents, hidden)
            nn.init.normal_(self.agent_embedding.weight, std=.02)
        self.action_encoder = nn.Sequential(
            linear(n_targets + 1, hidden, math.sqrt(2), bias=False),
            nn.GELU(), nn.LayerNorm(hidden))
        self.decoder_attention = Attention(hidden, causal=True)
        self.decoder_norm = nn.LayerNorm(hidden)
        self.cross_attention = Attention(hidden, causal=True)
        self.cross_norm = nn.LayerNorm(hidden)
        self.decoder_mlp = nn.Sequential(linear(hidden, hidden, math.sqrt(2)),
                                         nn.GELU(), linear(hidden, hidden))
        self.decoder_output_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(linear(hidden, hidden, math.sqrt(2)), nn.GELU(),
                                  nn.LayerNorm(hidden), linear(hidden, n_targets))

    def encode(self, obs):
        if self.n_agents is not None:
            representation = self.obs_encoder(obs[..., 0, :]).unsqueeze(-2)
            return representation + self.agent_embedding.weight
        representation = self.obs_encoder(obs)
        representation = self.encoder_norm(
            representation + self.encoder_attention(representation, representation))
        return self.encoder_output_norm(representation + self.encoder_mlp(representation))

    def decode(self, representation, shifted):
        actions = self.action_encoder(shifted)
        actions = self.decoder_norm(actions + self.decoder_attention(actions, actions))
        actions = self.cross_norm(representation + self.cross_attention(representation, actions))
        actions = self.decoder_output_norm(actions + self.decoder_mlp(actions))
        return self.head(actions)


class LegacyStrikeMATActorCritic(StrikeActorCritic):
    architecture = 'strike_mat_v1'
    one_hot_target_type = False
    relative_actor_coordinates = False
    algorithm = 'mat'
    autoregressive = True
    uses_previous_action_probabilities = False

    def __init__(self, n_targets, n_agents, hidden=64):
        super().__init__(n_targets, n_agents, hidden)
        self.actor_body = MATActor(self.obs_dim, n_targets, hidden)
        self.target_head = nn.Identity()

    def _shifted(self, obs, targets=None):
        shifted = obs.new_zeros((*obs.shape[:-1], self.n_targets + 1))
        shifted[..., 0, 0] = 1
        if targets is not None:
            shifted[..., 1:, 1:] = functional.one_hot(targets[..., :-1], self.n_targets)
        return shifted

    def _action_mask(self, obs, action_mask):
        mask = self.target_mask(obs) if action_mask is None else action_mask.clone()
        active = obs[..., 5:3 + 5 * self.n_agents:5]
        identities = obs[..., -self.n_agents:].argmax(-1)
        own_active = active.gather(-1, identities.unsqueeze(-1)).squeeze(-1) > .5
        mask = mask & own_active.unsqueeze(-1)
        mask[..., 0] |= ~own_active
        return mask

    def actor_logits(self, obs, targets=None):
        if obs.shape[-2] != self.n_agents:
            raise ValueError('MAT requires complete ordered agent groups')
        return self.actor_body.decode(self.actor_body.encode(obs), self._shifted(obs, targets))

    def distributions(self, obs, action_mask=None, targets=None):
        mask = self._action_mask(obs, action_mask)
        return Categorical(logits=self.actor_logits(obs, targets).masked_fill(~mask, -torch.inf))

    def act(self, obs, deterministic=False, action_mask=None):
        mask = self._action_mask(obs, action_mask)
        representation = self.actor_body.encode(obs)
        shifted = self._shifted(obs)
        targets, logps = [], []
        for agent in range(self.n_agents):
            logits = self.actor_body.decode(representation, shifted)[..., agent, :]
            distribution = Categorical(logits=logits.masked_fill(~mask[..., agent, :], -torch.inf))
            target = distribution.probs.argmax(-1) if deterministic else distribution.sample()
            targets.append(target)
            logps.append(distribution.log_prob(target))
            if agent + 1 < self.n_agents:
                shifted = shifted.clone()
                shifted[..., agent + 1, 1:] = functional.one_hot(target, self.n_targets)
        return strike_action(torch.stack(targets, -1).cpu().numpy()), {
            'logp': torch.stack(logps, -1), 'action_mask': mask}

    def evaluate_actions(self, obs, target, action_mask):
        distribution = self.distributions(obs, action_mask, target)
        return distribution.log_prob(target), distribution.entropy()


class LocalStrikeMATActorCritic(LegacyStrikeMATActorCritic):
    """Encode each drone's environment observation, without an appended ID."""

    architecture = 'strike_mat_v2'

    def __init__(self, n_targets, n_agents, hidden=64):
        super().__init__(n_targets, n_agents, hidden)
        self.obs_dim = SELF_FEATURES + TARGET_FEATURES * n_targets
        self.actor_body = MATActor(self.obs_dim, n_targets, hidden)

    def observation(self, world):
        return world._obs()

    def _action_mask(self, obs, action_mask):
        if action_mask is None:
            raise ValueError('MAT local observations require an explicit environment action mask')
        own_active = obs.ne(0).any(dim=-1)
        mask = action_mask.clone() & own_active.unsqueeze(-1)
        mask[..., 0] |= ~own_active
        return mask


class StrikeMATActorCritic(LegacyStrikeMATActorCritic):
    """Global-state projection with agent queries and an autoregressive decoder."""

    architecture = 'strike_mat_v4'
    one_hot_target_type = True

    def __init__(self, n_targets, n_agents, hidden=64):
        super().__init__(n_targets, n_agents, hidden)
        self.obs_dim = self.state_dim
        self.actor_body = MATActor(self.obs_dim, n_targets, hidden, n_agents=n_agents)

    def observation(self, world):
        return critic_state(world, one_hot=self.one_hot_target_type)

    def _action_mask(self, obs, action_mask):
        if action_mask is None:
            raise ValueError('Global-state MAT requires an explicit environment action mask')
        own_active = obs[..., 0, 5:3 + 5 * self.n_agents:5] > .5
        mask = action_mask.clone() & own_active.unsqueeze(-1)
        mask[..., 0] |= ~own_active
        return mask


class ScalarTypeStrikeMATActorCritic(StrikeMATActorCritic):
    architecture = 'strike_mat_v3'
    one_hot_target_type = False
