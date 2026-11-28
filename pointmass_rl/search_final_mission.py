"""Search-then-final_mission with frozen exploration and allocation policies.

Every drone starts under the frozen polygon-exploration policy. A learned
termination actor decides, per drone, when to stop exploring. A terminated
drone is handed to the frozen Target Allocation actor (time-limit filter and
score resolver) for the rest of the episode. Only the termination actor and its
state-value baseline are trained; the reward is the value earned from target progress.
"""
import argparse
import copy
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .belief_policy import choose_action, stack_observations, subset
from .belief_training import ExplorationRevisitGuard, _write_csv, load_belief_checkpoint
from .env import Config, PROGRESS_REWARD_RATE, final_mission_action, target_remaining_value
from .exploration import PolygonExplorationWorld
from .navigation import point_path
from .final_mission_ppo import (DecisionSchedule, checkpoint_payload, evaluation_resolved_action,
                         model_from_checkpoint, model_tensor)


DRONE_FEATURES = 15
TARGET_FEATURES = 9
PAIR_FEATURES = 6
GLOBAL_FEATURES = 10
RULES = ('late', 'greedy')
LEARNED = ('value', 'actor')  # Stop when the payoff beats continuing; stop when p(stop) > .5.
PROBABILITY_FLOOR = .02  # Both actions stay sampled, so the continuation value keeps its data.


def _pool_candidates(features, valid):
    """Mask padded candidates and return a fixed-size summary for each drone."""
    mask = valid.unsqueeze(-1)
    mean = (features*mask).sum(-2)/mask.sum(-2).clamp(min=1)
    maximum = features.masked_fill(~mask, -torch.inf).amax(-2)
    maximum = torch.where(mask.any(-2), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), -1)


class _AllocationView:
    """What the frozen allocation actor is allowed to see.

    Only terminated drones are active and only discovered live targets exist.
    Remaining time is clipped to the horizon the actor was trained with.
    """
    def __init__(self, world):
        self.world = world
        w = world._world
        self.n = w.n
        self.c = replace(w.c, horizon=world.allocation_horizon)

    def refresh(self):
        world, w = self.world, self.world._world
        remaining = world.c.horizon-w.t
        # t stays above zero so the decision schedule keeps teammates' intents.
        self.t = max(1, self.c.horizon-min(remaining, self.c.horizon))
        known = world.known_targets()
        self.agent_active = w.agent_active & world.final_mission_agent
        self.target_exists = known
        self.target_completed = w.target_completed
        self.target_type = np.where(known, w.target_type, 0)
        self.target_work_units = np.where(known, w.target_work_units, 0)
        self.targets = np.where(known[:, None], np.nan_to_num(w.targets), 0.)
        self.pos, self.vel = w.pos, w.vel*self.agent_active[:, None]
        self.final_mission_participants = w.final_mission_participants
        self.agent_final_mission_progress = w.agent_final_mission_progress
        self.target_assignment = w.target_assignment
        self.formation_one_initial_score, self.score = w.formation_one_initial_score, w.score
        return self

    def locked_targets(self):
        participating = self.agent_active & self.final_mission_participants.any(axis=0)
        return np.where(participating, self.final_mission_participants.argmax(axis=0), -1)


class SearchFinalMissionWorld(PolygonExplorationWorld):
    """Polygon exploration plus final_missions by drones that stopped exploring."""

    def __init__(self, config, allocation_horizon=100, decision_interval=20, forced_margin=5):
        super().__init__(config)
        self.allocation_horizon = allocation_horizon
        self.decision_interval = decision_interval
        self.forced_margin = forced_margin
        self.allocation_config = replace(self._world.c, horizon=allocation_horizon)
        self.final_mission_agent = np.zeros(self.n, dtype=bool)

    def reset(self, seed=None):
        self.final_mission_agent = np.zeros(self.n, dtype=bool)
        self.final_mission_paths = [None]*self.n
        self.next_ask = np.zeros(self.n, dtype=int)
        self.seen_event = np.full(self.n, -1, dtype=int)
        self.event = 0
        self.discoverer = np.full(self.c.n_targets, -1, dtype=int)
        self.found_step = np.full(self.c.n_targets, -1, dtype=int)
        self.credits = []
        self.stopped = np.zeros(self.n, dtype=bool)
        self.schedule = DecisionSchedule(self.allocation_config)
        self.view = None
        self._eta_step = -1
        observation = super().reset(seed)
        self.view = _AllocationView(self)
        return observation

    # ----- public state shared by both frozen policies and the new actor -----
    def known_targets(self):
        w = self._world
        return w.target_exists & w.discovered & ~w.target_completed & (w.target_work_units > 0)

    def final_mission_steps(self):
        """Steps for each drone to finish one work_units of each target on a Dubins path."""
        w, c = self._world, self.c
        if self._eta_step == w.t:
            return self._eta
        remaining = c.horizon-w.t
        per_step = c.final_mission_speed*c.dt
        eta = np.full((self.n, c.n_targets), np.inf)
        locks = w.locked_targets()
        # A point path never exceeds the straight line by more than this detour.
        detour = (2*math.pi+2)*c.belief_turn_radius
        for j in np.flatnonzero(self.known_targets()):
            for i in np.flatnonzero(w.agent_active):
                if locks[i] == j:
                    eta[i, j] = c.service_steps_per_unit-w.agent_final_mission_progress[i]
                    continue
                line = max(np.linalg.norm(w.targets[j]-w.pos[i])-c.final_mission_range, 0.)
                steps = lambda length: max(1, math.ceil(length/per_step-1e-9))+c.service_steps_per_unit-1
                if steps(line) > remaining+self.forced_margin+self.decision_interval:
                    eta[i, j] = steps(line)  # Already too far; exact length cannot help.
                elif steps(line+detour) <= remaining-self.allocation_horizon:
                    eta[i, j] = steps(line+detour)  # Far from the deadline; a bound suffices.
                else:
                    length = point_path(w.pos[i], float(w.heading[i]), w.targets[j],
                                        c.belief_turn_radius).length
                    eta[i, j] = steps(max(length-c.final_mission_range, 0.))
        self._eta, self._eta_step = eta, w.t
        return eta

    def residual_capacity(self):
        """Open final_mission slots per known target after current final_mission_agent choices."""
        w = self._world
        capacity = np.where(w.target_type == 1, 2, 1)
        capacity = np.minimum(capacity, w.target_work_units)*self.known_targets()
        chosen = self.schedule.targets
        for i in np.flatnonzero(w.agent_active & self.final_mission_agent):
            if chosen[i] >= 0:
                capacity[chosen[i]] -= 1
        return np.maximum(capacity, 0)

    def termination_state(self):
        """Which explorers may stop now, which must, and which are asked."""
        w = self._world
        remaining = self.c.horizon-w.t
        eta = self.final_mission_steps()
        open_slots = self.residual_capacity() > 0
        slack = np.where(open_slots[None], remaining-eta, -np.inf)
        best = slack.max(axis=1) if slack.shape[1] else np.full(self.n, -np.inf)
        explorer = w.agent_active & ~self.final_mission_agent
        eligible = explorer & (best >= 0)
        forced = eligible & (best <= self.forced_margin)
        ask = eligible & ~forced & ((self.seen_event != self.event) | (w.t >= self.next_ask))
        return eligible, forced, ask

    def admit(self, terminate):
        """Let explorers stop only while an open final_mission slot remains for them.

        Forced stops go first, then the shortest completion time. Each admitted
        drone reserves a slot at its quickest open target; the rest keep exploring.
        """
        eligible, forced, _ = self.termination_state()
        wanted = (np.asarray(terminate, dtype=bool) & eligible) | forced
        eta = self.final_mission_steps()
        slots = self.residual_capacity()
        remaining = self.c.horizon-self._world.t
        admitted = np.zeros(self.n, dtype=bool)
        for i in sorted(np.flatnonzero(wanted), key=lambda i: (not forced[i], eta[i].min(), int(i))):
            options = np.flatnonzero((slots > 0) & (eta[i] <= remaining))
            if options.size:
                slots[options[np.argmin(eta[i, options])]] -= 1
                admitted[i] = True
        return admitted

    def stop_payoff(self, agent, model):
        """Score this explorer would final_mission if it were handed over now.

        Asks the frozen allocation actor on a copy of its schedule; nothing in
        the episode changes. Zero when the assigned target has no open slot or
        cannot be finished in time.
        """
        payoff, _ = self._evaluate_handoff(agent, model, include_features=False)
        return payoff

    def _evaluate_handoff(self, agent, model, *, include_features):
        w = self._world
        open_slots = self.residual_capacity()
        final_mission_agent, schedule = self.final_mission_agent, self.schedule
        self.final_mission_agent, self.schedule = final_mission_agent.copy(), copy.deepcopy(schedule)
        self.final_mission_agent[agent] = True
        try:
            view = self.view.refresh()
            action, details = evaluation_resolved_action(model, view, self.schedule, deterministic=True)
            j = int(np.asarray(action['target']).reshape(-1)[agent])
            candidates = (view.target_exists & (open_slots > 0)
                          & (self.final_mission_steps()[agent] <= self.c.horizon-w.t))
            reachable = bool(candidates[j])
            features = None
            if include_features:
                obs = details['allocation_traces'][0]['obs'][0, agent:agent+1]
                tokens = model.actor_features(model_tensor(model, obs, dtype=torch.float32))
                selected = (np.arange(len(candidates)) == j) & candidates
                features = (tokens[0].cpu().numpy(), candidates, selected)
        finally:
            self.final_mission_agent, self.schedule = final_mission_agent, schedule
        if not reachable:
            return 0., features
        kind, work_units = w.target_type[j], w.target_work_units[j]
        return target_remaining_value(kind, work_units)-target_remaining_value(kind, work_units-1), features

    def allocation_mask(self, available):
        """Time-limit filter with Dubins completion times and the real deadline."""
        feasible = available & (self.final_mission_steps() <= self.c.horizon-self._world.t)
        empty = ~feasible.any(axis=-1)
        feasible[empty] = available[empty]
        return feasible

    # ----- exploration policy interface: terminated drones disappear -----
    def action_mask(self):
        mask, decision = super().action_mask()
        mask[self.final_mission_agent] = False
        mask[self.final_mission_agent, 0] = True
        decision[self.final_mission_agent] = False
        return mask, decision

    def observation(self):
        obs = super().observation()
        obs['drones'][self.final_mission_agent] = 0
        obs['active'] = obs['active'] & ~self.final_mission_agent
        obs['previous'] = obs['previous']*obs['active'][:, None]
        obs['previous_choice'] = obs['previous_choice']*obs['active'][:, None]
        return obs

    def termination_observation(self, payoff=None, *, include_background=True):
        w, c = self._world, self.c
        remaining = c.horizon-w.t
        scale = float(self.allocation_horizon)
        known = self.known_targets()
        explorer = w.agent_active & ~self.final_mission_agent
        final_mission_agent = w.agent_active & self.final_mission_agent
        eligible, forced, ask = self.termination_state()
        eta = self.final_mission_steps()
        capacity = self.residual_capacity()
        locked = w.final_mission_participants.any(axis=0)
        sweep = np.array([nav['index']/len(nav['waypoints']) if nav else 0. for nav in self.navigation])
        components = self.belief.components
        unseen_value = np.array([component['remaining']*target_remaining_value(component['type'], 2)
                                 for component in components])
        means = np.array([component['mean'] for component in components]).reshape(-1, 2)
        distance = np.linalg.norm(w.pos[:, None]-means[None], axis=-1)
        nearby = (unseen_value[None]*np.exp(-distance/20.)).sum(axis=1)/5.
        value = np.array([target_remaining_value(t, l) for t, l in zip(w.target_type, w.target_work_units)])
        # What stopping now can earn: one work_units of an open target reachable in time.
        gain = np.array([target_remaining_value(t, l)-target_remaining_value(t, l-1)
                         for t, l in zip(w.target_type, w.target_work_units)])
        open_now = (capacity > 0)[None] & (eta <= remaining)
        quickest = np.where(open_now, eta, np.inf).argmin(axis=1)
        richest = np.where(open_now, gain[None], -1.).argmax(axis=1)
        rows = np.arange(self.n)
        any_open = open_now.any(axis=1)
        handoff = np.column_stack((gain[quickest]/5., gain[richest]/5.,
            np.clip(np.where(any_open, eta[rows, richest], 0.)/scale, 0, 5)))*any_open[:, None]
        drones = np.column_stack((w.pos/c.size, np.cos(w.heading), np.sin(w.heading),
            explorer, final_mission_agent, locked & w.agent_active, sweep, eligible, forced, nearby,
            handoff, (np.zeros(self.n) if payoff is None else payoff)/5.)).astype(np.float32)
        drones *= w.agent_active[:, None]
        assigned = np.bincount(self.schedule.targets[final_mission_agent & (self.schedule.targets >= 0)],
                               minlength=c.n_targets)
        targets = np.column_stack((np.nan_to_num(w.targets)/c.size,
            (w.target_type[:, None] == np.arange(1, 4)), w.target_work_units/2., capacity/2.,
            assigned/2., gain/5.)).astype(np.float32)*known[:, None]
        relative = (np.nan_to_num(w.targets)[None]-w.pos[:, None])/c.size
        finite = np.where(np.isfinite(eta), eta, 0.)
        slack = remaining-finite
        pairs = np.concatenate((relative, np.linalg.norm(relative, axis=-1, keepdims=True),
            np.clip(finite/scale, 0, 5)[..., None], np.clip(slack/scale, -1, 5)[..., None],
            (np.isfinite(eta) & (slack >= 0))[..., None]), -1).astype(np.float32)
        pairs *= (known[None] & w.agent_active[:, None])[..., None]
        types = self.belief.remaining_types
        general = np.array([remaining/c.horizon, min(remaining, scale)/scale,
            explorer.sum()/self.n, final_mission_agent.sum()/self.n, known.sum()/c.n_targets,
            (value*known).sum()/(5.*c.n_targets), w.score/(5.*c.n_targets),
            *(types/c.n_targets)], dtype=np.float32)
        observation = dict(general=general, drones=drones, targets=targets, pairs=pairs,
                           active=w.agent_active.copy(), known=known, ask=ask, forced=forced)
        if include_background:
            observation['background'] = self.belief.background_region_masses().astype(np.float32)
        return observation

    # ----- transition -----
    def _allocate(self, model):
        """Frozen allocation actor with score resolver over terminated drones."""
        w = self._world
        raw = np.full(self.n, -1, dtype=int)
        final_mission_agents = w.agent_active & self.final_mission_agent
        if not final_mission_agents.any() or not self.known_targets().any():
            self.schedule.targets[~w.final_mission_participants.any(axis=0)] = -1
            return raw
        view = self.view.refresh()
        action, _ = evaluation_resolved_action(model, view, self.schedule, deterministic=True)
        chosen = np.asarray(action['target']).reshape(-1)
        valid = final_mission_agents & view.target_exists[chosen]
        raw[valid] = chosen[valid]
        self.schedule.targets[final_mission_agents & ~valid] = -1
        return raw

    def step(self, polygon_action, probabilities, terminate, allocation_model):
        """One physics step. terminate marks explorers that stop exploring now."""
        w, c = self._world, self.c
        if w.done:
            raise RuntimeError('Episode ended; call reset()')
        stopping = self.admit(terminate)
        self.stopped = stopping
        for i in np.flatnonzero(stopping):
            self.final_mission_agent[i] = True
            self.navigation[i] = None
            self.chosen[i] = -1
        if stopping.any():
            self.event += 1
        polygon_action = np.where(self.final_mission_agent, 0, np.asarray(polygon_action))
        probabilities = np.asarray(probabilities, dtype=np.float32).copy()
        probabilities[self.final_mission_agent] = np.eye(self.action_count, dtype=np.float32)[0]
        mask, decision = self.action_mask()
        if not mask[np.arange(self.n), polygon_action].all():
            raise ValueError('Expected one available polygon per exploring drone')
        self.previous = probabilities
        self.previous_choice = np.eye(self.action_count, dtype=np.float32)[polygon_action]
        self.chosen[decision] = polygon_action[decision]
        raw = self._allocate(allocation_model)
        before = w.discovered.copy()
        work_units_before, active_before = w.target_work_units.copy(), w.agent_active.copy()
        old_pos = w.pos.copy()
        budget = c.final_mission_speed*c.dt
        for i in np.flatnonzero(w.agent_active):
            if not self.final_mission_agent[i]:
                sweeping = self.navigation[i] is not None
                self._navigate_polygon(i, polygon_action[i])
                if sweeping and self.navigation[i] is None:
                    self.seen_event[i] = -1  # A finished sweep is a natural decision point.
                continue
            j = raw[i]
            if j < 0 or w.final_mission_participants[:, i].any():
                self.final_mission_paths[i] = None if j < 0 else self.final_mission_paths[i]
                continue
            path = self.final_mission_paths[i]
            if path is None or path[0] != j:
                path = (j, point_path(w.pos[i], float(w.heading[i]), w.targets[j], c.belief_turn_radius))
                self.final_mission_paths[i] = path
            w.pos[i], w.heading[i], _ = path[1].move(w.pos[i], float(w.heading[i]), budget)
            w.last_goal[i] = w.targets[j]
        w.vel = (w.pos-old_pos)/c.dt
        w.target_assignment.fill(False)
        selected = np.flatnonzero(w.agent_active & (raw >= 0))
        w.target_assignment[raw[selected], selected] = True
        w.last_progress_value_by_agent.fill(0)
        w.t += 1
        w._resolve_final_missions(raw)
        w.target_assignment[~w.target_exists | w.target_completed] = False
        w.vel[~w.agent_active] = 0
        w.last_agent_terminated = active_before & ~w.agent_active
        reward = float(PROGRESS_REWARD_RATE*w.last_progress_value_by_agent.sum())
        w.rewards_total += reward
        w.discounted_rewards_total += c.discount_gamma**(w.t-1)*reward
        explorers = w.agent_active & ~self.final_mission_agent
        if self.action_mask()[1].any():
            self._sense()
        else:
            self._sense(refresh=False)
        if ((w.discovered & ~before).any() or (w.target_work_units != work_units_before).any()
                or (active_before != w.agent_active).any()):
            self.event += 1
        # Per-drone credit this step as (drone, amount, step its target was found):
        # its own final_mission (found step -1), and final_missions by others on targets it found.
        for j in np.flatnonzero(w.discovered & ~before):
            seen = np.flatnonzero(w.visible[:, j] & active_before)
            if seen.size:
                self.discoverer[j] = seen[np.argmin(np.linalg.norm(w.pos[seen]-w.targets[j], axis=1))]
                self.found_step[j] = w.t
        self.credits = [(int(i), PROGRESS_REWARD_RATE*float(w.last_progress_value_by_agent[i]), -1)
                        for i in np.flatnonzero(w.last_progress_value_by_agent > 0)]
        for j in np.flatnonzero(w.target_work_units != work_units_before):
            finder = self.discoverer[j]
            final_mission_agents = (raw == j) & (w.last_progress_value_by_agent > 0)
            if finder >= 0 and not final_mission_agents[finder]:
                self.credits.append((int(finder), PROGRESS_REWARD_RATE*float(w.last_progress_value_by_agent[final_mission_agents].sum()),
                                     int(self.found_step[j])))
        all_targets_completed = bool(w.target_completed[w.target_exists].all())
        w.done = w.t >= c.horizon or not w.agent_active.any() or all_targets_completed
        if not explorers.any() and not self.known_targets().any():
            w.done = True  # Nobody searches and nothing known is left to final_mission.
        return reward, w.done

    def record_asked(self, asked):
        self.seen_event[asked] = self.event
        self.next_ask[asked] = self._world.t+self.decision_interval

    def metrics(self):
        w = self._world
        total = sum(target_remaining_value(t, 2) for t in w.target_type[w.target_exists])
        found = sum(target_remaining_value(t, 2) for t in w.target_type[w.target_exists & w.discovered])
        return dict(team_return=float(w.rewards_total), score=float(w.score),
                    total_value=float(total), discovered_value=float(found),
                    score_fraction=float(w.score/max(total, 1e-9)),
                    mission_success=bool(w.score >= w.formation_one_initial_score),
                    success_threshold=float(w.formation_one_initial_score),
                    terminated_drones=int(self.final_mission_agent.sum()),
                    retired_drones=int(w.initial_agent_count-w.agent_active.sum()),
                    agent_count=int(w.initial_agent_count), target_count=int(w.target_exists.sum()),
                    steps=int(w.t))


class _Encoder(nn.Module):
    def __init__(self, hidden, background_cells=0):
        super().__init__()
        self.pair = nn.Sequential(nn.Linear(TARGET_FEATURES+PAIR_FEATURES, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.background = (nn.Sequential(nn.Linear(background_cells, hidden), nn.SiLU())
                           if background_cells else None)
        extra = hidden if self.background is not None else 0
        self.drone = nn.Sequential(nn.Linear(DRONE_FEATURES+GLOBAL_FEATURES+2*hidden+extra, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU())

    def forward(self, obs):
        drones, targets, pairs = obs['drones'], obs['targets'], obs['pairs']
        n, k = drones.shape[1], targets.shape[1]
        mask = (obs['known'][:, None, :] & obs['active'][:, :, None])[..., None]
        tokens = self.pair(torch.cat((targets[:, None].expand(-1, n, -1, -1), pairs), -1))
        count = mask.sum(2).clamp(min=1)
        mean = (tokens*mask).sum(2)/count
        best = tokens.masked_fill(~mask, -1e9).max(2).values
        best = torch.where(mask.any(2), best, torch.zeros_like(best))
        general = obs['general'][:, None].expand(-1, n, -1)
        features = [drones, general, mean, best]
        if self.background is not None:
            mass = obs['background']
            # Scale by cell count so a uniform unit-mass prior starts at one per cell.
            background = self.background(mass * mass.shape[-1])
            features.append(background[:, None].expand(-1, n, -1))
        h = self.drone(torch.cat(features, -1))
        return h, _pool_team(h, obs['active'])


def _pool_team(features, active):
    mask = active.unsqueeze(-1)
    return (features*mask).sum(1)/mask.sum(1).clamp(min=1)


def _scaled_policy_token(tokens):
    """Preserve the direction and carry the original magnitude in a bounded channel."""
    rms = tokens.square().mean(dim=-1, keepdim=True).sqrt()
    magnitude = rms.log1p()
    return torch.cat((tokens/rms.clamp_min(1e-6), magnitude/(1+magnitude)), -1)


class _FrozenPolicyFusion(nn.Module):
    def __init__(self, hidden, feature_sizes):
        super().__init__()
        exploration_size, allocation_size = feature_sizes
        self.exploration = nn.Sequential(nn.Linear(exploration_size+2, hidden),
                                         nn.LayerNorm(hidden), nn.SiLU())
        self.allocation = nn.Sequential(
            nn.Linear(allocation_size+TARGET_FEATURES+PAIR_FEATURES+2, hidden),
            nn.LayerNorm(hidden), nn.SiLU())
        width = 4*hidden+DRONE_FEATURES+GLOBAL_FEATURES+1
        self.fuse = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())

    def forward(self, obs):
        drones = obs['drones']
        general = obs['general'][:, None].expand(-1, drones.shape[1], -1)
        exploration = self.exploration(torch.cat((_scaled_policy_token(obs['exploration_tokens']),
                                                 obs['exploration_available'].unsqueeze(-1)), -1))
        exploration = _pool_candidates(exploration, obs['exploration_valid'])
        targets = obs['targets'][:, None].expand(-1, drones.shape[1], -1, -1)
        allocation = self.allocation(torch.cat((_scaled_policy_token(obs['allocation_tokens']), targets, obs['pairs'],
                                               obs['allocation_selected'].unsqueeze(-1)), -1))
        allocation = _pool_candidates(allocation, obs['allocation_valid'])
        h = self.fuse(torch.cat((exploration, allocation, drones, general,
                                 obs['ask'].unsqueeze(-1)), -1))
        return h, _pool_team(h, obs['active'])


class TerminationActorCritic(nn.Module):
    """Per-drone stop probability and per-drone continuation value.

    Optimal stopping: the payoff of stopping is computed from the frozen
    allocation actor, so only the value of continuing to explore is learned.
    Actor and critic encoders are separate.
    """
    architecture = 'search_final_mission_termination_continuation'

    def __init__(self, hidden=64, background_cells=0, frozen_feature_sizes=None):
        super().__init__()
        self.hidden = hidden
        self.background_cells = background_cells
        self.frozen_feature_sizes = frozen_feature_sizes
        if frozen_feature_sizes is not None:
            self.architecture = 'search_final_mission_termination_scaled_candidate_fusion'
            self.actor = _FrozenPolicyFusion(hidden, frozen_feature_sizes)
            self.critic = _FrozenPolicyFusion(hidden, frozen_feature_sizes)
        else:
            self.actor = _Encoder(hidden, background_cells)
            self.critic = _Encoder(hidden, background_cells)
        if background_cells and frozen_feature_sizes is None:
            self.architecture = 'search_final_mission_termination_continuation_background'
        self.logit = nn.Sequential(nn.Linear(2*hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.value = nn.Sequential(nn.Linear(2*hidden+GLOBAL_FEATURES, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.logit[-1].weight, gain=.01)
        nn.init.zeros_(self.logit[-1].bias)  # Both actions are sampled equally at the start.

    def stop_probabilities(self, obs):
        h, team = self.actor(obs)
        logits = self.logit(torch.cat((h, team[:, None].expand_as(h)), -1)).squeeze(-1)
        return torch.sigmoid(logits).clamp(PROBABILITY_FLOOR, 1-PROBABILITY_FLOOR)

    def continuation(self, obs):
        """Score each drone still earns, by itself or through its finds, if it keeps exploring."""
        h, team = self.critic(obs)
        general = obs['general'][:, None].expand(-1, h.shape[1], -1)
        return self.value(torch.cat((h, team[:, None].expand_as(h), general), -1)).squeeze(-1)


class FrozenPolicies:
    """Both pretrained policies, loaded once and never updated."""
    def __init__(self, exploration_checkpoint, allocation_checkpoint, config, device='cpu'):
        self.exploration, _, _ = load_belief_checkpoint(exploration_checkpoint, config)
        self.exploration.to(device).eval().requires_grad_(False)
        allocation_config = replace(config, mode='known', belief_exploration_only=False)
        self.allocation = model_from_checkpoint(allocation_checkpoint, allocation_config, device=device)
        self.allocation.eval().requires_grad_(False)
        if (self.allocation.n_agents, self.allocation.n_targets) != (config.n_agents, config.n_targets):
            raise ValueError('Allocation checkpoint and config differ in drone/target capacity')
        # The environment supplies the time-limit filter with Dubins completion times.
        self.allocation.training_settings = dict(self.allocation.training_settings, deadline_mask=False)
        base = type(self.allocation).available_actions
        self.allocation.available_actions = lambda view: view.world.allocation_mask(base(self.allocation, view))

    @torch.no_grad()
    def explore(self, world, guard):
        mask, decision = world.action_mask()
        if not decision.any():
            action = mask.argmax(axis=1)
            return action, np.eye(world.action_count, dtype=np.float32)[action]
        observation = world.observation()
        reachable = world.entry_deadline_mask()
        action, probabilities = choose_action(self.exploration, observation, deterministic=True,
                                              resolver='score_time_limit', candidate_mask=reachable)
        guard.apply(world, observation['decision'], reachable, action, probabilities)
        return action, probabilities

    @torch.no_grad()
    def termination_inputs(self, world, ask, model):
        payoff = np.zeros(world.n)
        if model.frozen_feature_sizes is None:
            for agent in np.flatnonzero(ask):
                payoff[agent] = world.stop_payoff(agent, self.allocation)
            return payoff, world.termination_observation(payoff, include_background=bool(model.background_cells))

        # Termination can ask between polygon choices, when belief records are deferred.
        world.belief.refresh(world._world.t)
        observation = world.observation()
        belief = stack_observations([observation], device=next(self.exploration.parameters()).device)
        exploration = self.exploration.actor_features(belief)[0].cpu().numpy()

        shape = (world.n, world.c.n_targets)
        allocation = np.zeros((*shape, self.allocation.hidden), dtype=np.float32)
        allocation_valid = np.zeros(shape, dtype=bool)
        allocation_selected = np.zeros(shape, dtype=bool)
        for agent in np.flatnonzero(ask):
            payoff[agent], features = world._evaluate_handoff(agent, self.allocation, include_features=True)
            allocation[agent], allocation_valid[agent], allocation_selected[agent] = features
        state = world.termination_observation(payoff, include_background=False)
        state.update(exploration_tokens=exploration,
                     exploration_valid=observation['active'][:, None] & observation['valid'][None],
                     exploration_available=observation['mask'] & world.entry_deadline_mask(),
                     allocation_tokens=allocation, allocation_valid=allocation_valid,
                     allocation_selected=allocation_selected)
        return payoff, state


_worker = {}


def _initialize_worker(config, checkpoints, world_arguments, hidden, frozen_feature_sizes):
    torch.set_num_threads(1)
    _worker.update(frozen=FrozenPolicies(*checkpoints, config, 'cpu'),
                   world=SearchFinalMissionWorld(config, *world_arguments),
                   model=TerminationActorCritic(hidden, frozen_feature_sizes=frozen_feature_sizes))


def _worker_episode(task):
    weights, seed, rule = task
    if weights is not None:
        _worker['model'].load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
    return run_episodes(_worker['model'], _worker['frozen'], [_worker['world']], [seed],
                        rule=rule)


def collect(pool, model, seeds, rule=None):
    """One episode per seed across worker processes, in seed order."""
    weights = (None if rule in RULES else
               {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()})
    samples, rows = [], []
    for episode_samples, episode_rows in pool.map(
            _worker_episode, [(weights, int(seed), rule) for seed in seeds]):
        samples.extend(episode_samples)
        rows.extend(episode_rows)
    return samples, rows


@torch.no_grad()
def run_episodes(model, frozen, worlds, seeds, rule=None):
    """Complete one episode per world.

    rule None samples the actor (training). 'late' and 'greedy' are fixed
    baselines. 'value' and 'actor' are the deterministic learned rules.
    """
    device = next(model.parameters()).device if model is not None else 'cpu'
    samples, rows = [], []
    for world, seed in zip(worlds, seeds):
        world.reset(int(seed))
        guard = ExplorationRevisitGuard(world)
        pending = [None]*world.n
        trajectories = [[] for _ in range(world.n)]
        credits = [[] for _ in range(world.n)]
        done = False
        while not done:
            terminate = np.zeros(world.n, dtype=bool)
            asked_now = []
            _, _, ask = world.termination_state()
            if ask.any():
                if rule in RULES:
                    terminate = ask if rule == 'greedy' else terminate
                else:
                    payoff, state = frozen.termination_inputs(world, ask, model)
                    batch = stack_observations([state], device=device)
                    distribution = torch.distributions.Bernoulli(probs=model.stop_probabilities(batch)[0])
                    if rule == 'value':
                        choice = (torch.as_tensor(payoff, device=device) > model.continuation(batch)[0]).float()
                    elif rule == 'actor':
                        choice = (distribution.probs > .5).float()
                    else:
                        choice = distribution.sample()
                    logp = distribution.log_prob(choice).cpu().numpy()
                    stop_probability = distribution.probs.cpu().numpy()
                    terminate = choice.cpu().numpy().astype(bool) & ask
                    for agent in np.flatnonzero(ask):
                        if pending[agent] is not None:
                            trajectories[agent].append(pending[agent])
                        pending[agent] = dict(obs=state, agent=int(agent), action=float(terminate[agent]),
                                              old_logp=float(logp[agent]), payoff=float(payoff[agent]),
                                              old_probability=float(stop_probability[agent]),
                                              step=int(world._world.t))
                        asked_now.append(agent)
                world.record_asked(ask)
            action, probabilities = frozen.explore(world, guard)
            reward, done = world.step(action, probabilities, terminate, frozen.allocation)
            for agent in asked_now:
                pending[agent]['stopped'] = bool(world.stopped[agent])
            for agent, amount, found in world.credits:
                credits[agent].append((world._world.t, amount, found))
        for agent in range(world.n):
            if pending[agent] is not None:
                trajectories[agent].append(pending[agent])
            for sample in trajectories[agent]:
                # Own final_mission after this decision, plus final_missions by others on targets
                # found after it. Work on earlier discoveries continues regardless of the decision.
                sample['return'] = sum(amount for step, amount, found in credits[agent]
                                       if step > sample['step'] and (found < 0 or found > sample['step']))
            samples.extend(trajectories[agent])
        rows.append(dict(seed=int(seed), **world.metrics()))
    return samples, rows


def update(model, optimizer, samples, epochs, critic_epochs, width, entropy_coef, clip):
    """Fit the continuation value, then move the actor along payoff minus continuation."""
    device = next(model.parameters()).device
    batch = stack_observations([s['obs'] for s in samples], device=device)
    tensor = lambda key: torch.tensor([s[key] for s in samples], dtype=torch.float32, device=device)
    agents = torch.tensor([s['agent'] for s in samples], device=device)
    actions, old_logps, old_probabilities = tensor('action'), tensor('old_logp'), tensor('old_probability')
    returns, payoffs, stopped = tensor('return'), tensor('payoff'), tensor('stopped').bool()
    own = lambda values, ix: values[torch.arange(len(ix), device=device), agents[ix]]
    # Only drones that kept exploring show what continuing is worth.
    continued = torch.nonzero(~stopped).squeeze(-1)
    value_losses, policy_losses, entropies = [], [], []
    for _ in range(critic_epochs):
        for ix in continued[torch.randperm(len(continued), device=device)].split(width):
            loss = nn.functional.huber_loss(own(model.continuation(subset(batch, ix)), ix), returns[ix], delta=5.)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            value_losses.append(float(loss.detach()))
    with torch.no_grad():
        continuation = torch.cat([own(model.continuation(subset(batch, ix)), ix)
                                  for ix in torch.arange(len(samples), device=device).split(1024)])
        gap = payoffs-continuation
        # Value of the sampled action minus the policy's expected value, in units of one final_mission.
        advantages = (actions-old_probabilities)*gap/2.5
    for _ in range(epochs):
        for ix in torch.randperm(len(samples), device=device).split(width):
            distribution = torch.distributions.Bernoulli(probs=own(model.stop_probabilities(subset(batch, ix)), ix))
            ratio = (distribution.log_prob(actions[ix])-old_logps[ix]).exp()
            objective = torch.minimum(ratio*advantages[ix], ratio.clamp(1-clip, 1+clip)*advantages[ix])
            entropy = distribution.entropy().mean()
            loss = -objective.mean()-entropy_coef*entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            policy_losses.append(float(loss.detach()))
            entropies.append(float(entropy.detach()))
    mean = lambda values: float(values.mean()) if len(values) else float('nan')
    return dict(value_loss=float(np.mean(value_losses)) if value_losses else float('nan'),
                actor_loss=float(np.mean(policy_losses)), entropy=float(np.mean(entropies)),
                decision_samples=len(samples), stop_rate=float(actions.mean()),
                stop_payoff=mean(payoffs), continuation=mean(continuation),
                stop_preferred=float((gap > 0).float().mean()),
                continued_return=mean(returns[~stopped]),
                # How well the computed payoff matches what stopped drones then earned.
                stopped_return=mean(returns[stopped]), stopped_payoff=mean(payoffs[stopped]))


def summarize(rows):
    keys = ('team_return', 'score_fraction', 'mission_success', 'discovered_value',
            'terminated_drones', 'retired_drones', 'steps')
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def save(path, model, optimizer, config, metadata):
    torch.save(dict(architecture=model.architecture, hidden=model.hidden, state_dict=model.state_dict(),
                    background_cells=model.background_cells, frozen_feature_sizes=model.frozen_feature_sizes,
                    optimizer_state_dict=optimizer.state_dict(), config=asdict(config),
                    metadata=dict(metadata)), path)


def train(args):
    if args.episodes <= 0 or args.episodes_per_update <= 0 or args.checkpoint_interval < 0:
        raise ValueError('Episode counts must be positive and checkpoint interval must be nonnegative')
    config = Config.load(args.config)
    if config.belief_representation != 'gaussian_uniform':
        raise ValueError('Termination training requires gaussian_uniform')
    if args.horizon:
        config = replace(config, horizon=args.horizon)
    out = Path(args.out)
    if (out/'config.json').exists():
        raise SystemExit(f'Existing experiment: {out}. Choose a new output directory.')
    out.mkdir(parents=True, exist_ok=True)
    config.save(out/'config.json')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    checkpoints = (args.exploration_checkpoint, args.allocation_checkpoint)
    frozen_feature_sizes = tuple(checkpoint_payload(path).get('hidden', 64) for path in checkpoints)
    model = TerminationActorCritic(args.hidden, frozen_feature_sizes=frozen_feature_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    eval_seeds = [args.eval_seed+i for i in range(args.eval_episodes)]
    # Episodes run on CPU workers, each with its own frozen policies and world.
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
        initializer=_initialize_worker,
        initargs=(config, checkpoints, (args.allocation_horizon, args.decision_interval, args.forced_margin),
                  args.hidden, frozen_feature_sizes))
    metadata = dict(vars(args), device=str(device), episodes=0, decision_samples=0,
        trained='termination actor and per-drone continuation value only', frozen='exploration policy and allocation actor',
        termination_features='RMS-scaled frozen candidate features with compressed magnitudes, exploration availability and handoff feasibility',
        reward='value earned from target progress',
        stop_payoff='computed: one final_mission on the target the frozen allocation actor would assign now',
        continuation_target='undiscounted own credit to episode end: own final_mission plus final_missions by others on targets it finds later',
        actor_advantage='(action - stop probability) * (stop payoff - continuation value)',
        allocation='score resolver with Dubins time-limit filter; remaining time clipped to allocation horizon')
    (out/'training_metadata.json').write_text(json.dumps(metadata, indent=2, default=str)+'\n')
    evaluation_rows, update_rows, episode_rows = [], [], []
    for rule in RULES:
        _, rows = collect(pool, model, eval_seeds, rule=rule)
        evaluation_rows.append(dict(episodes=0, policy=rule, **summarize(rows)))
        print(f'Eval 0 [{rule}]: {evaluation_rows[-1]}', flush=True)
    _write_csv(out/'training_evaluations.csv', evaluation_rows)
    best = -math.inf

    def evaluate():
        nonlocal best
        for rule in LEARNED:
            _, rows = collect(pool, model, eval_seeds, rule=rule)
            row = dict(episodes=metadata['episodes'], policy=rule, **summarize(rows))
            evaluation_rows.append(row)
            print(f"Eval {metadata['episodes']} [{rule}]: {row}", flush=True)
            if row['team_return'] > best:
                best = row['team_return']
                metadata['best_rule'] = rule
                save(out/'best.pt', model, optimizer, config, metadata)
        _write_csv(out/'training_evaluations.csv', evaluation_rows)

    next_eval = args.eval_interval
    next_checkpoint = args.checkpoint_interval
    while metadata['episodes'] < args.episodes:
        completed = metadata['episodes']
        batch_size = min(args.episodes_per_update, args.episodes - completed)
        if next_checkpoint:
            # Stop collection at the checkpoint boundary so the snapshot is exact.
            batch_size = min(batch_size, next_checkpoint - completed)
        samples, rows = collect(pool, model, rng.integers(2**31, size=batch_size))
        metadata['episodes'] += len(rows)
        metadata['decision_samples'] += len(samples)
        episode_rows.extend(dict(episodes=metadata['episodes'], **row) for row in rows)
        _write_csv(out/'training_episodes.csv', episode_rows)
        if samples:
            diagnostics = update(model, optimizer, samples, args.epochs, args.critic_epochs,
                                 args.minibatch_size, args.entropy_coef, args.policy_clip)
            update_rows.append(dict(episodes=metadata['episodes'],
                                    train_return=float(np.mean([row['team_return'] for row in rows])),
                                    **diagnostics))
            print(f"Update {metadata['episodes']}/{args.episodes}: {update_rows[-1]}", flush=True)
            _write_csv(out/'training_updates.csv', update_rows)
        if next_checkpoint and metadata['episodes'] == next_checkpoint:
            path = out/f'checkpoint_{next_checkpoint}.pt'
            save(path, model, optimizer, config, metadata)
            print(f'Saved checkpoint {path}', flush=True)
            next_checkpoint += args.checkpoint_interval
        save(out/'latest.pt', model, optimizer, config, metadata)
        if args.eval_interval and metadata['episodes'] >= next_eval:
            evaluate()
            while next_eval <= metadata['episodes']:
                next_eval += args.eval_interval
    if args.eval_interval and (not evaluation_rows or evaluation_rows[-1]['episodes'] != metadata['episodes']):
        evaluate()
    pool.shutdown()


def main():
    p = argparse.ArgumentParser(description='Train only the exploration-termination actor.')
    p.add_argument('--exploration-checkpoint', required=True)
    p.add_argument('--allocation-checkpoint', required=True)
    p.add_argument('--config', default='configs/belief_exploration.json')
    p.add_argument('--out', required=True)
    p.add_argument('--episodes', type=int, default=20000)
    p.add_argument('--episodes-per-update', type=int, default=16)
    p.add_argument('--horizon', type=int, default=0, help='Override the mission horizon; 0 keeps the config')
    p.add_argument('--allocation-horizon', type=int, default=100,
                   help='Horizon the allocation actor was trained with; its time input is clipped to it')
    p.add_argument('--decision-interval', type=int, default=20,
                   help='Steps between termination questions when nothing changed')
    p.add_argument('--forced-margin', type=int, default=5,
                   help='Stop exploring when the last final_mission opportunity has this little slack')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--critic-epochs', type=int, default=4)
    p.add_argument('--minibatch-size', type=int, default=64)
    p.add_argument('--entropy-coef', type=float, default=.01)
    p.add_argument('--policy-clip', type=float, default=.2)
    p.add_argument('--eval-interval', type=int, default=320, help='Episodes between evaluations; 0 disables')
    p.add_argument('--checkpoint-interval', type=int, default=5000,
                   help='Training episodes between saved checkpoints; 0 disables periodic snapshots')
    p.add_argument('--eval-episodes', type=int, default=30)
    p.add_argument('--eval-seed', type=int, default=10000)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--workers', type=int, default=8, help='CPU processes running episodes')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    train(p.parse_args())


if __name__ == '__main__':
    main()
