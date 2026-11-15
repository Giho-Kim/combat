"""Full-area polygon exploration with the Target Allocation attention topology.

Candidates are a fixed rectangular partition of the spawn/search support.
The filter remains continuous; rectangles only define actions and quadrature.
"""
import math

import numpy as np
import torch
from torch import nn

from .attention_actor import AttentionActorBody
from .belief_env import BeliefWorld
from .belief_policy import BeliefActorCritic
from .env import split_strike_action, target_remaining_value
from .gaussian_belief import GAUSSIAN_FEATURES
from .graph_critic import GraphQBody
from .navigation import DubinsPath


DRONE_FEATURES = 14


class PolygonExplorationWorld(BeliefWorld):
    def __init__(self, config):
        super().__init__(config)
        self.action_count = config.belief_grid**2

    def reset(self, seed=None):
        observation = super().reset(seed)
        self._world.done = False
        return observation

    @property
    def polygons(self):
        # Stable candidate identities throughout an episode; no coverage gaps.
        return np.array([[lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]]
                         for lo, hi in self.belief.bounds])

    def action_mask(self):
        w = self._world
        mask = np.ones((self.n, self.action_count), dtype=bool)
        decision = w.agent_active.copy()
        for i, nav in enumerate(self.navigation):
            if nav is not None:
                mask[i] = False
                mask[i, nav['component']] = True
                decision[i] = False
        return mask, decision

    def observation(self):
        obs = super().observation()
        obs['drones'] = np.pad(obs['drones'], ((0, 0), (0, DRONE_FEATURES-9)))
        obs['beliefs'] = obs['beliefs'][:self.action_count].copy()
        obs['valid'] = np.ones(self.action_count, dtype=bool)
        # Keep all polygons available, even when almost no mass remains there.
        for k, (lo, hi) in enumerate(self.belief.bounds):
            obs['beliefs'][k, 12:14] = (lo+hi)/(2*self.c.size)
            obs['beliefs'][k, 15:17] = (hi-lo)/self.c.size
        for i, nav in enumerate(self.navigation):
            obs['drones'][i, 3] = float(nav is not None)
            obs['drones'][i, 4] = (nav['index']/len(nav['waypoints']) if nav else 0.)
            if nav:
                destination, heading = nav['waypoints'][nav['index']]
                obs['drones'][i, 9:11] = destination/self.c.size
                obs['drones'][i, 11:13] = [math.cos(heading), math.sin(heading)]
                obs['drones'][i, 13] = np.sign(nav['waypoints'][-1][0][1]-nav['waypoints'][0][0][1])
        return obs

    def _waypoints(self, component, position):
        lo, hi = self.belief.bounds[component]
        rows = max(2, math.ceil((hi[1]-lo[1])/self.c.belief_sweep_spacing))
        ys = lo[1]+(np.arange(rows)+.5)*(hi[1]-lo[1])/rows
        if abs(position[1]-ys[-1]) < abs(position[1]-ys[0]):
            ys = ys[::-1]
        left, right = lo[0]-self.c.sensor_range, hi[0]+self.c.sensor_range
        east = abs(position[0]-left) <= abs(position[0]-right)
        result = []
        for y in ys:
            start, end, angle = (left, right, 0.) if east else (right, left, math.pi)
            result.extend([(np.array([start, y]), angle), (np.array([end, y]), angle)])
            east = not east
        return result

    def _navigate_polygon(self, i, component):
        w, c = self._world, self.c
        nav = self.navigation[i]
        if nav is None:
            nav = dict(component=int(component), sweeping=True, index=0, path=None,
                       waypoints=self._waypoints(component, w.pos[i]))
            self.navigation[i] = nav
        budget = c.strike_speed*c.dt
        while budget > 1e-10 and nav['index'] < len(nav['waypoints']):
            destination, heading = nav['waypoints'][nav['index']]
            w.last_goal[i] = destination
            if nav['path'] is None:
                nav['path'] = DubinsPath(w.pos[i], w.heading[i], destination, heading,
                                         c.belief_turn_radius)
            w.pos[i], w.heading[i], budget = nav['path'].move(w.pos[i], w.heading[i], budget)
            if not nav['path'].done:
                break
            nav['path'] = None
            nav['index'] += 1
        if nav['index'] == len(nav['waypoints']):
            self.navigation[i] = None
            self.chosen[i] = -1

    def step(self, action, probabilities=None):
        w, c = self._world, self.c
        if w.done:
            raise RuntimeError('Episode ended; call reset()')
        proposed = split_strike_action(action)
        mask, decision = self.action_mask()
        if (proposed.shape != (self.n,) or not np.issubdtype(proposed.dtype, np.integer)
                or np.any(proposed < 0) or np.any(proposed >= self.action_count)
                or not mask[np.arange(self.n), proposed].all()):
            raise ValueError('Expected one available polygon per drone')
        if probabilities is None:
            probabilities = np.eye(self.action_count, dtype=np.float32)[proposed]
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if (probabilities.shape != mask.shape or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or not np.allclose(probabilities.sum(-1), 1., atol=1e-5)
                or (probabilities[~mask] > 1e-6).any()):
            raise ValueError('Probabilities must match the observed action mask')
        self.previous = probabilities.copy()
        self.previous_choice = np.eye(self.action_count, dtype=np.float32)[proposed]
        self.chosen[decision] = proposed[decision]
        before, old_pos = w.discovered.copy(), w.pos.copy()
        for i in np.flatnonzero(w.agent_active):
            self._navigate_polygon(i, proposed[i])
        w.vel = (w.pos-old_pos)/c.dt
        w.last_damage_by_agent.fill(0)
        w.last_agent_terminated.fill(False)
        w.t += 1
        self._sense()  # Full sensor footprint, including entry and turn arcs.
        newly_found = w.discovered & ~before & w.target_exists
        reward = sum(target_remaining_value(w.target_type[j], w.target_life[j])
                     for j in np.flatnonzero(newly_found))
        w.rewards_total += reward
        w.discounted_rewards_total += c.discount_gamma**(w.t-1)*reward
        complete = bool(w.discovered[w.target_exists].all())
        w.done = complete or w.t >= c.horizon
        return self.observation(), float(reward), complete, w.t >= c.horizon, self.metrics()

    def metrics(self):
        result = super().metrics()
        w = self._world
        found = sum(target_remaining_value(w.target_type[j], w.target_life[j])
                    for j in np.flatnonzero(w.discovered & w.target_exists))
        total = sum(target_remaining_value(w.target_type[j], w.target_life[j])
                    for j in np.flatnonzero(w.target_exists))
        result.update(score=float(found), success_threshold=float(total),
                      mission_success=bool(w.discovered[w.target_exists].all()),
                      discovered_value=float(found), consumed_objects=0)
        return result


class PolygonQBody(GraphQBody):
    """Original cross-attention and time-conditioned team readout, new inputs."""
    def __init__(self, hidden):
        super().__init__(hidden, self_attention=False)
        self.drone_encoder = nn.Sequential(nn.Linear(DRONE_FEATURES, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.target_encoder = nn.Sequential(nn.Linear(GAUSSIAN_FEATURES, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def encode(self, obs, action):
        drones, records = obs['drones'], obs['beliefs']
        alive, valid = obs['active'], obs['valid']
        k = records.shape[1]
        if action.shape != drones.shape[:2] or torch.any((action < 0) | (action >= k)):
            raise ValueError('Invalid polygon joint action')
        chosen = action[:, None, :] == torch.arange(k, device=action.device)[None, :, None]
        relative = records[:, :, None, 12:14]-drones[:, None, :, :2]
        edges = torch.cat((chosen[..., None].float(), obs['previous'].transpose(1, 2)[..., None],
                           obs['previous_choice'].transpose(1, 2)[..., None], relative), -1)
        mask = valid[:, :, None] & alive[:, None, :]
        edges = torch.where(mask[..., None], edges, torch.zeros_like(edges))
        time = self.time_encoder(obs['time'])
        d = self.drone_encoder(drones.masked_fill(~alive[..., None], 0))+time[:, None]
        target = self.target_encoder(records.masked_fill(~valid[..., None], 0))+time[:, None]
        reverse = torch.cat((edges[..., :3], -edges[..., 3:]), -1).transpose(1, 2)
        for target_read, drone_read in zip(self.target_reads, self.drone_reads):
            target = target_read(target, d, edges, mask)
            d = drone_read(d, target, reverse, mask.transpose(1, 2))
        return d, target, time, alive, valid, records


class ExplorationActorCritic(BeliefActorCritic):
    architecture = 'polygon_exploration_target_allocation'

    def __init__(self, hidden=64, features=GAUSSIAN_FEATURES):
        # Inherit distribution, COMA enumeration and target update, not encoders.
        nn.Module.__init__(self)
        if features != GAUSSIAN_FEATURES:
            raise ValueError('Polygon exploration uses continuous Gaussian/uniform belief')
        self.hidden, self.features = hidden, features
        self.actor_body = AttentionActorBody(hidden, time_embedding=False,
            query_width=features+DRONE_FEATURES+1, peer_width=DRONE_FEATURES+2, time_index=features)
        self.actor_head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor_head.weight, gain=.01)
        nn.init.zeros_(self.actor_head.bias)
        self.q_body = PolygonQBody(hidden)
        self.q_head = nn.Linear(hidden, 1)

    def actor_logits(self, obs):
        d = obs['drones']
        b, n, _ = d.shape
        k = obs['beliefs'].shape[1]
        candidates = obs['beliefs'][:, None].expand(b, n, k, -1).clone()
        candidates[..., :2] -= d[:, :, None, :2]
        candidates[..., 12:14] -= d[:, :, None, :2]
        own = d[:, :, 2:].clone()
        own[..., 7:9] -= d[..., :2]*(d[..., 3:4] > .5)
        query = torch.cat((candidates, obs['time'][:, None, None].expand(b, n, k, 1),
            obs['previous'][..., None], obs['previous_choice'][..., None],
            own[:, :, None].expand(b, n, k, DRONE_FEATURES-2)), -1)
        relative = d[:, None, :, :2]-d[:, :, None, :2]
        peer_state = d[:, None, :, 2:].expand(b, n, n, DRONE_FEATURES-2).clone()
        peer_state[..., 7:9] -= d[:, :, None, :2]*(peer_state[..., 1:2] > .5)
        physical = torch.cat((relative, peer_state), -1)
        peers = torch.cat((physical[:, :, None].expand(b, n, k, n, DRONE_FEATURES),
            obs['previous'].transpose(1, 2)[:, None, :, :, None].expand(b, n, k, n, 1),
            obs['previous_choice'].transpose(1, 2)[:, None, :, :, None].expand(b, n, k, n, 1)), -1)
        mask = (obs['active'][:, None, None, :] & obs['valid'][:, None, :, None]
                & ~torch.eye(n, dtype=torch.bool, device=d.device)[None, :, None, :])
        return self.actor_head(self.actor_body(query, peers, mask)).squeeze(-1)

    def joint_q(self, obs, action, detach_encoder=False):
        # Actor and critic encoders are separate, as in the reference policy.
        return self.q_head(self.q_body(obs, action)).squeeze(-1)
