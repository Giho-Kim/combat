"""Partially observed version of the point-mass game.

An approximate multi-object posterior contains a histogram Poisson intensity
for unseen objects and Gaussian, identified tracks for sensor detections.
This is not a full PMBM filter: game detections have reliable identities and
types, no clutter, and static positions. Both policy and Q see only the public
belief. The hidden game state is used solely by transitions and sensors.
"""
from dataclasses import replace

import numpy as np

from .env import Config, World, DAMAGE_REWARD_RATE, split_strike_action
from .navigation import DubinsPath, lawnmower_waypoints, point_path


BELIEF_FEATURES = 14


class _BeliefLayoutWorld(World):
    """Keep belief-mode spawns inside the search area, including members."""
    def _sample_point(self, min_base_distance=0.0):
        c = self.c
        lo = max(1.0, c.belief_search_margin)
        hi = c.size-lo
        corners = np.array([[lo,lo], [lo,hi], [hi,lo], [hi,hi]])
        for _ in range(2000):
            point = self.rng.uniform(lo, hi, 2)
            if np.linalg.norm(point-self.base) < min_base_distance:
                continue
            # Avoid a friendly centre from which the configured separation
            # cannot fit anywhere inside the smaller search area.
            if (c.formation_layout == 'configured' and min_base_distance > c.min_spawn_distance
                    and np.linalg.norm(corners-point, axis=1).max() < c.formation_separation):
                continue
            return point
        raise RuntimeError('Could not place entity inside belief search area')

    def _valid_spawn(self, point, min_base_distance):
        margin = self.c.belief_search_margin
        return (np.all(point >= margin) and np.all(point <= self.c.size-margin)
                and super()._valid_spawn(point, min_base_distance))


class TargetBelief:
    """Stable component slots: spatial intensity cells, then discovered tracks.

    Records: mean xy, covariance xx/xy/yy, type probabilities (3), expected
    life/2, expected object count, identified flag, time since observation,
    controller destination xy. The latter exposes the current support point.
    Positions/covariances use map-normalized coordinates. One intensity cell
    may contain any number of objects; one identified track contains one.
    """
    def __init__(self, config):
        self.c = config
        g, s = config.belief_grid, config.belief_subcells
        self.cells = g * g
        self.capacity = self.cells + config.n_targets
        centers = (np.arange(g) + .5) / g
        local = ((np.arange(s) + .5) / s - .5) / g
        offsets = np.stack(np.meshgrid(local, local), -1).reshape(-1, 2)
        self.points = np.stack(np.meshgrid(centers, centers), -1).reshape(-1, 1, 2) + offsets
        margin = config.belief_search_margin/config.size
        self.points = margin+(1-2*margin)*self.points
        expected = ((config.n_targets + config.min_targets) / 2
                    if config.randomize_counts else config.n_targets)
        self.weights = np.full(self.points.shape[:-1], expected / (g*g*s*s))
        self.tracks = []
        self.records = np.zeros((self.capacity, BELIEF_FEATURES), dtype=np.float32)
        self.valid = np.zeros(self.capacity, dtype=bool)
        self.goals = np.zeros((self.capacity, 2))
        self.refresh(0)

    def update(self, detection_map, observations, time):
        """Negative evidence thins unseen intensity; detections create tracks.

        observation handle is assigned by the game sensor only on visibility.
        It never appears in actor/Q inputs. Track slots are not recycled within
        an episode, so action history never points to a different object.
        """
        self.weights *= 1 - np.clip(detection_map, 0, 1)
        known = {track['handle']: track for track in self.tracks}
        variance = max((self.c.position_noise / self.c.size) ** 2, 1e-12)
        for handle, position, typ, life in observations:
            measurement = np.clip(position / self.c.size, 0, 1)
            if handle not in known:
                if len(self.tracks) >= self.c.n_targets:
                    raise RuntimeError('Detected objects exceed configured capacity')
                track = dict(handle=handle, mean=measurement, variance=variance,
                             type=int(typ), life=int(life), seen=time)
                self.tracks.append(track)
                known[handle] = track
            else:
                track = known[handle]
                gain = track['variance'] / (track['variance'] + variance)
                track['mean'] = track['mean'] + gain * (measurement - track['mean'])
                track['variance'] *= 1 - gain
                track.update(type=int(typ), life=int(life), seen=time)
        self.refresh(time)

    def refresh(self, time):
        self.records.fill(0)
        self.valid.fill(False)
        for j, (points, weights) in enumerate(zip(self.points, self.weights)):
            mass = weights.sum()
            if mass <= 1e-8:
                continue
            p = weights / mass
            mean = (points * p[:, None]).sum(0)
            centered = points - mean
            covariance = (centered * p[:, None]).T @ centered
            self.records[j, :12] = [*mean, covariance[0, 0], covariance[0, 1], covariance[1, 1],
                               1/3, 1/3, 1/3, 2/3, mass, 0, 0]
            self.valid[j] = True
            # Choose a support point, not a centroid inside an already cleared
            # hole. Equal-priority support points use stable geometric order.
            maximum = weights.max()
            possible = np.flatnonzero(weights >= maximum * .99)
            pick = possible[np.argmin(np.linalg.norm(points[possible] - mean, axis=1))]
            self.goals[j] = points[pick] * self.c.size
        for i, track in enumerate(self.tracks):
            j = self.cells + i
            typ = np.eye(3)[track['type'] - 1]
            self.records[j, :12] = [*track['mean'], track['variance'], 0, track['variance'],
                               *typ, track['life']/2, 1, 1,
                               (time-track['seen']) / self.c.horizon]
            self.valid[j] = track['life'] > 0
            self.goals[j] = track['mean'] * self.c.size
        self.records[self.valid, 12:14] = self.goals[self.valid] / self.c.size


class BeliefWorld:
    def __init__(self, config=None):
        self.c = config or Config(mode='belief')
        if self.c.mode != 'belief':
            raise ValueError('BeliefWorld requires mode=belief')
        if self.c.target_motion_scale != 0:
            raise ValueError('Initial belief mode models stationary game objects')
        self.n = self.c.n_agents
        # Composition keeps the historical fully observed World untouched.
        self._world = _BeliefLayoutWorld(replace(self.c, mode='known'))
        self.action_count = self.c.belief_grid**2 + self.c.n_targets

    @property
    def done(self):
        return self._world.done

    def reset(self, seed=None):
        self._world.reset(seed)
        w = self._world
        w.discovered.fill(False)
        w.coverage.fill(False)
        w.mem_pos.fill(0)
        w.mem_type.fill(0)
        w.mem_life.fill(0)
        self.sensor_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
        self.belief = TargetBelief(self.c)
        self.previous = np.zeros((self.n, self.action_count), dtype=np.float32)
        self.previous_choice = np.zeros_like(self.previous)
        self.chosen = np.full(self.n, -1, dtype=int)
        self.next_decision = np.zeros(self.n, dtype=int)
        self.navigation = [None for _ in range(self.n)]
        self.first_detection = None
        self._sense()
        return self.observation()

    def _visibility(self, points):
        w = self._world
        relative = points[None] - w.pos[:, None]
        distance = np.linalg.norm(relative, axis=-1)
        bearing = np.arctan2(relative[..., 1], relative[..., 0])
        angle = np.arctan2(np.sin(bearing-w.heading[:, None]), np.cos(bearing-w.heading[:, None]))
        return (w.agent_active[:, None] & (distance <= self.c.sensor_range)
                & ((np.abs(angle) <= np.deg2rad(self.c.sensor_fov_deg)/2) | (distance < 1e-6)))

    def _sense(self):
        w = self._world
        points = self.belief.points.reshape(-1, 2) * self.c.size
        observers = self._visibility(points).sum(0)
        pd = 1 - (1-self.c.detection_probability)**observers
        visibility = self._visibility(np.nan_to_num(w.targets)) & w.target_exists[None]
        detected = (visibility & (self.sensor_rng.random(visibility.shape) < self.c.detection_probability)).any(0)
        w.visible = visibility
        observations = []
        # Handles become observable only when the sensor detects the entity.
        # Geometric sorting avoids exposing hidden formation/slot ordering.
        ids = sorted(np.flatnonzero(detected), key=lambda j:tuple(w.targets[j]))
        for j in ids:
            position = w.targets[j] + self.sensor_rng.normal(0, self.c.position_noise, 2)
            observations.append((int(j), position, int(w.target_type[j]), int(w.target_life[j])))
        self.belief.update(pd.reshape(self.belief.weights.shape), observations, w.t)
        w.discovered |= detected
        if detected.any() and self.first_detection is None:
            self.first_detection = w.t

    def locked_components(self):
        raw = self._world.locked_targets()
        result = np.full(self.n, -1, dtype=int)
        for i, track in enumerate(self.belief.tracks):
            result[raw == track['handle']] = self.belief.cells + i
        return result

    def action_mask(self):
        w = self._world
        mask = np.broadcast_to(self.belief.valid, (self.n, self.action_count)).copy()
        locks = self.locked_components()
        sweeping = np.full(self.n, -1, dtype=int)
        for i, nav in enumerate(self.navigation):
            if (w.agent_active[i] and nav is not None and nav['sweeping']
                    and self.belief.valid[nav['component']]):
                sweeping[i] = nav['component']
        for i in np.flatnonzero(sweeping >= 0):
            j = sweeping[i]
            mask[:, j] = False
            mask[i, j] = True
        for j in range(self.belief.cells, self.belief.cells + len(self.belief.tracks)):
            life = int(round(self.belief.records[j, 8]*2))
            occupied = (locks == j).sum()
            if occupied >= life:
                mask[:, j] = locks == j
        decision = w.agent_active & (locks < 0)
        for i in range(self.n):
            held = self.chosen[i]
            if locks[i] >= 0:
                mask[i] = False
                mask[i, locks[i]] = True
            elif held >= 0 and mask[i, held] and (sweeping[i] >= 0 or self.c.commit_target or w.t < self.next_decision[i]):
                mask[i] = False
                mask[i, held] = True
                decision[i] = False
            if not w.agent_active[i] or not mask[i].any():
                mask[i] = False
                mask[i, 0] = True  # Finite categorical; invalid component is an idle action.
                decision[i] = False
        return mask, decision

    def observation(self):
        w = self._world
        alive = w.agent_active.astype(np.float32)
        # Heading is required because it affects the next observation footprint.
        drones = np.column_stack((w.pos/self.c.size, alive,
                                  w.strike_participants.any(0),
                                  w.agent_strike_progress/self.c.strike_steps_per_life,
                                  w.vel/self.c.strike_speed, np.cos(w.heading), np.sin(w.heading))).astype(np.float32)
        drones *= alive[:, None]
        mask, decision = self.action_mask()
        return dict(time=np.array([(self.c.horizon-w.t)/self.c.horizon], dtype=np.float32),
                    drones=drones, beliefs=self.belief.records.copy(), valid=self.belief.valid.copy(),
                    previous=self.previous*alive[:, None], previous_choice=self.previous_choice*alive[:, None],
                    mask=mask, decision=decision, active=w.agent_active.copy())

    def _navigate(self, i, component):
        """Follow an oriented path, then sweep a selected exploration region."""
        w, c = self._world, self.c
        radius = c.belief_turn_radius
        budget = c.strike_speed*c.dt
        if not self.belief.valid[component]:
            self.navigation[i] = None
            w.last_goal[i] = w.pos[i]
            return
        goal = self.belief.goals[component]
        is_region = component < self.belief.cells
        nav = self.navigation[i]
        if nav is None or nav['component'] != component:
            nav = dict(component=int(component), goal=goal.copy(), path=None,
                       sweeping=False, waypoints=[], index=0, arrived=False)
            self.navigation[i] = nav
        elif (not is_region or not c.belief_lawnmower) and np.linalg.norm(goal-nav['goal']) > c.strike_range*.25:
            nav['goal'] = goal.copy()
            nav['path'] = None
            nav['arrived'] = False

        if not is_region and w.strike_participants[:, i].any():
            w.last_goal[i] = nav['goal']
            return

        if nav['arrived']:
            # Stay at an unchanged destination until sensing or selection
            # supplies a new one. Do not manufacture a departure/return loop.
            w.last_goal[i] = nav['goal']
            return

        while budget > 1e-10:
            if nav['sweeping']:
                if nav['index'] >= len(nav['waypoints']):
                    # A complete pass does not erase any remaining belief.
                    nav['waypoints'] = lawnmower_waypoints(component, c, w.pos[i])
                    nav['index'] = 0
                destination, end_heading = nav['waypoints'][nav['index']]
            else:
                destination, end_heading = nav['goal'], None
            w.last_goal[i] = destination
            if nav['path'] is None:
                if end_heading is None and np.linalg.norm(destination-w.pos[i]) < 1e-8:
                    arrived = True
                else:
                    nav['path'] = (point_path(w.pos[i], w.heading[i], destination, radius)
                                   if end_heading is None else
                                   DubinsPath(w.pos[i], w.heading[i], destination, end_heading, radius))
                    arrived = False
            else:
                arrived = False
            if not arrived:
                w.pos[i], w.heading[i], budget = nav['path'].move(w.pos[i], w.heading[i], budget)
                arrived = nav['path'].done
            if not arrived:
                break
            nav['path'] = None
            if not is_region or not c.belief_lawnmower:
                # Resolve arrival/sensing this step before the next arc.
                nav['arrived'] = True
                break
            if not nav['sweeping']:
                nav['sweeping'] = True
                nav['waypoints'] = lawnmower_waypoints(component, c, w.pos[i])
                nav['index'] = 0
            else:
                nav['index'] += 1

    def step(self, action, probabilities=None):
        w, c = self._world, self.c
        if w.done:
            raise RuntimeError('Episode ended; call reset()')
        proposed = split_strike_action(action)
        if proposed.shape != (self.n,) or not np.issubdtype(proposed.dtype, np.integer) or np.any((proposed<0)|(proposed>=self.action_count)):
            raise ValueError('Expected one valid belief component ID per drone')
        mask, decision = self.action_mask()
        if not mask[np.arange(self.n), proposed].all():
            raise ValueError('Action violates current observed availability or commitment')
        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=np.float32)
            if (probabilities.shape != mask.shape or not np.isfinite(probabilities).all()
                    or (probabilities<0).any() or not np.allclose(probabilities.sum(-1), 1, atol=1e-5)
                    or (probabilities[~mask]>1e-6).any()):
                raise ValueError('Probabilities must match the observed action mask')
            self.previous = probabilities.copy()
        else:
            self.previous = np.eye(self.action_count, dtype=np.float32)[proposed]
        self.previous_choice = np.eye(self.action_count, dtype=np.float32)[proposed]
        self.chosen[decision] = proposed[decision]
        self.next_decision[decision] = w.t+c.decision_interval
        active_before = w.agent_active.copy()
        old_pos = w.pos.copy()
        raw = np.full(self.n, -1, dtype=int)
        for i in np.flatnonzero(w.agent_active):
            j = proposed[i]
            self._navigate(i, j)
            if not self.belief.valid[j]:
                continue
            if j >= self.belief.cells:
                raw[i] = self.belief.tracks[j-self.belief.cells]['handle']
        w.vel = (w.pos-old_pos)/c.dt
        w.target_assignment.fill(False)
        selected = np.flatnonzero(w.agent_active & (raw>=0))
        w.target_assignment[raw[selected], selected] = True
        w.last_damage_by_agent.fill(0)
        w.t += 1
        w._resolve_strikes(raw)
        # Task results are observed events, including the final participant's
        # report. No hidden object state is consulted by the learner.
        for track in self.belief.tracks:
            j = track['handle']
            if np.any(active_before & ~w.agent_active & (raw == j)):
                track.update(life=int(w.target_life[j]), seen=w.t)
        w.target_assignment[~w.target_exists | w.destroyed] = False
        w.vel[~w.agent_active] = 0
        w.last_agent_terminated = active_before & ~w.agent_active
        reward = float(DAMAGE_REWARD_RATE*w.last_damage_by_agent.sum())
        w.rewards_total += reward
        w.discounted_rewards_total += c.discount_gamma**(w.t-1)*reward
        if w.score>0 and w.first_score_step is None:
            w.first_score_step = w.t
        w.score_area += w.score/max(1,w.formation_one_initial_score)
        self._sense()
        w.done = w.t >= c.horizon
        return self.observation(), reward, False, w.done, self.metrics()

    def metrics(self):
        w = self._world
        return dict(team_return=w.rewards_total, discounted_team_return=w.discounted_rewards_total,
                    score=float(w.score), mission_success=bool(w.score>=w.formation_one_initial_score),
                    success_threshold=float(w.formation_one_initial_score),
                    discovery_fraction=float(w.discovered[w.target_exists].mean()),
                    discovered_objects=len(self.belief.tracks),
                    unseen_expected_count=float(self.belief.weights.sum()),
                    first_detection_step=self.first_detection, steps=w.t)
