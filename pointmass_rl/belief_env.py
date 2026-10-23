"""Partially observed version of the point-mass game.

An approximate multi-object posterior contains reported-object Gaussians and
a persistent uniform search weight, or a legacy spatial histogram. Detections
create tracks (point masses with a noiseless position sensor).
This is not a full PMBM filter: game detections have reliable identities and
types, no clutter, and static positions. Both policy and Q see only the public
belief. The hidden game state is used solely by transitions and sensors.
"""
from dataclasses import replace

import numpy as np

from .env import Config, World, DAMAGE_REWARD_RATE, split_strike_action
from .navigation import DubinsPath, lawnmower_waypoints, point_path
from .belief_intel import ReportPosterior
from .belief_sensor import detection_probabilities, cell_detection_probabilities
from .gaussian_belief import GaussianTargetBelief
from .task_automaton import TaskAutomata, TaskEvent, TaskPhase

# Public belief changes that may interrupt tasks under the event decision rule:
# a newly identified object, or the first valid track after none.
INTERRUPTS = ('track_new', 'contact')


BELIEF_FEATURES = 14


class _BeliefLayoutWorld(World):
    """Keep belief-mode spawns inside the search area, including members."""
    def _reset_once(self, seed, success_threshold=None, evaluation_task=None):
        super()._reset_once(seed, success_threshold, evaluation_task)
        self.initial_intel = None
        if not self.c.belief_initial_intel:
            return self.observation.copy()
        c = self.c
        # The historical F1 centre becomes the reported centre. F2 keeps its
        # sampled position. Only F1 is displaced after the report is generated.
        reported = self.formation_centers[0].copy()
        ids = np.flatnonzero(self.target_exists & (self.target_formation == 0))
        counts = np.bincount(self.target_type[ids], minlength=4)[1:4]
        # The simulator draws a latent scale once per episode. Sensors and
        # policy inputs never receive its realized value.
        self._intel_true_sigma = float(self.rng.lognormal(
            np.log(c.belief_intel_sigma), c.belief_intel_log_sigma_std))
        for _ in range(1000):
            center = reported + self.rng.normal(0., self._intel_true_sigma, 2)
            if not self._valid_spawn(center, c.min_spawn_distance):
                continue
            occupied = list(self.targets[self.target_exists & (self.target_formation != 0)])
            positions = []
            try:
                for _ in ids:
                    point = self._sample_cluster_member(center, c.target_formation_radius,
                                                       occupied, c.target_member_min_spacing)
                    positions.append(point)
                    occupied.append(point)
            except RuntimeError:
                continue
            self.formation_centers[0] = center
            self.targets[ids] = positions
            self.spawn_position[ids] = positions
            break
        else:
            raise RuntimeError('Could not place the reported formation within its uncertainty area')
        # This packet is the complete public report. It has no simulator IDs,
        # true centre, true sigma, true positions, or F2 composition.
        self.initial_intel = dict(position=reported.tolist(), type_counts=counts.tolist(),
                                  sigma_hint=c.belief_intel_sigma,
                                  log_sigma_std=c.belief_intel_log_sigma_std,
                                  member_radius=c.target_formation_radius)
        return self.observation.copy()

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
    def __init__(self, config, initial_intel=None):
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
        self.initial_intel = initial_intel
        self.report = (ReportPosterior(config, self.points, initial_intel)
                       if initial_intel is not None else None)
        if self.report is not None:
            self.weights = self.report.type_weights.sum(-1)
        self.initial_expected_count = float(self.weights.sum())
        self.heat_scale = max(float(self.weights.max()), 1e-12)
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
        known = {track['handle']: track for track in self.tracks}
        if self.report is not None:
            self.report.update(detection_map, [obs for obs in observations if obs[0] not in known])
            self.weights = self.report.type_weights.sum(-1)
        else:
            self.weights *= 1 - np.clip(detection_map, 0, 1)
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
            types = (self.report.type_weights[j].sum(0) / mass if self.report is not None
                     else np.full(3, 1/3))
            expected_life = float(types @ np.array([2., 1., 1.]))
            self.records[j, :12] = [*mean, covariance[0, 0], covariance[0, 1], covariance[1, 1],
                               *types, expected_life/2, mass, 0, 0]
            self.valid[j] = True
            # Choose a support point, not a centroid inside an already cleared
            # hole. Equal-priority support points use stable geometric order.
            maximum = weights.max()
            possible = np.flatnonzero(weights >= maximum * .99)
            pick = possible[np.argmin(np.linalg.norm(points[possible] - mean, axis=1))]
            if self.report is not None:
                # Shared-centre updates can slightly reorder neighbouring
                # peaks. Retain a still-promising destination to avoid a new
                # Dubins turn for each tiny change; cleared points never stick.
                distance = np.linalg.norm(points-self.goals[j]/self.c.size,axis=1)
                previous = int(np.argmin(distance))
                if distance[previous] < 1e-10 and weights[previous] >= .9*maximum:
                    pick = previous
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
        belief_class = (GaussianTargetBelief if self.c.belief_representation == 'gaussian_uniform'
                        else TargetBelief)
        self.belief = belief_class(self.c, w.initial_intel)
        self.previous = np.zeros((self.n, self.action_count), dtype=np.float32)
        self.previous_choice = np.zeros_like(self.previous)
        events = set(self.c.belief_decision_events)
        self.tasks = TaskAutomata(self.n, [e for e in TaskEvent if e.value in events])
        self.interrupts = events & set(INTERRUPTS)
        # Executor geometry only; the task phase lives in self.tasks.
        self.navigation = [None for _ in range(self.n)]
        self.first_detection = None
        self.known_tracks = self.valid_tracks = 0
        self._sense()
        self._sync_tasks()
        return self.observation()

    def _sync_tasks(self):
        """World changes become task events; new public information interrupts tasks."""
        self.tasks.sync(self._world.agent_active, self.locked_components())
        b = self.belief
        valid = int(b.valid[b.cells:b.cells + len(b.tracks)].sum())
        news = set()
        if len(b.tracks) > self.known_tracks:
            news.add('track_new')
        if self.valid_tracks == 0 and valid > 0:
            news.add('contact')
        self.known_tracks, self.valid_tracks = len(b.tracks), valid
        if news & self.interrupts:
            self.tasks.interrupt()

    @property
    def chosen(self):
        return self.tasks.held

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
        continuous = isinstance(self.belief, GaussianTargetBelief)
        if continuous:
            pd = None  # Filter integrates the continuous likelihood itself.
        elif self.c.belief_soft_detection:
            width = (self.c.size-2*self.c.belief_search_margin)/(
                self.c.belief_grid*self.c.belief_subcells)
            pd = cell_detection_probabilities(
                points, width, w.pos, w.heading, w.agent_active,
                self.c.sensor_range, self.c.sensor_fov_deg, self.c.detection_probability)
        elif self.belief.report is not None:
            # Use an enclosing circle to conservatively cover the complete
            # histogram square. Centre-only clearing can lose a continuous
            # target near a bin edge, despite its known existence.
            radius = self.belief.report.width/np.sqrt(2)
            delta = points[None]-w.pos[:,None]
            distance = np.linalg.norm(delta,axis=-1)
            bearing = np.arctan2(delta[...,1],delta[...,0])-w.heading[:,None]
            angle = np.abs(np.arctan2(np.sin(bearing),np.cos(bearing)))
            spread = np.arcsin(np.clip(radius/np.maximum(distance,1e-12),0,1))
            angle_ok = (np.ones_like(distance,dtype=bool) if self.c.sensor_fov_deg == 360
                        else (distance > radius)
                        & (angle+spread <= np.deg2rad(self.c.sensor_fov_deg)/2))
            observers = (w.agent_active[:,None] & angle_ok
                         & (distance+radius <= self.c.sensor_range)).sum(0)
            pd = 1 - (1-self.c.detection_probability)**observers
        else:
            observers = self._visibility(points).sum(0)
            pd = 1 - (1-self.c.detection_probability)**observers
        visibility = self._visibility(np.nan_to_num(w.targets)) & w.target_exists[None]
        if self.c.belief_soft_detection:
            target_pd = detection_probabilities(
                np.nan_to_num(w.targets), w.pos, w.heading, w.agent_active,
                self.c.sensor_range, self.c.sensor_fov_deg, self.c.detection_probability)
            target_pd *= w.target_exists[None]
        else:
            target_pd = visibility*self.c.detection_probability
        detected = (self.sensor_rng.random(visibility.shape) < target_pd).any(0)
        w.visible = visibility
        observations = []
        # Handles become observable only when the sensor detects the entity.
        # Geometric sorting avoids exposing hidden formation/slot ordering.
        ids = sorted(np.flatnonzero(detected), key=lambda j:tuple(w.targets[j]))
        for j in ids:
            position = w.targets[j] + self.sensor_rng.normal(0, self.c.position_noise, 2)
            observations.append((int(j), position, int(w.target_type[j]), int(w.target_life[j])))
        if continuous:
            # The background retains this continuous likelihood as history.
            # Capture poses now, rather than closing over the mutable world.
            sensor_positions = w.pos.copy()
            sensor_headings = w.heading.copy()
            sensor_active = w.agent_active.copy()
            def probability_at(points):
                probabilities = detection_probabilities(
                    points, sensor_positions, sensor_headings, sensor_active, self.c.sensor_range,
                    self.c.sensor_fov_deg, self.c.detection_probability)
                return 1-np.prod(1-probabilities, axis=0)
            self.belief.update(probability_at, observations, w.t)
        else:
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
        b = self.belief
        tracks = range(b.cells, b.cells + len(b.tracks))
        track_life = [int(round(b.records[j, 8]*2)) for j in tracks]
        return self.tasks.action_mask(b.valid, track_life, b.cells, self._world.t,
                                      initial_intel=self.c.belief_initial_intel,
                                      commit_target=self.c.commit_target,
                                      decision_rule=self.c.belief_decision_rule)

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
        tasks = self.tasks
        if not self.belief.valid[component]:
            if self.navigation[i] is not None:
                tasks.fire(i, TaskEvent.INVALIDATE)
            self.navigation[i] = None
            w.last_goal[i] = w.pos[i]
            return
        goal = self.belief.goals[component]
        is_region = component < self.belief.cells
        nav = self.navigation[i]
        if nav is None or nav['component'] != component:
            nav = dict(component=int(component), goal=goal.copy(), path=None,
                       waypoints=[], index=0)
            self.navigation[i] = nav
            tasks.fire(i, TaskEvent.SELECT, int(component))
        elif (not is_region or not c.belief_lawnmower
              or (c.belief_initial_intel and tasks.motion[i] != TaskPhase.SWEEPING)) and np.linalg.norm(goal-nav['goal']) > c.strike_range*.25:
            nav['goal'] = goal.copy()
            nav['path'] = None
            tasks.fire(i, TaskEvent.RETARGET)

        if not is_region and w.strike_participants[:, i].any():
            w.last_goal[i] = nav['goal']
            return

        if tasks.motion[i] == TaskPhase.ARRIVED:
            # Stay at an unchanged destination until sensing or selection
            # supplies a new one. Do not manufacture a departure/return loop.
            w.last_goal[i] = nav['goal']
            return

        while budget > 1e-10:
            if tasks.motion[i] == TaskPhase.SWEEPING:
                if nav['index'] >= len(nav['waypoints']):
                    # A complete pass does not erase any remaining belief.
                    nav['waypoints'] = lawnmower_waypoints(component, c, w.pos[i])
                    nav['index'] = 0
                    tasks.fire(i, TaskEvent.PASS_DONE)
                destination, end_heading = nav['waypoints'][nav['index']]
            else:
                destination, end_heading = nav['goal'], None
                if is_region and self.belief.report is not None:
                    # Approach a still-possible area from a sensor vantage
                    # point. Standing on its centre cannot clear the area with
                    # a forward-looking sensor. This controller is also used
                    # by the training environment, not just the replay.
                    support_radius = self.belief.report.width/np.sqrt(2)
                    half_angle = min(np.deg2rad(c.sensor_fov_deg)/2, np.pi/2)
                    nearest = support_radius/max(np.sin(half_angle),1e-9)
                    farthest = c.sensor_range-support_radius
                    if nearest < farthest:
                        distance = (nearest+farthest)/2
                        bearing = np.arctan2(*(nav['goal']-w.pos[i])[::-1])
                        if 'approach' not in nav or nav['path'] is None:
                            for offset in (0, np.pi/2, -np.pi/2, np.pi):
                                heading = bearing+offset
                                point = nav['goal']-distance*np.array([np.cos(heading),np.sin(heading)])
                                if ((point >= 0) & (point <= c.size)).all():
                                    nav['approach'] = (point,heading)
                                    break
                        destination, end_heading = nav.get('approach',(destination,None))
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
                tasks.fire(i, TaskEvent.ARRIVE)
                break
            if tasks.motion[i] != TaskPhase.SWEEPING:
                tasks.fire(i, TaskEvent.SWEEP)
                nav['waypoints'] = lawnmower_waypoints(component, c, w.pos[i])
                nav['index'] = 0
            else:
                nav['index'] += 1
                tasks.fire(i, TaskEvent.LEG_DONE)

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
        self.tasks.events.clear()
        self.tasks.commit(decision, proposed, w.t+c.decision_interval)
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
        self._sync_tasks()
        w.done = w.t >= c.horizon
        return self.observation(), reward, False, w.done, self.metrics()

    def unseen_expected_count(self):
        if isinstance(self.belief, GaussianTargetBelief):
            return float(self.belief.remaining_types.sum()+self.belief.background_mass)
        return float(self.belief.weights.sum())

    def metrics(self):
        w = self._world
        result = dict(team_return=w.rewards_total, discounted_team_return=w.discounted_rewards_total,
                    score=float(w.score), mission_success=bool(w.score>=w.formation_one_initial_score),
                    success_threshold=float(w.formation_one_initial_score),
                    discovery_fraction=float(w.discovered[w.target_exists].mean()),
                    discovered_objects=len(self.belief.tracks),
                    unseen_expected_count=self.unseen_expected_count(),
                    first_detection_step=self.first_detection, steps=w.t)
        if self.belief.report is not None:
            result.update(reported_unseen_expected_count=float(self.belief.report.remaining_types.sum()),
                          intel_sigma_posterior_mean=self.belief.report.sigma_mean,
                          intel_sigma_posterior_std=self.belief.report.sigma_std)
        if isinstance(self.belief, GaussianTargetBelief):
            result.update(reported_unseen_expected_count=float(self.belief.remaining_types.sum()),
                          background_search_mass=self.belief.background_mass,
                          all_reported_found_probability=float(self.belief.subset_probability[0]))
        return result
