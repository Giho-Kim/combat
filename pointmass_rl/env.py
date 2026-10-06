"""Requirement-aligned 2-D point-mass mission simulator.

The world deliberately keeps vehicle motion simple. Physical distances use
100 m units (the default 100 x 100 map is 10 km x 10 km), while formations,
sensing, jamming and strike semantics are represented explicitly.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

TARGET_FEATURES = 4
SELF_FEATURES = 3
SUCCESS_THRESHOLDS = (5, 8, 9, 11, 12)
DAMAGE_REWARD_RATE = 1.0
SUCCESS_BONUS = 0.0
EVALUATION_TASKS = (
    "type1_formation_two",
    "type1_formation_one_reachable",
    "type1_formation_one_unreachable",
)


def evaluation_task_for_episode(config, episode):
    if (config.horizon == 100 and config.n_targets == 5
            and config.min_targets == 5 and not config.randomize_counts
            and not config.randomize_target_composition):
        return EVALUATION_TASKS[episode % len(EVALUATION_TASKS)]
    return None
_TARGET_TYPES_BY_THRESHOLD = {
    5: ((2, 2, 3), (1, 1)),
    8: ((1, 2, 3), (1, 2)),
    9: ((1, 2, 2), (1, 3)),
    11: ((1, 1, 3), (2, 2)),
    12: ((1, 1, 2), (2, 3)),
}


def target_remaining_value(target_type, target_life):
    """Return the value still available from a target at its current life."""
    target_type, target_life = int(target_type), int(target_life)
    if target_life <= 0:
        return 0.0
    if target_type == 1:
        return 5.0 if target_life >= 2 else 2.5
    return float({2: 2, 3: 1}.get(target_type, 0))


def strike_action(target):
    """Build the only public action: one target ID per drone."""
    return {"target": np.asarray(target)}


def split_strike_action(action):
    """Return target IDs from the strike-only action mapping."""
    if not isinstance(action, dict) or set(action) != {"target"}:
        raise ValueError("Strike action must contain only target")
    return np.asarray(action["target"])


@dataclass
class Config:
    # Fixed tensor capacities; active counts are sampled per episode.
    n_agents: int = 6
    min_agents: int = 6
    n_targets: int = 5
    min_targets: int = 5
    randomize_counts: bool = False
    randomize_target_composition: bool = False
    horizon: int = 100
    dt: float = 1.0
    decision_interval: int = 1
    sticky_assignment: bool = False
    commit_target: bool = False
    size: float = 100.0
    strike_speed: float = 4.0 / 9.0  # 160 km/h
    sensor_range: float = 15.0
    sensor_fov_deg: float = 120.0
    position_noise: float = 0.02
    initial_intel_noise: float = 1.5
    min_spawn_distance: float = 30.0
    friendly_formation_radius: float = 1.0
    target_formation_radius: float = 6.5
    target_member_min_spacing: float = 3.0
    target_motion_scale: float = 0.0
    formation_separation: float = 35.0
    formation_distance_spread: float = 5.0
    formation_two_progress: float = 0.55
    formation_two_lateral_offset: float = 8.0
    strike_range: float = 0.05
    strike_probability: float = 1.0
    strike_steps_per_life: int = 10
    penalty_time: float = 0.1
    discount_gamma: float = 0.99
    gae_lambda: float = 0.95
    mission_failure_penalty: float = 100.0
    mission_success_reward: float = 100.0
    mode: str = 'known'
    belief_grid: int = 4
    belief_subcells: int = 8
    belief_turn_radius: float = 2.0  # Map units; 200 m in the game scale.
    belief_search_margin: float = 10.0  # 1 km on each side of the 10 km map.
    belief_lawnmower: bool = True
    detection_probability: float = 1.0
    formation_layout: str = 'legacy'

    def __post_init__(self):
        if self.mode not in ('known', 'belief'):
            raise ValueError('mode must be known or belief')
        if self.formation_layout not in ('legacy', 'configured'):
            raise ValueError('formation_layout must be legacy or configured')
        if type(self.belief_grid) is not int or not 1 <= self.belief_grid <= 8:
            raise ValueError('belief_grid must be an integer in 1..8')
        if type(self.belief_subcells) is not int or self.belief_subcells < 2:
            raise ValueError('belief_subcells must be an integer >= 2')
        if not np.isfinite(self.belief_turn_radius) or self.belief_turn_radius <= 0:
            raise ValueError('belief_turn_radius must be finite and positive')
        if type(self.belief_lawnmower) is not bool:
            raise ValueError('belief_lawnmower must be a boolean')
        if not np.isfinite(self.belief_search_margin) or self.belief_search_margin < 0:
            raise ValueError('belief_search_margin must be finite and nonnegative')
        if self.mode == 'belief' and 2*self.belief_search_margin >= self.size:
            raise ValueError('belief_search_margin must leave a nonempty search area')
        if not 0 < self.detection_probability <= 1:
            raise ValueError('detection_probability must be in (0, 1]')
        if type(self.decision_interval) is not int or self.decision_interval < 1:
            raise ValueError('decision_interval must be a positive integer')
        if not 1 <= self.n_agents <= 10:
            raise ValueError("n_agents capacity must be 1..10")
        if not 1 <= self.min_agents <= self.n_agents:
            raise ValueError("min_agents must be within agent capacity")
        if not 1 <= self.n_targets <= 20:
            raise ValueError("n_targets capacity must be 1..20")
        if not 1 <= self.min_targets <= self.n_targets:
            raise ValueError("min_targets must fit target capacity")
        if type(self.randomize_target_composition) is not bool:
            raise ValueError("randomize_target_composition must be a boolean")
        if self.randomize_target_composition and self.mode != 'known':
            raise ValueError("Random target composition currently requires known mode")
        if self.randomize_target_composition and self.min_targets < 3:
            raise ValueError("Random target composition needs at least three targets for two formations")
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2")
        positive = ("dt", "size", "strike_speed", "sensor_range",
                    "strike_range", "friendly_formation_radius",
                    "target_formation_radius", "formation_separation",
                    "formation_distance_spread",
                    "formation_two_lateral_offset",
                    "target_member_min_spacing")
        if any(getattr(self, key) <= 0 for key in positive):
            raise ValueError("physical scales and ranges must be positive")
        if self.target_motion_scale < 0:
            raise ValueError("target_motion_scale must be nonnegative")
        if not 0 <= self.strike_probability <= 1:
            raise ValueError("strike_probability must be in [0, 1]")
        if not isinstance(self.strike_steps_per_life, int) or self.strike_steps_per_life <= 0:
            raise ValueError("strike_steps_per_life must be a positive integer")
        if not 0 < self.sensor_fov_deg <= 360:
            raise ValueError("sensor_fov_deg must be in (0, 360]")
        if not 0 < self.formation_two_progress < 1:
            raise ValueError("formation_two_progress must be in (0, 1)")
        if min(self.position_noise, self.initial_intel_noise, self.penalty_time,
               self.mission_failure_penalty, self.mission_success_reward) < 0:
            raise ValueError("noise, drain and penalties must be nonnegative")
        if not 0 < self.discount_gamma <= 1:
            raise ValueError("discount_gamma must be in (0, 1]")
        if not 0 < self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be in (0, 1]")

    @property
    def speed(self):
        return self.strike_speed

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text())
        # Older saved runs contain reward coefficients that no longer apply.
        data.pop("damage_value_reward_scale", None)
        data.pop("damage_credit_scale", None)
        return cls(**data)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


class World:
    LAYOUT_RETRIES = 8

    def __init__(self, config=None):
        self.c = config or Config()
        if self.c.mode != 'known':
            raise ValueError('Use BeliefWorld for mode=belief')
        self.n = self.c.n_agents
        self.obs_dim = SELF_FEATURES + TARGET_FEATURES * self.c.n_targets

    def _count(self, low, high):
        low = min(low, high)
        return int(self.rng.integers(low, high + 1)) if self.c.randomize_counts else high

    def _sample_point(self, min_base_distance=0.0):
        c = self.c
        margin = 1.0
        for _ in range(2000):
            p = self.rng.uniform(margin, c.size - margin, 2)
            if np.linalg.norm(p - self.base) < min_base_distance:
                continue
            return p
        raise RuntimeError("Could not place entity; relax map constraints")

    def _valid_spawn(self, point, min_base_distance):
        if np.any(point < 0.2) or np.any(point > self.c.size - 0.2):
            return False
        if np.linalg.norm(point - self.base) < min_base_distance:
            return False
        return True

    def _sample_cluster_member(self, center, radius, occupied=(), min_spacing=0.0):
        for _ in range(1000):
            angle = self.rng.uniform(0, 2 * np.pi)
            distance = np.sqrt(self.rng.random()) * radius
            point = center + distance * np.array([np.cos(angle), np.sin(angle)])
            separated = not len(occupied) or np.all(
                np.linalg.norm(np.asarray(occupied) - point, axis=1) >= min_spacing)
            if separated and self._valid_spawn(point, self.c.min_spawn_distance):
                return point
        raise RuntimeError("Could not place formation member")

    def reset(self, seed=None, success_threshold=None, evaluation_task=None):
        """Reset the world, retrying deterministic layout generations on failure."""
        if evaluation_task is not None:
            if evaluation_task not in EVALUATION_TASKS:
                raise ValueError(f"Unknown evaluation task: {evaluation_task}")
            if success_threshold is not None:
                raise ValueError("Specify either evaluation_task or success_threshold")
            if (self.c.horizon != 100 or self.c.n_targets != 5
                    or self.c.min_targets != 5 or self.c.randomize_counts
                    or self.c.randomize_target_composition):
                raise ValueError("Evaluation tasks require five fixed targets and horizon 100")
        if (success_threshold is not None
                and success_threshold not in SUCCESS_THRESHOLDS):
            raise ValueError(
                f"success_threshold must be one of {SUCCESS_THRESHOLDS}")
        # Keep the first attempt identical to the historical seed mapping;
        # derive deterministic fallback layouts only if that attempt fails.
        attempt_seeds = [np.random.SeedSequence(seed)]
        attempt_seeds.extend(
            np.random.SeedSequence(seed).spawn(self.LAYOUT_RETRIES - 1)
        )
        last_error = None
        for attempt_seed in attempt_seeds:
            try:
                return self._reset_once(attempt_seed, success_threshold, evaluation_task)
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(
            f"Could not generate a valid scenario after {self.LAYOUT_RETRIES} attempts"
        ) from last_error

    def _reset_once(self, seed, success_threshold=None, evaluation_task=None):
        c = self.c
        seeds = seed.spawn(5)
        (self.rng, self.motion_rng, self.sensor_rng, self.combat_rng,
         self.allocation_rng) = [np.random.default_rng(s) for s in seeds]
        self.t, self.done = 0, False
        edge = 3.0
        self.base = self.rng.uniform(edge, c.size - edge, 2)
        actual_targets = self._count(c.min_targets, c.n_targets)
        actual_agents = self._count(c.min_agents, c.n_agents)
        self.agent_active = np.arange(self.n) < actual_agents
        self.initial_agent_count = actual_agents
        self.target_exists = np.zeros(c.n_targets, dtype=bool)
        self.target_exists[:actual_targets] = True

        self.friendly_center = self._sample_point(c.min_spawn_distance + c.friendly_formation_radius)
        self.pos = np.tile(self.base, (self.n, 1))
        for i in np.flatnonzero(self.agent_active):
            self.pos[i] = self._sample_cluster_member(self.friendly_center, c.friendly_formation_radius)
        self.vel = np.zeros((self.n, 2))
        self.heading = self.rng.uniform(-np.pi, np.pi, self.n)
        self.targets = np.full((c.n_targets, 2), np.nan)
        self.target_vel = np.zeros((c.n_targets, 2))
        self.target_base_vel = np.zeros((c.n_targets, 2))
        self.target_type = np.zeros(c.n_targets, dtype=int)
        self.target_life = np.zeros(c.n_targets, dtype=int)
        self.target_score = np.zeros(c.n_targets, dtype=int)
        self.spawn_step = np.full(c.n_targets, c.horizon + 1, dtype=int)
        self.spawn_position = np.full((c.n_targets, 2), np.nan)
        self.target_formation = np.zeros(c.n_targets, dtype=int)
        target_ids = np.flatnonzero(self.target_exists)
        formation_count = min(2, len(target_ids))
        # Keep formation records in fixed contiguous state-space sections:
        # formation 1 first, then formation 2. The optional varied scenario
        # samples every split with nonempty formations and strictly more in F1.
        if formation_count == 2:
            formation_one_count = (
                int(self.rng.integers(len(target_ids)//2 + 1, len(target_ids)))
                if c.randomize_target_composition else (len(target_ids) + 1) // 2)
            self.target_formation[target_ids[formation_one_count:]] = 1
        balanced_five_target_case = (
            not c.randomize_target_composition and len(target_ids) == 5
            and np.count_nonzero(self.target_formation[target_ids] == 0) == 3)
        threshold = None
        if balanced_five_target_case:
            if evaluation_task is not None:
                threshold = 5 if evaluation_task == EVALUATION_TASKS[0] else 12
            elif success_threshold is not None:
                threshold = success_threshold
            else:
                threshold = int(self.rng.choice(SUCCESS_THRESHOLDS))
        sampled_positions = None
        sampled_centers = None
        for _ in range(1000):
            centers, occupied = [], []
            candidate_positions = np.full((len(target_ids), 2), np.nan)
            try:
                for formation in range(formation_count):
                    center = None
                    if formation == 0:
                        if evaluation_task == EVALUATION_TASKS[1]:
                            separation, spread = 25.0, 2.0
                        elif evaluation_task == EVALUATION_TASKS[2]:
                            separation, spread = 54.0, 4.0
                        elif (evaluation_task is None and success_threshold is None
                              and balanced_five_target_case and c.formation_layout == 'legacy'):
                            # Train every B composition across the same near/far
                            # range; keep the fixed evaluation layouts unchanged.
                            separation, spread = 25.0, 33.0
                        else:
                            separation, spread = (c.formation_separation,
                                                  c.formation_distance_spread)
                        for _ in range(200):
                            candidate = self._sample_point(c.min_spawn_distance)
                            distance = np.linalg.norm(candidate - self.friendly_center)
                            if separation <= distance <= separation + spread:
                                center = candidate
                                break
                    else:
                        route = centers[0] - self.friendly_center
                        direction = route / np.linalg.norm(route)
                        lateral = np.array([-direction[1], direction[0]])
                        for side in self.rng.permutation([-1.0, 1.0]):
                            candidate = (self.friendly_center + c.formation_two_progress * route
                                         + side * c.formation_two_lateral_offset * lateral)
                            if self._valid_spawn(candidate, c.min_spawn_distance):
                                center = candidate
                                break
                    if center is None:
                        raise RuntimeError("Could not place target formation center")
                    centers.append(center)
                    member_indices = np.flatnonzero(
                        self.target_formation[target_ids] == formation)
                    for index in member_indices:
                        point = self._sample_cluster_member(
                            center, c.target_formation_radius, occupied,
                            c.target_member_min_spacing)
                        occupied.append(point)
                        candidate_positions[index] = point
            except RuntimeError:
                continue
            sampled_positions = candidate_positions
            sampled_centers = centers
            break
        else:
            raise RuntimeError("Could not place target formation")
        self.formation_centers = np.stack(sampled_centers)
        if balanced_five_target_case:
            # Shuffle within each formation while preserving its B-specific
            # type composition.
            formation_one_types, formation_two_types = _TARGET_TYPES_BY_THRESHOLD[threshold]
            formation_one_types = np.asarray(formation_one_types, dtype=int)
            formation_two_types = np.asarray(formation_two_types, dtype=int)
            self.rng.shuffle(formation_one_types)
            self.rng.shuffle(formation_two_types)
            type_pool = np.concatenate([formation_one_types, formation_two_types])
        else:
            if success_threshold is not None:
                raise ValueError(
                    "A fixed success_threshold requires the five-target 3+2 scenario")
            if c.randomize_target_composition:
                type_pool = self.rng.integers(1, 4, size=len(target_ids))
            else:
                type_pool = np.resize(np.array([1, 2, 3], dtype=int), len(target_ids))
                self.rng.shuffle(type_pool)
        common_heading = self.motion_rng.uniform(0, 2 * np.pi)
        for j, point, typ in zip(target_ids, sampled_positions, type_pool):
            formation = self.target_formation[j]
            typ = int(typ)
            self.target_type[j] = typ
            self.target_life[j] = 2 if typ == 1 else 1
            self.target_score[j] = {1: 5, 2: 2, 3: 1}[typ]
            self.spawn_position[j] = point
            self.spawn_step[j] = 0
            lo, hi = {1: (40, 60), 2: (60, 80), 3: (4, 10)}[typ]
            speed = self.motion_rng.uniform(lo, hi) / 360.0 * c.target_motion_scale
            heading = common_heading + self.motion_rng.normal(0, np.deg2rad(5))
            self.target_base_vel[j] = [np.cos(heading) * speed, np.sin(heading) * speed]
            self.target_vel[j] = self.target_base_vel[j]
        self.active = self.target_exists & (self.spawn_step == 0)
        self.targets[self.active] = self.spawn_position[self.active]

        self.mem_pos = np.zeros((self.n, c.n_targets, 2))
        self.mem_type = np.zeros((self.n, c.n_targets), dtype=int)
        self.mem_life = np.zeros((self.n, c.n_targets), dtype=int)
        self.coverage = np.zeros((self.n, 10, 10), dtype=bool)
        self.last_goal = self.pos.copy()
        self.discovered = np.zeros(c.n_targets, dtype=bool)
        self.destroyed = np.zeros(c.n_targets, dtype=bool)
        self.first_detection = 0
        self.first_score_step = None
        self.score = 0
        self.score_area = 0.0
        self.target_destroy_step = np.full(c.n_targets, -1, dtype=int)
        self.strike_progress = np.zeros(c.n_targets, dtype=int)
        self.agent_strike_progress = np.zeros(self.n, dtype=int)
        # Current target assignments include approaching and striking drones.
        # Assignment alone does not reserve a strike participation slot.
        self.target_assignment = np.zeros((c.n_targets, self.n), dtype=bool)
        # Each participant tracks its own strike duration. Late arrivals may
        # fill an open life slot but never inherit an earlier arrival's progress.
        self.strike_participants = np.zeros((c.n_targets, self.n), dtype=bool)
        self.last_agent_terminated = np.zeros(self.n, dtype=bool)
        self.initial_score = int(self.target_score[self.target_exists].sum())
        formation_one = self.target_exists & (self.target_formation == 0)
        self.formation_one_initial_score = int(self.target_score[formation_one].sum())
        self.strike_attempts = self.strike_hits = 0
        self.rewards_total = 0.0
        self.discounted_rewards_total = 0.0
        self._sense()
        self.discovered |= self.target_exists
        self.coverage[self.agent_active] = True
        self.observation = self._obs()
        return self.observation.copy()

    def _known_mask(self, i):
        return self.target_exists & ~self.destroyed

    def _sense(self):
        """Give every active drone exact current state for every live target."""
        self.perceived_pos = self.pos.copy()
        self.visible = (self.agent_active[:, None] & self.active[None]
                        & ~self.destroyed[None])
        for i, j in zip(*np.nonzero(self.visible)):
            self.mem_pos[i, j] = self.targets[j]
            self.mem_type[i, j] = self.target_type[j]
            self.mem_life[i, j] = self.target_life[j]

    def _obs(self):
        c = self.c
        obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        target_start = SELF_FEATURES
        reward_margin = ((self.formation_one_initial_score - self.score)
                         / self.formation_one_initial_score)
        for i in np.flatnonzero(self.agent_active):
            obs[i, :SELF_FEATURES] = [(c.horizon - self.t) / c.horizon,
                self.agent_strike_progress[i] / c.strike_steps_per_life
                if self.strike_participants[:, i].any() else 0.0,
                reward_margin]
            for j in np.flatnonzero(self._known_mask(i)):
                k = target_start + TARGET_FEATURES * j
                remaining_value = target_remaining_value(
                    self.mem_type[i, j], self.mem_life[i, j])
                capacity = min(2 if self.target_type[j] == 1 else 1,
                               int(self.target_life[j]))
                participant = self.strike_participants[j, i]
                occupied = int(np.count_nonzero(
                    self.agent_active & self.strike_participants[j]))
                in_range = (np.linalg.norm(self.pos[i] - self.targets[j])
                            <= c.strike_range)
                # Approaching drones do not reserve capacity. A live target is
                # masked for an outsider only when it reaches strike range and
                # actual participants have already filled every available slot.
                selectable = participant or occupied < capacity or not in_range
                obs[i, k:k + TARGET_FEATURES] = [
                    *((self.mem_pos[i, j] - self.perceived_pos[i]) / c.size),
                    remaining_value / 5.0, float(selectable)]
        return obs

    def resolve_setpoints(self, target_ids):
        """Resolve live targets; unavailable choices leave the drone in place."""
        target_ids = np.asarray(target_ids)
        goals = self.pos.copy()
        for i, target in enumerate(target_ids):
            if not self.agent_active[i]:
                continue
            if self._known_mask(i)[target]:
                goals[i] = self.mem_pos[i, target]
        return np.clip(goals, 0, self.c.size)

    def locked_targets(self):
        """Only participating drones are committed during an ongoing strike."""
        participating = self.agent_active & self.strike_participants.any(axis=0)
        return np.where(participating, self.strike_participants.argmax(axis=0), -1)

    def committed_targets(self, proposed_targets):
        """Keep each drone's current target until its strike action completes."""
        proposed_targets = np.asarray(proposed_targets)
        current = self.locked_targets()
        locked = ((current >= 0)
                  & self.target_exists[np.maximum(current, 0)]
                  & ~self.destroyed[np.maximum(current, 0)])
        return np.where(locked, current, proposed_targets)

    def _move_targets(self):
        c = self.c
        for j in np.flatnonzero(self.active & ~self.destroyed):
            scale = 0.5 if self.target_type[j] == 1 and self.target_life[j] == 1 else 1.0
            velocity = self.target_base_vel[j] * scale
            proposed = self.targets[j] + velocity * c.dt
            for axis in range(2):
                if proposed[axis] < 0 or proposed[axis] > c.size:
                    self.target_base_vel[j, axis] *= -1
            velocity = self.target_base_vel[j] * scale
            proposed = self.targets[j] + velocity * c.dt
            self.targets[j] = proposed
            self.target_vel[j] = velocity

    def _resolve_strikes(self, target_ids):
        for j in np.flatnonzero(self.active & ~self.destroyed):
            required = min(2 if self.target_type[j] == 1 else 1,
                           int(self.target_life[j]))
            open_slots = required - int(self.strike_participants[j].sum())
            if open_slots > 0:
                attackers = np.flatnonzero((target_ids == j) & self.agent_active
                                           & ~self.strike_participants[j])
                close = attackers[
                    np.linalg.norm(self.pos[attackers] - self.targets[j], axis=1)
                    <= self.c.strike_range]
                if len(close) > open_slots:
                    # Existing participants keep their slots. Only drones that
                    # arrive together for the remaining slots are sampled.
                    participants = self.allocation_rng.choice(
                        close, size=open_slots, replace=False)
                else:
                    participants = close
                self.strike_participants[j, participants] = True
                self.agent_strike_progress[participants] = 0
            current = self.strike_participants[j].copy()
            if not current.any():
                self.strike_progress[j] = 0
                continue
            # Every drone must serve the full strike duration from its own
            # arrival. Late arrivals do not inherit another drone's progress.
            self.agent_strike_progress[current] += 1
            finishing = current & (
                self.agent_strike_progress >= self.c.strike_steps_per_life)
            if not finishing.any():
                self.strike_progress[j] = int(
                    self.agent_strike_progress[current].max())
                continue
            self.strike_attempts += 1
            self.strike_participants[j, finishing] = False
            self.target_assignment[:, finishing] = False
            if self.combat_rng.random() >= self.c.strike_probability:
                self.agent_active[finishing] = False
                self.agent_strike_progress[finishing] = 0
                remaining = self.strike_participants[j]
                self.strike_progress[j] = (int(
                    self.agent_strike_progress[remaining].max())
                    if remaining.any() else 0)
                continue
            self.strike_hits += 1
            damage = min(self.target_life[j], int(finishing.sum()))
            old_life = int(self.target_life[j])
            self.target_life[j] -= damage
            self.mem_life[:, j] = self.target_life[j]
            damage_value = (target_remaining_value(self.target_type[j], old_life)
                            - target_remaining_value(
                                self.target_type[j], self.target_life[j]))
            self.score += damage_value
            self.last_damage_by_agent[finishing] += damage_value / finishing.sum()
            if self.target_life[j] <= 0:
                self.destroyed[j] = True
                self.active[j] = False
                self.target_destroy_step[j] = self.t
            self.agent_active[finishing] = False
            self.agent_strike_progress[finishing] = 0
            remaining = self.strike_participants[j]
            self.strike_progress[j] = (int(
                self.agent_strike_progress[remaining].max())
                if remaining.any() else 0)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode ended; call reset()")
        proposed_targets = split_strike_action(action)
        if proposed_targets.shape != (self.n,) or not np.issubdtype(proposed_targets.dtype, np.integer) or np.any((proposed_targets < 0) | (proposed_targets >= self.c.n_targets)):
            raise ValueError(f"Expected {self.n} target IDs")
        targets = self.committed_targets(proposed_targets)
        c = self.c
        reward = np.zeros(self.n, dtype=float)
        step_active = self.agent_active.copy()
        agent_count = int(step_active.sum())
        old_pos = self.pos.copy()
        goals = self.resolve_setpoints(targets)
        self.last_goal = goals.copy()
        self.target_assignment.fill(False)
        active_ids = np.flatnonzero(self.agent_active)
        self.target_assignment[targets[active_ids], active_ids] = True
        for i in np.flatnonzero(self.agent_active):
            delta = goals[i] - self.pos[i]
            dist = np.linalg.norm(delta)
            proposed = self.pos[i] + delta * min(1.0, c.strike_speed * c.dt / max(dist, 1e-12))
            self.pos[i] = np.clip(proposed, 0, c.size)
            moved = self.pos[i] - old_pos[i]
            if np.linalg.norm(moved) > 1e-12:
                self.heading[i] = np.arctan2(moved[1], moved[0])
        self.vel = (self.pos - old_pos) / c.dt
        self.t += 1
        score_before = self.score
        # Resolve an approach completed during this decision interval before
        # advancing the target to the next interval.
        self._resolve_strikes(targets)
        self.target_assignment[~self.target_exists | self.destroyed] = False
        # B is fixed by formation 1's initial type/life composition. D is
        # cumulative type-weighted life damage across both formations.
        margin = ((self.formation_one_initial_score - self.score)
                  / max(1, self.formation_one_initial_score))
        team_step_reward = -c.penalty_time * margin
        if score_before < self.formation_one_initial_score <= self.score:
            team_step_reward += c.mission_success_reward
        failure_end = (self.t >= c.horizon
                       and self.score < self.formation_one_initial_score)
        if failure_end:
            team_step_reward -= c.mission_failure_penalty
        if agent_count:
            reward[step_active] += team_step_reward / agent_count
        else:
            # Keep fixed-horizon team rewards observable after every drone has
            # been expended by distributing them over the fixed agent slots.
            reward += team_step_reward / self.n
        self.vel[~self.agent_active] = 0
        self.last_agent_terminated = step_active & ~self.agent_active
        if self.score > 0 and self.first_score_step is None:
            self.first_score_step = self.t
        self.score_area += self.score / max(1, self.formation_one_initial_score)
        self._move_targets()
        self._sense()
        self.discovered |= self.visible.any(axis=0)
        all_targets_destroyed = bool(np.all(~self.target_exists | self.destroyed))
        terminated = False
        truncated = self.t >= c.horizon
        self.done = truncated
        team_reward = float(reward.sum())
        self.rewards_total += team_reward
        self.discounted_rewards_total += (
            c.discount_gamma ** (self.t - 1) * team_reward)
        self.observation = self._obs()
        return self.observation.copy(), reward.astype(np.float32), terminated, truncated, self.metrics()

    def metrics(self):
        initial = self.target_exists
        return dict(team_return=self.rewards_total,
            discounted_team_return=self.discounted_rewards_total,
            score=float(self.score),
            baseline_score=float(self.formation_one_initial_score),
            mission_success=bool(self.score >= self.formation_one_initial_score),
            destroyed_fraction=float(self.destroyed[initial].mean()) if initial.any() else 1.0,
            discovery_fraction=float(self.discovered[self.target_exists].mean()),
            area_coverage=float(self.coverage.any(axis=0).mean()),
            first_detection_step=self.first_detection, active_agents=int(self.agent_active.sum()),
            first_score_step=self.first_score_step,
            score_auc=float(self.score_area / max(1, self.t)),
            targets=int(initial.sum()), success_threshold=float(self.formation_one_initial_score),
            strike_attempts=self.strike_attempts, strike_hits=self.strike_hits,
            agents_expended=int(self.initial_agent_count - self.agent_active.sum()),
            agent_terminated=self.last_agent_terminated.tolist(),
            steps=self.t)

    def snapshot(self):
        return dict(step=self.t, base=self.base.tolist(), drones=self.pos.tolist(), agent_active=self.agent_active.tolist(),
            targets=self.targets.tolist(), target_exists=self.target_exists.tolist(), active=self.active.tolist(),
            destroyed=self.destroyed.tolist(), target_formation=self.target_formation.tolist(),
            target_type=self.target_type.tolist(), target_life=self.target_life.tolist(),
            target_score=[target_remaining_value(typ, life) for typ, life in
                          zip(self.target_type, self.target_life)],
            target_destroy_step=self.target_destroy_step.tolist(),
            visible=self.visible.tolist(),
            goal_positions=self.last_goal.tolist(), strike_progress=self.strike_progress.tolist(),
            agent_strike_progress=self.agent_strike_progress.tolist(),
            target_assignment=self.target_assignment.tolist(),
            strike_participants=self.strike_participants.tolist(), metrics=self.metrics())
