"""Training-free, two-option receding-horizon planning for polygon search."""
import numpy as np

from .belief_sensor import detection_probabilities
from .navigation import DubinsPath


class BeliefLookahead:
    """Rank sweep sequences by predicted first-discovery value.

    Carry expected undiscovered value forward by multiplying by the sensor's
    miss probability. This is an open-loop approximation in belief space;
    real observations drive the belief at the next decision.
    """

    def __init__(self, world):
        if not world.c.belief_exploration_only or world.c.belief_representation != 'gaussian_uniform':
            raise ValueError('Belief lookahead requires Gaussian polygon exploration')
        self.world = world
        self.c = world.c
        self.points = (world.belief.points*self.c.size).reshape(-1, 2)
        self.bounds = world.belief.bounds
        self.centers = self.bounds.mean(axis=1)
        self.point_indices = np.arange(len(self.points)).reshape(len(self.bounds), -1)

    def _forecast(self, slot, position, heading, budget):
        """Estimate no-detection probabilities along the controller's route."""
        c = self.c
        stride = 6
        step_distance = c.final_mission_speed*c.dt
        position = position.copy()
        waypoints = self.world._waypoints(slot, position)
        index, path, elapsed = 0, None, 0
        miss = np.ones(len(self.points))
        while elapsed < budget and index < len(waypoints):
            steps = min(stride, budget-elapsed)
            requested = step_distance*steps
            distance = requested
            while distance > 1e-10 and index < len(waypoints):
                destination, end_heading = waypoints[index]
                if path is None:
                    path = DubinsPath(position, heading, destination, end_heading, c.belief_turn_radius)
                position, heading, distance = path.move(position, heading, distance)
                if path.done:
                    index += 1
                    path = None
            steps = max(1, int(np.ceil((requested-distance)/step_distance-1e-9)))
            elapsed += steps
            # Only nearby grid samples can receive sensor likelihood.
            cells = np.all(
                (self.bounds[:, 0] <= position+c.sensor_range)
                & (self.bounds[:, 1] >= position-c.sensor_range), axis=1)
            near = self.point_indices[cells].ravel()
            near = near[np.all(np.abs(self.points[near]-position) <= c.sensor_range, axis=1)]
            if len(near):
                points = self.points[near]
                if c.belief_soft_detection:
                    probability = detection_probabilities(
                        points, [position], [heading], [True], c.sensor_range,
                        c.sensor_fov_deg, c.detection_probability)[0]
                else:
                    delta = points-position
                    bearing = np.arctan2(delta[:, 1], delta[:, 0])-heading
                    angle = np.abs(np.arctan2(np.sin(bearing), np.cos(bearing)))
                    visible = (np.linalg.norm(delta, axis=1) <= c.sensor_range) & (
                        angle <= np.deg2rad(c.sensor_fov_deg)/2)
                    probability = visible*c.detection_probability
                miss[near] *= (1-probability)**steps
        return miss, position, heading, elapsed

    def _shortlist(self, value, position, candidates, width):
        mass = value.reshape(len(self.centers), -1).sum(axis=1)
        distance = np.linalg.norm(self.centers[candidates]-position, axis=1)
        rank = mass[candidates]/(1+distance/self.c.sensor_range)
        return candidates[np.argsort(-rank, kind='stable')[:width]]

    def choose(self, obs):
        world, c = self.world, self.c
        state = world._world
        action = obs['mask'].argmax(axis=1)
        if not obs['decision'].any():
            return action
        weights = world.belief.weights
        # Match the integrated expected count per cell while using the cheaper
        # display grid for within-cell density and the cell's type proportions.
        cell_value = obs['beliefs'][:, 9]*(obs['beliefs'][:, 5:8] @ np.array([5., 2., 1.]))
        totals = weights.sum(axis=1, keepdims=True)
        fractions = np.divide(weights, totals, out=np.full_like(weights, 1/weights.shape[1]), where=totals > 0)
        value = (fractions*cell_value[:, None]).ravel()
        reserved = set(action[obs['active'] & ~obs['decision']])
        remaining = c.horizon-state.t
        for agent in np.flatnonzero(obs['decision']):
            legal = np.flatnonzero(obs['mask'][agent])
            free = np.array([slot for slot in legal if slot not in reserved], dtype=int)
            candidates = free if len(free) else legal
            position, heading = state.pos[agent], state.heading[agent]
            best_score = -np.inf
            for first in self._shortlist(value, position, candidates, 4):
                miss, end, direction, used = self._forecast(first, position, heading, remaining)
                residual = value*miss
                immediate = value @ (1-miss)
                followup = 0.
                if used < remaining:
                    for second in self._shortlist(residual, end, candidates, 3):
                        next_miss, _, _, _ = self._forecast(second, end, direction, remaining-used)
                        followup = max(followup, residual @ (1-next_miss))
                # Discount only the continuation, favoring discoveries from the first sweep.
                score = immediate + 0.7*followup
                if score > best_score:
                    action[agent], best_score, best_miss = first, score, miss
            reserved.add(action[agent])
            value *= best_miss  # Avoid counting overlapping sensor coverage twice.
        return action
