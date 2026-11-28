from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pointmass_rl.belief_policy import choose_action
from pointmass_rl.env import Config, final_mission_action
from pointmass_rl.exploration import ExplorationPPOActorCritic, PolygonExplorationWorld
from pointmass_rl.navigation import DubinsPath, rectangle_entry_distance


class ExplorationDeadlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = replace(Config.load('configs/belief_exploration.json'),
                             n_agents=2, min_agents=2, final_mission_speed=1., randomize_counts=False)

    def world(self):
        world = PolygonExplorationWorld(self.config)
        world.reset(7)
        return world

    def test_straight_path_crossing_counts_before_waypoint_and_preserves_path(self):
        path = DubinsPath(np.array([0., 0.]), 0., np.array([20., 0.]), 0., 2.)
        segments = [segment.copy() for segment in path.segments]
        self.assertAlmostEqual(rectangle_entry_distance(path, [0., 0.], 0., [10., -.1], [11., .1]), 10.)
        self.assertEqual(path.segments, segments)

    def test_circular_entry_and_miss(self):
        path = SimpleNamespace(radius=2., segments=[[1, math.pi]])
        self.assertAlmostEqual(rectangle_entry_distance(path, [0., 0.], 0., [1., 1.], [3., 3.]), 2*math.pi/3)
        self.assertTrue(math.isinf(rectangle_entry_distance(path, [0., 0.], 0., [10., 10.], [11., 11.])))
        right = SimpleNamespace(radius=2., segments=[[-1, math.pi]])
        self.assertAlmostEqual(rectangle_entry_distance(right, [0., 0.], 0., [1., -3.], [3., -1.]), 2*math.pi/3)

    def test_entry_prediction_matches_controller_and_accounts_for_heading(self):
        estimates = []
        for heading in (0., math.pi):
            world = self.world()
            world._world.pos[0] = [0., 10.5]
            world._world.heading[0] = heading
            estimate = world.polygon_entry_steps(0, 0)
            lo, hi = world.belief.bounds[0]
            for actual in range(1, 100):
                world._navigate_polygon(0, 0)
                if np.all(world._world.pos[0] >= lo-1e-9) and np.all(world._world.pos[0] <= hi+1e-9):
                    break
            else:
                self.fail('Controller never entered the selected polygon')
            self.assertEqual(estimate, actual)
            estimates.append(estimate)
        self.assertEqual(estimates[0], 10)
        self.assertGreater(estimates[1], estimates[0])

    def test_deadline_includes_exact_entry_step(self):
        world = self.world()
        world._world.pos[0] = [0., 10.5]
        world._world.heading[0] = 0.
        # Keep another region feasible so the all-unreachable fallback cannot apply.
        with patch.object(world, 'polygon_entry_steps', side_effect=lambda agent, slot: 10 if slot == 0 else 0):
            world._world.t = world.c.horizon-10
            self.assertTrue(world.entry_deadline_mask()[0, 0])
            world._world.t += 1
            self.assertFalse(world.entry_deadline_mask()[0, 0])

    def test_entry_is_sufficient_without_time_for_full_sweep(self):
        world = self.world()
        world._world.pos[0] = [15., 15.]
        world._world.t = world.c.horizon-1
        mask = world.entry_deadline_mask()
        self.assertTrue(mask[0, 0])  # Already inside, although a whole sweep cannot finish.
        self.assertFalse(mask[0, 63])

    def test_locked_sweeps_and_all_unreachable_have_legal_actions(self):
        world = self.world()
        world._world.pos[:] = [0., 0.]
        world._world.t = world.c.horizon-1
        np.testing.assert_array_equal(world.entry_deadline_mask(), world.action_mask()[0])
        world._world.t = 0
        world.step(final_mission_action(np.array([0, 1])))
        world._world.t = world.c.horizon-1
        with patch.object(world, 'polygon_entry_steps', side_effect=AssertionError('Do not interrupt locked sweeps')):
            np.testing.assert_array_equal(world.entry_deadline_mask(), world.action_mask()[0])

    def test_score_reassignment_cannot_restore_deadline_rejected_candidate(self):
        world = self.world()
        obs = world.observation()
        model = ExplorationPPOActorCritic(hidden=16)
        logits = torch.zeros((1, 2, 64))
        logits[0, :, 0] = 100.  # Both prefer this unreachable polygon.
        logits[0, 0, 1], logits[0, 1, 1], logits[0, 1, 2] = 10., 9., 8.
        allowed = np.zeros_like(obs['mask'])
        allowed[:, 1:3] = True
        with patch.object(model, 'actor_logits', return_value=logits):
            raw, _ = choose_action(model, obs, deterministic=True, resolver='none')
            resolved, probabilities = choose_action(model, obs, deterministic=True,
                resolver='score_time_limit', candidate_mask=allowed)
        np.testing.assert_array_equal(raw, [0, 0])
        np.testing.assert_array_equal(resolved, [1, 2])
        self.assertTrue((probabilities[~allowed] == 0).all())
        np.testing.assert_allclose(probabilities.sum(-1), 1.)


if __name__ == '__main__':
    unittest.main()
