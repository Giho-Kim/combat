import unittest

import numpy as np

from pointmass_rl.env import Config, World, strike_action
from pointmass_rl.strike_guidance import progress_potential, avoidable_switches


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        self.world = World(Config(n_agents=5, min_agents=5))
        self.world.reset(7)
        self.world.targets[:] = [[20, 20], [30, 20], [40, 20], [50, 20], [60, 20]]
        self.world.pos[:] = [10, 20]
        self.world.target_type[:] = [1, 2, 3, 1, 2]
        self.world.target_life[:] = [2, 1, 1, 2, 1]
        self.world.target_assignment.fill(False)
        self.world.target_assignment[0, 0] = True

    def test_progress_deadline_capacity_and_terminal(self):
        w = self.world
        before = progress_potential(w)
        w.pos[0, 0] += 1
        self.assertGreater(progress_potential(w), before)
        w.t = 99
        self.assertLess(progress_potential(w), before)
        w.t = 0
        w.pos[:] = [10, 20]
        w.target_assignment[0] = True
        self.assertAlmostEqual(progress_potential(w), 2 * before)
        w.done = True
        self.assertEqual(progress_potential(w), 0)

    def test_switches_exempt_first_dead_or_unreachable_assignment(self):
        w = self.world
        targets = np.ones(w.n, dtype=int)
        self.assertEqual(avoidable_switches(w, targets), 1)
        w.t = 99
        self.assertEqual(avoidable_switches(w, targets), 0)
        w.t = 0
        w.destroyed[0] = True
        self.assertEqual(avoidable_switches(w, targets), 0)
        w.target_assignment.fill(False)
        self.assertEqual(avoidable_switches(w, targets), 0)

    def test_discounted_shaping_telescopes_through_completion(self):
        w = self.world
        initial = progress_potential(w)
        total = 0.0
        for step in range(w.c.horizon):
            old = progress_potential(w)
            w.step(strike_action(np.array([0, 0, 1, 3, 3])))
            shaping = w.c.discount_gamma * progress_potential(w) - old
            total += w.c.discount_gamma ** step * shaping
        self.assertAlmostEqual(total, -initial, places=7)
