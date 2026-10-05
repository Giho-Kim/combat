import unittest
import numpy as np

from pointmass_rl.env import Config, World
from pointmass_rl.strike_lookahead import (assignment_values, counterfactual_credit,
                                         completion_steps, deadline_action_mask)


class LookaheadCreditTests(unittest.TestCase):
    def setUp(self):
        self.world = World(Config(n_agents=5, min_agents=5))
        self.world.reset(7)
        self.world.pos[:] = [10, 10]
        self.world.targets[:] = [[20, 10], [20, 11], [20, 12], [20, 13], [20, 14]]
        self.world.target_type[:] = [1, 1, 2, 2, 3]
        self.world.target_life[:] = [2, 2, 1, 1, 1]

    def test_capacity_and_inactive_drones(self):
        crowded, spread = assignment_values(self.world, [[0]*5, [0, 0, 1, 1, 2]])
        self.assertLess(crowded, 5.)
        self.assertGreater(spread, 11.8)
        self.world.agent_active[4] = False
        self.assertLess(assignment_values(self.world, [[0, 0, 1, 1, 2]])[0], 10.)

    def test_unreachable_is_not_rewarded_and_state_not_mutated(self):
        before = self.world.pos.copy()
        self.world.targets[0] = [90, 90]
        self.assertLess(assignment_values(self.world, [[0]*5])[0], 1e-6)
        np.testing.assert_array_equal(before, self.world.pos)
        self.assertEqual(self.world.score, 0.)

    def test_counterfactual_baseline_has_zero_policy_mean(self):
        probs = np.full((5, 5), .2)
        credit = []
        for target in range(5):
            credit.append(counterfactual_credit(self.world, [0, 0, 1, 1, target], probs)[4])
        self.assertAlmostEqual(float(np.mean(credit)), 0., places=6)
        self.assertGreater(credit[2], credit[0])

    def test_training_smoke(self):
        from pointmass_rl.strike_ppo import train_mappo
        import torch
        config = Config(n_agents=2, min_agents=2, n_targets=3, min_targets=3,
                        horizon=4, gae_lambda=1.)
        model, _, _, _ = train_mappo(config, 40, rollout_steps=4, n_envs=2,
            epochs=1, eval_interval=0, algorithm='target_coma',
            coma_advantage='sampled_return', lookahead_credit=10.)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_deadline_mask_boundary_fallback_and_locked_progress(self):
        world = self.world
        world.t = 90
        world.targets[0] = world.pos[0] + [world.c.strike_speed, 0]
        world.targets[1] = world.pos[0] + [2 * world.c.strike_speed, 0]
        available = np.ones((5, 5), dtype=bool)
        mask = deadline_action_mask(world, available)
        self.assertTrue(mask[:, 0].all())
        self.assertFalse(mask[:, 1:].any())
        world.t = 99
        np.testing.assert_array_equal(deadline_action_mask(world, available), available)
        world.strike_participants[1, 0] = True
        world.agent_strike_progress[0] = 9
        self.assertEqual(completion_steps(world)[0, 1], 1)
        self.assertEqual(np.flatnonzero(deadline_action_mask(world, available)[0]).tolist(), [1])

    def test_checkpoint_preserves_execution_mask(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from pointmass_rl.strike_ppo import save_checkpoint, model_from_checkpoint
        from pointmass_rl.strike_target_actor import TargetScoringCOMAActorCritic
        model = TargetScoringCOMAActorCritic(5, 5)
        model.training_settings = {'deadline_mask': True}
        self.world.targets[0] = [90, 90]
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_checkpoint(path, model, {})
            restored = model_from_checkpoint(path, self.world.c)
        self.assertFalse(restored.available_actions(self.world)[:, 0].any())

    def test_validation_oracle_accounts_for_capacity_and_reachability(self):
        from scripts.verify_solve123 import physical_optimum
        self.assertEqual(physical_optimum(self.world), 12.)
        self.world.targets[:2] = [90, 90]
        self.assertEqual(physical_optimum(self.world), 5.)
