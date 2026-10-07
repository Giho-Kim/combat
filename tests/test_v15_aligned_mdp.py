import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pointmass_rl.env import Config, World
from pointmass_rl.policies import NearestTargetPolicy


class AlignedMDPTests(unittest.TestCase):
    def test_variable_changes_only_episode_composition(self):
        fixed = asdict(Config.load('configs/five_agents.json'))
        variable = asdict(Config.load('configs/v15_variable.json'))
        for key in ('n_agents', 'min_agents', 'n_targets', 'min_targets',
                    'randomize_counts', 'randomize_target_composition'):
            fixed.pop(key)
            variable.pop(key)
        self.assertEqual(fixed, variable)
        self.assertEqual(variable['horizon'], 100)
        self.assertEqual(variable['discount_gamma'], 1.0)
        self.assertEqual(variable['gae_lambda'], .97)
        self.assertEqual(variable['target_motion_scale'], 0.0)

    def test_reward_is_damage_for_every_step_and_episode(self):
        for path in ('configs/five_agents.json', 'configs/v15_variable.json'):
            for seed in range(10000, 10012):
                with self.subTest(config=path, seed=seed):
                    world = World(Config.load(path))
                    obs = world.reset(seed)
                    policy = NearestTargetPolicy(world.c, seed)
                    if world.c.randomize_counts:
                        f1 = np.count_nonzero(world.target_exists & (world.target_formation == 0))
                        f2 = np.count_nonzero(world.target_exists & (world.target_formation == 1))
                        self.assertGreater(f1, f2)
                        self.assertTrue(5 <= world.initial_agent_count <= 10)
                        self.assertTrue(5 <= world.target_exists.sum() <= 10)
                    total = 0.0
                    while not world.done:
                        before = world.score
                        obs, reward, terminated, truncated, metrics = world.step(policy.predict(obs))
                        self.assertFalse(terminated)
                        self.assertTrue(np.all(reward >= 0))
                        self.assertAlmostEqual(float(reward.sum()), world.score - before, places=5)
                        total += float(reward.sum())
                    self.assertTrue(truncated)
                    self.assertEqual(world.t, 100)
                    self.assertAlmostEqual(total, world.score, places=5)
                    self.assertAlmostEqual(metrics['team_return'], world.score, places=5)
                    self.assertAlmostEqual(metrics['discounted_team_return'], world.score, places=5)

    def test_old_saved_reward_fields_do_not_restore_penalties(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(dict(penalty_time=.1,
                mission_failure_penalty=100., mission_success_reward=100.)))
            config = Config.load(path)
            self.assertFalse(hasattr(config, 'penalty_time'))


if __name__ == '__main__':
    unittest.main()
