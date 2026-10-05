import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from pointmass_rl.env import Config, World, strike_action
from pointmass_rl.strike_ppo import (
    StrikeActorCritic, actor_observation, model_from_checkpoint, save_checkpoint,
)
from pointmass_rl.strike_target_actor import TargetScoringActorCritic
from scripts.sweep_200k import training_reward


class ExperimentalActorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = Config.load('configs/default.json')
        self.world = World(self.config)
        self.world.reset(7)
        self.obs = torch.from_numpy(actor_observation(self.world))

    def test_target_permutation_permutes_logits(self):
        model = TargetScoringActorCritic(5, 5)
        order = [2, 0, 4, 1, 3]
        permuted = self.obs.clone()
        start = 3 + 5 * 5
        permuted[:, start:start+30] = self.obs[:, start:start+30].reshape(5, 5, 6)[:, order].reshape(5, 30)
        permuted[:, start+30:-5] = self.obs[:, start+30:-5].reshape(5, 5, 5)[:, order].reshape(5, 25)
        torch.testing.assert_close(model.actor_logits(permuted), model.actor_logits(self.obs)[:, order])

    def test_temperature_and_target_actor_checkpoint_replay(self):
        for cls in (StrikeActorCritic, TargetScoringActorCritic):
            model = cls(5, 5)
            model.training_settings = {'policy_temperature': .2}
            with TemporaryDirectory() as directory:
                path = Path(directory)/'model.pt'
                save_checkpoint(path, model, {})
                restored = model_from_checkpoint(path, self.config)
                torch.testing.assert_close(model.distributions(self.obs).probs,
                                           restored.distributions(self.obs).probs)
            action, stats = model.act(self.obs)
            logp, _ = model.evaluate_actions(self.obs, torch.as_tensor(action['target']), stats['action_mask'])
            torch.testing.assert_close(logp, stats['logp'])

    def test_one_four_changes_reward_but_not_damage(self):
        world = self.world
        original_step = World.step
        world.target_type[0] = 1
        world.target_life[0] = 2
        world.agent_active[:] = False
        with training_reward('one_four'):
            for agent, expected_reward, expected_damage in [(0, .1, 2.5), (1, .4, 5.)]:
                world.agent_active[agent] = True
                world.pos[agent] = world.targets[0]
                world._sense()
                total = 0.
                for _ in range(world.c.strike_steps_per_life):
                    _, reward, *_ = world.step(strike_action(np.zeros(world.n, dtype=int)))
                    total += float(reward.sum())
                self.assertAlmostEqual(total, expected_reward, places=6)
                self.assertEqual(world.score, expected_damage)
        self.assertIs(World.step, original_step)
