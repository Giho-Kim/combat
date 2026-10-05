import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from pointmass_rl.env import Config, World
from pointmass_rl.strike_mat import StrikeMATActorCritic
from pointmass_rl.strike_ppo import (
    DecisionSchedule, critic_state, model_from_checkpoint, save_checkpoint,
    train_mappo,
)


class MATTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.config = Config(n_agents=5, min_agents=5)
        self.world = World(self.config)
        self.world.reset(10001)
        self.model = StrikeMATActorCritic(5, 5)
        self.obs = torch.from_numpy(self.model.observation(self.world))
        self.mask = torch.from_numpy(self.model.available_actions(self.world))

    def test_sequential_and_parallel_log_probabilities_match(self):
        observations = torch.stack([self.obs, self.obs])
        with torch.no_grad():
            for deterministic in (False, True):
                action, stats = self.model.act(observations, deterministic,
                                               action_mask=torch.stack([self.mask, self.mask]))
                logp, entropy = self.model.evaluate_actions(
                    observations, torch.from_numpy(action['target']), stats['action_mask'])
                torch.testing.assert_close(logp, stats['logp'], atol=1e-6, rtol=1e-5)
                self.assertTrue(torch.isfinite(entropy).all())

    def test_future_actions_do_not_leak_and_prefix_affects_later_logits(self):
        targets = torch.zeros(5, dtype=torch.long)
        changed = targets.clone()
        changed[2] = 1
        before = self.model.actor_logits(self.obs, targets)
        after = self.model.actor_logits(self.obs, changed)
        torch.testing.assert_close(before[:3], after[:3], atol=1e-7, rtol=0)
        self.assertGreater(float((before[3:] - after[3:]).abs().max().detach()), 1e-8)

    def test_masks_allow_shared_targets_and_force_inactive_and_locked_actions(self):
        mask = torch.zeros((5, 5), dtype=torch.bool)
        mask[:, 2] = True
        action, stats = self.model.act(self.obs, action_mask=mask)
        np.testing.assert_array_equal(action['target'], np.full(5, 2))
        self.world.agent_active[1] = False
        self.world.strike_participants[2, 0] = True
        self.world.target_assignment[2, 0] = True
        observations = torch.from_numpy(self.model.observation(self.world))
        available = self.model.available_actions(self.world)
        scheduled, _ = DecisionSchedule(self.config).mask(self.world, available)
        action, stats = self.model.act(observations, action_mask=torch.from_numpy(scheduled))
        self.assertEqual(action['target'][0], 2)
        self.assertEqual(action['target'][1], 0)
        logp, entropy = self.model.evaluate_actions(
            observations, torch.from_numpy(action['target']), stats['action_mask'])
        torch.testing.assert_close(logp, stats['logp'], atol=1e-6, rtol=1e-5)
        self.assertEqual(float(entropy[1].detach()), 0)

    def test_training_grouped_minibatches_and_checkpoint_roundtrip(self):
        for minibatch_size in (None, 3):
            with self.subTest(minibatch_size=minibatch_size):
                torch.manual_seed(7)
                initial = StrikeMATActorCritic(5, 5)
                model, _, evaluations, completed = train_mappo(
                    self.config, 60, rollout_steps=4, n_envs=2, epochs=1,
                    minibatch_size=minibatch_size, eval_interval=0, algorithm='mat')
                self.assertGreaterEqual(completed, 60)
                self.assertFalse(evaluations)
                self.assertTrue(any(not torch.equal(before, after) for before, after in
                                    zip(initial.actor_body.parameters(), model.actor_body.parameters())))
                self.assertTrue(any(not torch.equal(before, after) for before, after in
                                    zip(initial.critic_body.parameters(), model.critic_body.parameters())))
                self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
                with TemporaryDirectory() as directory:
                    path = Path(directory) / 'mat.pt'
                    save_checkpoint(path, model, {})
                    loaded = model_from_checkpoint(path, self.config)
                    self.assertIsInstance(loaded, StrikeMATActorCritic)
                    with torch.no_grad():
                        expected, _ = model.act(self.obs, deterministic=True, action_mask=self.mask)
                        actual, _ = loaded.act(self.obs, deterministic=True, action_mask=self.mask)
                    np.testing.assert_array_equal(expected['target'], actual['target'])

    def test_global_state_projection_preserves_assignments_and_distinguishes_agents(self):
        self.assertEqual(self.model.obs_dim, 83)
        np.testing.assert_array_equal(self.obs.numpy(), critic_state(self.world))
        self.world.target_assignment[2, 0] = True
        updated = self.model.observation(self.world)
        self.assertEqual(updated[0, -25:].reshape(5, 5)[2, 0], 1)
        self.assertFalse(hasattr(self.model.actor_body, 'encoder_attention'))
        with torch.no_grad():
            representation = self.model.actor_body.encode(self.obs)
        self.assertEqual(representation.shape, (5, 64))
        self.assertFalse(torch.equal(representation[0], representation[1]))

    def test_global_slot_mask_is_preserved_for_distant_drone(self):
        target = int(np.flatnonzero(self.world.target_type != 1)[0])
        self.world.strike_participants[target, 0] = True
        self.world.target_assignment[target, 0] = True
        available = self.model.available_actions(self.world)
        self.assertTrue(available[0, target])
        self.assertFalse(available[1:, target].any())


if __name__ == '__main__':
    unittest.main()
