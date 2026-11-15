from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pointmass_rl.env import Config, strike_action, target_remaining_value
from pointmass_rl.exploration import ExplorationActorCritic, PolygonExplorationWorld
from pointmass_rl.belief_policy import stack_observations
from pointmass_rl.belief_training import load_belief_checkpoint, save_belief_checkpoint


class PolygonExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = Config.load(Path(__file__).resolve().parents[1]/'configs/belief_exploration.json')

    def world(self):
        world = PolygonExplorationWorld(self.config)
        world.reset(10000)
        return world

    def test_partition_covers_search_support_without_overlap(self):
        world = self.world()
        bounds = world.belief.bounds
        area = np.prod(bounds[:, 1]-bounds[:, 0], axis=-1).sum()
        self.assertAlmostEqual(area, (self.config.size-2*self.config.belief_search_margin)**2)
        np.testing.assert_allclose(world.polygons.min(axis=(0, 1)), self.config.belief_search_margin)
        np.testing.assert_allclose(world.polygons.max(axis=(0, 1)), self.config.size-self.config.belief_search_margin)
        for i in range(len(bounds)):
            for j in range(i):
                intersection = np.maximum(0, np.minimum(bounds[i, 1], bounds[j, 1])-
                                           np.maximum(bounds[i, 0], bounds[j, 0]))
                self.assertEqual(np.prod(intersection), 0.)
        self.assertEqual(world.observation()['beliefs'].shape, (64, 18))

    def test_first_detection_value_paid_once_without_consumption(self):
        world = self.world()
        hidden = world._world
        hidden.discovered.fill(False)
        life, active = hidden.target_life.copy(), hidden.agent_active.copy()
        target = np.flatnonzero(hidden.target_exists)[0]
        expected = target_remaining_value(hidden.target_type[target], hidden.target_life[target])
        action = strike_action(np.arange(world.n))
        def detect():
            hidden.discovered[target] = True
        with patch.object(world, '_sense', side_effect=detect):
            _, first, _, _, info = world.step(action)
            _, repeated, _, _, _ = world.step(action)
        self.assertEqual(first, expected)
        self.assertEqual(repeated, 0.)
        self.assertEqual(info['consumed_objects'], 0)
        np.testing.assert_array_equal(hidden.target_life, life)
        np.testing.assert_array_equal(hidden.agent_active, active)
        self.assertEqual(world.observation()['mask'].shape[-1], 64)

    def test_completion_unlocks_for_new_selection(self):
        world = self.world()
        action = np.arange(world.n)
        world.step(strike_action(action))
        obs = world.observation()
        self.assertFalse(obs['decision'].any())
        np.testing.assert_array_equal(obs['mask'].sum(-1), 1)
        for i, nav in enumerate(world.navigation):
            nav.update(index=0, path=None,
                       waypoints=[(world._world.pos[i].copy(), world._world.heading[i])])
        world.step(strike_action(action))
        self.assertTrue(all(nav is None for nav in world.navigation))
        self.assertTrue(world.observation()['decision'].all())
        self.assertTrue(world.observation()['mask'].all())

    def test_policy_candidate_permutation_and_checkpoint(self):
        world = self.world()
        model = ExplorationActorCritic()
        obs = stack_observations([world.observation()])
        action = torch.arange(world.n)[None]
        perm = torch.randperm(64)
        inverse = torch.argsort(perm)
        changed = {key: value.clone() for key, value in obs.items()}
        for key in ('beliefs', 'valid'):
            changed[key] = changed[key][:, perm]
        for key in ('mask', 'previous', 'previous_choice'):
            changed[key] = changed[key][:, :, perm]
        torch.testing.assert_close(model.actor_logits(changed), model.actor_logits(obs)[:, :, perm])
        torch.testing.assert_close(model.joint_q(changed, inverse[action]), model.joint_q(obs, action))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            save_belief_checkpoint(path, model, deepcopy(model), self.config, {})
            loaded, config, payload = load_belief_checkpoint(path)
            self.assertTrue(config.belief_exploration_only)
            self.assertFalse(payload['training_settings']['shared_encoder'])
            torch.testing.assert_close(loaded.actor_logits(obs), model.actor_logits(obs))
            torch.testing.assert_close(loaded.joint_q(obs, action), model.joint_q(obs, action))

    def test_real_sensor_detects_outside_selected_polygon(self):
        world = self.world()
        w = world._world
        w.discovered.fill(False)
        w.pos[:] = [50., 49.]
        w.heading[:] = np.pi/2
        w.targets[0] = [50., 50.]
        # Fix only random draws; run the actual FOV and belief update.
        rng = unittest.mock.Mock(wraps=world.sensor_rng)
        rng.random.return_value = np.zeros((world.n, len(w.targets)))
        world.sensor_rng = rng
        with patch.object(world, '_navigate_polygon'):
            obs, reward, _, _, _ = world.step(strike_action(np.zeros(world.n, dtype=int)))
        self.assertTrue(w.discovered[0])
        self.assertGreater(reward, 0)
        self.assertTrue(any(track['handle'] == 0 for track in world.belief.tracks))
        self.assertEqual(obs['beliefs'].shape[0], world.action_count)

    def test_actor_and_critic_receive_finite_gradients(self):
        world = self.world()
        model = ExplorationActorCritic()
        obs = stack_observations([world.observation()])
        dist = model.distribution(obs)
        action = dist.sample()
        q = model.joint_q(obs, action)
        loss = (q-5.).square().mean()-dist.log_prob(action).mean()
        loss.backward()
        for module in (model.actor_body, model.actor_head, model.q_body, model.q_head):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)

    def test_hidden_truth_is_not_an_input(self):
        world = self.world()
        first = world.observation()
        world._world.targets[:] = 99.
        world._world.target_type[:] = 3
        world._world.target_life[:] = 1
        second = world.observation()
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)

    def test_drone_permutation(self):
        world = self.world()
        model = ExplorationActorCritic()
        obs = stack_observations([world.observation()])
        perm = torch.tensor([3, 1, 4, 0, 2])
        changed = {key: value.clone() for key, value in obs.items()}
        for key in ('drones', 'active', 'decision', 'mask', 'previous', 'previous_choice'):
            changed[key] = changed[key][:, perm]
        action = torch.arange(world.n)[None]
        torch.testing.assert_close(model.actor_logits(changed), model.actor_logits(obs)[:, perm])
        torch.testing.assert_close(model.joint_q(changed, action[:, perm]), model.joint_q(obs, action))

    def test_all_discovered_terminates(self):
        world = self.world()
        with patch.object(world, '_sense', side_effect=lambda: world._world.discovered.fill(True)):
            _, _, terminated, _, metrics = world.step(strike_action(np.zeros(world.n, dtype=int)))
        self.assertTrue(terminated)
        self.assertTrue(metrics['mission_success'])


if __name__ == '__main__':
    unittest.main()
