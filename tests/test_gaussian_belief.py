from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pointmass_rl.env import Config, strike_action
from pointmass_rl.gaussian_belief import GaussianTargetBelief
from pointmass_rl.belief_env import BeliefWorld
from pointmass_rl.belief_policy import BeliefActorCritic, choose_action, stack_observations
from pointmass_rl.belief_training import train_belief, load_belief_checkpoint, save_belief_checkpoint
from pointmass_rl.belief_report import frame


def empty_sensor(points):
    return np.zeros(len(points))


class GaussianBeliefTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **kwargs):
        return replace(Config.load('configs/belief.json'), **kwargs)

    def belief(self, counts=(1, 1, 1), **kwargs):
        report = dict(position=[50., 50.], type_counts=list(counts),
                      sigma_hint=10., log_sigma_std=.5, member_radius=6.5)
        return GaussianTargetBelief(self.config(**kwargs), report)

    def test_uniform_attributes_and_geometry(self):
        b = GaussianTargetBelief(self.config(belief_background_type_prior=(.6, .3, .1)))
        r = b.records[:b.cells]
        np.testing.assert_allclose(r[:, 9], 1/16)
        np.testing.assert_allclose(r[:, 14], 1/16)
        np.testing.assert_allclose(r[:, 5:8], np.tile([.6, .3, .1], (16, 1)))
        np.testing.assert_allclose(r[:, 8]*2, 1.6)
        np.testing.assert_allclose(r[:, 2], .2**2/12)
        np.testing.assert_allclose(r[:, 4], .2**2/12)
        self.assertTrue(b.valid[:b.cells].all())
        np.testing.assert_equal(r[:, 17], 0)

    def test_mixed_mass_and_attribute_conservation(self):
        b = self.belief()
        r = b.records[:b.cells]
        np.testing.assert_allclose(r[:, 5:8].sum(-1), 1)
        np.testing.assert_allclose((r[:, 5:8]*r[:, 9:10]).sum(0), np.full(3, 4/3), atol=1e-6)
        self.assertAlmostEqual(float(r[:, 9].sum()), 4.)
        np.testing.assert_allclose(r[:, 8]*2, r[:, 5:8]@np.array([2., 1., 1.]))

    def test_duplicate_sighting_does_not_consume_report_twice(self):
        b = self.belief()
        obs = [(987, np.array([50., 50.]), 1, 2)]
        b.update(empty_sensor, obs, 1)
        before = b.subset_probability.copy()
        b.update(empty_sensor, obs, 2)
        np.testing.assert_array_equal(before, b.subset_probability)
        self.assertEqual(len(b.tracks), 1)
        np.testing.assert_equal(b.records[b.cells, [2, 3, 4]], 0)

    def test_simultaneous_order_and_handle_independence(self):
        a, b = self.belief((2, 1, 0)), self.belief((2, 1, 0))
        obs = [(21, np.array([48., 49.]), 1, 2),
               (22, np.array([52., 53.]), 1, 2),
               (23, np.array([50., 51.]), 2, 1)]
        a.update(empty_sensor, obs, 1)
        b.update(empty_sensor, [(i+1000, p, t, l) for i, p, t, l in obs[::-1]], 1)
        np.testing.assert_allclose(a.subset_probability, b.subset_probability)
        np.testing.assert_allclose(a.records[:a.cells], b.records[:b.cells])
        self.assertTrue((a.remaining_types >= 0).all())
        self.assertEqual(len(a.tracks), 3)

    def test_one_report_cannot_be_consumed_twice(self):
        b = self.belief((1, 0, 0))
        b.update(empty_sensor, [(1, np.array([50., 50.]), 1, 2),
                                (2, np.array([50., 50.]), 1, 2)], 1)
        self.assertEqual(len(b.subset_probability), 2)
        self.assertGreaterEqual(b.remaining_types[0], 0)
        self.assertLessEqual(b.remaining_types[0], 1)
        self.assertEqual(len(b.tracks), 2)

    def test_background_only_type_does_not_change_report(self):
        b = self.belief((3, 0, 0))
        before = deepcopy(b.components)
        b.update(empty_sensor, [(7, np.array([50., 50.]), 2, 1)], 1)
        np.testing.assert_equal(b.remaining_types, [3, 0, 0])
        self.assertEqual(b.tracks[0]['initial_report_probability'], 0)
        for x, y in zip(before, b.components):
            np.testing.assert_array_equal(x['mean'], y['mean'])
            np.testing.assert_array_equal(x['covariance'], y['covariance'])

    def test_empty_observation_preserves_known_count_and_uniform(self):
        b = self.belief()
        def sensor(points):
            return .8*np.exp(-np.square(points-[45., 50.]).sum(-1)/8)
        old_mean = b.components[0]['mean'].copy()
        for t in range(10):
            b.update(sensor, [], t)
        np.testing.assert_allclose(b.remaining_types, [1, 1, 1])
        self.assertGreater(b.components[0]['mean'][0], old_mean[0])
        np.testing.assert_allclose(b.records[:b.cells, 14].sum(), 1)
        for component in b.components:
            self.assertGreater(np.linalg.eigvalsh(component['covariance']).min(), 0)

    def test_no_evidence_does_not_move_boundary_gaussian(self):
        b = self.belief()
        for x in b.components:
            x['mean'][:] = 11
        before = deepcopy(b.components)
        b.update(empty_sensor, [], 1)
        for x, y in zip(before, b.components):
            np.testing.assert_array_equal(x['mean'], y['mean'])
            np.testing.assert_array_equal(x['covariance'], y['covariance'])

    def test_uniform_survives_all_reports_removed(self):
        b = self.belief(belief_background_mass=0)
        b.update(empty_sensor, [(i, np.array([50., 50.]), i+1, 2 if i == 0 else 1)
                                for i in range(3)], 1)
        np.testing.assert_equal(b.remaining_types, 0)
        b.background_mass = 1
        b.refresh(1)
        self.assertTrue(b.valid[:b.cells].all())
        np.testing.assert_allclose(b.records[:b.cells, 9].sum(), 1)

    def test_subcells_have_no_effect_and_unobserved_truth_is_hidden(self):
        a, b = BeliefWorld(self.config(belief_subcells=2)), BeliefWorld(self.config(belief_subcells=12))
        a.reset(7); b.reset(7)
        for w in (a, b):
            w._world.pos[:] = 0
        a._world.targets[:] = 80
        b._world.targets[:] = 90
        b._world.target_type[:] = 3
        a._sense(); b._sense()
        for key, value in a.observation().items():
            np.testing.assert_array_equal(value, b.observation()[key])

    def test_actor_critic_training_checkpoint_and_replay(self):
        c = self.config(horizon=3, n_agents=2, min_agents=2)
        with tempfile.TemporaryDirectory() as out:
            _, episodes, _, steps = train_belief(c, 16, out, n_envs=1, rollout_steps=4,
                epochs=1, critic_epochs=1, minibatch_size=4, eval_interval=0)
            self.assertTrue(episodes)
            self.assertGreaterEqual(steps, 16)
            model, saved, _ = load_belief_checkpoint(Path(out)/'latest.pt')
            self.assertEqual(model.features, 18)
            w = BeliefWorld(saved); obs = w.reset(10000)
            while not w.done:
                action, probability = choose_action(model, obs, True, 'score')
                q = model.joint_q(stack_observations([obs]), torch.as_tensor(action[None]))
                self.assertTrue(torch.isfinite(q).all())
                obs, *_ = w.step(strike_action(action), probability)
            self.assertEqual(frame(w, obs)['background_search_mass'], 1.)
            with self.assertRaises(ValueError):
                load_belief_checkpoint(Path(out)/'latest.pt', replace(c, belief_representation='histogram'))
            wrong = BeliefActorCritic()
            with self.assertRaises(ValueError):
                save_belief_checkpoint(Path(out)/'bad.pt', wrong, deepcopy(wrong), c, {})

    def test_type_prior_validation(self):
        for prior in ((0, 0, 0), (.5, .5), (1.1, -.1, 0), (np.nan, 0, 1)):
            with self.assertRaises(ValueError):
                self.config(belief_background_type_prior=prior)
