import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pointmass_rl.belief_env import BeliefWorld, TargetBelief
from pointmass_rl.belief_policy import BeliefActorCritic, choose_action, stack_observations
from pointmass_rl.belief_training import command_evaluate, load_belief_checkpoint, train_belief, update_model
from pointmass_rl.cli import command_config
from pointmass_rl.env import Config, World, strike_action


class BeliefTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **kwargs):
        return replace(Config(mode='belief', n_agents=3, min_agents=3), **kwargs)

    def test_multiple_tracks_stable_slots_and_uncertainty(self):
        b = TargetBelief(self.config())
        zero = np.zeros_like(b.weights)
        observations = [(12, np.array([5., 5.]), 1, 2),
                        (7, np.array([6., 5.]), 2, 1),
                        (23, np.array([7., 5.]), 3, 1)]
        b.update(zero, observations, 0)
        self.assertEqual(len(b.tracks), 3)
        original = b.records[b.cells:b.cells+3].copy()
        b.update(zero, observations[::-1], 1)
        self.assertEqual([t['handle'] for t in b.tracks], [12, 7, 23])
        np.testing.assert_equal(b.records[b.cells:b.cells+3, :2], original[:, :2])
        self.assertTrue((b.records[b.cells:b.cells+3, 2] < original[:, 2]).all())
        self.assertTrue(b.valid[:b.cells].all())

    def test_negative_evidence_only_updates_observed_support(self):
        b = TargetBelief(self.config())
        before = b.weights.copy()
        detection = np.zeros_like(before)
        detection[0] = .5
        b.update(detection, [], 0)
        np.testing.assert_equal(b.weights[0], before[0]*.5)
        np.testing.assert_equal(b.weights[1:], before[1:])
        b.update(np.ones_like(before), [(1, np.array([5., 5.]), 2, 1)], 1)
        self.assertFalse(b.valid[:b.cells].any())
        self.assertTrue(b.valid[b.cells])

    def test_hidden_truth_does_not_enter_observation(self):
        w = BeliefWorld(self.config(sensor_range=.01))
        w.reset(41)
        w._world.pos[:] = 0
        w._world.targets[:] = 80
        w._sense()
        first = w.observation()
        w._world.targets[:] = 90
        w._world.target_type[:] = 1
        w._world.target_life[:] = 2
        w._sense()
        second = w.observation()
        self.assertEqual(len(w.belief.tracks), 0)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)
        model = BeliefActorCritic()
        a = torch.zeros((1, w.n), dtype=torch.long)
        torch.testing.assert_close(model.joint_q(stack_observations([first]), a),
                                   model.joint_q(stack_observations([second]), a))

    def test_sensor_reproducibility_and_detection(self):
        c = self.config(sensor_range=200, sensor_fov_deg=360)
        a, b = BeliefWorld(c), BeliefWorld(c)
        oa, ob = a.reset(8), b.reset(8)
        for key in oa:
            np.testing.assert_equal(oa[key], ob[key])
        self.assertEqual(len(a.belief.tracks), c.n_targets)
        self.assertFalse(oa['valid'][:a.belief.cells].any())

    def test_configured_formation_layout_is_spread_and_random(self):
        c = self.config(formation_layout='configured', formation_separation=60,
                        formation_distance_spread=20, formation_two_progress=.6,
                        formation_two_lateral_offset=22)
        centers = []
        for seed in range(30):
            world = BeliefWorld(c)
            world.reset(seed)
            friendly = world._world.friendly_center
            first, second = world._world.formation_centers
            route = first-friendly
            distance = np.linalg.norm(route)
            along = route/distance
            lateral = np.array([-along[1],along[0]])
            self.assertGreaterEqual(distance,60)
            self.assertLessEqual(distance,80)
            self.assertAlmostEqual(float(np.dot(second-friendly,along)),.6*distance)
            self.assertAlmostEqual(abs(float(np.dot(second-friendly,lateral))),22)
            self.assertGreater(np.linalg.norm(second-friendly),40)
            self.assertGreater(np.linalg.norm(first-second),30)
            centers.append(first)
        self.assertGreater(np.ptp(np.asarray(centers),axis=0).min(),30)

    def test_anonymous_choice_cannot_execute_undetected_task(self):
        w = BeliefWorld(self.config(sensor_range=.01))
        w.reset(7)
        j = int(np.flatnonzero(w.belief.valid)[0])
        w._world.pos[:] = w.belief.goals[j]
        w._world.targets[:] = w.belief.goals[j]
        # No identified track exists before this step, even at the same point.
        self.assertEqual(len(w.belief.tracks), 0)
        _, reward, _, _, _ = w.step(strike_action(np.full(w.n, j)))
        self.assertEqual(reward, 0)
        self.assertFalse(w._world.strike_participants.any())

    def test_task_completion_report_without_surviving_observer(self):
        c = self.config(n_agents=1, min_agents=1, sensor_range=200,
                        sensor_fov_deg=360, position_noise=0, strike_steps_per_life=1)
        w = BeliefWorld(c)
        w.reset(4)
        i = next(i for i, track in enumerate(w.belief.tracks) if track['life'] == 1)
        j = w.belief.cells+i
        w._world.pos[0] = w.belief.goals[j]
        obs, reward, _, _, _ = w.step(strike_action(np.array([j])))
        self.assertGreater(reward, 0)
        self.assertFalse(obs['active'].any())
        self.assertFalse(obs['valid'][j])
        self.assertEqual(w.belief.tracks[i]['life'], 0)
        self.assertTrue(obs['mask'].any(-1).all())

    def test_shared_encoder_receives_actor_and_q_gradients(self):
        w = BeliefWorld(self.config())
        b = stack_observations([w.reset(4)])
        m = BeliefActorCritic()
        action = m.distribution(b).sample()
        for objective in ('actor', 'q'):
            m.zero_grad(set_to_none=True)
            loss = (-m.distribution(b).log_prob(action).mean() if objective == 'actor'
                    else m.joint_q(b, action).sum())
            loss.backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.belief_encoder.parameters()), 0)
        m.zero_grad(set_to_none=True)
        m.joint_q(b, action, detach_encoder=True).sum().backward()
        self.assertTrue(all(p.grad is None for p in m.belief_encoder.parameters()))
        parameters = list(m.parameters())
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))

    def test_slot_and_drone_permutation(self):
        w = BeliefWorld(self.config())
        b = stack_observations([w.reset(3)])
        model = BeliefActorCritic()
        action = model.distribution(b).sample()
        k = b['beliefs'].shape[1]
        permutation = torch.randperm(k)
        reverse = torch.argsort(permutation)
        changed = {key: value.clone() for key, value in b.items()}
        for key in ('beliefs', 'valid'):
            changed[key] = changed[key][:, permutation]
        for key in ('previous', 'previous_choice', 'mask'):
            changed[key] = changed[key][:, :, permutation]
        torch.testing.assert_close(model.actor_logits(changed), model.actor_logits(b)[:, :, permutation])
        torch.testing.assert_close(model.joint_q(changed, reverse[action]), model.joint_q(b, action))
        p = torch.tensor([2, 0, 1])
        for key in ('drones', 'previous', 'previous_choice', 'mask', 'decision', 'active'):
            changed[key] = changed[key][:, p]
        torch.testing.assert_close(model.actor_logits(changed), model.actor_logits(b)[:, p][:, :, permutation])
        torch.testing.assert_close(model.joint_q(changed, reverse[action][:, p]), model.joint_q(b, action))

    def test_resolvers_and_variable_capacities(self):
        model = BeliefActorCritic()
        for n, t, g in ((1, 1, 2), (3, 5, 4), (8, 12, 3)):
            w = BeliefWorld(self.config(n_agents=n, min_agents=n, n_targets=t,
                min_targets=t, belief_grid=g, sensor_range=200, sensor_fov_deg=360))
            obs = w.reset(7)
            for resolver in ('none', 'score', 'distance'):
                action, prob = choose_action(model, obs, True, resolver)
                self.assertTrue(obs['mask'][np.arange(n), action].all())
                self.assertTrue(np.isfinite(prob).all())
                q = model.joint_q(stack_observations([obs]), torch.tensor(action)[None])
                self.assertTrue(torch.isfinite(q).all())

    def test_known_default_and_mode_guard(self):
        self.assertEqual(Config().mode, 'known')
        with self.assertRaises(ValueError):
            World(self.config())
        with self.assertRaises(ValueError):
            BeliefWorld(self.config(target_motion_scale=1))
        args = argparse.Namespace(config=None, mode='belief', model=None)
        self.assertEqual(command_config(args).mode, 'belief')
        self.assertEqual(command_config(args).n_agents, 5)

    def test_empty_belief_and_dead_agents_remain_finite(self):
        w = BeliefWorld(self.config())
        w.reset(7)
        w.belief.weights.fill(0)
        w.belief.refresh(0)
        w._world.agent_active.fill(False)
        b = stack_observations([w.observation()])
        m = BeliefActorCritic()
        dist = m.distribution(b)
        a = dist.sample()
        self.assertTrue(torch.isfinite(dist.probs).all())
        self.assertTrue(torch.isfinite(m.joint_q(b, a)).all())
        self.assertTrue(torch.isfinite(m.advantages(b, a, dist.probs)).all())
        self.assertFalse(b['decision'].any())

    def test_counterfactual_matches_direct_enumeration(self):
        w = BeliefWorld(self.config(n_agents=1, min_agents=1, belief_grid=1))
        b = stack_observations([w.reset(7)])
        # Add a second valid public candidate to exercise an actual alternative.
        b['valid'][0, 1] = True
        b['beliefs'][0, 1] = b['beliefs'][0, 0]
        b['beliefs'][0, 1, 0] += .2
        b['mask'][0, 0, 1] = True
        m = BeliefActorCritic()
        dist = m.distribution(b)
        action = dist.sample()
        actual = m.joint_q(b, action)
        expected = actual.clone()
        for j in b['mask'][0, 0].nonzero().flatten():
            alternative = torch.tensor([[int(j)]])
            expected -= dist.probs[0, 0, j]*m.joint_q(b, alternative)
        torch.testing.assert_close(m.advantages(b, action, dist.probs)[:, 0], expected)

    def test_q_warmup_does_not_change_behavior_policy(self):
        w = BeliefWorld(self.config())
        b = stack_observations([w.reset(7)])
        m = BeliefActorCritic()
        target = deepcopy(m).requires_grad_(False)
        dist = m.distribution(b)
        action = dist.sample()
        before = m.actor_logits(b).detach().clone()
        optimizer = torch.optim.Adam(m.parameters(), lr=.001)
        update_model(m, target, optimizer, b, action, dist.log_prob(action).detach(),
                     dist.probs.detach(), torch.ones(1), epochs=0, critic_epochs=2,
                     width=1, entropy_coef=.01, clip=.2)
        torch.testing.assert_close(m.actor_logits(b), before, rtol=0, atol=0)

    def test_decision_interval_and_random_active_counts(self):
        w = BeliefWorld(self.config(decision_interval=3, randomize_counts=True,
                                   min_agents=1, min_targets=1))
        obs = w.reset(7)
        m = BeliefActorCritic()
        action, prob = choose_action(m, obs)
        obs, *_ = w.step(strike_action(action), prob)
        held = obs['active'] & obs['valid'][action]
        self.assertFalse(obs['decision'][held].any())
        for i in np.flatnonzero(held):
            self.assertEqual(np.flatnonzero(obs['mask'][i]).tolist(), [int(action[i])])

    def test_training_checkpoint_and_terminal_rollouts(self):
        c = self.config(horizon=3, belief_grid=2)
        with tempfile.TemporaryDirectory() as directory:
            model, episodes, evaluations, steps = train_belief(c, 48, directory,
                n_envs=2, rollout_steps=4, epochs=1, critic_epochs=1, minibatch_size=4,
                eval_interval=0, checkpoint_interval=12)
            self.assertGreaterEqual(steps, 48)
            self.assertTrue(episodes)
            loaded, config, payload = load_belief_checkpoint(Path(directory)/'latest.pt')
            self.assertEqual(config, c)
            self.assertIn('optimizer_state_dict', payload)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key])
                self.assertTrue(torch.isfinite(value).all())
            other = BeliefWorld(replace(c, n_agents=2, min_agents=2, n_targets=7, min_targets=7))
            obs = other.reset(17)
            action, prob = choose_action(loaded, obs)
            other.step(strike_action(action), prob)
            evaluation = Path(directory)/'eval'
            command_evaluate(argparse.Namespace(model=str(Path(directory)/'latest.pt'),
                episodes=2, seed=30, out=str(evaluation)), c)
            html = (evaluation/'replay.html').read_text()
            payload = json.loads(html.split('const data=',1)[1].split(',sel=document',1)[0])
            self.assertEqual(set(payload['runs']),
                             {'random','nearest','type_priority','none','score','distance'})
            self.assertEqual(payload['seed'],31)
            self.assertTrue(all(len(frames)==c.horizon+1 for frames in payload['runs'].values()))
            self.assertIn('id="truth" type="checkbox"',html)
            self.assertEqual(sorted({item['formation'] for item in
                payload['runs']['none'][0]['truth']}),[1,2])
            self.assertEqual(len(payload['runs']['none'][0]['formation_centers']),2)
            self.assertTrue(all(frame['chosen'] is None for frame in
                                [frames[0] for frames in payload['runs'].values()]))
            with self.assertRaisesRegex(ValueError, 'Existing experiment'):
                train_belief(c, 1, directory)


if __name__ == '__main__':
    unittest.main()
