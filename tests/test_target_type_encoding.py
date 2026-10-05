import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from pointmass_rl.env import Config, World
from pointmass_rl.strike_ppo import (
    StrikeActorCritic, ScalarTypeStrikeActorCritic, CompactStrikeCOMAActorCritic,
    DecisionSchedule, _model_critic_state, _model_observation,
    actor_observation, critic_state, scheduled_action,
    model_from_checkpoint, save_checkpoint, train_mappo,
)
from pointmass_rl.strike_mat import StrikeMATActorCritic, ScalarTypeStrikeMATActorCritic
from pointmass_rl.strike_target_actor import (
    TargetScoringActorCritic, ScalarTypeTargetScoringActorCritic,
    TargetScoringCOMAActorCritic, LegacyV9TargetScoringCOMAActorCritic,
    LegacyV10TargetScoringCOMAActorCritic,
    LegacyV11TargetScoringCOMAActorCritic,
    DistanceTargetScoringCOMAActorCritic,
    PreviousChoiceActorTargetScoringCOMAActorCritic,
    PreviousProbabilityTargetScoringCOMAActorCritic,
    FullInputTargetScoringCOMAActorCritic,
    FullInputTargetScoringActorCritic, ProgressTargetScoringCOMAActorCritic,
    AliveStatusTargetScoringCOMAActorCritic,
)


class TargetTypeEncodingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = Config(n_agents=5, min_agents=5)
        self.world = World(self.config)
        self.world.reset(7)

    def test_actor_and_critic_type_and_life(self):
        world = self.world
        world.target_type[:] = [1, 2, 3, 1, 2]
        world.target_life[:] = [2, 1, 1, 1, 0]
        expected = np.eye(3)[world.target_type - 1]
        for obs in (actor_observation(world), critic_state(world)):
            records = obs[:, 28:58].reshape(5, 5, 6)
            for agent in range(5):
                np.testing.assert_array_equal(records[agent, :, 2:5], expected)
                np.testing.assert_array_equal(records[agent, :, 5], world.target_life / 2)
        world.target_exists[4] = False
        for obs in (actor_observation(world), critic_state(world)):
            np.testing.assert_array_equal(obs[:, 28:58].reshape(5, 5, 6)[:, 4, 2:5], 0)

    def test_compact_coma_matches_target_coma_information(self):
        scorer = PreviousProbabilityTargetScoringCOMAActorCritic(5, 5)
        compact = CompactStrikeCOMAActorCritic(5, 5)
        probabilities = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        self.world.agent_active[4] = False
        scorer_obs = torch.tensor(scorer.observation(self.world, probabilities))
        compact_obs = compact.observation(self.world, probabilities)
        captured = []
        handle = scorer.actor_body.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().numpy()))
        scorer.actor_logits(scorer_obs)
        handle.remove()
        features = captured[0]
        self.assertEqual(compact_obs.shape, (5, 68))
        for agent in range(5):
            targets = compact_obs[agent, 1:31].reshape(5, 6)
            own = compact_obs[agent, 31:36]
            peers = compact_obs[agent, 36:].reshape(4, 8)
            for target in range(5):
                expected = np.concatenate((targets[target], compact_obs[agent, :1],
                                           own[target:target + 1],
                                           np.column_stack((peers[:, :3], peers[:, 3 + target])).reshape(-1)))
                np.testing.assert_allclose(features[agent, target], expected)
        schedule = DecisionSchedule(self.config)
        np.testing.assert_array_equal(_model_critic_state(compact, self.world, schedule),
                                      _model_critic_state(scorer, self.world, schedule))
        self.assertEqual(compact.state_dim, scorer.state_dim)
        self.assertEqual(compact.q_body[0].normalized_shape,
                         scorer.q_body[0].normalized_shape)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'compact.pt'
            save_checkpoint(path, compact, {})
            restored = model_from_checkpoint(path, self.config)
            np.testing.assert_array_equal(restored.observation(self.world, probabilities),
                                          compact_obs)

    def test_masks_match_legacy_with_occupied_slots(self):
        world = self.world
        world.target_type[:] = [1, 2, 3, 1, 2]
        world.target_life[:] = [2, 1, 1, 1, 0]
        world.strike_participants[0, 0] = True
        world.target_assignment[0, 0] = True
        world.strike_participants[1, 1] = True
        world.target_assignment[1, 1] = True
        old = ScalarTypeStrikeActorCritic(5, 5).available_actions(world)
        for cls in (StrikeActorCritic, StrikeMATActorCritic, TargetScoringActorCritic):
            np.testing.assert_array_equal(cls(5, 5).available_actions(world), old)
        self.assertTrue(old[2, 0])
        self.assertFalse(old[2, 1])
        self.assertFalse(old[2, 4])

    def test_new_and_legacy_checkpoint_roundtrip(self):
        for cls in (StrikeActorCritic, ScalarTypeStrikeActorCritic,
                    StrikeMATActorCritic, ScalarTypeStrikeMATActorCritic,
                    TargetScoringActorCritic, ScalarTypeTargetScoringActorCritic,
                    FullInputTargetScoringActorCritic):
            with self.subTest(architecture=cls.architecture), TemporaryDirectory() as directory:
                model = cls(5, 5)
                path = Path(directory) / 'model.pt'
                save_checkpoint(path, model, {})
                restored = model_from_checkpoint(path, self.config)
                self.assertEqual(restored.one_hot_target_type, model.one_hot_target_type)
                self.assertEqual(restored.state_dim, model.state_dim)
                obs = torch.from_numpy(model.observation(self.world))
                np.testing.assert_array_equal(restored.observation(self.world), obs.numpy())
                mask = torch.from_numpy(model.available_actions(self.world))
                torch.manual_seed(4)
                _, before = model.act(obs, action_mask=mask)
                torch.manual_seed(4)
                _, after = restored.act(obs, action_mask=mask)
                torch.testing.assert_close(before['logp'], after['logp'])
                from pointmass_rl.strike_ppo import _model_critic_state, DecisionSchedule
                state = torch.from_numpy(_model_critic_state(
                    model, self.world, DecisionSchedule(self.config)))[None]
                torch.testing.assert_close(model.values(state), restored.values(state))

    def test_scorer_receives_each_other_drone_position_and_target_probability(self):
        model = TargetScoringActorCritic(5, 5)
        probabilities = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        obs = torch.from_numpy(model.observation(self.world, probabilities))
        captured = []
        handle = model.actor_body.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        model.actor_logits(obs.unsqueeze(0))
        handle.remove()
        features = captured[0][0]
        self.assertEqual(features.shape, (5, 5, 24))
        for agent in range(5):
            others = [i for i in range(5) if i != agent]
            for target in range(5):
                entries = features[agent, target, -16:].reshape(4, 4).numpy()
                np.testing.assert_allclose(entries[:, :2],
                    (self.world.pos[others] - self.world.pos[agent]) / self.config.size,
                    atol=1e-7)
                np.testing.assert_allclose(entries[:, 3], probabilities[others, target])

    def test_actor_idle_status_marks_locked_and_dead_teammates_unavailable(self):
        model = TargetScoringCOMAActorCritic(5, 5)
        self.world.strike_participants[0, 1] = True
        self.world.agent_active[2] = False
        captured = []
        handle = model.actor_body.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        model.actor_logits(torch.tensor(model.observation(self.world)).unsqueeze(0))
        handle.remove()
        # Actor 0 sees teammates 1,2,3,4 in drone-ID order.
        entries = captured[0][0, 0, 0, 9:29].reshape(4, 5)
        torch.testing.assert_close(entries[:, 2], torch.tensor([0., 0., 1., 1.]))

    def test_target_coma_actor_receives_previous_actual_choices(self):
        model = TargetScoringCOMAActorCritic(5, 5)
        schedule = DecisionSchedule(self.config)
        self.assertEqual(model.obs_dim, 123)
        self.assertEqual(model.actor_body[0].normalized_shape, (39,))
        obs = _model_observation(model, self.world, schedule)
        np.testing.assert_array_equal(obs[:, 88:113], 0)
        with torch.no_grad():
            action, _ = scheduled_action(model, self.world, schedule)
        np.testing.assert_array_equal(schedule.targets, action['target'])
        observed_choices = _model_observation(model, self.world, schedule)[0, 88:113]
        np.testing.assert_array_equal(observed_choices.reshape(5, 5).T.argmax(-1),
                                      action['target'])
        schedule.targets[:] = [2, 1, 4, 0, 3]
        obs = _model_observation(model, self.world, schedule)
        captured = []
        handle = model.actor_body.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        model.actor_logits(torch.tensor(obs))
        handle.remove()
        features = captured[0]
        for agent in range(5):
            others = [i for i in range(5) if i != agent]
            for target in range(5):
                self.assertEqual(float(features[agent, target, 8]),
                                 float(schedule.targets[agent] == target))
                peer_flags = features[agent, target, 9:29].reshape(4, 5)[:, 4]
                torch.testing.assert_close(peer_flags, torch.tensor(
                    [float(schedule.targets[i] == target) for i in others]))
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'v12.pt'
            save_checkpoint(path, model, {})
            restored = model_from_checkpoint(path, self.config)
            self.assertEqual(restored.architecture, 'strike_target_coma_v12')
            np.testing.assert_array_equal(_model_observation(restored, self.world, schedule), obs)
            state = torch.from_numpy(_model_critic_state(model, self.world, schedule))[None]
            action = torch.tensor([[0, 1, 2, 3, 4]])
            torch.testing.assert_close(restored.joint_q(state, action), model.joint_q(state, action))
            torch.testing.assert_close(restored.target_joint_q(state, action),
                                       model.target_joint_q(state, action))

    def test_target_coma_v9_checkpoint_still_loads(self):
        model = LegacyV9TargetScoringCOMAActorCritic(5, 5)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'v9.pt'
            save_checkpoint(path, model, {})
            restored = model_from_checkpoint(path, self.config)
        self.assertEqual(restored.architecture, 'strike_target_coma_v9')
        self.assertEqual(restored.q_body[0].normalized_shape, (136,))
        state = torch.zeros(2, restored.state_dim)
        action = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        torch.testing.assert_close(restored.joint_q(state, action), model.joint_q(state, action))

    def test_target_coma_discrete_q_selects_joint_action(self):
        model = LegacyV10TargetScoringCOMAActorCritic(3, 2)
        self.assertEqual(model.q_body[0].normalized_shape, (model.state_dim,))
        self.assertEqual(model.q_head.out_features, 9)
        state = torch.randn(2, model.state_dim)
        action = torch.tensor([[2, 1], [0, 2]])
        torch.testing.assert_close(model.joint_action_index(action), torch.tensor([7, 2]))
        torch.testing.assert_close(model.joint_q(state, action),
                                   model.q_values(state)[torch.arange(2), [7, 2]])
        torch.testing.assert_close(model.target_joint_q(state, action),
                                   model.target_q_values(state)[torch.arange(2), [7, 2]])

    def test_target_coma_discrete_counterfactual_q(self):
        model = LegacyV10TargetScoringCOMAActorCritic(3, 2)
        state = torch.zeros(1, model.state_dim)
        action = torch.tensor([[2, 1]])
        probabilities = torch.tensor([[[.2, .3, .5], [.5, .25, .25]]])
        with torch.no_grad():
            model.q_head.weight.zero_()
            model.q_head.bias.copy_(torch.tensor([
                a + 2 * b for a in range(3) for b in range(3)], dtype=torch.float32))
        torch.testing.assert_close(model.counterfactual_advantages(
            state, action, probabilities), torch.tensor([[.7, .5]]))
        torch.testing.assert_close(model.counterfactual_advantages(
            state, action, probabilities, sampled_returns=torch.tensor([10.])),
            torch.tensor([[6.7, 6.5]]))

    def test_target_coma_v7_checkpoint_still_loads(self):
        with TemporaryDirectory() as directory:
            model = PreviousChoiceActorTargetScoringCOMAActorCritic(5, 5)
            path = Path(directory) / 'v7.pt'
            save_checkpoint(path, model, {})
            restored = model_from_checkpoint(path, self.config)
            self.assertEqual(restored.obs_dim, 113)
            self.assertEqual(restored.state_dim, 76)
            self.assertEqual(restored.q_body[0].normalized_shape, (101,))

    def test_assignment_pooling_routes_only_live_selected_drones(self):
        model = TargetScoringCOMAActorCritic(3, 3)
        embeddings = torch.tensor([[[1., 2.], [3., 4.], [100., 200.]]],
                                  requires_grad=True)
        alive = torch.tensor([[1., 1., 0.]])
        pooled, counts = model.q_body.aggregate_assignments(
            embeddings, alive, torch.tensor([[0, 0, 1]]))
        torch.testing.assert_close(pooled, torch.tensor([[[4., 6.], [0., 0.], [0., 0.]]]))
        torch.testing.assert_close(counts, torch.tensor([[[2.], [0.], [0.]]]))
        pooled.sum().backward()
        torch.testing.assert_close(embeddings.grad, torch.tensor([[[1., 1.], [1., 1.], [0., 0.]]]))

    def test_pair_q_conditions_on_selected_target_and_preserves_target_slots(self):
        model = TargetScoringCOMAActorCritic(5, 5)
        state = torch.from_numpy(_model_critic_state(
            model, self.world, DecisionSchedule(self.config)))[None]
        action = torch.tensor([[0, 0, 1, 2, 3]])
        pairs, embeddings, team_inputs = [], [], []
        handles = [
            model.q_body.pair_encoder.register_forward_pre_hook(
                lambda module, args: pairs.append(args[0].detach())),
            model.q_body.pair_encoder.register_forward_hook(
                lambda module, args, result: embeddings.append(result.detach())),
            model.q_body.team_encoder.register_forward_pre_hook(
                lambda module, args: team_inputs.append(args[0].detach())),
        ]
        try:
            model.joint_q(state, action)
            changed = action.clone()
            changed[0, 1] = 4
            model.joint_q(state, changed)
        finally:
            for handle in handles:
                handle.remove()
        targets = state[:, 21:51].reshape(1, 5, 6)
        drones = state[:, 1:21].reshape(1, 5, 4)
        for index, choices in enumerate((action, changed)):
            selected = targets[:, choices[0]]
            torch.testing.assert_close(pairs[index][..., 16:22], selected)
            torch.testing.assert_close(pairs[index][..., 22:], selected[..., :2] - drones[..., :2])
            torch.testing.assert_close(team_inputs[index][:, :111], state)
            slots = team_inputs[index][:, 111:].reshape(1, 5, 65)
            for target in range(5):
                members = choices[0] == target
                torch.testing.assert_close(slots[:, target, :64], embeddings[index][:, members].sum(1))
                self.assertEqual(slots[0, target, 64].item(), members.sum().item())
        self.assertFalse(torch.allclose(embeddings[0][:, 1], embeddings[1][:, 1]))
        # Moving drone 1 from target 0 to 4 cannot alter other target slots.
        torch.testing.assert_close(team_inputs[0][:, 111:].reshape(1, 5, 65)[:, 1:4],
                                   team_inputs[1][:, 111:].reshape(1, 5, 65)[:, 1:4])

    def test_pair_q_gradients_and_target_soft_update(self):
        model = TargetScoringCOMAActorCritic(5, 5)
        state = torch.from_numpy(_model_critic_state(
            model, self.world, DecisionSchedule(self.config)))[None]
        action = torch.tensor([[0, 0, 1, 2, 3]])
        torch.testing.assert_close(model.joint_q(state, action), model.target_joint_q(state, action))
        before = [p.detach().clone() for p in model.target_q_body.parameters()]
        optimizer = torch.optim.Adam(model.q_body.parameters(), lr=.001)
        (model.joint_q(state, action) - 10).square().mean().backward()
        for encoder in (model.q_body.pair_encoder, model.q_body.team_encoder):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in encoder.parameters()))
        optimizer.step()
        model.update_target(.01)
        for old, online, target in zip(before, model.q_body.parameters(), model.target_q_body.parameters()):
            torch.testing.assert_close(target, .99 * old + .01 * online)
            self.assertFalse(target.requires_grad)
            self.assertIsNone(target.grad)

    def test_target_coma_v11_checkpoint_still_loads(self):
        model = LegacyV11TargetScoringCOMAActorCritic(5, 5)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'v11.pt'
            save_checkpoint(path, model, {})
            restored = model_from_checkpoint(path, self.config)
        self.assertEqual(restored.architecture, 'strike_target_coma_v11')
        state = torch.from_numpy(_model_critic_state(
            model, self.world, DecisionSchedule(self.config)))[None]
        action = torch.tensor([[0, 1, 2, 3, 4]])
        torch.testing.assert_close(restored.joint_q(state, action), model.joint_q(state, action))
        torch.testing.assert_close(restored.target_joint_q(state, action), model.target_joint_q(state, action))

    def test_training_warm_start_preserves_online_and_target_weights(self):
        config = Config(n_agents=2, min_agents=2, n_targets=3, min_targets=3, horizon=4)
        model = TargetScoringCOMAActorCritic(3, 2)
        with torch.no_grad():
            model.target_q_head.bias.fill_(2.)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'initial.pt'
            save_checkpoint(path, model, {})
            with patch('pointmass_rl.strike_ppo._coma_update', return_value={}):
                restored, _, _, _ = train_mappo(
                    config, 8, n_envs=2, rollout_steps=2, epochs=1,
                    eval_interval=0, algorithm='target_coma', entropy_coef=.022,
                    initial_checkpoint=path, target_coma_critic='pair')
            for name, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value)
            self.assertEqual(restored.training_settings['entropy_coef'], .022)
            self.assertTrue(restored.training_settings['optimizer_reinitialized'])
            self.assertEqual(restored.training_settings['initial_checkpoint'], str(path))

    def test_distance_actor_uses_observed_geometry_and_roundtrips(self):
        model = DistanceTargetScoringCOMAActorCritic(5, 5)
        self.world.agent_active[2] = False
        obs = torch.from_numpy(model.observation(self.world))
        captured = []
        handle = model.actor_body.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        model.actor_logits(obs)
        handle.remove()
        self.assertEqual(captured[0].shape, (5, 5, 44))
        distances = np.linalg.norm(self.world.targets[:, None, :] - self.world.pos[None, :, :], axis=-1)
        distances = distances / self.config.size * self.world.agent_active[None, :]
        for agent in range(5):
            order = [agent] + [peer for peer in range(5) if peer != agent]
            np.testing.assert_allclose(captured[0][agent, :, -5:], distances[:, order], atol=1e-7)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'v13.pt'
            save_checkpoint(path, model, {})
            restored = model_from_checkpoint(path, self.config)
        torch.testing.assert_close(restored.actor_logits(obs), model.actor_logits(obs))
        self.assertEqual(restored.architecture, 'strike_target_coma_v13')
        self.assertEqual(restored.q_body.team_encoder[0].normalized_shape, (436,))

    def test_assignment_q_counterfactual_matches_explicit_joint_evaluation(self):
        model = TargetScoringCOMAActorCritic(5, 5)
        state = torch.from_numpy(_model_critic_state(
            model, self.world, DecisionSchedule(self.config)))[None]
        action = torch.tensor([[0, 0, 1, 2, 3]])
        probabilities = torch.full((1, 5, 5), .2)
        actual = model.joint_q(state, action)
        expected = []
        for agent in range(5):
            alternatives = []
            for target in range(5):
                candidate = action.clone()
                candidate[:, agent] = target
                alternatives.append(model.joint_q(state, candidate))
            expected.append(actual - torch.stack(alternatives).mean(0))
        torch.testing.assert_close(model.counterfactual_advantages(state, action, probabilities),
                                   torch.stack(expected, -1), atol=1e-6, rtol=1e-5)
        self.assertGreater(float(torch.stack(expected).detach().abs().max()), 1e-7)
        # Inactive drone actions cannot alter either online or target Q.
        state[:, 1 + 4 * 4 + 2] = 0
        changed = action.clone()
        changed[:, 4] = 4
        torch.testing.assert_close(model.joint_q(state, action), model.joint_q(state, changed))
        torch.testing.assert_close(model.target_joint_q(state, action), model.target_joint_q(state, changed))

    def test_target_coma_v10_checkpoint_still_loads(self):
        model = LegacyV10TargetScoringCOMAActorCritic(5, 5)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'v10.pt'
            save_checkpoint(path, model, {})
            restored = model_from_checkpoint(path, self.config)
        state = torch.zeros(1, model.state_dim)
        torch.testing.assert_close(restored.q_values(state), model.q_values(state))

    def test_target_coma_v8_checkpoint_still_loads_without_velocity(self):
        from pointmass_rl.strike_target_actor import NoVelocityTargetScoringCOMAActorCritic
        model = NoVelocityTargetScoringCOMAActorCritic(5, 5)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'v8.pt'
            save_checkpoint(path, model, {})
            restored = model_from_checkpoint(path, self.config)
        self.assertFalse(restored.drone_velocity)
        self.assertEqual(restored.obs_dim, 113)
        self.assertEqual(restored.state_dim, 101)
        self.assertEqual(restored.actor_body[0].normalized_shape, (29,))
        obs = torch.tensor(restored.observation(self.world))
        torch.testing.assert_close(restored.actor_logits(obs), model.actor_logits(obs))
        self.world.vel[:] = self.config.strike_speed
        np.testing.assert_array_equal(restored.observation(self.world), obs.numpy())

    def test_actual_velocity_reaches_actor_and_q_in_correct_drone_order(self):
        model = TargetScoringCOMAActorCritic(5, 5)
        schedule = DecisionSchedule(self.config)
        velocities = np.array([[1., 0.], [0., -1.], [.3, .4], [-.6, .8], [0., 0.]],
                              dtype=np.float32)
        self.world.vel[:] = velocities * self.config.strike_speed
        self.world.agent_active[2] = False
        expected = velocities.copy()
        expected[2] = 0
        obs = _model_observation(model, self.world, schedule)
        np.testing.assert_allclose(obs[:, -10:], np.tile(expected.reshape(-1), (5, 1)))
        state = _model_critic_state(model, self.world, schedule)
        np.testing.assert_allclose(state[-10:].reshape(5, 2), expected)
        captured = []
        handle = model.actor_body.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        batched_obs = torch.tensor(np.stack([obs, obs]))
        logits = model.actor_logits(batched_obs)
        handle.remove()
        self.assertEqual(tuple(logits.shape), (2, 5, 5))
        for agent in range(5):
            order = [agent] + [peer for peer in range(5) if peer != agent]
            actual = captured[0][:, agent, :, -10:]
            desired = np.broadcast_to(expected[order].reshape(-1), (2, 5, 10))
            np.testing.assert_allclose(actual.numpy(), desired)
        # Velocities do not alter legal-target masks; both observation layouts
        # (raw availability records and velocity-augmented actor records) work.
        np.testing.assert_array_equal(model.target_mask(torch.tensor(obs)).numpy(),
                                      model.available_actions(self.world))

    def test_velocity_is_measured_motion_not_selected_target_direction(self):
        from pointmass_rl.env import strike_action
        model = TargetScoringCOMAActorCritic(5, 5)
        schedule = DecisionSchedule(self.config)
        np.testing.assert_array_equal(model.observation(self.world)[:, -10:], 0)
        before = self.world.pos.copy()
        self.world.step(strike_action(np.arange(5)))
        expected = ((self.world.pos - before) / self.config.dt / self.config.strike_speed)
        observed = _model_observation(model, self.world, schedule)
        np.testing.assert_allclose(observed[0, -10:].reshape(5, 2), expected, atol=1e-7)
        self.assertGreater(float(np.linalg.norm(expected)), 0.)
        schedule.targets[:] = [4, 3, 2, 1, 0]
        changed_choices = _model_observation(model, self.world, schedule)
        np.testing.assert_array_equal(changed_choices[:, -10:], observed[:, -10:])
        # Positions are relative in the actor, velocities stay in world axes.
        self.world.pos += [2, 3]
        self.world.targets += [2, 3]
        translated = _model_observation(model, self.world, schedule)
        np.testing.assert_allclose(translated, changed_choices, atol=1e-7)

    def test_target_coma_v6_checkpoint_still_loads(self):
        with TemporaryDirectory() as directory:
            model = PreviousProbabilityTargetScoringCOMAActorCritic(5, 5)
            path = Path(directory) / 'v6.pt'
            save_checkpoint(path, model, {})
            restored = model_from_checkpoint(path, self.config)
            self.assertEqual(restored.obs_dim, 88)
            self.assertEqual(restored.actor_body[0].normalized_shape, (24,))

    def test_target_coma_v5_checkpoint_keeps_alive_status(self):
        with TemporaryDirectory() as directory:
            model = AliveStatusTargetScoringCOMAActorCritic(5, 5)
            path = Path(directory) / 'legacy.pt'
            save_checkpoint(path, model, {})
            restored = model_from_checkpoint(path, self.config)
            self.assertFalse(restored.actor_idle_status)
            self.assertEqual(restored.actor_body[0].normalized_shape, (24,))

    def test_progress_affects_q_state_but_not_actor_scores(self):
        from pointmass_rl.strike_ppo import _model_critic_state, DecisionSchedule
        model = TargetScoringCOMAActorCritic(5, 5)
        schedule = DecisionSchedule(self.config)
        obs = torch.tensor(model.observation(self.world, schedule.previous_action_probabilities))
        logits = model.actor_logits(obs)
        before = _model_critic_state(model, self.world, schedule)
        self.world.agent_strike_progress[2] = 7
        changed = torch.tensor(model.observation(self.world, schedule.previous_action_probabilities))
        torch.testing.assert_close(model.actor_logits(changed), logits, atol=0, rtol=0)
        after = _model_critic_state(model, self.world, schedule)
        self.assertNotEqual(float(before[1 + 4 * 2 + 3]), float(after[1 + 4 * 2 + 3]))

    def test_inactive_intent_does_not_change_live_scores(self):
        model = TargetScoringActorCritic(5, 5)
        self.world.agent_active[4] = False
        first = np.full((5, 5), .2, dtype=np.float32)
        second = first.copy()
        second[4] = [1, 0, 0, 0, 0]
        logits = [model.actor_logits(torch.tensor(model.observation(self.world, p)))
                  for p in (first, second)]
        torch.testing.assert_close(logits[0][:4], logits[1][:4], atol=0, rtol=0)

    def test_target_coma_critic_intent_alignment_and_episode_reset(self):
        from pointmass_rl.strike_target_actor import TargetScoringCOMAActorCritic
        from pointmass_rl.strike_ppo import _model_critic_state, DecisionSchedule
        model = TargetScoringCOMAActorCritic(5, 5)
        schedule = DecisionSchedule(self.config)
        schedule.previous_action_probabilities[:] = np.eye(5)
        state = _model_critic_state(model, self.world, schedule)
        self.assertEqual(model.state_dim, 111)
        self.assertEqual(model.obs_dim, 123)
        np.testing.assert_allclose(state[51:76], .2)
        np.testing.assert_array_equal(state[76:101], 0)
        self.world.t = 1
        self.world.agent_active[4] = False
        schedule.targets[:] = [2, 1, 4, 0, 3]
        state = _model_critic_state(model, self.world, schedule)
        self.assertEqual(len(state[:51]), 51)
        expected = np.eye(5, dtype=np.float32)
        expected[4] = 0
        np.testing.assert_array_equal(state[51:76].reshape(5, 5), expected.T)
        choices = np.zeros((5, 5), dtype=np.float32)
        choices[np.arange(4), schedule.targets[:4]] = 1
        np.testing.assert_array_equal(state[76:101].reshape(5, 5), choices.T)
        self.assertEqual(model.q_body.team_encoder[0].normalized_shape, (436,))

    def test_compact_critic_ignores_score_and_approach_assignment(self):
        from pointmass_rl.strike_ppo import _model_critic_state, DecisionSchedule
        model = TargetScoringCOMAActorCritic(5, 5)
        schedule = DecisionSchedule(self.config)
        before = _model_critic_state(model, self.world, schedule)
        self.world.score = 3
        self.world.formation_one_initial_score = 12
        self.world.target_assignment[1, 2] = True
        np.testing.assert_array_equal(_model_critic_state(model, self.world, schedule), before)
        self.world.strike_participants[1, 2] = True
        np.testing.assert_array_equal(_model_critic_state(model, self.world, schedule), before)

    def test_target_coma_v2_checkpoint_still_loads(self):
        with TemporaryDirectory() as directory:
            model = FullInputTargetScoringCOMAActorCritic(5, 5)
            path = Path(directory) / 'legacy.pt'
            save_checkpoint(path, model, {})
            restored = model_from_checkpoint(path, self.config)
            self.assertEqual(restored.state_dim, 108)
            torch.testing.assert_close(
                model.actor_logits(torch.from_numpy(model.observation(self.world))),
                restored.actor_logits(torch.from_numpy(restored.observation(self.world))))

    def test_target_coma_v4_checkpoint_still_loads(self):
        with TemporaryDirectory() as directory:
            model = ProgressTargetScoringCOMAActorCritic(5, 5)
            path = Path(directory) / 'legacy.pt'
            save_checkpoint(path, model, {})
            restored = model_from_checkpoint(path, self.config)
            self.assertEqual(restored.actor_body[0].normalized_shape, (28,))
            self.assertEqual(restored.state_dim, 76)
