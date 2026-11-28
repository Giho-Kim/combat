import unittest

import numpy as np

from pointmass_rl.env import (Config, EVALUATION_TASKS, SELF_FEATURES, SUCCESS_THRESHOLDS,
                              TARGET_FEATURES, World, final_mission_action,
                              target_remaining_value)
from pointmass_rl.policies import (ApproximateDPPolicy, HeuristicPolicy, NearestTargetPolicy,
                                   RandomPolicy, TypePriorityPolicy)


def compact(**kwargs):
    values = dict(n_agents=3, min_agents=3, n_targets=2, min_targets=2,
                  randomize_counts=False, horizon=20)
    values.update(kwargs)
    return Config(**values)


def action(world, target=0):
    return final_mission_action(np.full(world.n, target, dtype=int))


class WorldTests(unittest.TestCase):
    def test_target_count_can_exceed_available_agents(self):
        c = Config(n_agents=1, min_agents=1, n_targets=2, min_targets=2)
        self.assertEqual(c.n_targets, 2)

    def test_final_mission_action_is_the_only_schema(self):
        w = World(compact())
        obs = w.reset(12)
        self.assertEqual(w.obs_dim, 3 + 4 * w.c.n_targets)
        self.assertFalse(hasattr(w, "selected_target"))
        result = w.step(action(w))
        self.assertEqual(obs.shape, (w.n, w.obs_dim))
        self.assertEqual(result[1].shape, (w.n,))
        with self.assertRaisesRegex(ValueError, "only target"):
            w.step({"task": np.zeros(w.n, dtype=int), "target": np.zeros(w.n, dtype=int)})

    def test_policy_keeps_final_mission_lock_without_previous_target_observation(self):
        w = World(compact(n_agents=1, min_agents=1))
        obs = w.reset(12)
        policy = RandomPolicy(w.c, seed=12)
        policy.locked_target[0] = 1
        obs[0, 1] = 0.5
        self.assertEqual(int(policy.predict(obs)["target"][0]), 1)

    def test_assignment_is_separate_from_final_mission_participation(self):
        w = World(compact(n_agents=1, min_agents=1, n_targets=1, min_targets=1,
                          final_mission_range=1, service_steps_per_unit=2))
        w.reset(12)
        w.pos[0] = w.targets[0] + [10, 0]
        w.step(action(w))
        self.assertTrue(w.target_assignment[0, 0])
        self.assertFalse(w.final_mission_participants[0, 0])
        w.pos[0] = w.targets[0]
        w.step(action(w))
        self.assertTrue(w.target_assignment[0, 0])
        self.assertTrue(w.final_mission_participants[0, 0])
        w.step(action(w))
        self.assertFalse(w.target_assignment.any())
        self.assertFalse(w.final_mission_participants.any())

    def test_unavailable_target_choice_is_an_idle_step(self):
        w = World(compact(n_agents=1, min_agents=1, n_targets=2, min_targets=2))
        w.reset(12)
        w.target_work_units[0] = 0
        w.target_completed[0] = True
        w.active[0] = False
        before = w.pos.copy()
        _, reward, _, _, _ = w.step(action(w, 0))
        np.testing.assert_array_equal(w.pos, before)
        self.assertFalse(w.target_assignment.any())
        self.assertAlmostEqual(float(reward.sum()), 0.0)

    def test_target_is_locked_until_final_mission_action_completes(self):
        w = World(compact(final_mission_probability=0))
        w.reset(13)
        w.target_type[0] = 2
        w.step(action(w, 0))
        w.step(action(w, 1))
        np.testing.assert_array_equal(w.locked_targets(), np.full(w.n, -1))
        w.pos[0] = w.targets[0]
        w.mem_pos[:, 0] = w.targets[0]
        w.step(action(w, 0))
        self.assertEqual(w.locked_targets()[0], 0)
        for _ in range(w.c.service_steps_per_unit - 1):
            w.step(action(w, 1))
        self.assertFalse(w.agent_active[0])
        self.assertEqual(w.locked_targets()[0], -1)
        w.coverage[0, 0, 0] = True
        w.step(action(w, 1))
        np.testing.assert_array_equal(w.vel[0], [0, 0])
        self.assertGreaterEqual(w.metrics()['area_coverage'], .01)

    def test_seed_shape_bounds_and_counts(self):
        c = Config(horizon=5)
        a, b = World(c), World(c)
        np.testing.assert_array_equal(a.reset(10), b.reset(10))
        self.assertGreaterEqual(a.agent_active.sum(), 3)
        self.assertLessEqual(a.agent_active.sum(), 10)
        self.assertGreaterEqual(a.target_exists.sum(), 3)
        self.assertGreaterEqual(a.agent_active.sum(), a.target_exists.sum())
        for _ in range(c.horizon):
            oa, ra, ta, ua, _ = a.step(action(a))
            ob, rb, tb, ub, _ = b.step(action(b))
            np.testing.assert_array_equal(oa, ob)
            np.testing.assert_array_equal(ra, rb)
            self.assertTrue(np.isfinite(oa).all())
            self.assertTrue(((a.pos >= 0) & (a.pos <= c.size)).all())
            if ta or ua:
                break

    def test_default_horizon_and_start_distance(self):
        c = Config()
        self.assertEqual(c.n_agents, 6)
        self.assertEqual(c.gae_lambda, 0.97)
        self.assertEqual(c.horizon, 100)
        self.assertFalse(c.randomize_counts)
        w = World(c)
        w.reset(16)
        self.assertEqual(int(w.agent_active.sum()), c.n_agents)
        self.assertEqual(int(w.target_exists.sum()), c.n_targets)
        per_work_units_values = []
        for j in np.flatnonzero(w.target_exists):
            if w.target_type[j] == 1:
                per_work_units_values.extend([2, 3])
            else:
                per_work_units_values.append(w.target_score[j])
        achievable = sum(sorted(per_work_units_values, reverse=True)[:c.n_agents])
        self.assertGreater(achievable, w.formation_one_initial_score)
        self.assertEqual(len(w.formation_centers), 2)
        np.testing.assert_array_equal(w.target_formation, [0, 0, 0, 1, 1])
        route = w.formation_centers[0] - w.agent_start_center
        distance = np.linalg.norm(route)
        self.assertGreaterEqual(distance, c.formation_separation - c.formation_distance_spread)
        self.assertLessEqual(distance, c.formation_separation + c.formation_distance_spread)
        direction = route / distance
        second = w.formation_centers[1] - w.agent_start_center
        self.assertAlmostEqual(float(second @ direction),
                               c.formation_two_progress * distance)
        lateral = second - (second @ direction) * direction
        self.assertAlmostEqual(float(np.linalg.norm(lateral)),
                               c.formation_two_lateral_offset)
        points = w.spawn_position[w.target_exists]
        pairwise = np.linalg.norm(points[:, None] - points[None], axis=2)
        pairwise += np.eye(len(points)) * 1e9
        self.assertTrue((pairwise >= c.target_member_min_spacing).all())
        for j in np.flatnonzero(w.target_exists):
            center = w.formation_centers[w.target_formation[j]]
            self.assertLessEqual(np.linalg.norm(w.spawn_position[j] - center),
                                 c.target_formation_radius)

    def test_observation_exposes_type_weighted_remaining_work_units(self):
        w = World(compact())
        obs = w.reset(4)
        records = obs[0, SELF_FEATURES:SELF_FEATURES + TARGET_FEATURES * w.c.n_targets]
        records = records.reshape(w.c.n_targets, TARGET_FEATURES)
        expected = np.array([target_remaining_value(typ, work_units)
                             for typ, work_units in zip(w.target_type, w.target_work_units)]) / 5
        np.testing.assert_allclose(records[:, 2], expected)
        np.testing.assert_allclose(records[:, 3], 1.0)

        type_one = int(np.flatnonzero(w.target_type == 1)[0])
        w.target_work_units[type_one] = 1
        w.mem_work_units[:, type_one] = 1
        records = w._obs()[0, SELF_FEATURES:].reshape(w.c.n_targets, TARGET_FEATURES)
        self.assertAlmostEqual(float(records[type_one, 2]), 2.5 / 5)

    def test_full_type_two_final_mission_is_masked_only_at_final_mission_range(self):
        w = World(compact(n_targets=1, min_targets=1, final_mission_range=1,
                          service_steps_per_unit=10))
        w.reset(31)
        w.target_type[0], w.target_work_units[0], w.target_score[0] = 2, 1, 2
        w.mem_type[:, 0], w.mem_work_units[:, 0] = 2, 1
        w.pos[0] = w.targets[0]
        w._sense()
        obs, _, _, _, _ = w.step(action(w))
        records = obs[:, SELF_FEATURES:].reshape(w.n, 1, TARGET_FEATURES)
        self.assertEqual(records[0, 0, 3], 1.0)
        np.testing.assert_array_equal(records[1:, 0, 3], np.ones(w.n - 1))
        w.pos[1] = w.targets[0]
        w._sense()
        records = w._obs()[:, SELF_FEATURES:].reshape(w.n, 1, TARGET_FEATURES)
        self.assertEqual(records[1, 0, 3], 0.0)
        self.assertEqual(records[2, 0, 3], 1.0)

    def test_approaching_drone_does_not_reserve_target_capacity(self):
        w = World(compact(n_targets=1, min_targets=1, final_mission_range=.01))
        w.reset(32)
        w.target_type[0], w.target_work_units[0], w.target_score[0] = 2, 1, 2
        w.mem_type[:, 0], w.mem_work_units[:, 0] = 2, 1
        obs, _, _, _, _ = w.step(action(w, 0))
        records = obs[:, SELF_FEATURES:].reshape(w.n, 1, TARGET_FEATURES)
        np.testing.assert_array_equal(records[:, 0, 3], np.ones(w.n))

    def test_observation_contains_normalized_remaining_time(self):
        w = World(compact(horizon=20))
        obs = w.reset(15)
        np.testing.assert_allclose(obs[w.agent_active, 0], 1.0)
        obs, _, _, _, _ = w.step(action(w))
        np.testing.assert_allclose(obs[w.agent_active, 0], 19 / 20)

    def test_observation_contains_normalized_reward_margin(self):
        w = World(compact())
        obs = w.reset(15)
        np.testing.assert_allclose(obs[w.agent_active, 2], 1.0)
        w.score = w.formation_one_initial_score + 1
        obs = w._obs()
        expected = -1 / w.formation_one_initial_score
        np.testing.assert_allclose(obs[w.agent_active, 2], expected)

    def test_two_final_mission_agents_complete_type_one_target(self):
        w = World(compact(n_agents=4, min_agents=4, n_targets=1, min_targets=1,
                          horizon=40,
                          final_mission_probability=1, final_mission_range=1))
        w.reset(5)
        w.target_type[0], w.target_work_units[0], w.target_score[0] = 1, 2, 5
        w.initial_score = w.formation_one_initial_score = 5
        w.targets[0] = np.array([50., 50.])
        w.target_base_vel[0] = 0
        w.target_vel[0] = 0
        w.mem_pos[:, 0] = w.targets[0]
        w.pos[:] = [[50, 50], [50, 50], [30, 30], [40, 40]]
        _, reward, terminated, _, _ = w.step(action(w))
        self.assertAlmostEqual(float(reward.sum()), 0.0)
        self.assertFalse(terminated)

        w.pos[:2] = w.targets[0]
        for _ in range(w.c.service_steps_per_unit - 2):
            _, reward, terminated, _, _ = w.step(action(w))
            self.assertAlmostEqual(float(reward.sum()), 0.0)
        _, reward, terminated, _, _ = w.step(action(w))
        self.assertAlmostEqual(float(reward.sum()), 5.0)
        self.assertEqual(w.target_work_units[0], 0)
        self.assertFalse(terminated)
        np.testing.assert_array_equal(w.agent_active, [False, False, True, True])
        np.testing.assert_allclose(w.last_progress_value_by_agent, [2.5, 2.5, 0, 0])
        self.assertTrue(np.all(w.observation[1] == 0))

        metrics = w.metrics()
        self.assertTrue(metrics["mission_success"])
        self.assertEqual(w.score, 5)
        self.assertAlmostEqual(metrics["team_return"], 5.0)
        self.assertAlmostEqual(
            metrics["discounted_team_return"],
            5.0 * w.c.discount_gamma ** (w.t - 1))
        np.testing.assert_array_equal(w.agent_active, [False, False, True, True])

    def test_type_one_single_final_missions_deal_one_work_units_each(self):
        w = World(compact(n_agents=2, min_agents=2, n_targets=1, min_targets=1,
                          service_steps_per_unit=2, final_mission_probability=1,
                          final_mission_range=1))
        w.reset(25)
        w.target_type[0], w.target_work_units[0], w.target_score[0] = 1, 2, 5
        w.initial_score = w.formation_one_initial_score = 5
        w.targets[0] = [50, 50]
        w.target_base_vel[0] = 0
        w.pos[:] = [[50, 50], [20, 20]]
        w._sense()
        w.step(action(w))
        self.assertEqual(w.final_mission_participants[0].sum(), 1)
        _, reward, terminated, _, _ = w.step(action(w))
        self.assertFalse(terminated)
        self.assertEqual(w.target_work_units[0], 1)
        self.assertAlmostEqual(float(reward.sum()), 2.5)
        np.testing.assert_allclose(w.last_progress_value_by_agent, [2.5, 0])
        self.assertEqual(w.snapshot()['target_score'][0], 2.5)
        remaining = ApproximateDPPolicy._remaining_value(
            w._obs()[1, SELF_FEATURES:].reshape(1, TARGET_FEATURES))
        np.testing.assert_allclose(remaining, [2.5])
        np.testing.assert_array_equal(w.agent_active, [False, True])
        w.pos[1] = w.targets[0]
        w._sense()
        w.step(action(w))
        np.testing.assert_allclose(w.last_progress_value_by_agent, [0, 0])
        _, reward, terminated, _, _ = w.step(action(w))
        self.assertFalse(terminated)
        self.assertEqual(w.target_work_units[0], 0)
        self.assertAlmostEqual(float(reward.sum()), 2.5)
        self.assertEqual(w.score, 5)
        np.testing.assert_allclose(w.last_progress_value_by_agent, [0, 2.5])
        self.assertFalse(w.agent_active.any())

    def test_failed_final_mission_has_no_extra_penalty_at_horizon(self):
        w = World(compact(n_agents=1, min_agents=1, n_targets=1, min_targets=1,
                          final_mission_probability=0, final_mission_range=1,
                          service_steps_per_unit=1, horizon=3))
        w.reset(21)
        w.target_type[0], w.target_work_units[0] = 2, 1
        w.mem_type[:, 0], w.mem_work_units[:, 0] = 2, 1
        w.pos[0] = w.targets[0]
        w.mem_pos[0, 0] = w.targets[0]
        obs, reward, terminated, truncated, metrics = w.step(action(w))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(w.agent_active[0])
        self.assertTrue(np.all(obs[0] == 0))
        self.assertEqual(metrics["agents_expended"], 1)
        self.assertEqual(metrics["agent_terminated"], [True])
        self.assertFalse(metrics["mission_success"])
        self.assertAlmostEqual(float(reward.sum()), 0.0, places=6)

        _, reward, terminated, truncated, _ = w.step(action(w))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(float(reward.sum()), 0.0, places=6)

        _, reward, terminated, truncated, _ = w.step(action(w))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(float(reward.sum()), 0.0, places=6)

    def test_type_two_final_mission_participation_is_capped_at_one(self):
        w = World(compact(n_targets=1, min_targets=1,
                          final_mission_probability=1, final_mission_range=1,
                          service_steps_per_unit=1))
        w.reset(22)
        w.target_type[0], w.target_work_units[0] = 2, 1
        w.mem_type[:, 0], w.mem_work_units[:, 0] = 2, 1
        w.pos[:2] = w.targets[0]
        w.pos[2] = w.targets[0] + [10, 10]
        w.mem_pos[:, 0] = w.targets[0]
        w.step(action(w))
        self.assertEqual(int(w.agent_active[:2].sum()), 1)
        self.assertTrue(w.agent_active[2])
        self.assertEqual(w.metrics()["agents_expended"], 1)
        np.testing.assert_array_equal(w.locked_targets(), [-1, -1, -1])

    def test_type_one_final_mission_participation_is_capped_at_two(self):
        w = World(compact(n_targets=1, min_targets=1,
                          final_mission_probability=0, final_mission_range=1,
                          service_steps_per_unit=1))
        w.reset(24)
        w.target_type[0], w.target_work_units[0] = 1, 2
        w.mem_type[:, 0], w.mem_work_units[:, 0] = 1, 2
        w.pos[:] = w.targets[0]
        w.mem_pos[:, 0] = w.targets[0]
        w.step(action(w))
        self.assertEqual(np.count_nonzero(~w.agent_active), 2)
        self.assertEqual(np.count_nonzero(w.agent_active), 1)
        self.assertEqual(w.metrics()["agents_expended"], 2)
        self.assertEqual(w.locked_targets()[2], -1)

    def test_late_drone_serves_its_own_full_final_mission_duration(self):
        w = World(compact(n_targets=1, min_targets=1,
                          final_mission_probability=1, final_mission_range=1,
                          service_steps_per_unit=2))
        w.reset(26)
        w.target_type[0], w.target_work_units[0], w.target_score[0] = 1, 2, 5
        w.initial_score = w.formation_one_initial_score = 5
        w.targets[0] = [50, 50]
        w.target_base_vel[0] = 0
        w.pos[:] = [[50, 50], [20, 20], [30, 30]]
        w._sense()
        w.step(action(w))
        # Two more drones arrive while the first drone is already striking.
        w.pos[1:] = w.targets[0]
        w._sense()
        _, reward, terminated, _, _ = w.step(action(w))
        self.assertFalse(terminated)
        self.assertEqual(w.target_work_units[0], 1)
        np.testing.assert_array_equal(w.agent_active, [False, True, True])
        self.assertEqual(w.metrics()['agents_expended'], 1)
        late_participants = np.flatnonzero(w.final_mission_participants[0])
        self.assertEqual(len(late_participants), 1)
        late = int(late_participants[0])
        self.assertIn(late, (1, 2))
        self.assertEqual(w.agent_final_mission_progress[late], 1)
        self.assertAlmostEqual(float(reward.sum()), 2.5)

        _, reward, terminated, _, _ = w.step(action(w))
        self.assertFalse(terminated)
        self.assertEqual(w.target_work_units[0], 0)
        self.assertFalse(w.agent_active[late])
        self.assertEqual(np.count_nonzero(w.agent_active), 1)
        self.assertEqual(w.metrics()['agents_expended'], 2)
        self.assertEqual(w.last_progress_value_by_agent[late], 2.5)
        self.assertEqual(np.count_nonzero(w.last_progress_value_by_agent), 1)
        self.assertAlmostEqual(float(reward.sum()), 2.5)

    def test_snapshot_records_loiter_visualization_state(self):
        w = World(compact(n_agents=1, min_agents=1, n_targets=1, min_targets=1,
                          final_mission_probability=1, final_mission_range=1,
                          service_steps_per_unit=3))
        w.reset(31)
        w.pos[0] = w.targets[0]
        w.mem_pos[0, 0] = w.targets[0]
        w.step(action(w))
        frame = w.snapshot()
        self.assertEqual(frame["final_mission_progress"], [1])
        self.assertEqual(frame["agent_final_mission_progress"], [1])
        self.assertEqual(frame["final_mission_participants"], [[True]])

    def test_all_active_drones_always_know_exact_live_target_state(self):
        w = World(compact(n_agents=1, min_agents=1, n_targets=1, min_targets=1,
                          sensor_range=.1, sensor_fov_deg=1,
                          initial_intel_noise=50, position_noise=50))
        w.reset(23)
        w.pos[0] = [50, 50]
        w.targets[0] = [90, 90]
        w.target_base_vel[0] = 0
        w.target_vel[0] = 0
        w.heading[0] = 0
        w._sense()
        self.assertTrue(w.visible[0, 0])
        np.testing.assert_array_equal(w.mem_pos[0, 0], w.targets[0])
        self.assertEqual(w.metrics()['discovery_fraction'], 1)
        self.assertEqual(w.metrics()['first_detection_step'], 0)

    def test_no_progress_has_zero_reward_before_horizon(self):
        c = compact(horizon=2)
        w = World(c)
        w.reset(10)
        _, first_reward, _, _, _ = w.step(action(w))
        _, reward, _, truncated, _ = w.step(action(w))
        self.assertTrue(truncated)
        self.assertAlmostEqual(float(first_reward.sum()), 0.0)
        self.assertAlmostEqual(float(reward.sum()), 0.0)

    def test_all_groups_contribute_to_success_and_reward(self):
        w = World(compact(n_agents=1, min_agents=1, n_targets=2, min_targets=2,
                          final_mission_probability=1, final_mission_range=1, horizon=5,
                          service_steps_per_unit=1))
        w.reset(27)
        w.target_formation[:] = [0, 1]
        w.target_type[:] = [3, 1]
        w.target_work_units[:] = [1, 2]
        w.target_score[:] = [1, 5]
        w.initial_score, w.formation_one_initial_score = 6, 1
        w.targets[1] = [50, 50]
        w.target_base_vel[:] = 0
        w.pos[0] = w.targets[1]
        w._sense()
        _, reward, terminated, _, metrics = w.step(action(w, 1))
        self.assertFalse(terminated)
        self.assertEqual(w.target_work_units[1], 1)
        self.assertEqual(w.score, 2.5)
        self.assertTrue(metrics['mission_success'])
        self.assertEqual(metrics['success_threshold'], 1)
        self.assertAlmostEqual(float(reward.sum()), 2.5)

        w = World(compact(n_agents=1, min_agents=1, n_targets=2, min_targets=2,
                          final_mission_probability=1, final_mission_range=1, horizon=5,
                          service_steps_per_unit=1))
        w.reset(27)
        w.target_formation[:] = [0, 1]
        w.target_type[:] = [3, 1]
        w.target_work_units[:] = [1, 2]
        w.target_score[:] = [1, 5]
        w.initial_score, w.formation_one_initial_score = 6, 4
        w.targets[1] = [50, 50]
        w.target_base_vel[:] = 0
        w.pos[0] = w.targets[1]
        w._sense()
        _, reward, terminated, _, _ = w.step(action(w, 1))
        self.assertFalse(terminated)
        self.assertAlmostEqual(float(reward.sum()), 2.5)

    def test_threshold_crossing_does_not_create_persistent_reward(self):
        w = World(compact(n_agents=2, min_agents=2, n_targets=2, min_targets=2,
                          final_mission_probability=1, final_mission_range=1, horizon=3,
                          service_steps_per_unit=1))
        w.reset(27)
        w.target_formation[:] = [0, 1]
        w.target_type[:] = [3, 1]
        w.target_work_units[:] = [1, 2]
        w.target_score[:] = [1, 5]
        w.initial_score, w.formation_one_initial_score = 6, 1
        w.targets[1] = [50, 50]
        w.target_base_vel[:] = 0
        w.pos[:] = [[50, 50], [10, 10]]
        w._sense()

        _, reward, _, _, metrics = w.step(action(w, 1))
        self.assertTrue(metrics['mission_success'])
        self.assertGreater(metrics['score'], metrics['baseline_score'])
        self.assertAlmostEqual(float(reward.sum()), 2.5)

        _, reward, _, _, _ = w.step(action(w, 0))
        self.assertAlmostEqual(float(reward.sum()), 0.0)

    def test_lookahead_uses_new_per_step_reward(self):
        c = compact(n_agents=1, min_agents=1, n_targets=1, min_targets=1,
                    horizon=4, final_mission_range=1, service_steps_per_unit=1)
        policy = ApproximateDPPolicy(c)
        policy.success_threshold = 1
        policy._current_progress = np.zeros(1)
        record = np.array([[1 / c.size, 0, .4, 1]])
        objective = policy._objective(
            np.array([0]), np.array([0]), [record], np.array([2]), 0, 4)[0]
        expected = 2 * c.discount_gamma ** 2
        self.assertAlmostEqual(objective, expected)

    def test_equal_baseline_is_success(self):
        w = World(compact(n_agents=1, min_agents=1, n_targets=1, min_targets=1,
                          final_mission_probability=1, final_mission_range=1, horizon=2,
                          service_steps_per_unit=1))
        w.reset(34)
        w.target_type[0], w.target_work_units[0], w.target_score[0] = 2, 1, 2
        w.mem_type[:, 0], w.mem_work_units[:, 0] = 2, 1
        w.initial_score = w.formation_one_initial_score = 2
        w.pos[0] = w.targets[0]
        w._sense()

        _, reward, _, truncated, metrics = w.step(action(w))
        self.assertFalse(truncated)
        self.assertEqual(metrics['score'], metrics['baseline_score'])
        self.assertTrue(metrics['mission_success'])
        self.assertAlmostEqual(float(reward.sum()), 2.0)

        _, reward, _, truncated, metrics = w.step(action(w))
        self.assertTrue(truncated)
        self.assertTrue(metrics['mission_success'])
        self.assertAlmostEqual(float(reward.sum()), 0.0)

    def test_positive_margin_does_not_terminate_with_forces_remaining(self):
        w = World(compact(n_agents=2, min_agents=2, n_targets=2, min_targets=2,
                          final_mission_probability=1, final_mission_range=1, horizon=5,
                          service_steps_per_unit=1))
        w.reset(33)
        w.target_formation[:] = [0, 1]
        w.target_type[:] = [3, 1]
        w.target_work_units[:] = [1, 2]
        w.target_score[:] = [1, 5]
        w.formation_one_initial_score = 1
        w.targets[1] = [50, 50]
        w.target_base_vel[:] = 0
        w.pos[:] = [[50, 50], [10, 10]]
        w._sense()
        _, _, terminated, _, metrics = w.step(action(w, 1))
        self.assertTrue(metrics['mission_success'])
        self.assertFalse(terminated)
        self.assertTrue(w.agent_active[1])

    def test_target_types_work_units_speed_and_scores(self):
        w = World(compact(target_motion_scale=.05))
        w.reset(14)
        self.assertEqual(set(w.target_type[w.target_exists]), {1, 2})
        for j in np.flatnonzero(w.target_exists):
            typ = w.target_type[j]
            self.assertEqual(w.target_work_units[j], 2 if typ == 1 else 1)
            self.assertEqual(w.target_score[j], {1: 5, 2: 2, 3: 1}[typ])
            kmh = np.linalg.norm(w.target_vel[j]) * 360 / w.c.target_motion_scale
            lo, hi = {1: (40, 60), 2: (60, 80), 3: (4, 10)}[typ]
            self.assertTrue(lo <= kmh <= hi)

    def test_targets_are_stationary_by_default(self):
        w = World(compact())
        w.reset(14)
        np.testing.assert_array_equal(w.target_vel, 0)
        before = w.targets.copy()
        w._move_targets()
        np.testing.assert_array_equal(w.targets, before)

    def test_default_targets_mix_all_types_evenly(self):
        w = World(Config())
        w.reset(19)
        counts = np.bincount(w.target_type[w.target_exists], minlength=4)
        np.testing.assert_array_equal(counts[1:], [2, 2, 1])

    def test_success_threshold_case_can_be_fixed(self):
        w = World(Config())
        for index, threshold in enumerate(SUCCESS_THRESHOLDS):
            w.reset(100 + index, success_threshold=threshold)
            self.assertEqual(w.formation_one_initial_score, threshold)
            counts = np.bincount(w.target_type[w.target_exists], minlength=4)
            np.testing.assert_array_equal(counts[1:], [2, 2, 1])

    def test_three_evaluation_tasks_have_required_types_and_distances(self):
        c = Config(n_agents=5, min_agents=5, horizon=100)
        center_ranges = ((35, 40), (25, 27), (54, 58))
        for index, task in enumerate(EVALUATION_TASKS):
            w = World(c)
            w.reset(100 + index, evaluation_task=task)
            center_distance = np.linalg.norm(w.formation_centers[0] - w.agent_start_center)
            self.assertGreaterEqual(center_distance, center_ranges[index][0])
            self.assertLessEqual(center_distance, center_ranges[index][1])
            formation_one = w.target_formation == 0
            formation_two = w.target_formation == 1
            if index == 0:
                np.testing.assert_array_equal(sorted(w.target_type[formation_one]), [2, 2, 3])
                np.testing.assert_array_equal(sorted(w.target_type[formation_two]), [1, 1])
                self.assertEqual(w.formation_one_initial_score, 5)
            else:
                np.testing.assert_array_equal(sorted(w.target_type[formation_one]), [1, 1, 2])
                np.testing.assert_array_equal(sorted(w.target_type[formation_two]), [2, 3])
                self.assertEqual(w.formation_one_initial_score, 12)
                distances = np.linalg.norm(
                    w.targets[formation_one][None] - w.pos[:, None], axis=-1)
                if index == 1:
                    self.assertLessEqual(
                        distances.max(),
                        c.final_mission_speed * c.dt * (c.horizon - c.service_steps_per_unit))
                else:
                    self.assertGreater(
                        distances.min(), c.final_mission_speed * c.dt * c.horizon + c.final_mission_range)

    def test_heuristic_spreads_single_step_targets(self):
        w = World(compact(n_targets=3, min_targets=3))
        w.reset(17)
        w.target_type[:] = 2
        w.target_work_units[:] = 1
        w.target_score[:] = 2
        w.mem_type[:] = 2
        w.mem_work_units[:] = 1
        targets = HeuristicPolicy(w.c).predict(w._obs())["target"]
        self.assertEqual(len(np.unique(targets[w.agent_active])), 3)

    def test_heuristic_forms_two_drone_type_one_teams(self):
        w = World(compact(n_agents=4, min_agents=4, n_targets=2, min_targets=2))
        w.reset(18)
        w.target_type[:] = 1
        w.target_work_units[:] = 2
        w.target_score[:] = 5
        w.mem_type[:] = 1
        w.mem_work_units[:] = 2
        targets = HeuristicPolicy(w.c).predict(w._obs())["target"]
        counts = np.bincount(targets[w.agent_active], minlength=2)
        np.testing.assert_array_equal(counts, [2, 2])

    def test_capacity_aware_nearest_and_type_priority_rules(self):
        c = compact(n_agents=4, min_agents=4, n_targets=3, min_targets=3)
        obs = np.zeros((4, SELF_FEATURES + 3 * TARGET_FEATURES), dtype=np.float32)
        obs[:, 0] = 1.0
        distances = np.array([[.06, .05, .50],
                              [.07, .30, .50],
                              [.08, .09, .50],
                              [.40, .50, .05]])
        for j, value in enumerate((1.0, .4, .2)):
            k = SELF_FEATURES + j * TARGET_FEATURES
            obs[:, k] = distances[:, j]
            obs[:, k + 2] = value
            obs[:, k + 3] = 1.0

        nearest = NearestTargetPolicy(c)
        priority = TypePriorityPolicy(c)
        np.testing.assert_array_equal(nearest.predict(obs)["target"], [1, 0, 0, 2])
        np.testing.assert_array_equal(priority.predict(obs)["target"], [0, 0, 1, 2])
        for policy in (nearest, priority):
            counts = np.bincount(policy.assigned_target, minlength=3)
            np.testing.assert_array_equal(counts, [2, 1, 1])
            changed = obs.copy()
            changed[:, SELF_FEATURES] = .9
            np.testing.assert_array_equal(policy.predict(changed)["target"],
                                          policy.assigned_target)

    def test_type_one_with_one_work_units_reserves_one_slot(self):
        c = compact(n_agents=3, min_agents=3, n_targets=3, min_targets=3)
        obs = np.zeros((3, SELF_FEATURES + 3 * TARGET_FEATURES), dtype=np.float32)
        obs[:, 0] = 1.0
        for j, value in enumerate((.6, .4, .2)):
            k = SELF_FEATURES + j * TARGET_FEATURES
            obs[:, k] = .1 + j * .1
            obs[:, k + 2] = value
            obs[:, k + 3] = 1.0
        for policy_class in (NearestTargetPolicy, TypePriorityPolicy):
            targets = policy_class(c).predict(obs)["target"]
            np.testing.assert_array_equal(np.bincount(targets, minlength=3), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
