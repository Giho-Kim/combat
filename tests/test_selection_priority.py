import unittest

import numpy as np
import torch

from pointmass_rl.env import Config, World
from pointmass_rl.strike_ppo import (
    DecisionSchedule, StrikeActorCritic, allocated_actions,
)
from pointmass_rl.strike_mat import StrikeMATActorCritic


@unittest.skip('legacy distance-priority rejection/reselection resolver was removed')
class SelectionPriorityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = Config(n_agents=5, min_agents=5, n_targets=3, min_targets=3,
                             decision_interval=5)
        self.world = World(self.config)
        self.world.reset(7)
        self.world.targets[:] = [[0, 0], [20, 0], [30, 0]]
        self.world.pos[:] = [[5, 0], [1, 0], [3, 0], [2, 0], [4, 0]]
        self.world.target_type[:] = [1, 1, 2]
        self.world.target_life[:] = [2, 2, 1]
        self.world._sense()

    def model(self, model_class=StrikeActorCritic):
        model = model_class(3, 5)
        head = model.actor_body.head[-1] if model.autoregressive else model.target_head
        with torch.no_grad():
            head.weight.zero_()
            head.bias.copy_(torch.tensor([3., 2., 1.]))
        return model

    def test_nearest_wins_and_losers_reselect_without_advancing_time(self):
        for model_class in (StrikeActorCritic, StrikeMATActorCritic):
            model = self.model(model_class)
            schedule = DecisionSchedule(self.config)
            with torch.no_grad():
                action, traces = allocated_actions(model, [self.world], [schedule], True)
                for trace in traces:
                    logp, _ = model.evaluate_actions(torch.from_numpy(trace['obs']),
                        torch.from_numpy(trace['target']), torch.from_numpy(trace['action_mask']))
                    np.testing.assert_allclose(logp.numpy(), trace['logp'], atol=1e-6)
            np.testing.assert_array_equal(action['target'][0], [1, 0, 2, 0, 1])
            np.testing.assert_array_equal(traces[0]['decision'][0], [False, True, False, True, False])
            np.testing.assert_array_equal(traces[1]['decision'][0], [True, False, False, False, True])
            np.testing.assert_array_equal(traces[2]['decision'][0], [False, False, True, False, False])
            decisions = np.stack([trace['decision'][0] for trace in traces])
            selected = np.stack([trace['target'][0] for trace in traces])
            np.testing.assert_array_equal(decisions.sum(axis=0), np.ones(5))
            for agent in range(5):
                self.assertEqual(selected[decisions[:, agent], agent].item(),
                                 action['target'][0, agent])
            self.assertEqual(len(traces), 3)
            self.assertEqual(self.world.t, 0)
            np.testing.assert_array_equal(schedule.targets, action['target'][0])
            self.assertFalse(self.world.strike_participants.any())

    def test_existing_participant_keeps_slot_even_when_farther(self):
        self.world.strike_participants[0, 0] = True
        self.world.target_assignment[0, 0] = True
        with torch.no_grad():
            action, traces = allocated_actions(self.model(), [self.world],
                                               [DecisionSchedule(self.config)], True)
        np.testing.assert_array_equal(np.flatnonzero(action['target'][0] == 0), [0, 1])
        self.assertFalse(traces[0]['decision'][0, 0])

    def test_hold_is_interrupted_for_rejected_drone_and_ties_use_agent_index(self):
        self.world.pos[:] = [1, 0]
        self.world.t = 1
        schedule = DecisionSchedule(self.config)
        schedule.targets[:] = 0
        schedule.next_decision[:] = 5
        with torch.no_grad():
            action, traces = allocated_actions(self.model(), [self.world], [schedule], True)
        np.testing.assert_array_equal(action['target'][0], [0, 0, 1, 1, 2])
        self.assertFalse(traces[0]['decision'].any())
        np.testing.assert_array_equal(traces[1]['decision'][0], [False, False, True, True, False])
        np.testing.assert_array_equal(traces[2]['decision'][0], [False, False, False, False, True])
        np.testing.assert_array_equal(schedule.next_decision, [5, 5, 6, 6, 6])

    def test_insufficient_total_capacity_is_explicit(self):
        self.world.target_life[:] = 1
        with self.assertRaisesRegex(ValueError, 'one live target slot'):
            allocated_actions(self.model(), [self.world], [DecisionSchedule(self.config)], True)

    def test_committed_target_survives_expiry_and_policy_change(self):
        self.config.commit_target = True
        schedule = DecisionSchedule(self.config)
        model = self.model()
        with torch.no_grad():
            first, _ = allocated_actions(model, [self.world], [schedule], True)
            self.world.target_assignment.fill(False)
            self.world.target_assignment[first['target'][0], np.arange(5)] = True
            self.world.t = 20
            model.target_head.bias.copy_(torch.tensor([1., 2., 3.]))
            second, traces = allocated_actions(model, [self.world], [schedule], True)
        np.testing.assert_array_equal(first['target'], second['target'])
        self.assertFalse(any(trace['decision'].any() for trace in traces))
        # Invalid assignments are released, and resetting releases all commitments.
        available = model.available_actions(self.world)
        available[0, first['target'][0, 0]] = False
        _, decisions = schedule.mask(self.world, available)
        self.assertTrue(decisions[0])
        self.world.t = 0
        _, decisions = schedule.mask(self.world, model.available_actions(self.world))
        self.assertTrue(decisions.all())

    def test_commit_protects_approach_without_sticky_flag(self):
        self.config.commit_target = True
        self.world.t = 10
        self.world.target_assignment.fill(False)
        self.world.target_assignment[0, [0, 4]] = True
        schedule = DecisionSchedule(self.config)
        schedule.targets[[0, 4]] = 0
        with torch.no_grad():
            action, _ = allocated_actions(self.model(), [self.world], [schedule], True)
        np.testing.assert_array_equal(np.flatnonzero(action['target'][0] == 0), [0, 4])

    def test_sticky_approachers_keep_slots_against_closer_newcomer(self):
        self.config.sticky_assignment = True
        self.world.pos[:] = [[0, 0], [1, 0], [2, 0], [30, 0], [40, 0]]
        self.world.target_assignment.fill(False)
        self.world.target_assignment[0, :2] = True
        self.world.target_assignment[1, 3:] = True
        self.world.t = 1
        schedule = DecisionSchedule(self.config)
        schedule.targets[:] = [0, 0, 0, 1, 1]
        schedule.next_decision[:] = 5
        with torch.no_grad():
            action, traces = allocated_actions(self.model(), [self.world], [schedule], True)
        np.testing.assert_array_equal(action['target'][0], [0, 0, 2, 1, 1])
        self.assertFalse(traces[1]['action_mask'][0, 2, 1])
        self.assertEqual(self.world.t, 1)

    def test_sticky_does_not_force_actor_to_keep_old_target(self):
        self.config.sticky_assignment = True
        self.world.target_assignment[2, 1] = True
        with torch.no_grad():
            action, _ = allocated_actions(self.model(), [self.world],
                                          [DecisionSchedule(self.config)], True)
        self.assertEqual(action['target'][0, 1], 0)

    def test_later_candidate_displaces_earlier_winner_and_winner_reselects(self):
        self.world.pos[:] = [[0, 0], [1, 0], [2, 0], [30, 0], [40, 0]]
        self.world.t = 1
        for model_class in (StrikeActorCritic, StrikeMATActorCritic):
            model = self.model(model_class)
            schedule = DecisionSchedule(self.config)
            schedule.targets[:] = [0, 0, 0, 1, 1]
            schedule.next_decision[:] = 5
            with torch.no_grad():
                action, traces = allocated_actions(model, [self.world], [schedule], True)
                for trace in traces:
                    logp, _ = model.evaluate_actions(torch.from_numpy(trace['obs']),
                        torch.from_numpy(trace['target']), torch.from_numpy(trace['action_mask']))
                    np.testing.assert_allclose(logp.numpy(), trace['logp'], atol=1e-6)
            np.testing.assert_array_equal(traces[0]['target'][0], [0, 0, 0, 1, 1])
            self.assertTrue(traces[1]['action_mask'][0, 2, 1])
            np.testing.assert_array_equal(traces[1]['target'][0], [0, 0, 1, 1, 1])
            np.testing.assert_array_equal(traces[2]['decision'][0], [False, False, False, False, True])
            np.testing.assert_array_equal(action['target'][0], [0, 0, 1, 1, 2])
            np.testing.assert_array_equal(schedule.next_decision, [5, 5, 6, 5, 6])
            self.assertEqual(self.world.t, 1)

    def test_later_candidate_cannot_displace_actual_participant(self):
        self.world.pos[:] = [[0, 0], [1, 0], [2, 0], [30, 0], [40, 0]]
        self.world.strike_participants[1, 4] = True
        self.world.target_assignment[1, 4] = True
        self.world.t = 1
        schedule = DecisionSchedule(self.config)
        schedule.targets[:] = [0, 0, 0, 1, 1]
        schedule.next_decision[:] = 5
        with torch.no_grad():
            action, traces = allocated_actions(self.model(), [self.world], [schedule], True)
        self.assertFalse(traces[1]['action_mask'][0, 2, 1])
        np.testing.assert_array_equal(action['target'][0], [0, 0, 2, 1, 1])


class DirectJointActionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = Config(n_agents=5, min_agents=5, n_targets=3, min_targets=3,
                             decision_interval=1)
        self.world = World(self.config)
        self.world.reset(7)
        self.model = StrikeActorCritic(3, 5)
        with torch.no_grad():
            self.model.target_head.weight.zero_()
            self.model.target_head.bias.copy_(torch.tensor([3., 2., 1.]))

    def test_all_first_choices_pass_through_without_rejection(self):
        with torch.no_grad():
            action, traces = allocated_actions(
                self.model, [self.world], [DecisionSchedule(self.config)], True)
        self.assertEqual(len(traces), 1)
        np.testing.assert_array_equal(action['target'][0], np.zeros(5, dtype=int))
        self.assertTrue(traces[0]['decision'][0].all())

    def test_evaluation_resolver_distributes_by_distance_only_when_enabled(self):
        from pointmass_rl.strike_ppo import scheduled_action
        self.world.targets[:] = [[0, 0], [20, 0], [30, 0]]
        self.world.pos[:] = [[5, 0], [1, 0], [3, 0], [2, 0], [4, 0]]
        self.world.target_type[:] = [1, 1, 2]
        self.world.target_life[:] = [2, 2, 1]
        self.world._sense()
        with torch.no_grad():
            direct, _ = scheduled_action(self.model, self.world,
                                         DecisionSchedule(self.config))
            resolved, _ = scheduled_action(self.model, self.world,
                DecisionSchedule(self.config), resolver=True)
        np.testing.assert_array_equal(direct['target'], [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(resolved['target'], [1, 0, 2, 0, 1])
        self.assertEqual(self.world.t, 0)

    def test_evaluation_resolver_handles_insufficient_slots(self):
        from pointmass_rl.strike_ppo import scheduled_action
        self.world.target_life[:] = 1
        with torch.no_grad():
            resolved, _ = scheduled_action(self.model, self.world,
                DecisionSchedule(self.config), resolver=True)
        self.assertTrue(np.all((resolved['target'] >= 0) & (resolved['target'] < 3)))

    def test_capacity_shortage_does_not_trigger_reselection(self):
        self.world.target_life[:] = 1
        with torch.no_grad():
            action, traces = allocated_actions(
                self.model, [self.world], [DecisionSchedule(self.config)], True)
        self.assertEqual(len(traces), 1)
        np.testing.assert_array_equal(action['target'][0], np.zeros(5, dtype=int))

    def test_schedule_hold_still_forces_the_previous_action(self):
        self.world.t = 1
        schedule = DecisionSchedule(self.config)
        schedule.interval = 5
        schedule.targets[:] = 2
        schedule.next_decision[:] = 5
        with torch.no_grad():
            action, traces = allocated_actions(self.model, [self.world], [schedule], True)
        np.testing.assert_array_equal(action['target'][0], np.full(5, 2))
        self.assertFalse(traces[0]['decision'].any())


if __name__ == '__main__':
    unittest.main()
