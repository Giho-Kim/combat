import json
import unittest

import numpy as np

from pointmass_rl.task_automaton import MOTION_TRANSITIONS, TaskAutomata, TaskEvent, TaskPhase
from tests.belief_trace import EVENT_GOLDEN, GOLDEN, compute, compute_event


class TaskAutomatonTests(unittest.TestCase):
    def test_motion_path_through_sweep(self):
        tasks = TaskAutomata(2)
        tasks.fire(0, TaskEvent.SELECT, 3)
        self.assertEqual(tasks.phases()[0], TaskPhase.TRANSIT)
        tasks.fire(0, TaskEvent.SWEEP)
        tasks.fire(0, TaskEvent.PASS_DONE)
        self.assertEqual(tasks.phases()[0], TaskPhase.SWEEPING)
        np.testing.assert_array_equal(tasks.sweeping(np.ones(4, dtype=bool)), [3, -1])
        np.testing.assert_array_equal(tasks.sweeping(np.zeros(4, dtype=bool)), [-1, -1])
        tasks.fire(0, TaskEvent.INVALIDATE)
        self.assertEqual(tasks.phases()[0], TaskPhase.IDLE)
        self.assertEqual(tasks.component[0], -1)

    def test_undefined_transitions_raise(self):
        tasks = TaskAutomata(1)
        for event in (TaskEvent.ARRIVE, TaskEvent.SWEEP, TaskEvent.RETARGET, TaskEvent.PASS_DONE):
            with self.assertRaises(ValueError):
                tasks.fire(0, event)
        tasks.fire(0, TaskEvent.SELECT, 1)
        tasks.fire(0, TaskEvent.ARRIVE)
        with self.assertRaises(ValueError):
            tasks.fire(0, TaskEvent.SWEEP)
        self.assertNotIn((TaskPhase.SWEEPING, TaskEvent.RETARGET), MOTION_TRANSITIONS)

    def test_world_sync_emits_lock_release_deactivate(self):
        tasks = TaskAutomata(3)
        active = np.ones(3, dtype=bool)
        tasks.sync(active, np.array([-1, 17, -1]))
        self.assertEqual(tasks.phases().tolist(), [TaskPhase.IDLE, TaskPhase.ENGAGED, TaskPhase.IDLE])
        active[1] = False
        tasks.sync(active, np.array([-1, -1, -1]))
        self.assertEqual(tasks.events[-2:], [(1, TaskEvent.RELEASE, None), (1, TaskEvent.DEACTIVATE, None)])
        self.assertEqual(tasks.phases()[1], TaskPhase.INACTIVE)
        with self.assertRaises(ValueError):
            tasks.fire(1, TaskEvent.SELECT, 0)

    def test_engaged_agent_is_held_and_full_track_is_closed(self):
        tasks = TaskAutomata(3)
        tasks.sync(np.ones(3, dtype=bool), np.array([5, -1, -1]))
        valid = np.ones(6, dtype=bool)
        mask, decision = tasks.action_mask(valid, [1], 5, 0, initial_intel=True, commit_target=False)
        self.assertEqual(np.flatnonzero(mask[0]).tolist(), [5])
        self.assertFalse(mask[1:, 5].any())
        self.assertEqual(decision.tolist(), [False, True, True])

    def test_event_rule_redecides_only_after_termination(self):
        valid = np.ones(4, dtype=bool)
        mask = lambda tasks, t, rule: tasks.action_mask(valid, [], 4, t, initial_intel=True,
                                                         commit_target=False, decision_rule=rule)
        tasks = TaskAutomata(1)
        tasks.commit(np.array([True]), np.array([2]), until=100)
        tasks.fire(0, TaskEvent.SELECT, 2)
        for event in (TaskEvent.RETARGET, TaskEvent.SWEEP):
            if event is TaskEvent.SWEEP:
                tasks.fire(0, TaskEvent.SELECT, 2)
            tasks.fire(0, event)
            self.assertFalse(mask(tasks, 5, 'event')[1][0], event)
        tasks.fire(0, TaskEvent.PASS_DONE)  # A whole pass is too long a task by default.
        self.assertFalse(mask(tasks, 5, 'event')[1][0])
        tasks.fire(0, TaskEvent.LEG_DONE)
        self.assertTrue(mask(tasks, 5, 'event')[1][0])
        self.assertFalse(mask(tasks, 5, 'interval')[1][0])  # Interval rule ignores events.
        tasks.commit(np.array([True]), np.array([2]), until=100)
        self.assertFalse(tasks.due[0])
        self.assertFalse(mask(tasks, 5, 'event')[1][0])
        self.assertTrue(mask(tasks, 100, 'event')[1][0])  # The interval is an upper bound.
        with self.assertRaises(ValueError):
            mask(tasks, 5, 'whenever')

    def test_interrupt_frees_only_unengaged_agents(self):
        tasks = TaskAutomata(3, terminations=())
        tasks.sync(np.array([True, True, False]), np.array([-1, 6, -1]))
        tasks.interrupt()
        self.assertEqual(tasks.due.tolist(), [True, False, False])

    def test_event_rule_decisions_have_a_reason(self):
        from dataclasses import replace
        from pointmass_rl.belief_env import BeliefWorld
        from pointmass_rl.env import Config, strike_action
        from tests.belief_trace import _choose, scenarios
        with self.assertRaises(ValueError):
            Config(mode='belief', belief_decision_rule='always')
        for name in ('gaussian_intel', 'histogram_plain'):
            config = replace(scenarios()[name], belief_decision_rule='event', decision_interval=10**6)
            w = BeliefWorld(config)
            obs, rng = w.reset(3), np.random.default_rng(0)
            last, decisions = np.zeros(w.n, dtype=int), 0
            while not w.done:
                t = w.tasks
                for i in np.flatnonzero(obs['decision']):
                    if t.held[i] < 0 or t.due[i]:
                        continue
                    # Otherwise the held component itself must have become unavailable.
                    due, until = t.due.copy(), t.hold_until.copy()
                    t.hold_until[:] = 10**9
                    held_mask = w.action_mask()[0]
                    t.due[:], t.hold_until[:] = due, until
                    self.assertFalse(held_mask[i, t.held[i]], (name, w._world.t, i))
                decisions += int(obs['decision'].sum())
                last = _choose(rng, obs['mask'], w.belief.cells, last)
                obs, *_ = w.step(strike_action(last))
            self.assertLess(decisions, w.n*config.horizon/10, name)

    def test_belief_world_matches_pre_automaton_golden_traces(self):
        # Digests and phase letters were recorded from the implementation in
        # which this state lived in ad hoc navigation dictionaries.
        golden, new = json.loads(GOLDEN.read_text()), compute()
        seen = set()
        for name, seeds in golden.items():
            for seed, expected in seeds.items():
                with self.subTest(scenario=name, seed=seed):
                    self.assertEqual(new[name][seed], expected)
                seen.update(''.join(expected['phases']))
        self.assertEqual(seen, set('XE-TSA'))


    def test_event_rule_matches_reference_traces(self):
        # Recorded with the first event-rule implementation, whose interrupts
        # came from a separate belief-event detector; BeliefWorld now detects
        # new tracks itself and must reproduce those rollouts exactly.
        golden, new = json.loads(EVENT_GOLDEN.read_text()), compute_event()
        for name, seeds in golden.items():
            for seed, expected in seeds.items():
                with self.subTest(scenario=name, seed=seed):
                    self.assertEqual(new[name][seed], expected)


if __name__ == '__main__':
    unittest.main()
