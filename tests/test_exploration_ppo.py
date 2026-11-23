from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pointmass_rl.belief_policy import choose_action, stack_observations
from pointmass_rl.belief_training import command_evaluate, load_belief_checkpoint
from pointmass_rl.env import Config
from pointmass_rl.exploration import ExplorationPPOActorCritic, PolygonExplorationWorld
from pointmass_rl.exploration_ppo import collect_decisions, evaluate_exploration, train_exploration_ppo, update_decisions


class ScriptedOptions:
    """Two drones completing options every 3/5 seconds; delayed team rewards."""
    n, action_count = 2, 64

    def __init__(self, config, template):
        self.c, self.template = config, template

    def reset(self, seed=None):
        self.t = 0
        self.chosen = np.zeros(self.n, dtype=int)
        return self.observation()

    def observation(self):
        obs = {k: v.copy() for k, v in self.template.items()}
        obs['decision'] = self.t % np.array([3, 5]) == 0
        obs['time'][:] = 1-self.t/130
        obs['mask'][:] = False
        for agent in range(self.n):
            if obs['decision'][agent]:
                obs['mask'][agent] = True
            else:
                obs['mask'][agent, self.chosen[agent]] = True
        return obs

    def step(self, action, probabilities):
        # strike_action contains navigation setpoints too; its first column is the slot.
        from pointmass_rl.env import split_strike_action
        self.chosen = split_strike_action(action)
        self.t += 1
        reward = 7. if self.t == 100 else 2. if self.t == 130 else 0.
        done = self.t == 130
        return self.observation(), reward, False, done, dict(team_return=9.)


class ExplorationPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = replace(Config.load('configs/belief_exploration.json'), n_agents=2, min_agents=2,
                             randomize_counts=False)
        cls.template = PolygonExplorationWorld(cls.config).reset(7)

    def test_asynchronous_options_retain_delayed_rewards_and_terminal_partial_sweep(self):
        model = ExplorationPPOActorCritic(hidden=16)
        world = ScriptedOptions(self.config, self.template)
        with patch.object(model, 'joint_q', side_effect=AssertionError('No Q targets')):
            samples, episodes, steps = collect_decisions(model, [world], np.random.default_rng(7), 1)
        self.assertEqual(steps, 260)
        self.assertEqual(len(episodes), 1)
        first = [s for s in samples if s['agent'] == 0]
        second = [s for s in samples if s['agent'] == 1]
        self.assertEqual(len(first), 44)
        self.assertEqual(len(second), 26)
        self.assertEqual(first[-1]['duration'], 1)  # Timeout interrupts a 3-second option.
        self.assertEqual(first[0]['return'], 9.)  # Includes discovery 100 seconds later.
        self.assertEqual(second[0]['return'], 9.)
        self.assertEqual(first[-1]['return'], 2.)
        self.assertEqual(sum(s['duration'] for s in first), 130)
        self.assertEqual(sum(s['duration'] for s in second), 130)
        self.assertEqual(sum(s['reward'] for s in first), 9.)
        self.assertEqual(sum(s['reward'] for s in second), 9.)
        for s in samples:
            self.assertTrue(s['obs']['decision'][s['agent']])

    def test_discount_uses_elapsed_physics_steps_inside_and_across_options(self):
        config = replace(self.config, discount_gamma=.99)
        model = ExplorationPPOActorCritic(hidden=16)
        samples, _, _ = collect_decisions(model, [ScriptedOptions(config, self.template)], np.random.default_rng(7), 1)
        for agent in range(2):
            first = next(s for s in samples if s['agent'] == agent)
            self.assertAlmostEqual(first['return'], 7*.99**99 + 2*.99**129, places=6)

    def test_ppo_updates_actor_and_state_value_without_counterfactual_q(self):
        torch.manual_seed(7)
        model = ExplorationPPOActorCritic(hidden=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
        samples, _, _ = collect_decisions(model, [ScriptedOptions(self.config, self.template)], np.random.default_rng(7), 1)
        before_actor = model.actor_head.weight.detach().clone()
        before_value = model.q_head.weight.detach().clone()
        with patch.object(model, 'joint_q', side_effect=AssertionError('No joint Q')), \
             patch.object(model, 'advantages', side_effect=AssertionError('No COMA')):
            diagnostics = update_decisions(model, optimizer, samples, 1, 1, 32, .01, .2)
        self.assertTrue(all(np.isfinite(v) for v in diagnostics.values()))
        self.assertFalse(torch.equal(before_actor, model.actor_head.weight))
        self.assertFalse(torch.equal(before_value, model.q_head.weight))

    def test_real_world_training_checkpoint_and_evaluation_action(self):
        config = replace(self.config, horizon=3)
        with tempfile.TemporaryDirectory() as directory:
            model, episodes, evaluations, steps = train_exploration_ppo(config, 1, directory,
                n_envs=1, rollout_decisions=1, epochs=1, critic_epochs=1, eval_interval=1,
                eval_episodes=1, eval_workers=1, device='cpu')
            self.assertEqual(steps, 6)  # Finish the episode despite the requested budget.
            self.assertEqual(len(episodes), 1)
            self.assertEqual([row['policy'] for row in evaluations], ['none', 'score_time_limit'])
            _, _, best = load_belief_checkpoint(Path(directory)/'best.pt')
            self.assertEqual(best['metadata']['best_policy'], 'score_time_limit')
            loaded, restored_config, payload = load_belief_checkpoint(Path(directory)/'exploration_ppo.pt')
            self.assertEqual(restored_config, config)
            self.assertEqual(payload['architecture'], 'polygon_exploration_decision_ppo')
            self.assertIn('Monte Carlo', payload['training_settings']['update'])
            obs = PolygonExplorationWorld(config).reset(7)
            batch = stack_observations([obs])
            torch.testing.assert_close(loaded.actor_logits(batch), model.actor_logits(batch))
            torch.testing.assert_close(loaded.state_value(batch), model.state_value(batch))
            for resolver in ('none', 'score', 'distance'):
                action, probabilities = choose_action(loaded, obs, deterministic=True, resolver=resolver)
                self.assertTrue(obs['mask'][np.arange(2), action].all())
                np.testing.assert_allclose(probabilities.sum(-1), 1., atol=1e-6)
            self.assertTrue((Path(directory)/'latest.pt').exists())
            evaluation_out = Path(directory)/'evaluation'
            command_evaluate(SimpleNamespace(model=str(Path(directory)/'exploration_ppo.pt'),
                episodes=1, seed=10000, out=str(evaluation_out)), config)
            summary = json.loads((evaluation_out/'summary.json').read_text())
            self.assertEqual(list(summary['summaries']), ['none', 'score_time_limit'])

    def test_parallel_evaluation_preserves_serial_metrics_seeds_and_order(self):
        torch.manual_seed(7)
        model = ExplorationPPOActorCritic(hidden=16)
        config = replace(self.config, horizon=4)
        serial = evaluate_exploration(model, config, episodes=3, seed=10000, workers=1)
        parallel = evaluate_exploration(model, config, episodes=3, seed=10000, workers=2)
        self.assertEqual(serial, parallel)
        self.assertEqual(list(parallel), ['none', 'score_time_limit'])
        for summary, rows in parallel.values():
            self.assertEqual([row['seed'] for row in rows], [10000, 10001, 10002])
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(np.isfinite(value) for value in summary.values()))

    def test_evaluation_rejects_nonpositive_worker_counts(self):
        with self.assertRaises(ValueError):
            evaluate_exploration(ExplorationPPOActorCritic(hidden=16), self.config, workers=0)


if __name__ == '__main__':
    unittest.main()
