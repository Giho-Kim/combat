import unittest
from dataclasses import replace

import numpy as np

from pointmass_rl.belief_env import BeliefWorld
from pointmass_rl.belief_training import coordination_counts, coordination_metrics, evaluate_belief
from tests.belief_trace import scenarios


class CoordinationMetricTests(unittest.TestCase):
    def test_over_capacity_counts(self):
        w = BeliefWorld(scenarios()['gaussian_intel'])
        obs = w.reset(25)
        regions = np.flatnonzero(obs['valid'][:w.belief.cells])
        counts = coordination_counts(obs, np.full(w.n, regions[0]))
        self.assertEqual((counts['agent_steps'], counts['over_capacity']), (w.n, w.n-1))
        spread = regions[np.arange(w.n) % len(regions)]
        self.assertEqual(coordination_counts(obs, spread)['over_capacity'], max(0, w.n-len(regions)))
        self.assertEqual(coordination_counts(obs, spread)['decisions'], int(obs['decision'].sum()))
        metrics = coordination_metrics(dict(agent_steps=10, over_capacity=2, decisions=5))
        self.assertEqual((metrics['over_capacity_rate'], metrics['decision_rate']), (.2, .5))
        self.assertTrue(np.isnan(coordination_metrics(dict(agent_steps=0, over_capacity=0, decisions=0))['decision_rate']))

    def test_evaluation_reports_coordination(self):
        summary, rows = evaluate_belief(None, replace(scenarios()['gaussian_intel'], horizon=150),
                                        episodes=2, seed=25, baseline='nearest')
        for key in ('over_capacity_rate', 'decision_rate'):
            self.assertIn(key, summary)
            self.assertIn(key, rows[0])
        self.assertGreater(summary['over_capacity_rate'], 0)  # Nearest ignores teammates.


if __name__ == '__main__':
    unittest.main()
