import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

try:
    import torch
    from pointmass_rl.strike_ppo import (StrikeActorCritic, load_checkpoint,
                                        save_checkpoint, train_mappo, critic_state, _advantages,
                                        _scale_actor_advantage, _summarize)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from pointmass_rl.env import Config, World


@unittest.skipUnless(HAS_TORCH, "RL extra absent")
class StrikePolicyTests(unittest.TestCase):
    def test_equal_team_advantages_keep_policy_signal(self):
        for value in (-4., 4.):
            advantage = torch.full((5,), value)
            scaled = _scale_actor_advantage(advantage, torch.ones(5, dtype=torch.bool))
            self.assertTrue(torch.isfinite(scaled).all())
            self.assertTrue((scaled.sign() == advantage.sign()).all())
            self.assertTrue((scaled.abs() > 0).all())

    def test_damage_credit_scale_must_be_nonnegative(self):
        with self.assertRaises(ValueError):
            Config(damage_credit_scale=-1)
        with self.assertRaises(ValueError):
            Config(gae_lambda=0)

    def test_binary_success_summary_is_raw_mean(self):
        rows = [dict(team_return=0, mission_success=value, score=0,
                     baseline_score=1, destroyed_fraction=0, score_auc=0, steps=1)
                for value in (0, 0, 1, 1, 1)]
        self.assertAlmostEqual(_summarize(rows)["mission_success"], .6)

    def test_team_value_and_delayed_failure_credit(self):
        w = World(Config())
        obs = w.reset(8)
        model = StrikeActorCritic(w.obs_dim, w.c.n_targets, w.n)
        state = torch.as_tensor(critic_state(w, obs))
        values = model.values(state)
        torch.testing.assert_close(values, values[0].expand_as(values))
        # Drone 0 disappeared after step 0; a later team penalty must still
        # affect the return of its earlier decision.
        reward = np.array([[1., 1.], [-5., -5.]])
        zeros = np.zeros_like(reward)
        done = np.array([[0., 0.], [1., 1.]])
        _, returns = _advantages(reward, zeros, zeros, done, gamma=1, gae_lambda=1)
        np.testing.assert_allclose(returns[0], [-4., -4.])

    def test_shapes_ranges_mask_and_finite_statistics(self):
        c = Config(horizon=5)
        w = World(c)
        obs = torch.as_tensor(w.reset(2), dtype=torch.float32)
        model = StrikeActorCritic(w.obs_dim, c.n_targets, c.n_agents)
        action, stats = model.act(obs)
        target = action["target"]
        self.assertEqual(target.shape, (c.n_agents,))
        self.assertTrue(np.all((target >= 0) & (target < c.n_targets)))
        for value in stats.values():
            self.assertTrue(torch.isfinite(value).all())
        replay_logp, _ = model.evaluate_actions(
            obs, torch.as_tensor(target), stats["action_mask"])
        torch.testing.assert_close(replay_logp, stats["logp"])
        self.assertTrue(np.all(stats["action_mask"].numpy()[
            np.arange(c.n_agents), target]))
        locked = action["target"].copy()
        locked_action, _ = model.act(obs + torch.randn_like(obs) * .01,
                                     deterministic=True, locked_target=locked)
        active = np.any(obs.numpy() != 0, axis=1)
        np.testing.assert_array_equal(locked_action["target"][active], locked[active])
        self.assertTrue(np.isfinite(w.step(action)[0]).all())

    def test_centralized_critic_uses_joint_observation(self):
        c = Config(horizon=5)
        w = World(c)
        obs = torch.as_tensor(w.reset(3), dtype=torch.float32)
        model = StrikeActorCritic(w.obs_dim, c.n_targets, c.n_agents)
        state = torch.as_tensor(critic_state(w, obs.numpy()))
        original = model.values(state)
        changed_obs = obs.clone()
        changed_obs[1, 0] += .5
        changed_state = torch.as_tensor(critic_state(w, changed_obs.numpy()))
        changed = model.values(changed_state)
        self.assertFalse(torch.allclose(original, changed))
        actor_before = model.distributions(obs).probs[0]
        actor_after = model.distributions(changed_obs).probs[0]
        torch.testing.assert_close(actor_before, actor_after)

        w.strike_progress[0] = 9
        progress_state = torch.as_tensor(critic_state(w, obs.numpy()))
        self.assertAlmostEqual(float(progress_state[0, -(c.n_targets + 1)]), .9, places=6)
        self.assertFalse(torch.allclose(original, model.values(progress_state)))
        torch.testing.assert_close(actor_before, model.distributions(obs).probs[0])

        w.score = w.formation_one_initial_score / 2
        margin_state = torch.as_tensor(critic_state(w, obs.numpy()))
        self.assertAlmostEqual(float(margin_state[0, -1]), .5, places=6)

    def test_short_strike_mappo_update_and_checkpoint(self):
        c = Config(n_agents=3, min_agents=3, n_targets=2, min_targets=2,
                   randomize_counts=False, horizon=12)
        model, episodes, evaluations, completed = train_mappo(
            c, total_agent_transitions=36, seed=4, rollout_steps=6,
            epochs=1, minibatch_size=16, eval_interval=18, eval_episodes=1)
        self.assertIsInstance(model, StrikeActorCritic)
        self.assertGreaterEqual(completed, 36)
        self.assertTrue(episodes)
        self.assertEqual(len(evaluations), 9)
        self.assertEqual({row["agent_transitions"] for row in evaluations}, {0, 18, 36})
        self.assertEqual({row["policy"] for row in evaluations},
                         {"random", "heuristic", "mappo"})
        self.assertIn("score_auc", evaluations[0])
        self.assertIn("team_return", evaluations[0])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "strike.pt"
            save_checkpoint(path, model, {"test": True})
            restored = StrikeActorCritic(model.obs_dim, model.n_targets, model.n_agents)
            self.assertEqual(load_checkpoint(path, restored), {"test": True})


if __name__ == "__main__":
    unittest.main()
