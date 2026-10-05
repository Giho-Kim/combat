import unittest
from unittest.mock import patch

import torch

from pointmass_rl.env import Config, World
from pointmass_rl.strike_ppo import _actor_event, train_mappo, _ppo_update


class ActorEventTests(unittest.TestCase):
    def test_pre_completion_start_and_final_states(self):
        world = World(Config(n_agents=5, min_agents=5))
        world.reset(7)
        self.assertTrue(_actor_event(world))
        world.t = 2
        self.assertFalse(_actor_event(world))
        world.strike_participants[0, 0] = True
        world.agent_strike_progress[0] = world.c.strike_steps_per_life - 2
        self.assertFalse(_actor_event(world))
        world.agent_strike_progress[0] += 1
        self.assertTrue(_actor_event(world))
        world.agent_active[0] = False
        self.assertFalse(_actor_event(world))
        world.t = world.c.horizon - 1
        self.assertTrue(_actor_event(world))

    def test_filter_preserves_rollout_returns_and_critic(self):
        config = Config(n_agents=3, min_agents=3, n_targets=2, min_targets=2,
                        horizon=6)
        batches = []
        def update(model, optimizers, batch, epochs, minibatch_size):
            batches.append({key: value.clone() for key, value in batch.items()})
            return _ppo_update(model, optimizers, batch, epochs, minibatch_size)
        for mode in ('all', 'events'):
            with patch('pointmass_rl.strike_ppo._ppo_update', side_effect=update):
                model, _, _, _ = train_mappo(config, 18, seed=4, rollout_steps=6,
                    epochs=1, n_envs=1, eval_interval=0, actor_samples=mode)
            self.assertEqual(model.training_settings['actor_sample_mode'], mode)
        full, sparse = batches
        for key in ('obs', 'target', 'logp', 'action_mask', 'advantage',
                    'state', 'returns', 'critic_active'):
            torch.testing.assert_close(full[key], sparse[key])
        self.assertLess(int(sparse['decision'].sum()), int(full['decision'].sum()))
        self.assertGreater(int(sparse['decision'].sum()), 0)
        self.assertFalse((sparse['decision'] & ~full['decision']).any())
