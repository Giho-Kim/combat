import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from pointmass_rl.attention_actor import PeerCrossAttention
from pointmass_rl.env import Config, World
from pointmass_rl.strike_ppo import model_from_checkpoint, save_checkpoint, train_mappo
from pointmass_rl.strike_target_actor import AttentionTargetScoringCOMAActorCritic


def example(n, t):
    drones = torch.rand(2, n, 5)
    drones[..., 2] = 1
    targets = torch.rand(2, t, 6)
    targets[..., 2:5] = torch.nn.functional.one_hot(torch.arange(t) % 3, 3)
    identity = torch.zeros(2, n)
    identity[:, 0] = 1
    return [torch.rand(2, 3), drones, targets, torch.rand(2, t, n),
            identity, torch.rand(2, t, n), torch.rand(2, n, 2)]


def pack(fields):
    return torch.cat([x.flatten(1) for x in fields], -1)


class AttentionActorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def test_whole_checkpoint_resizes(self):
        model = AttentionTargetScoringCOMAActorCritic(5, 5)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'actor.pt'
            save_checkpoint(path, model, {})
            for n, t in [(1, 1), (2, 3), (5, 5), (8, 12)]:
                restored = model_from_checkpoint(path, Config(n_agents=n, min_agents=n,
                                                             n_targets=t, min_targets=t))
                obs = pack(example(n, t))
                actual = restored.actor_logits(obs)
                self.assertEqual(actual.shape, (2, t))
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, model.actor_logits(obs, n_agents=n, n_targets=t))
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, restored.state_dict()[key])
                world = World(Config(n_agents=n, min_agents=n, n_targets=t, min_targets=t))
                world.reset(seed=7)
                observation = torch.as_tensor(restored.observation(world), dtype=torch.float32)
                _, info = restored.act(observation, deterministic=True)
                self.assertEqual(info['probs'].shape, (n, t))
                self.assertTrue(torch.isfinite(info['probs']).all())
                torch.testing.assert_close(info['probs'].sum(-1), torch.ones(n))

    def test_permutations(self):
        model = AttentionTargetScoringCOMAActorCritic(4, 3)
        fields = example(3, 4)
        m, d, t, p, identity, prev, v = fields
        dp, tp = torch.tensor([2, 0, 1]), torch.tensor([3, 1, 0, 2])
        changed = [m, d[:, dp], t[:, tp], p[:, tp][:, :, dp], identity[:, dp],
                   prev[:, tp][:, :, dp], v[:, dp]]
        torch.testing.assert_close(model.actor_logits(pack(changed)),
                                   model.actor_logits(pack(fields))[:, tp])

    def test_padding_and_ignored_fields(self):
        model = AttentionTargetScoringCOMAActorCritic(3, 2)
        fields = example(2, 3)
        expected = model.actor_logits(pack(fields))
        m, d, t, p, identity, prev, v = fields
        m[:, 1:] = 999
        d[..., 4] = 999
        torch.testing.assert_close(model.actor_logits(pack(fields)), expected)
        padded = [m, torch.nn.functional.pad(d, (0, 0, 0, 2)),
                  torch.nn.functional.pad(t, (0, 0, 0, 2)),
                  torch.nn.functional.pad(p, (0, 2, 0, 2)),
                  torch.nn.functional.pad(identity, (0, 2)),
                  torch.nn.functional.pad(prev, (0, 2, 0, 2)),
                  torch.nn.functional.pad(v, (0, 0, 0, 2))]
        padded[1][:, 2:, :2] = 999
        padded[6][:, 2:] = 999
        torch.testing.assert_close(model.actor_logits(pack(padded), n_agents=4, n_targets=5)[:, :3], expected)

    def test_gradients_and_prior_intent(self):
        model = AttentionTargetScoringCOMAActorCritic(3, 2)
        fields = example(2, 3)
        before = model.actor_logits(pack(fields))
        fields[3][:, 0, 1] += 10
        fields[5][:, 1, 0] += 10
        after = model.actor_logits(pack(fields))
        self.assertGreater((after - before).abs().max().item(), 1e-7)
        after.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            and p.grad.abs().sum() > 0 for p in model.actor_body.parameters()))

    def test_duplicate_messages_and_empty_peers(self):
        layer = PeerCrossAttention(16)
        query, peer = torch.randn(2, 3, 16), torch.randn(2, 3, 1, 16)
        one = layer.aggregate(query, peer, torch.ones(2, 3, 1))
        two = layer.aggregate(query, peer.repeat(1, 1, 2, 1), torch.ones(2, 3, 2))
        torch.testing.assert_close(two, 2 * one)
        torch.testing.assert_close(layer.aggregate(query, peer, torch.zeros(2, 3, 1)), torch.zeros_like(one))

    def test_no_targets_fallback_and_invalid_width(self):
        model = AttentionTargetScoringCOMAActorCritic(3, 2)
        fields = example(2, 3)
        fields[2].zero_()
        probabilities = model.distributions(pack(fields)).probs
        torch.testing.assert_close(probabilities, torch.tensor([[1., 0., 0.], [1., 0., 0.]]))
        with self.assertRaises(ValueError):
            model.actor_logits(pack(fields)[..., :-1])

    def test_training_default(self):
        config = Config(n_agents=2, min_agents=2, n_targets=3, min_targets=3, horizon=4)
        model, _, _, _ = train_mappo(config, 24, n_envs=2, rollout_steps=4, epochs=1,
                                    critic_epochs=1, eval_interval=0, algorithm='target_coma')
        self.assertEqual(model.architecture, 'strike_target_coma_v15')
        self.assertFalse(model.training_settings['commit_target'])
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
