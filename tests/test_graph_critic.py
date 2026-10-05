import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from pointmass_rl.graph_critic import GraphQBody, RelationAttention
from pointmass_rl.strike_target_actor import GraphTargetScoringCOMAActorCritic, TargetScoringCOMAActorCritic
from pointmass_rl.strike_ppo import save_checkpoint, model_from_checkpoint, train_mappo
from pointmass_rl.env import Config


def pack(time, drones, targets, probabilities, previous):
    return torch.cat((time, drones[..., :4].flatten(-2), targets.flatten(-2),
                      probabilities.transpose(-1, -2).flatten(-2),
                      previous.transpose(-1, -2).flatten(-2), drones[..., 4:].flatten(-2)), -1)


def example(n, t, batch=2):
    drones = torch.rand(batch, n, 6)
    drones[..., 2] = 1
    targets = torch.rand(batch, t, 6)
    targets[..., 2:5] = torch.nn.functional.one_hot(torch.arange(t) % 3, 3)
    probabilities = torch.randn(batch, n, t).softmax(-1)
    action = torch.arange(n).repeat(batch, 1) % t
    previous = torch.nn.functional.one_hot(action, t).float()
    return (torch.ones(batch, 1), drones, targets, probabilities, previous), action


class GraphCriticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)

    def test_same_weights_accept_different_slot_counts(self):
        model = GraphTargetScoringCOMAActorCritic(5, 5)
        other = GraphTargetScoringCOMAActorCritic(12, 8)
        other.q_body.load_state_dict(model.q_body.state_dict())
        other.q_head.load_state_dict(model.q_head.state_dict())
        for n, t in [(1, 1), (2, 3), (5, 5), (8, 12)]:
            fields, action = example(n, t)
            state = pack(*fields)
            actual = model.joint_q(state, action)
            self.assertEqual(actual.shape, (2,))
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, other.joint_q(state, action))
            torch.testing.assert_close(actual, model.target_joint_q(state, action))

    def test_actor_initialization_matches_v12(self):
        torch.manual_seed(7)
        old = TargetScoringCOMAActorCritic(5, 5)
        torch.manual_seed(7)
        new = GraphTargetScoringCOMAActorCritic(5, 5)
        for name in ('actor_body', 'target_head'):
            for key, value in getattr(old, name).state_dict().items():
                torch.testing.assert_close(getattr(new, name).state_dict()[key], value)
        self.assertEqual(old.obs_dim, new.obs_dim)

    def test_aggregation_preserves_duplicate_message_multiplicity(self):
        attention = RelationAttention(16)
        query, source, edges = torch.randn(1, 1, 16), torch.randn(1, 1, 16), torch.randn(1, 1, 1, 5)
        one = attention.aggregate(query, source, edges, torch.ones(1, 1, 1, dtype=torch.bool))
        two = attention.aggregate(query, source.repeat(1, 2, 1), edges.repeat(1, 1, 2, 1),
                                  torch.ones(1, 1, 2, dtype=torch.bool))
        torch.testing.assert_close(two, 2 * one)
        self.assertTrue((one > 0).all())

    def test_joint_permutation_invariance(self):
        body = GraphQBody()
        fields, action = example(3, 4)
        time, drones, targets, probs, previous = fields
        dp, tp = torch.tensor([2, 0, 1]), torch.tensor([3, 1, 0, 2])
        reordered = pack(time, drones[:, dp], targets[:, tp], probs[:, dp][:, :, tp], previous[:, dp][:, :, tp])
        new_action = torch.argsort(tp)[action[:, dp]]
        torch.testing.assert_close(body(pack(*fields), action), body(reordered, new_action), atol=2e-5, rtol=1e-5)

    def test_padding_and_dead_actions_do_not_change_q(self):
        model = GraphTargetScoringCOMAActorCritic(3, 2)
        fields, action = example(2, 3)
        time, drones, targets, probs, previous = fields
        expected = model.joint_q(pack(*fields), action)
        padded = pack(time, torch.cat((drones, torch.zeros(2, 2, 6)), 1),
                      torch.cat((targets, torch.zeros(2, 2, 6)), 1),
                      torch.nn.functional.pad(probs, (0, 2, 0, 2)),
                      torch.nn.functional.pad(previous, (0, 2, 0, 2)))
        padded_action = torch.cat((action, torch.zeros(2, 2, dtype=torch.long)), 1)
        for evaluator in (model.joint_q, model.target_joint_q):
            torch.testing.assert_close(evaluator(padded, padded_action), expected, atol=2e-5, rtol=1e-5)
            padded_action[:, -2:] = 4
            torch.testing.assert_close(evaluator(padded, padded_action), expected, atol=2e-5, rtol=1e-5)
        drones[..., 2] = 0
        self.assertTrue(torch.isfinite(model.joint_q(pack(time, drones, targets, probs, previous), action)).all())

    def test_counterfactual_runtime_sizes_and_action_sensitivity(self):
        model = GraphTargetScoringCOMAActorCritic(5, 5)
        fields, action = example(2, 3)
        state = pack(*fields)
        probabilities = torch.full((2, 2, 3), 1 / 3)
        expected = []
        actual = model.joint_q(state, action)
        for i in range(2):
            alternatives = []
            for j in range(3):
                changed = action.clone()
                changed[:, i] = j
                alternatives.append(model.joint_q(state, changed))
            expected.append(actual - torch.stack(alternatives).mean(0))
        advantage = model.counterfactual_advantages(state, action, probabilities)
        torch.testing.assert_close(advantage, torch.stack(expected, -1), atol=2e-5, rtol=1e-4)
        self.assertGreater(advantage.abs().max().item(), 1e-6)

    def test_gradients_target_update_and_checkpoint(self):
        model = GraphTargetScoringCOMAActorCritic(3, 2)
        fields, action = example(2, 3)
        loss = (model.joint_q(pack(*fields), action) - 3).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.q_body.parameters()))
        before = [p.clone() for p in model.target_q_body.parameters()]
        optimizer = torch.optim.Adam(model.q_body.parameters(), lr=.001)
        optimizer.step()
        model.update_target(.01)
        for old, online, target in zip(before, model.q_body.parameters(), model.target_q_body.parameters()):
            torch.testing.assert_close(target, old * .99 + online * .01)
            self.assertFalse(target.requires_grad)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.pt'
            save_checkpoint(path, model, {})
            restored = model_from_checkpoint(path, Config(n_agents=2, min_agents=2, n_targets=3, min_targets=3))
            resized = GraphTargetScoringCOMAActorCritic(7, 4)
            actor_before = {k: v.clone() for k, v in resized.actor_body.state_dict().items()}
            resized.load_critic_checkpoint(path)
            larger_fields, larger_action = example(4, 7)
            larger_state = pack(*larger_fields)
            torch.testing.assert_close(resized.joint_q(larger_state, larger_action),
                                       model.joint_q(larger_state, larger_action))
            torch.testing.assert_close(resized.target_joint_q(larger_state, larger_action),
                                       model.target_joint_q(larger_state, larger_action))
            for k, v in resized.actor_body.state_dict().items():
                torch.testing.assert_close(v, actor_before[k])
        torch.testing.assert_close(restored.joint_q(pack(*fields), action), model.joint_q(pack(*fields), action))
        torch.testing.assert_close(restored.target_joint_q(pack(*fields), action), model.target_joint_q(pack(*fields), action))

    def test_training_defaults_to_graph_with_original_actor_and_reselection(self):
        config = Config(n_agents=2, min_agents=2, n_targets=3, min_targets=3, horizon=4)
        model, _, _, _ = train_mappo(config, 24, n_envs=2, rollout_steps=4, epochs=1,
                                    critic_epochs=1, eval_interval=0, algorithm='target_coma',
                                    target_coma_actor='mlp')
        self.assertEqual(model.architecture, 'strike_target_coma_v14')
        self.assertFalse(model.training_settings['commit_target'])
        self.assertFalse(model.actor_target_distances)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
