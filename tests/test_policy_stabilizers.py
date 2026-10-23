from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pointmass_rl.belief_env import BeliefWorld
from pointmass_rl.belief_policy import stack_observations
from pointmass_rl.belief_training import (build_belief_model, load_belief_checkpoint, policy_kl,
                                          save_belief_checkpoint, update_model)
from pointmass_rl.env import Config, strike_action


class PolicyStabilizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **kwargs):
        return replace(Config(mode='belief', n_agents=3, min_agents=3, horizon=40), **kwargs)

    def batch(self, config, steps=8):
        w = BeliefWorld(config)
        obs = [w.reset(7)]
        for _ in range(steps-1):
            obs.append(w.step(strike_action(np.argmax(obs[-1]['mask'], 1)))[0])
        return stack_observations(obs)

    def test_validation(self):
        for change in (dict(belief_logit_cap=-1.), dict(belief_target_kl=float('nan')),
                       dict(belief_separate_critic_encoder=1)):
            with self.subTest(change), self.assertRaises(ValueError):
                Config(mode='belief', **change)

    def test_logit_cap_bounds_logits_and_keeps_order(self):
        c = self.config()
        b = self.batch(c, 1)
        torch.manual_seed(0)
        free = build_belief_model(c)
        capped = build_belief_model(replace(c, belief_logit_cap=3.))
        capped.load_state_dict(free.state_dict())
        torch.nn.init.normal_(free.actor_head.weight, std=200.)
        capped.actor_head.load_state_dict(free.actor_head.state_dict())
        allowed = b['mask']
        raw, bounded = free.actor_logits(b), capped.actor_logits(b)
        gap = raw.masked_fill(~allowed, -torch.inf).amax(-1, keepdim=True)-raw
        self.assertGreater(float(gap[allowed].max()), 3.)
        self.assertTrue(((bounded >= -3.) & (bounded <= 0.))[allowed].all())
        for row in range(allowed.shape[1]):
            r, q = raw[0, row][allowed[0, row]], bounded[0, row][allowed[0, row]]
            self.assertTrue((q[r.argsort()].diff() >= 0).all())  # Monotone; float ties allowed.

    def test_logit_cap_ignores_a_common_shift(self):
        # Regression: capping raw logits saturated every option to +cap once
        # they drifted upward together, which made the policy exactly uniform.
        c = replace(self.config(), belief_logit_cap=3.)
        b = self.batch(c, 1)
        m = build_belief_model(c)
        torch.nn.init.normal_(m.actor_head.weight, std=200.)
        before = m.distribution(b).probs
        with torch.no_grad():
            m.actor_head.bias += 1000.
        torch.testing.assert_close(m.distribution(b).probs, before)
        self.assertLess(float(m.distribution(b).entropy()[b['mask'].sum(-1) > 1].mean()),
                        float(torch.log(b['mask'].sum(-1).float())[b['mask'].sum(-1) > 1].mean()) - .1)

    def test_separate_critic_encoder_isolates_actor_from_q_loss(self):
        c = self.config(belief_separate_critic_encoder=True)
        b = self.batch(c, 1)
        m = build_belief_model(c)
        action = m.distribution(b).sample()
        actor_params = {n for n, _ in m.named_parameters() if n.startswith(('belief_encoder', 'actor_'))}
        for detach in (False, True):  # Warm-up detaching must still train the critic encoder.
            m.zero_grad(set_to_none=True)
            m.joint_q(b, action, detach_encoder=detach).sum().backward()
            grads = {n: p.grad for n, p in m.named_parameters()}
            self.assertTrue(all(grads[n] is None for n in actor_params))
            self.assertGreater(sum(float(g.abs().sum()) for n, g in grads.items()
                                   if n.startswith('q_belief_encoder')), 0)
        m.zero_grad(set_to_none=True)
        m.distribution(b).log_prob(action).sum().backward()
        self.assertTrue(all(p.grad is None for n, p in m.named_parameters() if n.startswith('q_')))

    def test_kl_measure_and_early_stop(self):
        c = self.config()
        b = self.batch(c)
        b['decision'] = b['active'].clone()
        m = build_belief_model(c)
        dist = m.distribution(b)
        self.assertAlmostEqual(float(policy_kl(dist.probs.detach(), dist, b['decision'])), 0., places=6)
        action = dist.sample()
        args = (b, action, dist.log_prob(action).detach(), dist.probs.detach(), torch.ones(len(action)))
        def run(target_kl):
            model = deepcopy(m)
            optimizer = torch.optim.Adam(model.parameters(), lr=.05)
            return update_model(model, deepcopy(model).requires_grad_(False), optimizer, *args, epochs=4,
                                critic_epochs=1, width=2, entropy_coef=.01, clip=.2, target_kl=target_kl)
        free, stopped = run(0.), run(1e-9)
        self.assertFalse(free['kl_stopped'])
        self.assertEqual(free['actor_steps'], 16)
        self.assertTrue(stopped['kl_stopped'])
        self.assertLess(stopped['actor_steps'], free['actor_steps'])
        self.assertGreater(free['policy_kl'], 1e-9)

    def test_checkpoint_round_trip_and_mismatch(self):
        c = self.config(belief_logit_cap=3., belief_separate_critic_encoder=True, belief_target_kl=.02)
        m = build_belief_model(c)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'m.pt'
            save_belief_checkpoint(path, m, deepcopy(m), c, {})
            loaded, config, payload = load_belief_checkpoint(path)
            self.assertEqual((loaded.logit_cap, loaded.separate_critic_encoder), (3., True))
            for key, value in m.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key])
            with self.assertRaisesRegex(ValueError, 'logit cap or critic encoder'):
                load_belief_checkpoint(path, replace(c, belief_logit_cap=0.))


    def test_experiment_configs_with_automaton_input_off_still_load(self):
        from pointmass_rl.env import without_experiment_fields
        data = dict(vars(self.config()), belief_automaton_input='none')
        self.assertEqual(Config(**without_experiment_fields(data)), self.config())
        with self.assertRaisesRegex(ValueError, 'automaton-state'):
            without_experiment_fields(dict(data, belief_automaton_input='self'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_text(json.dumps(dict(data, belief_background_type_prior=list(data['belief_background_type_prior']))))
            loaded = Config.load(path)  # JSON turns the type-prior tuple into a list.
            self.assertEqual(replace(loaded, belief_background_type_prior=tuple(loaded.belief_background_type_prior)),
                             self.config())


if __name__ == '__main__':
    unittest.main()
