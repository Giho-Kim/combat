from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pointmass_rl.belief_env import BeliefWorld, TargetBelief
from pointmass_rl.belief_policy import BeliefActorCritic, choose_action, stack_observations
from pointmass_rl.belief_training import load_belief_checkpoint, save_belief_checkpoint, train_belief
from pointmass_rl.env import Config, strike_action


class ReportBeliefTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **kwargs):
        # This suite covers the legacy report histogram, independently of
        # whichever representation the current training config selects.
        return replace(Config.load('configs/belief.json'), belief_representation='histogram', **kwargs)

    def report(self, counts=(1,1,1)):
        return dict(position=[50.,50.], type_counts=list(counts),
                    sigma_hint=10., log_sigma_std=.5, member_radius=6.5)

    def test_report_precedes_displacement_and_preserves_second_formation(self):
        c = self.config()
        offsets, scales = [], []
        for seed in range(20):
            old, new = BeliefWorld(replace(c,belief_initial_intel=False)), BeliefWorld(c)
            old.reset(seed)
            new.reset(seed)
            w, report = new._world, new.belief.initial_intel
            np.testing.assert_equal(report['position'],old._world.formation_centers[0])
            np.testing.assert_equal(w.friendly_center,old._world.friendly_center)
            second = w.target_exists & (w.target_formation == 1)
            first = w.target_exists & (w.target_formation == 0)
            np.testing.assert_equal(w.targets[second],old._world.targets[second])
            np.testing.assert_equal(report['type_counts'],np.bincount(w.target_type[first],minlength=4)[1:])
            offset = np.linalg.norm(w.formation_centers[0]-report['position'])
            offsets.append(offset)
            scales.append(w._intel_true_sigma)
            self.assertTrue((w.targets[w.target_exists] >= c.belief_search_margin).all())
            self.assertTrue((w.targets[w.target_exists] <= c.size-c.belief_search_margin).all())
            self.assertEqual(set(report),{'position','type_counts','sigma_hint','log_sigma_std','member_radius'})
            for position in w.targets[first]:
                likelihood = new.belief.report._measurement_likelihood(position)
                self.assertGreater(float(np.sum(likelihood*new.belief.report.spatial)),0.)
        self.assertLess(min(offsets),10.)
        self.assertGreater(max(offsets),20.)
        self.assertGreater(np.ptp(scales),1.)

    def test_gaussian_prior_has_a_central_peak_and_continuous_tails(self):
        b = TargetBelief(self.config(),self.report())
        distance = np.linalg.norm(b.points*100-[50,50],axis=-1)
        self.assertGreater(b.report.spatial[distance < 5].mean(),
                           b.report.spatial[(distance > 15) & (distance < 20)].mean())
        self.assertTrue((b.report.spatial > 0).all())
        np.testing.assert_allclose(b.report.scale_posterior,b.report.scale_prior)
        self.assertGreater(b.report.sigma_std,0.)

    def test_empty_centre_updates_unknown_gaussian_scale(self):
        b = TargetBelief(self.config(),self.report())
        initial_sigma = b.report.sigma_mean
        scanned = (np.linalg.norm(b.points*100-[50,50],axis=-1) < 15).astype(float)
        b.update(scanned,[],1)
        self.assertGreater(b.report.sigma_mean,initial_sigma)
        np.testing.assert_allclose(b.report.remaining_types,[1,1,1])
        self.assertAlmostEqual(b.report.scale_posterior.sum(),1.)

    def test_detection_localizes_other_members_and_updates_scale(self):
        b = TargetBelief(self.config(n_targets=3,min_targets=3),self.report((2,0,0)))
        # Isolate the reported formation to inspect the conditional update.
        b.report.unreported_likelihood = 0.
        original_sigma = b.report.sigma_mean
        position = np.array([75.,50.])
        original_mean = (b.points*100*b.report.spatial[...,None]).sum((0,1))
        b.update(np.zeros_like(b.weights),[(8,position,1,2)],1)
        updated_mean = (b.points*100*b.report.spatial[...,None]).sum((0,1))
        self.assertLess(np.linalg.norm(updated_mean-position),np.linalg.norm(original_mean-position))
        self.assertGreater(updated_mean[0],65.)
        self.assertGreater(b.report.sigma_mean,original_sigma)
        np.testing.assert_allclose(b.report.remaining_types,[1,0,0])

    def test_empty_observations_redistribute_known_existence_and_types(self):
        belief = TargetBelief(self.config(),self.report())
        before = belief.report.spatial.copy()
        miss = (belief.points[...,0] < .5).astype(float)
        belief.update(miss,[],1)
        np.testing.assert_allclose(belief.report.remaining_types,[1,1,1])
        self.assertAlmostEqual(belief.report.spatial.sum(),1.)
        self.assertTrue((belief.report.spatial[miss == 1] == 0).all())
        self.assertGreater(belief.report.spatial[miss == 0].sum(),before[miss == 0].sum())
        self.assertAlmostEqual(belief.weights.sum(),3.)
        np.testing.assert_allclose(belief.records[:belief.cells,9],belief.weights.sum(-1))
        self.assertTrue(np.isfinite(belief.records).all())

    def test_report_detection_transfers_to_track_once(self):
        c = self.config(n_targets=1,min_targets=1)
        b = TargetBelief(c,self.report((1,0,0)))
        index = np.unravel_index(b.weights.argmax(),b.weights.shape)
        position = b.points[index]*c.size
        observation = [(178,position,1,2)]
        b.update(np.zeros_like(b.weights),observation,1)
        self.assertAlmostEqual(b.weights.sum(),0.)
        self.assertEqual(len(b.tracks),1)
        self.assertEqual(b.records[b.cells,9],1.)
        b.update(np.zeros_like(b.weights),observation,2)
        self.assertEqual(len(b.tracks),1)
        self.assertAlmostEqual(b.report.remaining_types.sum(),0.)

    def test_ambiguous_detection_marginalizes_source_instead_of_reading_id(self):
        c = self.config(n_targets=2,min_targets=2)
        a = TargetBelief(c,self.report((1,0,0)))
        b = deepcopy(a)
        index = np.unravel_index(a.weights.argmax(),a.weights.shape)
        position = a.points[index]*c.size
        a.update(np.zeros_like(a.weights),[(1,position,1,2)],1)
        b.update(np.zeros_like(b.weights),[(900,position,1,2)],1)
        self.assertGreater(a.report.remaining_types[0],0.)
        self.assertLess(a.report.remaining_types[0],1.)
        np.testing.assert_equal(a.records,b.records)
        np.testing.assert_equal(a.weights,b.weights)

    def test_complete_detection_removes_unseen_belief_at_varied_counts(self):
        for total in (1,5,9):
            c = self.config(n_targets=total,min_targets=total,sensor_range=200,sensor_fov_deg=360,
                            belief_soft_detection=False,detection_probability=1.)
            w = BeliefWorld(c)
            obs = w.reset(8)
            self.assertEqual(len(w.belief.tracks),total)
            self.assertFalse(obs['valid'][:w.belief.cells].any())
            self.assertAlmostEqual(w.belief.report.remaining_types.sum(),0.)
            self.assertAlmostEqual(w.belief.weights.sum(),0.)

    def test_report_filter_and_network_do_not_read_unobserved_truth(self):
        w = BeliefWorld(self.config())
        obs = w.reset(41)
        self.assertEqual(len(w.belief.tracks),0)
        reference = deepcopy(w)
        corners = np.array([[10,10],[10,90],[90,10],[90,90]])
        far = corners[np.argmax(np.linalg.norm(corners-w._world.friendly_center,axis=1))]
        w._world.targets[:] = far
        w._world.target_type[:] = 3
        w._world.target_life[:] = 1
        w._world._intel_true_sigma *= 10
        w._sense()
        reference._sense()
        obs = reference.observation()
        changed = w.observation()
        for key in obs:
            np.testing.assert_equal(obs[key],changed[key],err_msg=key)
        model = BeliefActorCritic()
        original, modified = stack_observations([obs]),stack_observations([changed])
        action = model.distribution(original).sample()
        torch.testing.assert_close(model.actor_logits(original),model.actor_logits(modified))
        torch.testing.assert_close(model.joint_q(original,action),model.joint_q(modified,action))

    def test_lawnmower_does_not_lock_out_new_decisions(self):
        w = BeliefWorld(self.config())
        w.reset(7)
        slot = int(np.flatnonzero(w.belief.valid[:w.belief.cells])[0])
        w.chosen[0] = slot
        w.navigation[0] = dict(component=slot,sweeping=True)
        mask, decision = w.action_mask()
        self.assertTrue(decision[0])
        self.assertGreater(mask[0].sum(),1)
        self.assertFalse(mask[1:,slot].any())

    def test_partial_sensor_coverage_cannot_erase_an_entire_bin(self):
        w = BeliefWorld(self.config(n_agents=1,min_agents=1))
        w.reset(7)
        b = w.belief
        index = np.unravel_index(b.report.spatial.argmax(),b.weights.shape)
        center = b.points[index]*w.c.size
        w._world.pos[0] = center-[4.9,0]
        w._world.heading[0] = 0.
        w._world.targets[:] = center+[10.,10.]
        self.assertTrue(w._visibility(center[None])[0,0])
        w._sense()
        self.assertGreater(b.report.spatial[index],0.)

    def test_dubins_search_approaches_a_viewpoint_that_clears_the_candidate(self):
        w = BeliefWorld(self.config(n_agents=1,min_agents=1,belief_lawnmower=False,
                                   belief_soft_detection=False,detection_probability=1.))
        w.reset(7)
        b = w.belief
        slot = int(np.argmax(b.weights.sum(-1)))
        goal = b.goals[slot].copy()
        index = int(np.argmin(np.linalg.norm(b.points[slot]*w.c.size-goal,axis=-1)))
        w._world.pos[0] = goal-[6.,0]
        w._world.heading[0] = 0.
        for _ in range(30):
            if b.weights[slot,index] == 0:
                break
            w.step(strike_action(np.array([slot])))
        self.assertEqual(b.weights[slot,index],0.)

    def test_unreported_type_does_not_consume_reported_roster(self):
        b = TargetBelief(self.config(),self.report((2,1,0)))
        index = np.unravel_index(b.weights.argmax(),b.weights.shape)
        position = b.points[index]*b.c.size
        b.update(np.zeros_like(b.weights),[(5,position,3,1)],1)
        np.testing.assert_allclose(b.report.remaining_types,[2,1,0])

    def test_checkpoint_preserves_scenario_and_old_config_stays_legacy(self):
        model = BeliefActorCritic()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            c = self.config()
            save_belief_checkpoint(path,model,deepcopy(model),c,{})
            _,loaded,payload = load_belief_checkpoint(path)
            self.assertEqual(loaded,c)
            for key in ('belief_initial_intel','belief_intel_sigma','belief_intel_log_sigma_std',
                        'belief_soft_detection'):
                payload['config'].pop(key)
            torch.save(payload,path)
            _,legacy,_ = load_belief_checkpoint(path)
            self.assertFalse(legacy.belief_initial_intel)
            self.assertFalse(legacy.belief_soft_detection)

    def test_new_scenario_training_updates_and_execution(self):
        c = self.config(horizon=3,n_agents=3,min_agents=1,randomize_counts=True,min_targets=3)
        with tempfile.TemporaryDirectory() as directory:
            model,episodes,_,steps = train_belief(c,32,directory,n_envs=2,rollout_steps=4,
                epochs=1,critic_epochs=1,minibatch_size=4,eval_interval=0)
            self.assertTrue(episodes)
            self.assertGreaterEqual(steps,32)
            loaded,config,_ = load_belief_checkpoint(Path(directory)/'latest.pt')
            self.assertTrue(config.belief_initial_intel)
            for parameter in model.parameters():
                self.assertTrue(torch.isfinite(parameter).all())
            w = BeliefWorld(config)
            obs = w.reset(10000)
            while not w.done:
                action,prob = choose_action(loaded,obs,True,'score')
                obs,*_ = w.step(strike_action(action),prob)


if __name__ == '__main__':
    unittest.main()
