from dataclasses import replace
import unittest

import numpy as np

from pointmass_rl.belief_env import BeliefWorld, TargetBelief
from pointmass_rl.belief_sensor import detection_probabilities, cell_detection_probabilities
from pointmass_rl.env import Config


class SoftSensorTests(unittest.TestCase):
    def field(self, points, **kwargs):
        args=dict(positions=np.array([[0.,0.]]),headings=np.array([0.]),
                  active=np.array([True]),sensor_range=5.,fov_degrees=80.,peak_probability=.9)
        args.update(kwargs)
        return detection_probabilities(np.asarray(points),**args)

    def test_distance_and_angle_taper_continuously(self):
        p=self.field([[0.,0.],[2.5,0.],[4.999,0.],[5.,0.],[5.001,0.]])[0]
        np.testing.assert_allclose(p[:2],[.9,.45])
        self.assertGreater(p[2],0)
        self.assertLess(p[2],1e-6)
        np.testing.assert_equal(p[3:],0.)
        angle=np.deg2rad([0.,20.,39.999,40.,40.001])
        p=self.field(2.5*np.column_stack((np.cos(angle),np.sin(angle))))[0]
        np.testing.assert_allclose(p[:2],[.45,.225])
        self.assertLess(p[2],1e-8)
        self.assertLess(p[3],1e-20)
        self.assertEqual(p[4],0)
        self.assertEqual(self.field([[1.,0.]],active=np.array([False]))[0,0],0)

    def test_cell_average_includes_partial_coverage_and_overlapping_drones(self):
        kwargs=dict(positions=np.array([[0.,0.],[0.,0.]]),headings=np.zeros(2),
                    active=np.ones(2,dtype=bool),sensor_range=5.,fov_degrees=80.,peak_probability=.9)
        center=np.array([[4.8,0.]])
        actual=cell_detection_probabilities(center,1.,**kwargs)[0]
        offsets=(np.arange(101)+.5)/101-.5
        xx,yy=np.meshgrid(offsets,offsets)
        samples=center+np.column_stack((xx.ravel(),yy.ravel()))
        pd=detection_probabilities(samples,**kwargs)
        reference=np.mean(1-np.prod(1-pd,axis=0))
        self.assertGreater(actual,0)
        self.assertLess(actual,1)
        self.assertAlmostEqual(actual,reference,delta=.001)

    def test_empty_sensing_reduces_mass_gradually_without_erasing_support(self):
        config=replace(Config.load('configs/belief.json'),n_agents=1,min_agents=1)
        world=BeliefWorld(config);world.reset(7)
        report=dict(position=[50.,50.],type_counts=[1,1,1],sigma_hint=10.,
                    log_sigma_std=.5,member_radius=6.5)
        world.belief=TargetBelief(config,report)
        world._world.pos[:]=[47.,50.];world._world.heading[:]=0.
        world._world.targets[:]=[85.,85.]
        points=world.belief.points.reshape(-1,2)*config.size
        pd=cell_detection_probabilities(points,2.5,world._world.pos,world._world.heading,
                world._world.agent_active,config.sensor_range,config.sensor_fov_deg,config.detection_probability)
        measured=pd.reshape(world.belief.weights.shape)>.1
        before=world.belief.weights.copy()
        world._sense()
        once=world.belief.weights.copy()
        self.assertTrue((once>0).all())
        self.assertLess(once[measured].sum(),before[measured].sum())
        self.assertAlmostEqual(once.sum(),3.)
        for _ in range(4):
            world._sense()
        self.assertTrue((world.belief.weights>0).all())
        self.assertLess(world.belief.weights[measured].sum(),once[measured].sum())
        np.testing.assert_allclose(world.belief.report.remaining_types,[1,1,1])

    def test_real_detection_uses_spatial_probability_not_uniform_peak(self):
        class FixedRandom:
            def random(self,shape):
                return np.full(shape,.4)
            def normal(self,mean,std,size):
                return np.zeros(size)
        config=replace(Config.load('configs/belief.json'),n_agents=1,min_agents=1)
        world=BeliefWorld(config);world.reset(7)
        world._world.pos[:]=[50.,50.];world._world.heading[:]=0.
        world._world.targets[:]=[54.9,50.]
        world._world.targets[0]=[51.,50.]
        world.sensor_rng=FixedRandom()
        world._sense()
        self.assertEqual([t['handle'] for t in world.belief.tracks],[0])

    def test_soft_settings_require_imperfect_peak(self):
        with self.assertRaises(ValueError):
            Config(mode='belief',belief_soft_detection=True,detection_probability=1.)


if __name__=='__main__':
    unittest.main()
