import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from pointmass_rl.env import Config,World
from pointmass_rl.strike_ppo import DecisionSchedule,_model_critic_state,save_checkpoint,model_from_checkpoint
from pointmass_rl.temporal_critic import TemporalActorCritic,TemporalGraphQBody,trajectory_labels

class TemporalCriticTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);torch.manual_seed(7)
 def sample(self):
  c=Config(n_agents=3,min_agents=3,n_targets=3,min_targets=3,horizon=12);w=World(c);w.reset(7)
  m=TemporalActorCritic(3,3,horizon=12);s=torch.tensor(_model_critic_state(m,w,DecisionSchedule(c)))[None];a=torch.tensor([[0,1,2]])
  return c,m,s,a
 def test_vectorized_masses_match_count_recurrence(self):
  log0=torch.randn(2,3,7,3,dtype=torch.float64).log_softmax(-1);p12=torch.rand(2,3,7,dtype=torch.float64)*.8
  first,second=TemporalGraphQBody.event_masses(log0,torch.log1p(-p12),p12.log())
  p0=torch.ones(2,3,dtype=torch.float64);p1=torch.zeros_like(p0);p2=torch.zeros_like(p0);f=[];g=[]
  for k in range(7):
   p=log0[:,:,k].exp();f.append(p0*(p[...,1]+p[...,2]));g.append(p0*p[...,2]+p1*p12[:,:,k])
   p2=p2+g[-1];p1=p1*(1-p12[:,:,k])+p0*p[...,1];p0=p0*p[...,0]
  torch.testing.assert_close(first,torch.stack(f,-1),rtol=1e-12,atol=1e-12);torch.testing.assert_close(second,torch.stack(g,-1),rtol=1e-12,atol=1e-12)
  self.assertTrue(torch.all(second.cumsum(-1)<=first.cumsum(-1)+1e-12))
 def test_episode_reset_and_cutoff_are_censored(self):
  damage=np.array([[[1,0]],[[0,1]],[[0,0]],[[1,1]]]);done=np.array([[0],[1],[0],[0]],bool)
  r=trajectory_labels(damage,done,6)
  np.testing.assert_array_equal(r['temporal_observed'][0],[1,1,0,0,0,0]);np.testing.assert_array_equal(r['temporal_observed'][2],[1,1,0,0,0,0])
  np.testing.assert_array_equal(r['temporal_previous'][0,:,1],[1,0]);np.testing.assert_array_equal(r['temporal_next'][2,:,1],[1,1])
 def test_value_limits_and_event_loss_gradient(self):
  c,m,s,a=self.sample();log0,l11,l12,life,raw=m.q_body.event_logs(s,a);f,g=m.q_body.event_masses(log0,l11,l12)
  self.assertTrue(torch.all(g[life<2]==0));self.assertTrue(torch.all(f.sum(-1)<=1+1e-5));self.assertTrue(torch.all(g.sum(-1)<=1+1e-5))
  maximum=(life*(raw[...,2:5]*torch.tensor([2.5,2.,1.])).sum(-1)).sum(-1)
  q=m.joint_q(s,a);self.assertTrue(torch.all(q>=0));self.assertTrue(torch.all(q<=maximum+1e-4))
  zero=s.clone();zero[:,0]=0;self.assertEqual(float(m.joint_q(zero,a).detach()),0.)
  dead=s.clone();dead[:,3:1+4*3:4]=0;self.assertEqual(float(m.joint_q(dead,a).detach()),0.)
  prev=torch.zeros(1,3,12,dtype=torch.long);nxt=prev.clone();nxt[:,:,2:]=1;prev[:,:,3:]=1;observed=torch.ones(1,12,dtype=torch.bool)
  loss=m.q_body.completion_loss(s,a,prev,nxt,observed);loss.backward()
  self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.q_body.parameters()))
 def test_counterfactual_matches_brute_force(self):
  _,m,s,a=self.sample();p=torch.full((1,3,3),1/3.)
  with torch.no_grad():
   actual=m.joint_q(s,a);expected=[]
   for i in range(3):
    values=[]
    for k in range(3):z=a.clone();z[:,i]=k;values.append(m.joint_q(s,z))
    expected.append(actual-torch.stack(values,-1).mean(-1))
   torch.testing.assert_close(m.counterfactual_advantages(s,a,p),torch.stack(expected,-1),atol=1e-5,rtol=1e-5)
 def test_target_permutation_preserves_q(self):
  _,m,s,a=self.sample();n=t=3;perm=torch.tensor([2,0,1]);inverse=torch.argsort(perm)
  z=s.clone();end=1+4*n
  z[:,end:end+6*t]=s[:,end:end+6*t].reshape(1,t,6)[:,perm].reshape(1,-1)
  for start in [end+6*t,end+6*t+n*t]:z[:,start:start+n*t]=s[:,start:start+n*t].reshape(1,t,n)[:,perm].reshape(1,-1)
  torch.testing.assert_close(m.joint_q(s,a),m.joint_q(z,inverse[a]),atol=1e-5,rtol=1e-5)
 def test_gamma_and_remaining_time_select_reward_prefix(self):
  _,m,s,a=self.sample();h=12
  # Almost deterministic first completion at delta 2, second at delta 4.
  p=torch.zeros(1,3,h,3,dtype=s.dtype)+1e-8;p[...,0]=1-2e-8
  p[:,:,1,:]=torch.tensor([1e-8,1-2e-8,1e-8]);p12=torch.full((1,3,h),1e-8);p12[:,:,3]=1-1e-7
  raw=torch.zeros(1,3,6);raw[:,:,2]=1;raw[:,:,-1]=1
  from unittest.mock import patch
  m.q_body.gamma=.5
  with patch.object(m.q_body,'event_logs',return_value=(p.log(),torch.log1p(-p12),p12.log(),torch.full((1,3),2),raw)):
   expected=3*2.5*(.5+.5**3)
   self.assertAlmostEqual(float(m.joint_q(s,a).detach()),expected,places=4)
   short=s.clone();short[:,0]=3/h
   self.assertAlmostEqual(float(m.joint_q(short,a).detach()),3*2.5*.5,places=4)
 def test_checkpoint_roundtrip_and_count_transfer(self):
  c,m,s,a=self.sample();m.training_settings={'temporal_horizon':12,'temporal_gamma':1.}
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'model.pt';save_checkpoint(p,m,{})
   clone=model_from_checkpoint(p,c)
   torch.testing.assert_close(m.joint_q(s,a),clone.joint_q(s,a))
   other=model_from_checkpoint(p,Config(n_agents=4,min_agents=4,n_targets=5,min_targets=5,horizon=12));self.assertEqual(other.n_agents,4)

if __name__=='__main__':unittest.main()
