"""Deadline-aware Q from ordered target-completion event distributions.

A target has 0/1/2 future completed units. Learned time-indexed transition
probabilities describe 0->0/1/2 and 1->1/2; 2 is absorbing. Thus the second
completion cannot precede the first, and remaining life bounds total reward.
No travel-time calculation, oracle, assignment persistence or planner is used.
"""
from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .graph_critic import GraphQBody
from .strike_target_actor import TargetAllocationActorCritic

class TemporalGraphQBody(GraphQBody):
    def __init__(self,hidden=64,horizon=100,gamma=1.,self_attention=False):
        super().__init__(hidden,self_attention=self_attention)
        self.horizon=int(horizon);self.gamma=float(gamma)
        for name in ('team_query','team_key','team_value','team_encoder'):delattr(self,name)
        self.event_head=nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,4*self.horizon))
        nn.init.orthogonal_(self.event_head[-1].weight,gain=.01)
        # Horizon-scaled low event hazards prevent an initially saturated Q.
        with torch.no_grad():self.event_head[-1].bias.copy_(torch.tensor([math.log(max(1,horizon)),0.,0.,-math.log(max(1,horizon))]).repeat(horizon))

    def event_logs(self,state,action):
        _,target,time,_,exists,raw=self.encode(state,action)
        logits=self.event_head(torch.cat((target,time.unsqueeze(-2).expand_as(target)),-1)).reshape(*target.shape[:-1],self.horizon,4)
        life=(raw[...,-1]*2).round().long().clamp(0,2)*exists.long()
        first=logits[...,:3].clone()
        first[...,2]=first[...,2].masked_fill(life.unsqueeze(-1)<2,-torch.inf)
        log0=F.log_softmax(first,-1)
        log12=F.logsigmoid(logits[...,3]);log11=F.logsigmoid(-logits[...,3])
        log12=log12.masked_fill(life.unsqueeze(-1)<2,-torch.inf)
        log11=torch.where(life.unsqueeze(-1)<2,torch.zeros_like(log11),log11)
        return log0,log11,log12,life,raw

    @staticmethod
    def event_masses(log0,log11,log12):
        # Closed-form propagation avoids a Python loop over every future timestep.
        logp0=log0[...,0].cumsum(-1)
        logp0prev=F.pad(logp0[...,:-1],(1,0),value=0.)
        stay1=log11.cumsum(-1)
        logp1=stay1+torch.logcumsumexp(logp0prev+log0[...,1]-stay1,-1)
        p1prev=F.pad(logp1[...,:-1].exp(),(1,0),value=0.)
        first=logp0prev.exp()*(log0[...,1].exp()+log0[...,2].exp())
        second=logp0prev.exp()*log0[...,2].exp()+p1prev*log12.exp()
        return first,second

    def forward(self,state,action):
        log0,log11,log12,life,raw=self.event_logs(state,action)
        first,second=self.event_masses(log0,log11,log12)
        remaining=(state[...,0]*self.horizon).round().long().clamp(0,self.horizon)
        offsets=torch.arange(self.horizon,device=state.device)
        before_deadline=offsets<remaining[...,None]
        discount=torch.pow(torch.as_tensor(self.gamma,device=state.device,dtype=state.dtype),offsets)
        expected=((first+second)*before_deadline.unsqueeze(-2)*discount).sum(-1)
        value=(raw[...,2:5]*torch.tensor([2.5,2.,1.],device=state.device,dtype=state.dtype)).sum(-1)
        n=action.shape[-1]
        has_agent=(state[...,1:1+4*n].reshape(*state.shape[:-1],n,4)[...,2]>.5).any(-1)
        return (expected*value*(life>0)).sum(-1)*has_agent

    def completion_loss(self,state,action,previous_count,next_count,observed):
        log0,log11,log12,life,_=self.event_logs(state,action)
        prev=previous_count.long();nxt=next_count.long()
        log_likelihood=torch.where(prev==0,log0.gather(-1,nxt.clamp(0,2).unsqueeze(-1)).squeeze(-1),torch.where(nxt>prev,log12,log11))
        valid=observed.unsqueeze(-2)&(prev<life.unsqueeze(-1))&(life.unsqueeze(-1)>0)
        # Sum the observed path log-likelihood; average over supervised targets.
        supervised=valid.any(-1)
        if not supervised.any():return self.event_head[-1].weight.sum()*0
        terms=torch.where(valid,log_likelihood,torch.zeros_like(log_likelihood))
        return -terms.sum(-1)[supervised].mean()

class TemporalActorCritic(TargetAllocationActorCritic):
    architecture='target_allocation_temporal'
    def __init__(self,n_targets,n_agents,hidden=64,horizon=100,gamma=1.,actor_time_embedding=True):
        super().__init__(n_targets,n_agents,hidden,actor_time_embedding)
        self.q_body=TemporalGraphQBody(hidden,horizon,gamma)
        self.target_q_body=deepcopy(self.q_body).requires_grad_(False)
        self.q_head=nn.Identity();self.target_q_head=nn.Identity()
    def joint_q(self,state,joint_action):return self.q_body(state,joint_action)
    def target_joint_q(self,state,joint_action):return self.target_q_body(state,joint_action)
    @torch.no_grad()
    def counterfactual_advantages(self,state,joint_action,action_probabilities,sampled_returns=None):
        b,n,t=action_probabilities.shape
        alive=state[:,1:1+4*n].reshape(b,n,4)[:,:,2]>.5
        batch,agent,target=((action_probabilities>0)&alive[:,:,None]).nonzero(as_tuple=True)
        q=state.new_zeros((b,n,t))
        for start in range(0,len(batch),256):
            rows=batch[start:start+256];i=agent[start:start+256];k=target[start:start+256]
            action=joint_action[rows].clone();action[torch.arange(len(rows),device=state.device),i]=k
            q[rows,i,k]=self.joint_q(state[rows],action)
        actual=q.gather(-1,joint_action.unsqueeze(-1)).squeeze(-1)
        outcome=actual if sampled_returns is None else sampled_returns[:,None]
        return (outcome-(q*action_probabilities).sum(-1))*alive


def trajectory_labels(damage,done,horizon):
    """Observed paths from every rollout state; rollout cutoffs are right-censored.

    No observation beyond an environment reset or beyond the actual collected
    suffix is labelled as a non-completion.
    """
    damage=np.asarray(damage,dtype=np.int64);done=np.asarray(done,dtype=bool)
    steps,envs,targets=damage.shape
    previous=np.zeros((steps,envs,targets,horizon),np.int8)
    following=np.zeros_like(previous);observed=np.zeros((steps,envs,horizon),bool)
    for start in range(steps):
        count=np.zeros((envs,targets),np.int64);live=np.ones(envs,bool)
        for offset in range(min(horizon,steps-start)):
            t=start+offset;previous[start,:,:,offset]=count
            count=np.minimum(2,count+damage[t]*live[:,None]);following[start,:,:,offset]=count
            observed[start,:,offset]=live;live=live&~done[t]
            if not live.any():break
    return dict(temporal_previous=previous.reshape(-1,targets,horizon),temporal_next=following.reshape(-1,targets,horizon),temporal_observed=observed.reshape(-1,horizon))
