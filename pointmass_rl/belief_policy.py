"""Shared belief encoder with a decentralized actor and joint-action Q."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .attention_actor import PeerCrossAttention
from .graph_critic import RelationAttention
from .belief_env import BELIEF_FEATURES


def stack_observations(observations):
    return {k:torch.as_tensor(np.stack([o[k] for o in observations])) for k in observations[0]}


def subset(batch, indices):
    return {k:v[indices] for k,v in batch.items()}


class BeliefEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(BELIEF_FEATURES, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, records, valid):
        records = torch.where(valid[..., None], records, torch.zeros_like(records))
        return self.net(records) * valid[..., None]


class BeliefActorCritic(nn.Module):
    architecture = 'belief_target_coma_v1'

    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        # ONE module, ONE parameter set, used by both online actor and Q.
        self.belief_encoder = BeliefEncoder(hidden)
        self.actor_query = nn.Sequential(nn.Linear(hidden+12, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.actor_peer = nn.Sequential(nn.Linear(11, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.actor_attention = nn.ModuleList([PeerCrossAttention(hidden) for _ in range(2)])
        self.actor_head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor_head.weight, gain=.01)
        nn.init.zeros_(self.actor_head.bias)
        self.q_drone = nn.Sequential(nn.Linear(9, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.q_time = nn.Linear(1, hidden)
        self.q_target_reads = nn.ModuleList([RelationAttention(hidden) for _ in range(2)])
        self.q_drone_reads = nn.ModuleList([RelationAttention(hidden) for _ in range(2)])
        self.q_gate = nn.Linear(hidden, 1)
        self.q_value = nn.Linear(hidden, hidden)
        self.q_head = nn.Sequential(nn.Linear(2*hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def encode_beliefs(self, obs):
        return self.belief_encoder(obs['beliefs'], obs['valid'])

    def actor_logits(self, obs):
        d = obs['drones']
        b, n, _ = d.shape
        k = obs['beliefs'].shape[1]
        encoded = self.encode_beliefs(obs)
        relative = obs['beliefs'][:, None, :, 12:14] - d[:, :, None, :2]
        own = d[:, :, None, 2:].expand(b, n, k, 7)
        time = obs['time'][:, None, None].expand(b, n, k, 1)
        query = self.actor_query(torch.cat((encoded[:, None].expand(b,n,k,-1), relative, own,
                                           time, obs['previous'][..., None], obs['previous_choice'][..., None]), -1))
        peer_relative = d[:, None, :, :2]-d[:, :, None, :2]
        peer_physical = torch.cat((peer_relative, d[:, None, :, 2:].expand(b,n,n,7)), -1)
        peer = torch.cat((peer_physical[:, :, None].expand(b,n,k,n,9),
                          obs['previous'].transpose(1,2)[:, None, :, :, None].expand(b,n,k,n,1),
                          obs['previous_choice'].transpose(1,2)[:, None, :, :, None].expand(b,n,k,n,1)), -1)
        peer = self.actor_peer(peer)
        mask = ((d[:,:,2]>.5)[:,None,None,:] & obs['valid'][:,None,:,None]
                & ~torch.eye(n,dtype=torch.bool,device=d.device)[None,:,None,:])
        for attention in self.actor_attention:
            query = attention(query,peer,mask)
        return self.actor_head(query).squeeze(-1)

    def distribution(self, obs):
        return Categorical(logits=self.actor_logits(obs).masked_fill(~obs['mask'], -torch.inf))

    def joint_q(self, obs, action, detach_encoder=False):
        drones = obs['drones']
        alive, valid = drones[...,2]>.5, obs['valid']
        tokens = self.encode_beliefs(obs)
        if detach_encoder:
            tokens = tokens.detach()
        n,k = drones.shape[1],tokens.shape[1]
        if action.shape != drones.shape[:2] or torch.any((action<0)|(action>=k)):
            raise ValueError('Q action shape or component IDs are invalid')
        chosen = action[:,None,:]==torch.arange(k,device=action.device)[None,:,None]
        relative = obs['beliefs'][:,:,None,12:14]-drones[:,None,:,:2]
        edges = torch.cat((chosen[...,None].float(),obs['previous'].transpose(1,2)[...,None],
                           obs['previous_choice'].transpose(1,2)[...,None],relative),-1)
        mask = valid[:,:,None] & alive[:,None,:]
        edges = torch.where(mask[...,None],edges,torch.zeros_like(edges))
        d = self.q_drone(torch.where(alive[...,None],drones,torch.zeros_like(drones)))
        time = self.q_time(obs['time'])
        d,tokens = d+time[:,None],tokens+time[:,None]
        reverse = torch.cat((edges[...,:3],-edges[...,3:]),-1).transpose(1,2)
        for target_read,drone_read in zip(self.q_target_reads,self.q_drone_reads):
            tokens = target_read(tokens,d,edges,mask)
            d = drone_read(d,tokens,reverse,mask.transpose(1,2))
        merged = torch.cat((d,tokens),1)
        valid_tokens = torch.cat((alive,valid),1)
        message = self.q_gate(merged).sigmoid()*nn.functional.softplus(self.q_value(merged))
        pooled = (message*valid_tokens[...,None]).sum(1)
        return self.q_head(torch.cat((time,pooled),-1)).squeeze(-1)

    @torch.no_grad()
    def advantages(self, obs, actions, probabilities, sampled_returns=None):
        actual = self.joint_q(obs,actions) if sampled_returns is None else sampled_returns
        result = torch.zeros_like(actions,dtype=torch.float32)
        # Enumerate only valid alternatives; chunk expansion to bound memory.
        for agent in range(actions.shape[1]):
            rows, candidates = obs['mask'][:,agent].nonzero(as_tuple=True)
            baseline = torch.zeros_like(actual)
            for start in range(0,len(rows),64):
                ix, alternative = rows[start:start+64],candidates[start:start+64]
                changed = actions[ix].clone()
                changed[:,agent] = alternative
                q = self.joint_q(subset(obs,ix),changed)
                baseline.index_add_(0,ix,probabilities[ix,agent,alternative]*q)
            result[:,agent] = actual-baseline
        return result

    @torch.no_grad()
    def soft_update(self, source, tau=.01):
        for destination, online in zip(self.parameters(),source.parameters()):
            destination.lerp_(online,tau)


@torch.no_grad()
def choose_action(model, obs, deterministic=False, resolver='none'):
    batch = stack_observations([obs])
    logits = model.actor_logits(batch)[0]
    masks = torch.as_tensor(obs['mask']).clone()
    distribution = Categorical(logits=logits.masked_fill(~masks,-torch.inf))
    action = distribution.probs.argmax(-1) if deterministic else distribution.sample()
    if resolver not in ('none','score','distance'):
        raise ValueError('Unknown resolver')
    if resolver != 'none':
        rejected = torch.zeros_like(masks)
        # Each search component has one destination, so assign one drone to it.
        # Identified objects retain one strike slot per remaining life.
        for _ in range(masks.shape[0]*masks.shape[1]+1):
            changed = False
            for j in range(masks.shape[1]):
                if not obs['valid'][j]:
                    continue
                ids = [i for i in range(len(action)) if obs['active'][i] and int(action[i])==j]
                capacity = (int(round(obs['beliefs'][j,8]*2))
                            if obs['beliefs'][j,10]>=.5 else 1)
                def priority(i):
                    value = (-float(logits[i,j]) if resolver=='score' else
                             float(np.linalg.norm(obs['drones'][i,:2]-obs['beliefs'][j,12:14])))
                    return (bool(obs['decision'][i]), not bool(obs['drones'][i,3]),value,i)
                ids.sort(key=priority)
                for i in ids[capacity:]:
                    if not obs['decision'][i]:
                        continue
                    rejected[i,j]=True
                    alternatives = torch.as_tensor(obs['mask'][i]) & ~rejected[i]
                    if alternatives.any():
                        masks[i]=alternatives
                        dist = Categorical(logits=logits[i].masked_fill(~alternatives,-torch.inf))
                        action[i]=dist.probs.argmax() if deterministic else dist.sample()
                        changed=True
            if not changed:
                break
        distribution = Categorical(logits=logits.masked_fill(~masks,-torch.inf))
    return action.numpy(),distribution.probs.numpy()
