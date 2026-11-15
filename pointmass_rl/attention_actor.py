"""Slot-independent target scoring from own query and observed peer tokens.

No current joint action enters this actor. Sigmoid-gated positive messages
retain peer multiplicity without an explicit count feature.
"""
import torch
from torch import nn


class PeerCrossAttention(nn.Module):
    def __init__(self, hidden=64, heads=4):
        super().__init__()
        if hidden % heads:
            raise ValueError('hidden must be divisible by heads')
        self.heads, self.width = heads, hidden // heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.ff = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def aggregate(self, query, peers, mask):
        q = self.query(query).reshape(*query.shape[:-1], self.heads, self.width).unsqueeze(-3)
        k = self.key(peers).reshape(*peers.shape[:-1], self.heads, self.width)
        v = torch.nn.functional.softplus(self.value(peers)).reshape_as(k)
        gate = torch.sigmoid((q * k).sum(-1) / self.width ** .5)
        return (gate.unsqueeze(-1) * v * mask[..., None, None]).sum(-3).flatten(-2)

    def forward(self, query, peers, mask):
        message = self.aggregate(query, peers, mask)
        return query + self.ff(torch.cat((query, message), -1))


class AttentionActorBody(nn.Module):
    def __init__(self, hidden=64, heads=4, layers=2, time_embedding=False,
                 query_width=13, peer_width=7, time_index=6):
        super().__init__()
        # Query: target6, time1, own previous probability/choice2,
        # own alive/idle2 and velocity2. Peer: xy2, idle1, velocity2, intent2.
        self.time_embedding = time_embedding
        self.time_index = time_index
        self.query_encoder = nn.Sequential(nn.Linear(query_width-int(time_embedding), hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.peer_encoder = nn.Sequential(nn.Linear(peer_width, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        if time_embedding:
            self.time_encoder = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([PeerCrossAttention(hidden, heads) for _ in range(layers)])

    def forward(self, query_features, peer_features, mask):
        if self.time_embedding:
            index = self.time_index
            time = self.time_encoder(query_features[..., index:index+1])
            entity = torch.cat((query_features[..., :index], query_features[..., index+1:]), -1)
            query = self.query_encoder(entity) + time
            peers = self.peer_encoder(peer_features)
        else:
            # Restore the original scalar-time architecture for old checkpoints.
            query = self.query_encoder(query_features)
            peers = self.peer_encoder(peer_features)
        for layer in self.layers:
            query = layer(query, peers, mask)
        return query
