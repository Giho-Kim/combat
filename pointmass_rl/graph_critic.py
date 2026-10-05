"""Size-independent relational critic with unnormalized, gated attention.

Sigmoid gates do not divide by neighborhood size: duplicating an identical
message doubles its aggregate. No count feature or slot-ID embedding is used.
"""
import torch
from torch import nn


class RelationAttention(nn.Module):
    def __init__(self, hidden, heads=4):
        super().__init__()
        if hidden % heads:
            raise ValueError('hidden must be divisible by heads')
        self.heads, self.width = heads, hidden // heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.edge_key = nn.Linear(5, hidden)
        self.edge_value = nn.Linear(5, hidden)
        # Do not normalize the aggregate: its magnitude contains multiplicity.
        self.update = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden))

    def aggregate(self, query, source, edges, mask):
        shape = (*edges.shape[:-1], self.heads, self.width)
        q = self.query(query).unsqueeze(-2).reshape(
            *query.shape[:-1], 1, self.heads, self.width)
        k = (self.key(source).unsqueeze(-3) + self.edge_key(edges)).reshape(shape)
        v = torch.nn.functional.softplus(
            self.value(source).unsqueeze(-3) + self.edge_value(edges)).reshape(shape)
        gates = torch.sigmoid((q * k).sum(-1) / self.width ** .5)
        messages = gates.unsqueeze(-1) * v * mask[..., None, None]
        return messages.sum(dim=-3).flatten(-2)

    def forward(self, query, source, edges, mask):
        message = self.aggregate(query, source, edges, mask)
        return query + self.update(torch.cat((query, message), dim=-1))


class GraphQBody(nn.Module):
    """All parameter shapes are independent of drone/target slot counts."""
    def __init__(self, hidden=64, heads=4, rounds=2):
        super().__init__()
        self.drone_encoder = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.target_encoder = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.time_encoder = nn.Linear(1, hidden)
        self.target_reads = nn.ModuleList([RelationAttention(hidden, heads) for _ in range(rounds)])
        self.drone_reads = nn.ModuleList([RelationAttention(hidden, heads) for _ in range(rounds)])
        self.team_query = nn.Parameter(torch.zeros(hidden))
        self.team_key = nn.Linear(hidden, hidden)
        self.team_value = nn.Linear(hidden, hidden)
        self.team_encoder = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    @staticmethod
    def dimensions(state, joint_action):
        n = joint_action.shape[-1]
        remaining = state.shape[-1] - 1 - 6 * n
        t, remainder = divmod(remaining, 6 + 2 * n)
        if n < 1 or t < 1 or remainder or state.shape[:-1] != joint_action.shape[:-1]:
            raise ValueError('Invalid graph critic state/action shape')
        if joint_action.dtype not in (torch.int32, torch.int64):
            raise ValueError('joint_action must contain integer target IDs')
        if torch.any((joint_action < 0) | (joint_action >= t)):
            raise ValueError('joint_action contains an invalid target')
        return n, t

    def forward(self, state, joint_action):
        n, t = self.dimensions(state, joint_action)
        drone_end, target_end = 1 + 4 * n, 1 + 4 * n + 6 * t
        drones = state[..., 1:drone_end].reshape(*state.shape[:-1], n, 4)
        targets = state[..., drone_end:target_end].reshape(*state.shape[:-1], t, 6)
        velocity = state[..., -2 * n:].reshape(*state.shape[:-1], n, 2)
        probability = state[..., target_end:target_end + n * t].reshape(*state.shape[:-1], t, n)
        previous = state[..., target_end + n * t:target_end + 2 * n * t].reshape(*state.shape[:-1], t, n)
        alive = drones[..., 2] > .5
        exists = targets[..., 2:5].sum(-1) > .5
        mask = exists.unsqueeze(-1) & alive.unsqueeze(-2)
        chosen = joint_action.unsqueeze(-2) == torch.arange(t, device=state.device).unsqueeze(-1)
        relative = targets[..., :, None, :2] - drones[..., None, :, :2]
        edges = torch.cat((chosen.to(state.dtype).unsqueeze(-1),
                           probability.unsqueeze(-1), previous.unsqueeze(-1), relative), -1)
        edges = torch.where(mask.unsqueeze(-1), edges, torch.zeros_like(edges))
        drones = torch.where(alive.unsqueeze(-1), torch.cat((drones, velocity), -1),
                             torch.zeros_like(torch.cat((drones, velocity), -1)))
        targets = torch.where(exists.unsqueeze(-1), targets, torch.zeros_like(targets))
        d, target = self.drone_encoder(drones), self.target_encoder(targets)
        time = self.time_encoder(state[..., :1])
        d, target = d + time.unsqueeze(-2), target + time.unsqueeze(-2)
        reverse = torch.cat((edges[..., :3], -edges[..., 3:]), -1).transpose(-3, -2)
        for target_read, drone_read in zip(self.target_reads, self.drone_reads):
            target = target_read(target, d, edges, mask)
            d = drone_read(d, target, reverse, mask.transpose(-1, -2))
        tokens = torch.cat((d, target), dim=-2)
        valid = torch.cat((alive, exists), dim=-1)
        query = time + self.team_query
        gates = torch.sigmoid((self.team_key(tokens) * query.unsqueeze(-2)).sum(-1)
                              / query.shape[-1] ** .5)
        messages = torch.nn.functional.softplus(self.team_value(tokens))
        summary = (messages * (gates * valid).unsqueeze(-1)).sum(-2)
        return self.team_encoder(torch.cat((time, summary), -1))
