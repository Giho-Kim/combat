"""Physical damage upper bound for Target Allocation validation."""
import itertools

import numpy as np

from .strike_lookahead import completion_steps


def physical_optimum(world):
    """Enumerate feasible assignments for stationary deterministic targets."""
    if world.c.target_motion_scale != 0 or world.c.strike_probability != 1:
        raise ValueError('Physical oracle requires stationary deterministic targets')
    if world.c.n_targets ** world.n > 1_000_000:
        raise ValueError('Exact audit enumeration is limited to 1,000,000 joint assignments')
    assignments = np.asarray(list(itertools.product(range(world.c.n_targets), repeat=world.n)))
    feasible = completion_steps(world) <= world.c.horizon - world.t
    values = np.zeros(len(assignments))
    for target in np.flatnonzero(world.target_exists & ~world.destroyed):
        count = ((assignments == target) & feasible[:, target] & world.agent_active).sum(-1)
        count = np.minimum(count, world.target_life[target])
        values += count * {1: 2.5, 2: 2., 3: 1.}[int(world.target_type[target])]
    return float(values.max())
