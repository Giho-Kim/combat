"""Training-only progress shaping and avoidable reassignment cost."""
import numpy as np

from .env import PROGRESS_REWARD_RATE


def arrival_steps(world):
    distance = np.linalg.norm(world.pos[:, None] - world.targets[None], axis=-1)
    travel = np.maximum(distance - world.c.final_mission_range, 0) / (
        world.c.final_mission_speed * world.c.dt)
    eta = travel + world.c.service_steps_per_unit
    for agent, target in enumerate(world.locked_targets()):
        if target >= 0:
            eta[agent, target] = max(
                0, world.c.service_steps_per_unit - world.agent_final_mission_progress[agent])
    return np.nan_to_num(eta, nan=np.inf)


def progress_potential(world):
    """Value-weighted proximity of existing, capacity-limited assignments.

    Terminal potential is zero, so gamma*Phi(next)-Phi(current) telescopes
    across the complete episode. No expert action or imitation target is used.
    """
    if world.done:
        return 0.0
    eta = arrival_steps(world)
    remaining = world.c.horizon - world.t
    per_work_units = np.select([world.target_type == 1, world.target_type == 2],
                         [2.5, 2.0], default=1.0) * PROGRESS_REWARD_RATE
    potential = 0.0
    for target in np.flatnonzero(world.target_exists & ~world.target_completed):
        agents = np.flatnonzero(world.target_assignment[target] & world.agent_active)
        capacity = min(int(world.target_work_units[target]),
                       2 if world.target_type[target] == 1 else 1)
        times = np.sort(eta[agents, target])[:capacity]
        # Smooth deadline penalty avoids a reward cliff when ETA crosses horizon.
        proximity = np.exp(-times / 20.0)
        feasibility = np.exp(-np.maximum(times - remaining, 0) / 5.0)
        potential += per_work_units[target] * float(np.sum(proximity * feasibility))
    return potential


def avoidable_switches(world, targets):
    """Charge only leaving a live, reachable assignment while free to move."""
    assigned = world.target_assignment.T
    old_targets = assigned.argmax(axis=-1)
    agents = np.arange(world.n)
    eta = arrival_steps(world)
    eligible = (world.agent_active & (world.locked_targets() < 0)
                & assigned.any(axis=-1)
                & world.target_exists[old_targets] & ~world.target_completed[old_targets]
                & (world.target_work_units[old_targets] > 0)
                & (eta[agents, old_targets] <= world.c.horizon - world.t))
    return int(np.count_nonzero(eligible & (old_targets != targets)))
