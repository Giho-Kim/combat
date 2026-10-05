"""Optional model-assisted policy credit, not an execution-time planner.

Score the actor's sampled joint actions by the damage obtainable if those
assignments are continued. Counterfactual scores hold teammates' actions fixed.
This is auxiliary, biased guidance: it is NOT the learned COMA Q or a claim that
the policy will actually keep its current assignments. Evaluation never uses it.
"""
import numpy as np


def completion_steps(world):
    """Exact earliest strike completion for stationary point-mass targets."""
    distance = np.linalg.norm(world.pos[:, None] - world.targets[None], axis=-1)
    travel = np.maximum(distance - world.c.strike_range, 0) / (
        world.c.strike_speed * world.c.dt)
    eta = np.maximum(1., np.ceil(travel - 1e-9)) + world.c.strike_steps_per_life - 1
    for agent, target in enumerate(world.locked_targets()):
        if target >= 0:
            eta[agent, target] = world.c.strike_steps_per_life - world.agent_strike_progress[agent]
    return np.nan_to_num(eta, nan=np.inf)


def deadline_action_mask(world, available):
    """Remove impossible completions only when a feasible legal action exists.

    No idle action is invented if everything is unreachable. This is an explicit
    model-based execution constraint, used identically in training/evaluation.
    """
    if world.c.target_motion_scale != 0:
        raise ValueError('deadline_mask requires stationary targets')
    feasible = available & (completion_steps(world) <= world.c.horizon - world.t)
    empty = ~feasible.any(axis=-1)
    feasible[empty] = available[empty]
    return feasible


def assignment_values(world, assignments):
    """Smooth deadline-aware damage forecast for a batch of joint assignments."""
    assignments = np.asarray(assignments, dtype=int)
    distance = np.linalg.norm(world.pos[:, None] - world.targets[None], axis=-1)
    travel = np.maximum(distance - world.c.strike_range, 0) / (
        world.c.strike_speed * world.c.dt)
    eta = np.maximum(1., travel) + world.c.strike_steps_per_life - 1
    for agent, target in enumerate(world.locked_targets()):
        if target >= 0:
            eta[agent, target] = world.c.strike_steps_per_life - world.agent_strike_progress[agent]
    eta = np.nan_to_num(eta, nan=1e6)
    remaining = world.c.horizon - world.t
    # Smoothness provides credit near deadlines; a small time preference breaks
    # equal-damage ties. Neither term changes the environment's reward or D.
    feasibility = 1 / (1 + np.exp(np.clip((eta - remaining) / 2., -60, 60)))
    feasibility *= np.exp(-.03 * eta / world.c.horizon)
    result = np.zeros(assignments.shape[:-1], dtype=np.float32)
    for target in np.flatnonzero(world.target_exists & ~world.destroyed):
        capacity = min(int(world.target_life[target]), 2 if world.target_type[target] == 1 else 1)
        if capacity <= 0:
            continue
        assigned = (assignments == target) & world.agent_active
        contribution = assigned * feasibility[:, target]
        best = np.sort(contribution, axis=-1)[..., -capacity:]
        per_life = {1: 2.5, 2: 2., 3: 1.}[int(world.target_type[target])]
        result += per_life * best.sum(axis=-1)
    return result


def counterfactual_credit(world, joint_action, probabilities):
    """Expected-baseline marginal credit for each sampled agent action."""
    action = np.asarray(joint_action)
    alternatives = np.broadcast_to(action, (world.n, world.c.n_targets, world.n)).copy()
    agents = np.arange(world.n)
    targets = np.arange(world.c.n_targets)
    alternatives[agents[:, None], targets[None], agents[:, None]] = targets
    values = assignment_values(world, alternatives)
    actual = values[agents, action]
    return actual - (np.asarray(probabilities) * values).sum(axis=-1)
