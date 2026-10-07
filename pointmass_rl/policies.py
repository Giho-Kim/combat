"""Baselines for strike-target prioritization."""
import itertools

import numpy as np

from .env import (DAMAGE_REWARD_RATE, SELF_FEATURES, TARGET_FEATURES,
                  strike_action)


def _records(row, n_targets):
    end = SELF_FEATURES + TARGET_FEATURES * n_targets
    return row[SELF_FEATURES:end].reshape(n_targets, TARGET_FEATURES)


def _current_locks(obs, previous_targets):
    """A progressing strike stays on the target chosen by this policy last step."""
    return np.where(obs[:, 1] > 0, previous_targets, -1)


class RandomPolicy:
    def __init__(self, config, seed=0):
        self.c = config
        self.rng = np.random.default_rng(seed)
        self.locked_target = np.full(config.n_agents, -1, dtype=int)

    def reset(self):
        self.locked_target.fill(-1)

    def predict(self, obs):
        self.locked_target = _current_locks(obs, self.locked_target)
        targets = np.zeros(len(obs), dtype=int)
        for i, row in enumerate(obs):
            known = np.flatnonzero(_records(row, self.c.n_targets)[:, 3] > 0)
            if len(known):
                if self.locked_target[i] in known:
                    targets[i] = self.locked_target[i]
                else:
                    targets[i] = self.rng.choice(known)
                    self.locked_target[i] = targets[i]
        return strike_action(targets)


class HeuristicPolicy:
    """Coordinate distinct targets, using two-drone teams for type 1."""
    def __init__(self, config, seed=0):
        self.c = config
        self.locked_target = np.full(config.n_agents, -1, dtype=int)

    def reset(self):
        self.locked_target.fill(-1)

    def predict(self, obs):
        self.locked_target = _current_locks(obs, self.locked_target)
        targets = np.zeros(len(obs), dtype=int)
        active = np.flatnonzero(np.any(obs != 0, axis=1))
        records = [_records(row, self.c.n_targets) for row in obs]

        # Preserve assignments whose targets are still selectable.
        free = []
        assigned = np.zeros(self.c.n_targets, dtype=int)
        for i in active:
            known = np.flatnonzero(records[i][:, 3] > 0)
            if self.locked_target[i] in known:
                targets[i] = self.locked_target[i]
                assigned[targets[i]] += 1
            else:
                self.locked_target[i] = -1
                free.append(i)

        # Allocate whole strike teams. A type-1 target receives two drones;
        # other targets receive one. Filled targets leave the candidate set,
        # which spreads the formation whenever useful work remains.
        while free:
            candidates = []
            for j in range(self.c.n_targets):
                observers = [i for i in free if records[i][j, 3] > 0]
                if not observers:
                    continue
                sample = records[observers[0]][j]
                weighted_life = float(sample[2])
                # Normalized remaining values uniquely encode the valid cases:
                # type 1 life 2/1 -> 1/.6, type 2 -> .4, type 3 -> .2.
                required = 2 if weighted_life > .75 else 1
                needed = max(0, required - assigned[j])
                if needed == 0 or len(observers) < needed:
                    continue
                distances = sorted(
                    ((np.linalg.norm(records[i][j, :2]) * self.c.size, i)
                     for i in observers), key=lambda item: item[0])
                team = [i for _, i in distances[:needed]]
                eta = np.mean([distance for distance, _ in distances[:needed]]) / self.c.strike_speed
                if weighted_life > .75:
                    per_drone_value = 2.5
                else:
                    per_drone_value = weighted_life * 5.0
                priority = per_drone_value * np.power(.99, eta)
                candidates.append((priority, -eta, j, team))
            if not candidates:
                break
            _, _, target, team = max(candidates, key=lambda item: item[:3])
            for i in team:
                targets[i] = target
                self.locked_target[i] = target
                free.remove(i)
                assigned[target] += 1

        # Extra drones are unavoidable when there are more drones than useful
        # target slots; attach each to its nearest live target.
        for i in free:
            known = np.flatnonzero(records[i][:, 3] > 0)
            if len(known):
                distance = np.linalg.norm(records[i][known, :2], axis=1)
                target = int(known[np.argmin(distance)])
                targets[i] = target
                self.locked_target[i] = target
        return strike_action(targets)


class _CapacityRulePolicy:
    """Keep approach assignments and fill whole strike teams without overflow."""
    def __init__(self, config, seed=0):
        self.c = config
        self.assigned_target = np.full(config.n_agents, -1, dtype=int)

    def reset(self):
        self.assigned_target.fill(-1)

    def _priority(self, remaining_value):
        return 0

    def predict(self, obs):
        records = [_records(row, self.c.n_targets) for row in obs]
        active = np.flatnonzero(np.any(obs != 0, axis=1))
        targets = np.zeros(len(obs), dtype=int)
        assigned = np.zeros(self.c.n_targets, dtype=int)
        free = []

        # Approaching drones reserve slots just like strike participants.
        for i in active:
            j = self.assigned_target[i]
            if j >= 0 and records[i][j, 3] > 0:
                capacity = 2 if records[i][j, 2] > .75 else 1
                if assigned[j] < capacity:
                    targets[i] = j
                    assigned[j] += 1
                    continue
            self.assigned_target[i] = -1
            free.append(i)

        while free:
            candidates = []
            for j in range(self.c.n_targets):
                available = [i for i in free if records[i][j, 3] > 0]
                if not available:
                    continue
                value = float(records[available[0]][j, 2])
                capacity = 2 if value > .75 else 1
                needed = capacity - assigned[j]
                if needed <= 0 or len(available) < needed:
                    continue
                distances = sorted(
                    (float(np.linalg.norm(records[i][j, :2])), i) for i in available)
                team = [i for _, i in distances[:needed]]
                candidates.append((self._priority(value), distances[0][0], j, team))
            if not candidates:
                break
            _, _, j, team = min(candidates, key=lambda item: item[:3])
            for i in team:
                targets[i] = j
                self.assigned_target[i] = j
                assigned[j] += 1
                free.remove(i)

        # The fixed five-target scenario has more life slots than drones. In
        # smaller configurable scenarios, an excess drone has no idle action.
        # Point it at a live target without reserving an additional slot.
        for i in free:
            available = np.flatnonzero(records[i][:, 3] > 0)
            if len(available):
                distances = np.linalg.norm(records[i][available, :2], axis=1)
                targets[i] = int(available[np.argmin(distances)])
        return strike_action(targets)


class NearestTargetPolicy(_CapacityRulePolicy):
    """Fill the closest available target team first."""


class TypePriorityPolicy(_CapacityRulePolicy):
    """Fill type 1, then type 2, then type 3 teams; use distance within type."""
    def _priority(self, remaining_value):
        if remaining_value >= .5:
            return 0  # Type 1, whether it has one or two life remaining.
        return 1 if remaining_value > .3 else 2


class ApproximateDPPolicy:
    """Centralized model-based assignment planner for the stationary scenario.

    The continuous trajectory is approximated by each drone's predicted strike
    completion time. Feasible joint target assignments are exhaustively scored
    with the environment's discounted damage rewards. The
    plan is recomputed only when a target becomes invalid or a strike lock
    changes it.
    """
    def __init__(self, config, seed=0):
        self.c = config
        self.plan = np.full(config.n_agents, -1, dtype=int)
        self.initial_total_value = None
        self.last_remaining = None

    def reset(self):
        self.plan.fill(-1)
        self.initial_total_value = None
        self.last_remaining = None

    @staticmethod
    def _remaining_value(records):
        return np.rint(records[:, 2] * 10.0) / 2.0

    @staticmethod
    def _capacity(value):
        return 2 if value == 5 else (1 if value > 0 else 0)

    def _completion_delay(self, record, progress):
        if progress > 0:
            completed = int(round(progress * self.c.strike_steps_per_life))
            return max(1, self.c.strike_steps_per_life - completed)
        distance = float(np.linalg.norm(record[:2]) * self.c.size)
        travel = max(1, int(np.ceil(
            max(0.0, distance - self.c.strike_range)
            / (self.c.strike_speed * self.c.dt))))
        return travel + self.c.strike_steps_per_life - 1

    def _events(self, assignment, active, records, remaining):
        events = []
        for target in np.flatnonzero(remaining > 0):
            members = [i for i in active if assignment[i] == target]
            if not members:
                continue
            # Locked strike progress is carried in the self record, not the
            # target record.
            delays = sorted(self._completion_delay(
                records[i][target], self._current_progress[i]) for i in members)
            value = float(remaining[target])
            if value == 5:
                if len(delays) >= 2 and delays[0] == delays[1]:
                    events.append((delays[0], 5.0))
                else:
                    events.append((delays[0], 2.5))
                    if len(delays) >= 2:
                        events.append((delays[1], 2.5))
            else:
                events.append((delays[0], float(value)))
        return events

    def _objective(self, assignment, active, records, remaining, damage_so_far,
                   remaining_steps):
        grouped = {}
        for delay, value in self._events(assignment, active, records, remaining):
            grouped[delay] = grouped.get(delay, 0.0) + value
        score = float(damage_so_far)
        objective = 0.0
        total_delay = 0.0
        gamma = self.c.discount_gamma

        for delay in sorted(grouped):
            if delay > remaining_steps:
                break
            value = grouped[delay]
            objective += (gamma ** (delay - 1)
                          * DAMAGE_REWARD_RATE * value)
            score += value
            total_delay += delay * value
        return objective, score, -total_delay

    def _make_plan(self, obs):
        active = np.flatnonzero(np.any(obs != 0, axis=1))
        assignment = np.full(self.c.n_agents, -1, dtype=int)
        if not len(active):
            return assignment
        records = [_records(row, self.c.n_targets) for row in obs]
        remaining = self._remaining_value(records[active[0]])
        if self.initial_total_value is None:
            self.initial_total_value = float(remaining.sum())
        damage_so_far = self.initial_total_value - float(remaining.sum())
        remaining_steps = int(round(obs[active[0], 0] * self.c.horizon))
        locks = _current_locks(obs, self.plan)
        self._current_progress = obs[:, 1].copy()
        counts = np.zeros(self.c.n_targets, dtype=int)
        free = []
        for i in active:
            if locks[i] >= 0:
                assignment[i] = locks[i]
                counts[locks[i]] += 1
            else:
                free.append(int(i))
        choices = [np.flatnonzero(records[i][:, 3] > 0).tolist() for i in free]
        best_key = None
        best_assignment = None
        for proposed in itertools.product(*choices):
            candidate_counts = counts.copy()
            feasible = True
            for target in proposed:
                candidate_counts[target] += 1
                if candidate_counts[target] > self._capacity(remaining[target]):
                    feasible = False
                    break
            if not feasible:
                continue
            candidate = assignment.copy()
            for i, target in zip(free, proposed):
                candidate[i] = target
            key = self._objective(
                candidate, active, records, remaining, damage_so_far,
                remaining_steps)
            if best_key is None or key > best_key:
                best_key, best_assignment = key, candidate
        if best_assignment is not None:
            self.last_remaining = remaining.copy()
            return best_assignment
        # This should only occur when useful target slots are fewer than active
        # drones. Keep the action valid and let the environment reject overflow.
        for i in free:
            valid = np.flatnonzero(records[i][:, 3] > 0)
            if len(valid):
                distance = np.linalg.norm(records[i][valid, :2], axis=1)
                assignment[i] = int(valid[np.argmin(distance)])
        self.last_remaining = remaining.copy()
        return assignment

    def predict(self, obs):
        active = np.flatnonzero(np.any(obs != 0, axis=1))
        locks = _current_locks(obs, self.plan)
        needs_plan = self.initial_total_value is None
        if len(active):
            remaining = self._remaining_value(
                _records(obs[active[0]], self.c.n_targets))
            needs_plan = (needs_plan or self.last_remaining is None
                          or not np.array_equal(remaining, self.last_remaining))
        for i in active:
            records = _records(obs[i], self.c.n_targets)
            target = self.plan[i]
            if locks[i] >= 0 and target != locks[i]:
                needs_plan = True
                break
            if target < 0 or records[target, 3] <= 0:
                needs_plan = True
                break
        if needs_plan:
            self.plan = self._make_plan(obs)
        targets = np.where(self.plan >= 0, self.plan, 0)
        targets = np.where(locks >= 0, locks, targets)
        return strike_action(targets.astype(int))


class StrikeMAPPOPolicy:
    def __init__(self, path, config, resolver=True, device='cpu'):
        from .strike_ppo import model_from_checkpoint
        self.model = model_from_checkpoint(path, config, device=device)
        self.model.eval()
        self.c = config
        self.resolver = resolver
        self.reset()

    def reset(self):
        from .strike_ppo import DecisionSchedule
        self.schedule = DecisionSchedule(self.c)

    def predict(self, obs, world):
        import torch
        from .strike_ppo import scheduled_action
        with torch.no_grad():
            action, _ = scheduled_action(self.model, world, self.schedule, resolver=self.resolver)
            return action
