"""Baselines for strike-target prioritization."""
import numpy as np

from .env import SELF_FEATURES, TARGET_FEATURES, strike_action


def _records(row, n_targets):
    end = SELF_FEATURES + TARGET_FEATURES * n_targets
    return row[SELF_FEATURES:end].reshape(n_targets, TARGET_FEATURES)


def _current_locks(obs, config):
    locks = np.full(config.n_agents, -1, dtype=int)
    for i, row in enumerate(obs):
        encoded = row[1]
        if row[2] > 0:
            locks[i] = int(round(encoded * config.n_targets)) - 1
    return locks


class RandomPolicy:
    def __init__(self, config, seed=0):
        self.c = config
        self.rng = np.random.default_rng(seed)
        self.locked_target = np.full(config.n_agents, -1, dtype=int)

    def reset(self):
        self.locked_target.fill(-1)

    def predict(self, obs):
        self.locked_target = _current_locks(obs, self.c)
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
        self.locked_target = _current_locks(obs, self.c)
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
                # type 1 life 2/1 -> 1/.5, type 2 -> .4, type 3 -> .2.
                required = 2 if weighted_life > .75 else 1
                needed = max(0, required - assigned[j])
                if needed == 0 or len(observers) < needed:
                    continue
                distances = sorted(
                    ((np.linalg.norm(records[i][j, :2]) * self.c.size, i)
                     for i in observers), key=lambda item: item[0])
                team = [i for _, i in distances[:needed]]
                eta = np.mean([distance for distance, _ in distances[:needed]]) / self.c.strike_speed
                per_drone_value = 2.5 if weighted_life >= .5 else weighted_life * 5.0
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


class StrikeMAPPOPolicy:
    def __init__(self, path, config):
        from .strike_ppo import StrikeActorCritic, load_checkpoint
        from .env import World
        self.model = StrikeActorCritic(World(config).obs_dim, config.n_targets, config.n_agents)
        load_checkpoint(path, self.model)
        self.model.eval()
        self.c = config
        self.locked_target = np.full(config.n_agents, -1, dtype=int)

    def reset(self):
        self.locked_target.fill(-1)

    def predict(self, obs):
        import torch
        with torch.no_grad():
            self.locked_target = _current_locks(obs, self.c)
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            mask = self.model.target_mask(obs_t).numpy()
            valid = ((self.locked_target >= 0)
                     & mask[np.arange(len(obs)), np.maximum(self.locked_target, 0)])
            self.locked_target[~valid] = -1
            action, _ = self.model.act(obs_t, deterministic=True,
                                       locked_target=self.locked_target)
            self.locked_target = action["target"].copy()
        return action
