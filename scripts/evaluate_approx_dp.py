#!/usr/bin/env python3
"""Evaluate the centralized approximate-DP assignment baseline."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pointmass_rl.env import (Config, EVALUATION_TASKS, SUCCESS_THRESHOLDS, World,
                              evaluation_task_for_episode)
from pointmass_rl.policies import ApproximateDPPolicy


def evaluate(config, episodes, seed):
    task_suite = evaluation_task_for_episode(config, 0) is not None
    cycle = len(EVALUATION_TASKS) if task_suite else len(SUCCESS_THRESHOLDS)
    if episodes % cycle:
        raise ValueError(f"episodes must be a multiple of {cycle}")
    rows = []
    for episode in range(episodes):
        episode_seed = seed + episode
        task = evaluation_task_for_episode(config, episode)
        threshold = (None if task is not None else
                     SUCCESS_THRESHOLDS[episode % len(SUCCESS_THRESHOLDS)])
        world = World(config)
        obs = world.reset(episode_seed, success_threshold=threshold,
                          evaluation_task=task)
        policy = ApproximateDPPolicy(config, episode_seed)
        policy.reset()
        while not world.done:
            obs, _, _, _, _ = world.step(policy.predict(obs))
        rows.append(dict(episode=episode, seed=episode_seed,
                         evaluation_task=task, **world.metrics()))
    return rows


def statistics(rows):
    result = {}
    excluded = {"episode", "seed", "evaluation_task", "agent_terminated"}
    for key in rows[0]:
        if key in excluded or rows[0][key] is None:
            continue
        sample = np.asarray([float(row[key]) for row in rows], dtype=float)
        std = float(sample.std(ddof=1)) if len(sample) > 1 else 0.0
        result[key] = {
            "mean": float(sample.mean()),
            "std": std,
            "ci95_halfwidth": float(1.96 * std / np.sqrt(len(sample))),
            "n": len(sample),
        }
    for threshold in sorted({int(row["success_threshold"]) for row in rows}):
        sample = np.asarray([
            float(row["mission_success"]) for row in rows
            if int(row["success_threshold"]) == threshold], dtype=float)
        result[f"success_b{threshold}"] = {
            "mean": float(sample.mean()),
            "std": float(sample.std(ddof=1)) if len(sample) > 1 else 0.0,
            "n": len(sample),
        }
    for task in EVALUATION_TASKS:
        sample = np.asarray([float(row["mission_success"]) for row in rows
                             if row["evaluation_task"] == task], dtype=float)
        if len(sample):
            result[f"success_{task}"] = {
                "mean": float(sample.mean()),
                "std": float(sample.std(ddof=1)) if len(sample) > 1 else 0.0,
                "n": len(sample),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/five_agents.json")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--out", type=Path,
                        default=Path("runs/train_decentralized/approx_dp_evaluation.json"))
    args = parser.parse_args()
    config = Config.load(args.config)
    rows = evaluate(config, args.episodes, args.seed)
    summary = statistics(rows)
    payload = {
        "policy": "centralized_approximate_dp",
        "episodes": args.episodes,
        "seed": args.seed,
        "config": args.config,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    episode_path = args.out.with_name("approx_dp_episodes.csv")
    with episode_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))
    print(episode_path)


if __name__ == "__main__":
    main()
