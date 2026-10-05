#!/usr/bin/env python3
"""Plot evaluation and rollout learning curves from a completed training run."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


BG = "#ffffff"
PANEL = "#f8fafc"
GRID = "#e2e8f0"
TEXT = "#0f172a"
MUTED = "#64748b"
MAPPO = "#2563eb"
GREEDY = "#059669"
LOOKAHEAD = "#7c3aed"
NEAREST = "#64748b"

plt.rcParams.update({
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": MUTED,
    "axes.titlecolor": TEXT,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "font.size": 10,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
})


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values(rows, key):
    def numeric(value):
        if value == "True":
            return 1.0
        if value == "False":
            return 0.0
        return float(value)
    return np.asarray([numeric(row[key]) for row in rows], dtype=float)


def rolling(values_array, window):
    if len(values_array) < window:
        window = max(1, len(values_array))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values_array, kernel, mode="valid"), window


def rolling_stats(values_array, window):
    mean, window = rolling(values_array, window)
    squared, _ = rolling(np.square(values_array), window)
    std = np.sqrt(np.maximum(0.0, squared - np.square(mean)))
    return mean, std, window


def style_axis(ax, percent=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.spines[["left", "bottom"]].set_linewidth(.8)
    ax.grid(axis="y", color=GRID, linewidth=.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=3, width=.8, color="#94a3b8")
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda value, _: "0" if value == 0 else f"{value / 1000:.0f}k"))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:.0f}%"))


def endpoint(ax, x, y, color=MAPPO):
    ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(-5, 8),
                textcoords="offset points", ha="right", color=color,
                fontsize=9, fontweight="bold")


def plot_evaluation(run_dir):
    rows = read_rows(run_dir / "training_evaluations.csv")
    policies = {name: [row for row in rows if row["policy"] == name]
                for name in ("nearest", "type_priority", "heuristic", "mappo")}
    lookahead_path = run_dir / "approx_dp_evaluation.json"
    lookahead = (json.loads(lookahead_path.read_text(encoding="utf-8"))["summary"]
                 if lookahead_path.exists() else None)
    mappo = policies["mappo"]
    x = values(mappo, "agent_transitions")
    metrics = [
        ("team_return", "Return"),
        ("mission_success", "Mission success"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), layout="constrained")
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, metrics):
        y = (values(mappo, "score") / values(mappo, "baseline_score")
             if key == "score_ratio" else values(mappo, key))
        ax.plot(x, y, color=MAPPO, marker="o", markerfacecolor="white",
                markeredgewidth=1.4, markersize=4.5, linewidth=2.4, label="MAPPO")
        for name, label, color in (("nearest", "Nearest", NEAREST),
                                   ("type_priority", "Type priority", GREEDY),
                                   ("heuristic", "Greedy", GREEDY)):
            if policies[name]:
                baseline = policies[name][0]
                baseline_value = (float(baseline["score"]) / float(baseline["baseline_score"])
                                  if key == "score_ratio" else float(baseline[key]))
                ax.axhline(baseline_value, color=color, linestyle=(0, (4, 3)),
                           linewidth=1.5, label=label)
        if lookahead is not None:
            lookahead_value = (lookahead["score"]["mean"] / lookahead["baseline_score"]["mean"]
                               if key == "score_ratio" else lookahead[key]["mean"])
            ax.axhline(lookahead_value, color=LOOKAHEAD, linestyle=(0, (2, 2)),
                       linewidth=1.7, label="Lookahead")
        ax.set_title(title, loc="left", pad=10)
        ax.set_xlabel("Agent transitions")
        style_axis(ax, key in {"mission_success", "destroyed_fraction"})
        endpoint(ax, x, y)
        if key in {"mission_success", "destroyed_fraction", "score_auc"}:
            ax.set_ylim(-.03, 1.05)
    b_ax = axes[-1]
    task_keys = [key for key in mappo[0] if key.startswith("success_type1_")]
    if task_keys:
        for task, color in zip(task_keys, ("#6b4c9a", "#2b8cbe", "#d64545")):
            b_ax.plot(x, values(mappo, task), marker="o",
                      markerfacecolor="white", markersize=4, linewidth=2,
                      color=color, label=task.removeprefix("success_").replace("_", " "))
        b_ax.set_title("MAPPO success by task", loc="left", pad=10)
    else:
        for threshold, color in zip((5, 8, 9, 11, 12),
                                    ("#6b4c9a", "#2b8cbe", "#34a853", "#f39c12", "#d64545")):
            b_ax.plot(x, values(mappo, f"success_b{threshold}"), marker="o",
                      markerfacecolor="white", markersize=4, linewidth=2,
                      color=color, label=f"B={threshold}")
        b_ax.set_title("MAPPO success by B", loc="left", pad=10)
    b_ax.set_xlabel("Agent transitions")
    b_ax.set_ylim(-.03, 1.05)
    style_axis(b_ax, percent=True)
    b_ax.legend(ncol=2, fontsize=12, loc="lower right",
                handlelength=2.4, columnspacing=1.4)
    policy_handles, policy_labels = axes[0].get_legend_handles_labels()
    fig.legend(policy_handles, policy_labels,
               loc="lower center", bbox_to_anchor=(.5, -.075), ncol=3,
               fontsize=15, handlelength=3.0, columnspacing=2.2)
    output = run_dir / "learning_curves_eval.png"
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_training(run_dir):
    rows = read_rows(run_dir / "training_episodes.csv")
    evaluation_rows = read_rows(run_dir / "training_evaluations.csv")
    baselines = {name:next((row for row in evaluation_rows
                            if row["policy"] == name), None)
                 for name in ("nearest", "type_priority", "heuristic")}
    lookahead_path = run_dir / "approx_dp_evaluation.json"
    lookahead = (json.loads(lookahead_path.read_text(encoding="utf-8"))["summary"]
                 if lookahead_path.exists() else None)
    x = values(rows, "agent_transitions")
    metrics = [
        (values(rows, "team_return"), "team_return", "Return"),
        (values(rows, "mission_success"), "mission_success", "Mission success"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), layout="constrained")
    axes = axes.ravel()
    for ax, (y, key, title) in zip(axes, metrics):
        ax.scatter(x, y, s=7, alpha=.10, color=MAPPO, edgecolors="none")
        smooth, spread, window = rolling_stats(y, 50)
        smooth_x = x[window - 1:]
        ax.fill_between(smooth_x, smooth - spread, smooth + spread,
                        color=MAPPO, alpha=.10, linewidth=0)
        ax.plot(smooth_x, smooth, linewidth=2.5, color=MAPPO,
                label=f"MAPPO ({window}-episode mean)")
        for name, label, color in (("nearest", "Nearest", NEAREST),
                                   ("type_priority", "Type priority", GREEDY),
                                   ("heuristic", "Greedy", GREEDY)):
            if baselines[name] is not None:
                ax.axhline(float(baselines[name][key]), color=color,
                           linestyle=(0, (4, 3)), linewidth=1.5, label=label)
        if lookahead is not None:
            ax.axhline(lookahead[key]["mean"], color=LOOKAHEAD,
                       linestyle=(0, (2, 2)), linewidth=1.7, label="Lookahead")
        ax.set_title(title, loc="left", pad=10)
        ax.set_xlabel("Agent transitions")
        style_axis(ax, title == "Mission success")
        if title == "Mission success":
            ax.set_ylim(-.03, 1.05)
    b_ax = axes[-1]
    success = values(rows, "mission_success")
    thresholds = values(rows, "success_threshold").astype(int)
    for threshold, color in zip((5, 8, 9, 11, 12),
                                ("#6b4c9a", "#2b8cbe", "#34a853", "#f39c12", "#d64545")):
        selected = thresholds == threshold
        bx, by = x[selected], success[selected]
        smooth, window = rolling(by, 20)
        b_ax.plot(bx[window - 1:], smooth, linewidth=2.1, color=color,
                  label=f"B={threshold}")
    b_ax.set_title("Training success by B", loc="left", pad=10)
    b_ax.set_xlabel("Agent transitions")
    b_ax.set_ylim(-.03, 1.05)
    style_axis(b_ax, percent=True)
    b_ax.legend(ncol=2, fontsize=12, loc="lower right",
                handlelength=2.4, columnspacing=1.4)
    mean_handles, mean_labels = axes[0].get_legend_handles_labels()
    fig.legend(mean_handles, mean_labels,
               loc="lower center", bbox_to_anchor=(.5, -.075), ncol=3,
               fontsize=15, handlelength=3.0, columnspacing=2.2)
    output = run_dir / "learning_curves_train.png"
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(plot_evaluation(args.run_dir))
    print(plot_training(args.run_dir))


if __name__ == "__main__":
    main()
