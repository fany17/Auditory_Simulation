"""Figures for the frozen D26 scalar result tables."""
from collections import defaultdict
from math import fsum

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = (1024, 2026, 31415)
ARMS = ("HH", "CH", "CC")
COLORS = {"HH": "#245a83", "CH": "#a85e24", "CC": "#397552", "H": "#245a83", "C": "#397552"}
READOUT_COLORS = {"raw": "#657888", "affine": "#ac631f", "Ccap": "#245a83"}


def avg(values):
    return fsum(values) / len(values)


def score_plot(folder, summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    rows = summary["primary"]
    for axis, metric, ylabel in zip(axes, ("r", "MSE"), ("49-person macro Pearson r", "49-person macro MSE")):
        for j, (readout, color) in enumerate(READOUT_COLORS.items()):
            for i, arm in enumerate(ARMS):
                values = [r[metric] for r in rows if r["arm"] == arm and r["readout"] == readout]
                center = i+(j-1)*.22
                axis.scatter([center-.04, center, center+.04], values, s=24, alpha=.62, color=color)
                axis.scatter(center, avg(values), s=55, marker="D", color=color, zorder=3,
                             label=readout if i == 0 else None)
        anchor = avg([r[metric] for r in summary["anchor"]])
        axis.axhline(anchor, color="#434a50", ls="--", lw=1., label="Original FT+Ccap mean")
        if metric == "r":
            top = max(r["mean_r"] for r in summary["historical_references"]["arms"])
            axis.axhline(top, color="#a12a53", ls=":", lw=1.2, label="Best prior frozen arm")
        axis.set(xticks=range(3), xticklabels=ARMS, ylabel=ylabel)
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=7, loc="best")
    fig.savefig(folder / "val_full_matrix.png", dpi=175)
    plt.close(fig)


def objective_plot(folder, epochs, costs):
    groups = defaultdict(list)
    for row in epochs:
        groups[row["seed"], row["stage"], row["branch"]].append(row)
    selected = {(r["seed"], r["stage"], r["branch"]): r for r in costs["stages"]}
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout="constrained")
    for i, seed in enumerate(SEEDS):
        for j, (stage, title) in enumerate((("CL", "Contrastive CE"), ("HEAD", "Frozen encoder + head"), ("FT", "Joint FT"))):
            axis = axes[i, j]
            for branch in ("H", "C") if stage == "CL" else ARMS:
                rows = sorted(groups[seed, stage, branch], key=lambda r: r["epoch"])
                axis.plot([r["epoch"] for r in rows], [r["val"] for r in rows], label=branch, color=COLORS[branch])
                train = [r for r in rows if "train" in r]
                axis.plot([r["epoch"] for r in train], [r["train"] for r in train],
                          color=COLORS[branch], ls="--", lw=.8, alpha=.65)
                chosen = selected[seed, stage, branch]["used_model_epoch"]
                point = next(r for r in rows if r["epoch"] == chosen)
                axis.scatter(chosen, point["val"], s=25, color=COLORS[branch], zorder=3)
            axis.set_title(f"Seed {seed} / {title}", fontsize=10)
            axis.grid(alpha=.2)
            if i == 2:
                axis.set_xlabel("Epoch (zero-based)")
            if i == 0:
                axis.legend(fontsize=8)
    fig.savefig(folder / "training_objectives.png", dpi=170)
    plt.close(fig)


def intervention_plot(folder, summary):
    pairs = summary["paired"]
    scope = [r for r in pairs if r["scope"] == "pure_memory"]
    strata = summary["strata"]
    contrasts = ("near-far", "near-copy")
    colors = ("#245a83", "#a85e24")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained", gridspec_kw={"width_ratios": [1, 1.38]})
    for i, metric in enumerate(("r", "MSE")):
        left, right = axes[i]
        for j, (comparison, color) in enumerate(zip(contrasts, colors)):
            values = [next(r[f"delta_{metric}"] for r in scope if r["seed"] == seed and r["comparison"] == comparison)
                      for seed in SEEDS]
            left.bar([x+(j-.5)*.32 for x in range(3)], values, .28, color=color, label=comparison)
        left.set(xticks=range(3), xticklabels=SEEDS, ylabel=f"Pure-memory Δ {metric}")
        left.axhline(0, color="#555e68", lw=.8)
        left.grid(axis="y", alpha=.2)
        if i == 0:
            left.legend(fontsize=8)
        stories = list(summary["geometry"]["story_people"])
        for j, (comparison, color) in enumerate(zip(contrasts, colors)):
            route = comparison.split("-")[1]
            for k, story in enumerate(stories):
                by_route = {condition: {r["seed"]:r[metric] for r in strata if r["arm"] == "HH"
                                         and r["scope"] == "pure_memory" and r["stratum"] == "story"
                                         and r["label"] == story and r["route"] == condition}
                            for condition in ("near", route)}
                values = [by_route["near"][seed]-by_route[route][seed] for seed in SEEDS]
                pos = k+(j-.5)*.31
                right.scatter(values, [pos]*3, color=color, s=21, alpha=.5)
                right.scatter(avg(values), pos, color=color, s=40, marker="D")
        right.axvline(0, color="#555e68", lw=.8)
        right.set(yticks=range(len(stories)), yticklabels=stories, xlabel=f"Pure-memory Δ {metric}")
        right.grid(axis="x", alpha=.2)
    fig.savefig(folder / "hh_memory_interventions.png", dpi=170)
    plt.close(fig)


def draw(folder, summary, epochs):
    score_plot(folder, summary)
    objective_plot(folder, epochs, summary["actual_costs"])
    intervention_plot(folder, summary)
