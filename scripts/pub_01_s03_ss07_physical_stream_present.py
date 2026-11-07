"""Produce D19 tables and static figures from completed scalar evidence on 2203."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from pub_01_s03_ss07_physical_stream_meta import ROOT, ROLES, SEEDS, read_json, write_json


COLORS = ("#005a8c", "#bc412b", "#617924")
LABELS = ("C · Current", "H · EEG history", "P · Prediction prior")


def csv_file(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract(fit, receipt):
    seed_rows, epochs, people, gradients, pairs = [], [], [], [], []
    anchor_diagnostics = []
    for seed in SEEDS:
        anchor_file = read_json(fit / f"seed{seed}_ANCHOR.json")
        anchor = anchor_file["anchor"]
        for row in anchor_file["autocorrelation"]:
            anchor_diagnostics.append(dict(seed=seed, **row))
        base = {row["subject"]: row for row in anchor["val"]["people"]}
        for role in ROLES:
            folder = fit / f"seed{seed}_{role}"
            result = read_json(folder / "RESULT.json")
            track = result["epochs"]
            assert [row["epoch"] for row in track] == list(range(-1, 20))
            best = max(track, key=lambda row: row["val"]["r"])
            trained = max(track[1:], key=lambda row: row["val"]["r"])
            assert result["selected_epoch"] == best["epoch"]
            selected = result["selected"]
            pair = result["val_pair"]
            assert selected["r"] == best["val"]["r"] and len(selected["people"]) == 49
            assert len(selected["blocks"]) == 752
            row = dict(seed=seed, role=role, selected_epoch=best["epoch"],
                       anchor_r=anchor["val"]["r"], selected_r=selected["r"],
                       selected_delta_r=selected["r"] - anchor["val"]["r"],
                       best_trained_epoch=trained["epoch"], best_trained_r=trained["val"]["r"],
                       best_trained_delta_r=trained["val"]["r"] - anchor["val"]["r"],
                       last_val_r=track[-1]["val"]["r"],
                       selected_val_MSE=selected["MSE"], anchor_val_MSE=anchor["val"]["MSE"],
                       selected_val_active_MSE=selected["active_MSE"],
                       selected_residual_RMS=selected["residual_RMS"],
                       near_minus_older=None if pair is None else pair["near"]["r"] - pair["older"]["r"],
                       older_pair_status=("not_applicable" if pair is None else
                                          "anchor_only" if best["epoch"] == -1 else "trained"),
                       anchor_train_active_MSE=anchor["train"]["active_MSE"],
                       last_train_active_MSE=track[-1]["train"]["active_MSE"])
            seed_rows.append(row)
            old_people = {} if pair is None else {p["subject"]: p for p in pair["older"]["people"]}
            for item in selected["people"]:
                subject = item["subject"]
                people.append(dict(seed=seed, role=role, subject=subject, blocks=item["blocks"],
                                   selected_epoch=best["epoch"], r=item["r"],
                                   anchor_r=base[subject]["r"], delta_r=item["r"] - base[subject]["r"],
                                   MSE=item["MSE"], anchor_MSE=base[subject]["MSE"],
                                   older_r=None if pair is None else old_people[subject]["r"],
                                   older_pair_status=row["older_pair_status"]))
            for tick in track:
                train = tick.get("train", {})
                val = tick["val"]
                epochs.append(dict(seed=seed, role=role, epoch=tick["epoch"], val_r=val["r"],
                                   val_delta_r=val["r"] - anchor["val"]["r"],
                                   val_MSE=val["MSE"], val_active_MSE=val["active_MSE"],
                                   val_residual_RMS=val["residual_RMS"],
                                   online_TRAIN_MSE=tick.get("online_TRAIN_MSE"),
                                   fixed_TRAIN_r=train.get("r"), fixed_TRAIN_MSE=train.get("MSE"),
                                   fixed_TRAIN_active_MSE=train.get("active_MSE")))
                if tick["epoch"] >= 0:
                    grad = read_json(folder / f"epoch{tick['epoch']:02}_FIRST_BATCH.json")
                    named = grad.pop("gradients")
                    gradients.append(dict(seed=seed, role=role, **grad, **named))
        item = receipt["seeds"][str(seed)]
        pairs.append(dict(seed=seed, **item["margins"],
                          all_H_margins_positive=item["all_H_margins_positive"]))
    return seed_rows, epochs, people, gradients, pairs, anchor_diagnostics


def figures(out, seed_rows, epochs, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    by_run = defaultdict(list)
    for row in epochs:
        by_run[row["seed"], row["role"]].append(row)
    by_selection = {(row["seed"], row["role"]): row for row in seed_rows}

    def panel(ax, seed, metric, label, scale=1., fixed=False, select=False):
        ax.set_title(f"Seed {seed}", loc="left", fontweight="bold")
        ax.set_xlabel("Epoch (-1 is the frozen anchor)")
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#e4e9ee", linewidth=.7)
        ax.axhline(0, color="#728394", lw=.8)
        for color, role, name in zip(COLORS, ROLES, LABELS):
            track = [row for row in by_run[seed, role] if row[metric] is not None]
            baseline = track[0][metric] if fixed else 0.
            x = [row["epoch"] for row in track]
            y = [(row[metric] - baseline) * scale for row in track]
            ax.plot(x, y, color=color, label=name, lw=1.5, marker="o" if fixed else None)
            if select:
                chosen = by_selection[seed, role]["selected_epoch"]
                ax.scatter(chosen, y[x.index(chosen)], color=color, edgecolor="white", s=39, zorder=4)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.65), sharey=True, layout="constrained")
    for ax, seed in zip(axes, SEEDS):
        panel(ax, seed, "val_delta_r", "VAL Δr × 1,000", scale=1000, select=True)
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle("Validation relative to the same-seed frozen anchor", fontweight="bold")
    fig.savefig(out / "01_val_trajectories.png", dpi=180, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey="row", layout="constrained")
    for i, seed in enumerate(SEEDS):
        panel(axes[0, i], seed, "fixed_TRAIN_active_MSE", "Fixed TRAIN ΔMSE × 1,000",
              scale=1000, fixed=True)
        panel(axes[1, i], seed, "val_residual_RMS", "VAL active residual RMS")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Training fit and physical residual amplitude", fontweight="bold")
    fig.savefig(out / "02_train_and_residual.png", dpi=180, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.05), layout="constrained")
    for j, (key, title) in enumerate((("selected_delta_r", "Selected including frozen anchor"),
                                       ("best_trained_delta_r", "Best among trained epochs only"))):
        ax = axes[j]
        for i, (role, color, name) in enumerate(zip(ROLES, COLORS, LABELS)):
            rows = [by_selection[seed, role] for seed in SEEDS]
            ys = [r[key] * 1000 for r in rows]
            ax.scatter([k + (i - 1) * .19 for k in range(3)], ys, color=color, s=44, label=name)
        ax.axhline(0, color="#728394", lw=.8)
        ax.set_xticks(range(3), [str(seed) for seed in SEEDS])
        ax.set_xlabel("Seed")
        ax.set_ylabel("VAL Δr × 1,000")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#e4e9ee", linewidth=.7)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out / "03_selection.png", dpi=180, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    comparisons = [("H_minus_anchor", "H − anchor"), ("H_minus_C", "H − C"),
                   ("H_minus_P", "H − P"), ("H_near_minus_older", "H near − older")]
    for i, row in enumerate(pairs):
        ax.scatter([k + (i - 1) * .14 for k in range(len(comparisons))],
                   [1000 * row[key] for key, _ in comparisons],
                   s=52, label=f"Seed {row['seed']}", marker=("o", "s", "^")[i])
    ax.set_xticks(range(len(comparisons)), [label for _, label in comparisons])
    ax.axhline(0, color="#728394", lw=.8)
    ax.set_ylabel("Selected VAL Δr × 1,000")
    ax.set_title("Prespecified history contrasts", loc="left", fontweight="bold")
    if all(row["selected_epoch"] == -1 for row in seed_rows):
        ax.text(.02, .87, "All selections fall back to the anchor. No trained near/older pair.",
                transform=ax.transAxes, color="#354f66", fontsize=9)
    ax.grid(axis="y", color="#e4e9ee", linewidth=.7)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "04_history_gate.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    fit = ROOT / "fit"
    receipt = read_json(fit / "COMPLETED.json")
    assert receipt["passed"] and receipt["counters"]["formal_fits_completed"] == 9
    assert not (fit / "FAILURE.json").exists()
    out = ROOT / "presentation"
    out.mkdir()
    seed_rows, epochs, people, gradients, pairs, anchor = extract(fit, receipt)
    for name, rows in (("SEED_RESULTS", seed_rows), ("EPOCH_METRICS", epochs), ("PEOPLE", people),
                       ("FIRST_BATCH", gradients), ("HISTORY_CONTRASTS", pairs), ("ANCHOR_AUTOCORRELATION", anchor)):
        csv_file(out / (name + ".csv"), rows)
    figures(out, seed_rows, epochs, pairs)
    stats = {}
    for role in ROLES:
        rows = [r for r in seed_rows if r["role"] == role]
        stats[role] = dict(selected_r_mean=mean(r["selected_r"] for r in rows),
                           anchor_r_mean=mean(r["anchor_r"] for r in rows),
                           selected_epochs=[r["selected_epoch"] for r in rows],
                           best_trained_delta_r_mean=mean(r["best_trained_delta_r"] for r in rows),
                           last_train_active_MSE_change_mean=mean(r["last_train_active_MSE"] -
                                                                 r["anchor_train_active_MSE"] for r in rows))
    write_json(out / "SUMMARY.json", dict(formal=receipt, seed_rows=seed_rows,
                                         contrasts=pairs, role_means=stats))


if __name__ == "__main__":
    main()
