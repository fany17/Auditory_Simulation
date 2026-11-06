"""Compare the completed memoryless control with frozen development endpoints."""

import csv
from collections import Counter, defaultdict
from math import fsum

from pub_01_s03_ss07_memoryless_output import D20, D22, PARENT, ROLE, ROOT, SEEDS
from pub_01_s03_ss07_recent_prediction_context import read, write


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_contributions(anchor, chosen):
    original = {(b["record"], b["start"]): b["r"] for b in anchor["blocks"]}
    counts = Counter(b["subject"] for b in chosen["blocks"])
    parts = {"mixed": defaultdict(list), "pure": defaultdict(list)}
    for b in chosen["blocks"]:
        name = "mixed" if b["start"] < 1344 else "pure"
        parts[name][b["subject"]].append(b["r"] - original[b["record"], b["start"]])
    result = {}
    for name, rows in parts.items():
        result[name + "_blocks"] = sum(map(len, rows.values()))
        result[name + "_contribution"] = fsum(fsum(v) / counts[s] for s, v in rows.items()) / len(counts)
        result[name + "_within_group_macro_delta"] = fsum(fsum(v) / len(v) for v in rows.values()) / len(rows)
    assert abs(result["mixed_contribution"] + result["pure_contribution"] - (chosen["r"] - anchor["r"])) < 1e-14
    return result


def collect(fit):
    grid, people, epochs, split = [], [], [], []
    d21 = read(PARENT / "s03_output_ridge_20261106/fit/COMPLETED.json")["selections"]
    d20 = read(D20 / "fit/COMPLETED.json")["seeds"]
    for seed in SEEDS:
        anchor = read(fit / f"seed{seed}_ANCHOR.json")["anchor"]["val"]
        base_people = {r["subject"]: r["r"] for r in anchor["people"]}
        for role, stage in (("S", fit), ("B", D22 / "fit")):
            result = read(stage / f"seed{seed}_{ROLE}/RESULT.json")
            track = result["epochs"]
            assert [r["epoch"] for r in track] == list(range(-1, 20))
            best = max(track[1:], key=lambda r: r["val"]["r"])
            selected = max(track, key=lambda r: r["val"]["r"])
            val = result["selected"]
            assert track[0]["val"]["r"] == anchor["r"] and selected["epoch"] == result["selected_epoch"]
            old = result["val_pair"]["older"]
            grid.append(dict(seed=seed, role=role, selected_epoch=selected["epoch"], r=val["r"],
                             anchor_r=anchor["r"], delta_r=val["r"] - anchor["r"],
                             D21_A_r=d21[str(seed)]["A"]["r"], D21_F_r=d21[str(seed)]["F"]["r"],
                             D20_r=d20[str(seed)]["selected_r"], best_trained_epoch=best["epoch"],
                             best_trained_r=best["val"]["r"], last_r=track[-1]["val"]["r"],
                             selected_val_active_MSE=val["active_MSE"],
                             anchor_val_active_MSE=anchor["active_MSE"],
                             selected_block_MSE=val["MSE"],
                             anchor_TRAIN_active_MSE=track[0]["train"]["active_MSE"],
                             last_TRAIN_active_MSE=track[-1]["train"]["active_MSE"],
                             last_val_active_MSE=track[-1]["val"]["active_MSE"],
                             last_val_residual_RMS=track[-1]["val"]["residual_RMS"],
                             secondary_slot="identical_current" if role == "S" else "lag288",
                             secondary_r=None if selected["epoch"] == -1 else old["r"]))
            split.append(dict(seed=seed, role=role, **split_contributions(anchor, val)))
            far_people = {r["subject"]: r["r"] for r in old["people"]}
            for row in val["people"]:
                subject = row["subject"]
                people.append(dict(seed=seed, role=role, selected_epoch=selected["epoch"], **row,
                                   anchor_r=base_people[subject], delta_r=row["r"] - base_people[subject],
                                   secondary_r=None if selected["epoch"] == -1 else far_people[subject]))
            for point in track:
                r = point["val"]["r"]
                epochs.append(dict(seed=seed, role=role, epoch=point["epoch"], val_r=r, delta_r=r - anchor["r"],
                                   val_active_MSE=point["val"]["active_MSE"],
                                   val_residual_RMS=point["val"]["residual_RMS"],
                                   train_active_MSE=point.get("train", {}).get("active_MSE")))
    return grid, people, epochs, split


def chart(out, grid, epochs, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {"S": "#005a8c", "B": "#a85421"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, sharey="row", figsize=(12, 6), layout="constrained")
    for i, seed in enumerate(SEEDS):
        for j, (key, ylabel) in enumerate((("delta_r", "Development VAL Δr × 1,000"),
                                         ("train_active_MSE", "Fixed TRAIN ΔMSE × 1,000"))):
            ax = axes[j, i]
            ax.axhline(0., lw=.8, color="#66798a")
            for role in ("S", "B"):
                rows = [r for r in epochs if r["seed"] == seed and r["role"] == role and r[key] is not None]
                baseline = rows[0][key]
                ax.plot([r["epoch"] for r in rows], [1000 * (r[key] - baseline) for r in rows],
                        label=role, color=palette[role], marker="o" if j else None, lw=1.6)
            if j == 0:
                row = next(r for r in grid if r["seed"] == seed)
                ax.axhline(1000 * (row["D21_F_r"] - row["anchor_r"]), linestyle="--", color="#527528",
                           linewidth=1, label="D21 F selected")
                ax.axhline(1000 * (row["D21_A_r"] - row["anchor_r"]), linestyle=":", color="#6b7e90",
                           linewidth=1, label="D21 A selected")
            ax.grid(axis="y", linewidth=.65, color="#e4e9ee")
            ax.set_title(f"Seed {seed}", loc="left", fontweight="bold")
            ax.set_xlabel("Epoch (-1 is the frozen anchor)")
            ax.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(out / "VAL_TRAIN_TRAJECTORIES.png", dpi=180, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, layout="constrained")
    for ax, seed in zip(axes, SEEDS):
        for i, role in enumerate(("S", "B")):
            row = next(r for r in split if r["seed"] == seed and r["role"] == role)
            for k, (key, symbol) in enumerate((("mixed_contribution", "o"), ("pure_contribution", "s"))):
                ax.scatter(k + .15 * (i - .5), 1000 * row[key], color=palette[role], marker=symbol, s=54,
                           label=role if k == 0 else None)
        ax.axhline(0., lw=.8, color="#66798a")
        ax.set_xticks((0, 1), ("Mixed prefix", "Pure active"))
        ax.set_title(f"Seed {seed}", loc="left", fontweight="bold")
        ax.set_ylabel("Contribution to selected Δr × 1,000")
        ax.grid(axis="y", linewidth=.65, color="#e4e9ee")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out / "PREFIX_CONTRIBUTIONS.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    fit = ROOT / "fit"
    done = read(fit / "COMPLETED.json")
    assert done["passed"] and done["counters"]["formal_fits_completed"] == 3
    assert not (fit / "FAILURE.json").exists()
    out = ROOT / "presentation"
    out.mkdir()
    grid, people, epochs, split = collect(fit)
    write_csv(out / "SEED_RESULTS.csv", grid)
    write_csv(out / "PEOPLE.csv", people)
    write_csv(out / "EPOCH_METRICS.csv", epochs)
    write(out / "PREFIX_SPLIT.json", split)
    chart(out, grid, epochs, split)


if __name__ == "__main__":
    main()
