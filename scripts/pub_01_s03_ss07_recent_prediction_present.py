"""Summarize completed D20 and frozen D19 P scalar outputs on 2203."""

import csv

from pub_01_s03_ss07_recent_prediction_context import ROLE, ROOT, SEEDS, far_folder, read


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gather(fit):
    epochs, seed_rows, people = [], [], []
    for seed in SEEDS:
        paths = {
            "recent": fit / f"seed{seed}_{ROLE}" / "RESULT.json",
            "far": far_folder(seed) / "RESULT.json",
        }
        anchor_r = read(fit / f"seed{seed}_ANCHOR.json")["anchor"]["val"]["r"]
        for kind, path in paths.items():
            result = read(path)
            track = result["epochs"]
            assert [r["epoch"] for r in track] == list(range(-1, 20))
            best = max(track[1:], key=lambda r: r["val"]["r"])
            selected = max(track, key=lambda r: r["val"]["r"])
            assert track[0]["val"]["r"] == anchor_r and selected["epoch"] == result["selected_epoch"]
            pair = result["val_pair"]
            seed_rows.append(dict(seed=seed, history=kind, anchor_r=anchor_r, selected_epoch=selected["epoch"],
                                  selected_r=selected["val"]["r"], best_trained_epoch=best["epoch"],
                                  best_trained_r=best["val"]["r"], last_val_r=track[-1]["val"]["r"],
                                  anchor_TRAIN_active_MSE=track[0]["train"]["active_MSE"],
                                  last_TRAIN_active_MSE=track[-1]["train"]["active_MSE"],
                                  last_VAL_active_MSE=track[-1]["val"]["active_MSE"],
                                  last_VAL_residual_RMS=track[-1]["val"]["residual_RMS"],
                                  selected_paired_r=None if selected["epoch"] == -1 else pair["older"]["r"]))
            paired_by_person = {r["subject"]: r["r"] for r in pair["older"]["people"]}
            for row in result["selected"]["people"]:
                people.append(dict(seed=seed, history=kind, selected_epoch=selected["epoch"],
                                   **row, paired_r=None if selected["epoch"] == -1 else paired_by_person[row["subject"]]))
            for point in track:
                epochs.append(dict(seed=seed, history=kind, epoch=point["epoch"],
                                   val_r=point["val"]["r"], delta_r=point["val"]["r"] - anchor_r,
                                   val_active_MSE=point["val"]["active_MSE"],
                                   val_residual_RMS=point["val"]["residual_RMS"],
                                   train_active_MSE=point.get("train", {}).get("active_MSE")))
    return seed_rows, epochs, people


def chart(out, epochs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey="row", layout="constrained")
    for i, seed in enumerate(SEEDS):
        for j, (key, ylabel) in enumerate((("delta_r", "VAL Δr × 1,000"),
                                         ("train_active_MSE", "Fixed TRAIN ΔMSE × 1,000"))):
            ax = axes[j, i]
            ax.axhline(0., lw=.8, color="#66798a")
            ax.grid(axis="y", linewidth=.65, color="#e4e9ee")
            for role, color, label in (("recent", "#005a8c", "D20 · lag 1"),
                                       ("far", "#bc412b", "D19 P · lag 288")):
                rows = [r for r in epochs if r["seed"] == seed and r["history"] == role and r[key] is not None]
                baseline = rows[0][key]
                ax.plot([r["epoch"] for r in rows], [1000 * (r[key] - baseline) for r in rows],
                        label=label, color=color, marker="o" if j else None, lw=1.6)
            ax.set_title(f"Seed {seed}", loc="left", fontweight="bold")
            ax.set_xlabel("Epoch (-1 is the frozen anchor)")
            ax.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(out / "VAL_TRAIN_TRAJECTORIES.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    fit = ROOT / "fit"
    receipt = read(fit / "COMPLETED.json")
    assert receipt["passed"] and receipt["counters"]["formal_fits_completed"] == 3
    assert not (fit / "FAILURE.json").exists()
    out = ROOT / "presentation"
    out.mkdir()
    seed_rows, epochs, people = gather(fit)
    write_csv(out / "SEED_RESULTS.csv", seed_rows)
    write_csv(out / "EPOCH_METRICS.csv", epochs)
    write_csv(out / "PEOPLE.csv", people)
    chart(out, epochs)


if __name__ == "__main__":
    main()
