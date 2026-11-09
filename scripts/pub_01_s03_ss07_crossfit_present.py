"""Summarize completed D24 scalar metrics and emit static figures only."""
import csv
import json
import os
from pathlib import Path
from statistics import mean

ROOT = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_source_family_crossfit_20261106")
SATA = ROOT.parent
SEEDS = (1024, 2026, 31415)
ARMS = ("B_IS", "S_IS", "B_OOF", "S_OOF")
CONTRASTS = (("B_OOF", "B_IS"), ("B_OOF", "S_OOF"), ("S_OOF", "S_IS"), ("B_IS", "S_IS"))
ROLE = "P_past_prediction"
COLORS = {"B_IS": "#a95520", "S_IS": "#005d86", "B_OOF": "#a95520", "S_OOF": "#005d86"}


def read(path):
    return json.loads(path.read_text())


def csv_write(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    seed_rows, curves, people, comparisons = [], [], [], []
    previous = {"B": SATA/"s03_output_query_20261106/fit",
                "S": SATA/"s03_memoryless_output_20261106/fit"}
    full = SATA/"s03_physical_stream_20261106/fit"
    for seed in SEEDS:
        folder = ROOT/"downstream_formal"/f"seed{seed}"
        anchor = read(full/f"seed{seed}_ANCHOR.json")["anchor"]["val"]
        base_people = {p["subject"]: p for p in anchor["people"]}
        chosen = {}
        for arm in ARMS:
            result = read(folder/arm/f"seed{seed}_{ROLE}/RESULT.json")
            track = result["epochs"]
            assert [r["epoch"] for r in track] == list(range(-1, 20))
            selected = result["selected"]
            best = max(track[1:], key=lambda row: row["val"]["r"])
            chosen[arm] = selected, track[-1]["val"]
            old = read(previous[arm[0]]/f"seed{seed}_{ROLE}/RESULT.json")
            secondary = result["val_pair"]["older"]["r"]
            assert track[0]["val"]["r"] == anchor["r"]
            base_train = track[0]["train"]["active_MSE"]
            seed_rows.append(dict(seed=seed, arm=arm, selected_epoch=result["selected_epoch"],
                selected_r=selected["r"], anchor_r=anchor["r"], selected_delta=selected["r"]-anchor["r"],
                best_trained_epoch=best["epoch"], best_trained_r=best["val"]["r"],
                last_r=track[-1]["val"]["r"], last_delta=track[-1]["val"]["r"]-anchor["r"],
                epochs_above_anchor=sum(row["val"]["r"] > anchor["r"] for row in track[1:]),
                selected_active_MSE=selected["active_MSE"], anchor_active_MSE=anchor["active_MSE"],
                last_active_MSE=track[-1]["val"]["active_MSE"],
                anchor_TRAIN_active_MSE=base_train, last_TRAIN_active_MSE=track[-1]["train"]["active_MSE"],
                selected_lag288_delta=None if arm[0] != "B" or result["selected_epoch"] < 0 else selected["r"]-secondary,
                D22_or_D23_selected_r=old["selected"]["r"]))
            assert len(selected["people"]) == len(base_people) == 49 and len(selected["blocks"]) == 752
            for p in selected["people"]:
                b = base_people[p["subject"]]
                people.append(dict(seed=seed, arm=arm, subject=p["subject"], blocks=p["blocks"],
                                   selected_r=p["r"], anchor_r=b["r"], delta=p["r"]-b["r"]))
            for row in track:
                train = row.get("train", {}).get("active_MSE")
                curves.append(dict(seed=seed, arm=arm, epoch=row["epoch"], val_r=row["val"]["r"],
                    val_delta=row["val"]["r"]-anchor["r"], val_active_MSE=row["val"]["active_MSE"],
                    online_TRAIN_MSE=row.get("online_TRAIN_MSE"), fixed_TRAIN_active_MSE=train,
                    fixed_TRAIN_MSE_delta=None if train is None else train-base_train,
                    selected=row["epoch"] == result["selected_epoch"]))
        for left, right in CONTRASTS:
            a, alast = chosen[left]
            b, blast = chosen[right]
            by_b = {p["subject"]: p["r"] for p in b["people"]}
            by_a = {p["subject"]: p["r"] for p in a["people"]}
            differences = [by_a[name]-by_b[name] for name in sorted(by_a)]
            comparisons.append(dict(seed=seed, contrast=f"{left}-{right}", selected_r=a["r"]-b["r"],
                  last_r=alast["r"]-blast["r"], selected_active_MSE=a["active_MSE"]-b["active_MSE"],
                  subjects_positive=sum(x>0 for x in differences), subjects_negative=sum(x<0 for x in differences)))
    return seed_rows, curves, people, comparisons


def figures(out, curves, comparisons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, sharey="row", figsize=(12, 6), layout="constrained")
    for j, seed in enumerate(SEEDS):
        for i, (metric, label) in enumerate((("val_delta", "Development VAL delta r (x1000)"),
                                             ("fixed_TRAIN_MSE_delta", "Fixed TRAIN delta active MSE (x1000)"))):
            ax = axes[i, j]
            ax.axhline(0, color="#738295", lw=.8)
            for arm in ARMS:
                rows = [row for row in curves if row["seed"] == seed and row["arm"] == arm and row[metric] is not None]
                ax.plot([r["epoch"] for r in rows], [1000*r[metric] for r in rows], color=COLORS[arm],
                        linestyle="-" if arm.endswith("OOF") else "--", label=arm,
                        marker="o" if i else None, lw=1.5)
                if i == 0:
                    chosen = next(r for r in rows if r["selected"])
                    ax.scatter(chosen["epoch"], 1000*chosen[metric], s=34,
                               facecolors="white", edgecolors=COLORS[arm], zorder=3)
            ax.set_title(f"Seed {seed}", loc="left", fontweight="bold")
            ax.set_xlabel("Epoch (-1 = lane anchor)" if i else "Epoch (-1 = shared anchor)")
            if j == 0:
                ax.set_ylabel(label)
            ax.grid(axis="y", lw=.5, color="#e3e8ee")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(out/"CROSSFIT_TRAJECTORIES.png", dpi=180, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, layout="constrained")
    names = [f"{a}-{b}" for a,b in CONTRASTS]
    for ax, key, title in zip(axes, ("selected_r", "last_r"), ("Selected checkpoint", "Last epoch")):
        for i, name in enumerate(names):
            rows = [row for row in comparisons if row["contrast"] == name]
            y = [1000*row[key] for row in rows]
            ax.scatter([i-.14,i,i+.14], y, color=["#005a83","#8e4826","#597d31"], s=34)
            ax.hlines(mean(y), i-.23, i+.23, color="#263c4b", lw=1.6)
        ax.axhline(0,color="#738295",lw=.8)
        ax.set_xticks(range(4), [name.replace("-","\nminus\n") for name in names])
        ax.set_title(title,loc="left",fontweight="bold")
        ax.grid(axis="y",lw=.5,color="#e3e8ee")
    axes[0].set_ylabel("Paired development VAL delta r (x1000)")
    fig.savefig(out/"CROSSFIT_PAIRED_CONTRASTS.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    terminal = read(ROOT/"downstream_queue_v03/TERMINAL.json")
    audit = read(ROOT/"coordinator_downstream_audit_v02/COMPLETED.json")
    assert terminal["passed"] and audit["passed"] and terminal["fits"] == audit["fits"] == 12
    assert audit["counters"]["neural_forwards"] == audit["counters"]["label_arrays_read"] == audit["counters"]["fitting_updates"] == 0
    matrix = read(ROOT/"UPSTREAM_RECOVERY_TERMINAL.json")
    replay = read(ROOT/"coordinator_upstream_audit_v03/COMPLETED.json")
    assert matrix["passed"] and replay["passed"] and len(matrix["jobs"]) == len(replay["jobs"]) == 9
    seed_rows, curves, people, comparisons = collect()
    out = ROOT/"presentation_v01"
    out.mkdir()
    os.environ["MPLCONFIGDIR"] = str(out/"matplotlib_cache")
    csv_write(out/"SEED_RESULTS.csv", seed_rows)
    csv_write(out/"EPOCH_METRICS.csv", curves)
    csv_write(out/"PEOPLE.csv", people)
    csv_write(out/"PAIRED_CONTRASTS.csv", comparisons)
    figures(out, curves, comparisons)
    completed = [read(ROOT/"downstream_formal"/f"seed{seed}"/"COMPLETED.json") for seed in SEEDS]
    arm_keys = ("selected_r", "selected_delta", "last_r", "last_delta", "selected_active_MSE",
                "anchor_active_MSE", "last_active_MSE", "anchor_TRAIN_active_MSE", "last_TRAIN_active_MSE")
    summary = dict(
        anchors={arm: {key: mean(row[key] for row in seed_rows if row["arm"] == arm) for key in arm_keys}
                 for arm in ARMS},
        contrasts={name: {key: mean(row[key] for row in comparisons if row["contrast"] == name)
                          for key in ("selected_r", "last_r", "selected_active_MSE")}
                   for name in [f"{a}-{b}" for a,b in CONTRASTS]},
        costs=dict(upstream_jobs=9, upstream_wall_sum_seconds=sum(j["wall_seconds"] for j in matrix["jobs"].values()),
                   upstream_CPU_seconds=sum(j["CPU_seconds"] for j in matrix["jobs"].values()),
                   downstream_fits=12, downstream_CPU_seconds=sum(j["CPU_seconds"] for j in completed),
                   downstream_wall_sum_seconds=sum(j["elapsed_seconds"] for j in completed),
                   downstream_updates=sum(j["counters"]["optimizer_updates"] for j in completed),
                   downstream_backward_points=sum(j["counters"]["backward_points"] for j in completed),
                   independent_V03=replay["counters"], independent_V02_before_stop_NN_windows=1377,
                   first_failure_and_precision_diagnosis_NN_windows=1076,
                   independent_downstream=audit["counters"]),
        presentation_rows=dict(seed=len(seed_rows), curves=len(curves), people=len(people), pairs=len(comparisons)),
    )
    with (out/"SUMMARY.json").open("x") as stream:
        json.dump(summary,stream,indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
