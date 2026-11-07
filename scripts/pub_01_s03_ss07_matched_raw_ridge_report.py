"""Validate and summarize completed D28 block scores without fitting or reading signals."""
import csv
import json
from collections import Counter, defaultdict
from math import fsum, isclose, isfinite
from pathlib import Path
from shutil import disk_usage

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_matched_raw_ridge_20261107")
FORMAL = ROOT / "formal"
ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100.)


def read(name, folder=FORMAL):
    return json.loads((folder / name).read_text())


def average(values):
    return fsum(values) / len(values)


def metrics(rows):
    people = defaultdict(list)
    for row in rows:
        people[row["subject"]].append(row)
    return dict(people=len(people), blocks=len(rows),
                **{k: average([average([r[k] for r in b]) for b in people.values()])
                   for k in ("r", "MSE")})


def agrees(a, b):
    return all(isclose(a[k], b[k], rel_tol=0., abs_tol=1e-13) for k in ("r", "MSE"))


def save_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    finished = read("COMPLETED.json")
    assert finished["passed"] and finished["TRAIN_records"] == 426 and finished["TRAIN_points"] == 24297126
    assert finished["distinct_fit_statistics"] == finished["eigensystems"] == 50
    assert finished["ridge_candidates"] == 450 and finished["VAL_candidate_blocks"] == 13536
    assert finished["VAL_records"] == finished["VAL_people"] == 49 and finished["VAL_eligible_blocks"] == 752
    assert all(finished[k] == 0 for k in ("gpu", "new_neural_forwards", "new_neural_optimizations", "old_test_reads"))
    inv = read("INVENTORY.json", ROOT)
    report = read("SELECTED_VAL.json")
    chosen = report["alpha_by_fit"]
    roster = inv["personal_subjects"]
    families = set(inv["train_source_families"])
    assert set(chosen) == {"G", *roster}
    with (FORMAL / "VAL_BLOCKS.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected, same_alpha = defaultdict(list), defaultdict(list)
    fit_groups = defaultdict(list)
    mask_seen = defaultdict(set)
    for r in rows:
        r["alpha"], r["r"], r["MSE"] = (float(r[k]) for k in ("alpha", "r", "MSE"))
        assert all(isfinite(r[k]) for k in ("r", "MSE"))
        assert -1 <= r["r"] <= 1 and r["MSE"] >= 0
        key = r["kind"], r["alpha"]
        block = r["record"], r["start"], r["samples"]
        assert block not in mask_seen[key]
        mask_seen[key].add(block)
        same_alpha[key].append(r)
        fit = "G" if r["kind"] == "G" else r["subject"]
        fit_groups[fit, r["alpha"]].append(r)
        if r["alpha"] == chosen[fit]:
            selected[r["kind"]].append(r)
    assert len(fit_groups) == 450 and len(same_alpha) == 18 and len(rows) == 13536
    assert all(m == mask_seen["G", ALPHAS[0]] and len(m) == 752 for m in mask_seen.values())
    with (FORMAL / "CANDIDATE_SCORES.csv").open(newline="") as stream:
        candidate_rows = list(csv.DictReader(stream))
    assert len(candidate_rows) == 450
    native = {}
    for row in candidate_rows:
        key = row["fit"], float(row["alpha"])
        assert key not in native
        native[key] = {k: float(row[k]) for k in ("r", "MSE")}
        rebuilt = metrics(fit_groups[key])
        assert agrees(native[key], rebuilt)
        assert int(row["participants"]) == rebuilt["people"] and int(row["blocks"]) == rebuilt["blocks"]
    assert all(max(ALPHAS, key=lambda a: native[fit, a]["r"]) == alpha for fit, alpha in chosen.items())
    people = []
    for person in sorted(roster):
        entry = next(row for row in inv["val"] if row["subject"] == person)
        g = metrics([r for r in selected["G"] if r["subject"] == person])
        i = metrics([r for r in selected["I"] if r["subject"] == person])
        people.append(dict(subject=person, story=entry["story"], blocks=g["blocks"], personal_alpha=chosen[person],
                           G_r=g["r"], I_r=i["r"], I_minus_G_r=i["r"]-g["r"],
                           G_MSE=g["MSE"], I_MSE=i["MSE"], I_minus_G_MSE=i["MSE"]-g["MSE"]))
    strata = []
    kinds = ("G", "I")
    overall = {k: metrics(selected[k]) for k in kinds}
    stories = sorted({row["story"] for row in inv["val"]})
    for kind in kinds:
        assert agrees(overall[kind], report["metrics"][kind]) and agrees(overall[kind], finished["scores"][kind])
        for field, names in (("story", stories), ("family", ("shared", "unseen"))):
            for name in names:
                matching = [r for r in selected[kind] if
                            (r["story"] == name if field == "story" else
                             ((r["source_family"] in families) == (name == "shared")))]
                sub = metrics(matching)
                assert agrees(sub, report["metrics"][kind]["by_"+field][name])
                strata.append(dict(kind=kind, stratum=field, name=name, **sub))
    curve = [dict(kind=kind, alpha=alpha, **metrics(same_alpha[kind, alpha]))
             for kind in kinds for alpha in ALPHAS]
    moments = read("MOMENT_LEDGER.json")
    files = sorted(FORMAL.glob("VAL/RECORD_*.json"))
    assert len(files) == 49
    decomposition = []
    for fit in ["G", *roster]:
        with np.load(FORMAL / "coefficients" / (fit + ".npz"), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
        assert metadata["fit"] == fit and metadata["decompositions"] == 1 and metadata["ridge_candidates"] == 9
        decomposition.append(metadata)
    assert len(decomposition) == 50
    data = dict(overall=overall, delta_I_minus_G={k:overall["I"][k]-overall["G"][k] for k in ("r","MSE")},
                G_alpha=chosen["G"], I_alpha_counts=dict(sorted(Counter(chosen[p] for p in roster).items())),
                G_alpha_is_grid_endpoint=chosen["G"] in (ALPHAS[0], ALPHAS[-1]),
                I_low_endpoint_count=sum(chosen[p] == ALPHAS[0] for p in roster),
                I_high_endpoint_count=sum(chosen[p] == ALPHAS[-1] for p in roster),
                I_person_r_improved=sum(r["I_minus_G_r"] > 0 for r in people),
                I_person_r_equal=sum(r["I_minus_G_r"] == 0 for r in people),
                I_person_MSE_reduced=sum(r["I_minus_G_MSE"] < 0 for r in people),
                I_person_MSE_equal=sum(r["I_minus_G_MSE"] == 0 for r in people),
                strata=strata, common_alpha_curve=curve,
                costs=dict(finished=finished, record_statistics_CPU_seconds=fsum(r["CPU_seconds"] for r in moments["record_ledger"]),
                           record_statistics_wall_seconds=fsum(r["wall_seconds"] for r in moments["record_ledger"]),
                           global_merge_CPU_seconds=moments["merge_CPU_seconds"],
                           eigensystem_stage_CPU_seconds=fsum(d["CPU_seconds"] for d in decomposition),
                           eigensystem_stage_wall_seconds=fsum(d["elapsed_seconds"] for d in decomposition),
                           coefficient_fit_metadata=decomposition,
                           VAL_record_CPU_seconds=fsum(json.loads(p.read_text())["CPU_seconds"] for p in files),
                           full_formal_file_bytes=sum(p.stat().st_size for p in FORMAL.rglob("*") if p.is_file()),
                           selected_coefficients_bytes=(FORMAL / "SELECTED_COEFFICIENTS.npz").stat().st_size,
                           current_SATA_free_bytes=disk_usage(ROOT).free))
    destination = ROOT / "report01"
    destination.mkdir()
    save_json(destination / "RESULT_SUMMARY.json", data)
    save_csv(destination / "PER_PERSON.csv", people)
    save_csv(destination / "STRATA.csv", strata)
    save_csv(destination / "COMMON_ALPHA_CURVE.csv", curve)
    figures(destination, curve, people, data, stories)


def figures(destination, curve, people, data, stories):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    styles = {"G": ("#25597b", "Shared model (G)"), "I": ("#b16a19", "Personal models (I), common alpha")}
    for axis, key, ylabel in zip(axes, ("r", "MSE"), ("Participant-macro Pearson r", "Participant-macro MSE")):
        for kind, (color, label) in styles.items():
            series = [r for r in curve if r["kind"] == kind]
            axis.semilogx([r["alpha"] for r in series], [r[key] for r in series], "o-", color=color, label=label)
        axis.axhline(data["overall"]["I"][key], color=styles["I"][0], ls=":", label="I, per-person selected alpha")
        axis.axvline(data["G_alpha"], color=styles["G"][0], alpha=.35, ls="--")
        axis.set(xlabel="Ridge alpha", ylabel=ylabel)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(destination / "alpha_curves.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for axis, key, xlabel in zip(axes, ("I_minus_G_r", "I_minus_G_MSE"), ("Personal minus shared Pearson r", "Personal minus shared MSE")):
        for j, story in enumerate(stories):
            points = [r for r in people if r["story"] == story]
            axis.scatter([r[key] for r in points], [j]*len(points), alpha=.72, s=24, color="#25597b")
            axis.scatter([average([r[key] for r in points])], [j], color="#b16a19", marker="D", s=30, zorder=3)
        axis.set(xlabel=xlabel, yticks=range(len(stories)),
                 yticklabels=[s+"  (n="+str(sum(r["story"] == s for r in people))+")" for s in stories])
        axis.axvline(0., color="#47545d", ls="--", lw=1)
        axis.grid(axis="x", alpha=.2)
    fig.savefig(destination / "paired_by_story.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    summarize()

