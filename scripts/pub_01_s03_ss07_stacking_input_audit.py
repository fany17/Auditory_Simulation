"""Audit frozen public TRAIN/VAL lineage on server2203 from metadata only."""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import mean, median

HOME = Path("/home/fanyu/auditory_simulation_m6a/pub_01")
SATA = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01")
OLD = HOME / "s03_strong_baselines_20261015/transformer_available01"
BASE = HOME / "s03_incremental_continuity_20261101"
D8 = HOME / "s03_finetuned_readout_transfer_20261101"
RUNS = {key: SATA / (f"s03_{name}_20261106") for key, name in (
    ("D19", "physical_stream"), ("D20", "recent_prediction_context"),
    ("D22", "output_query"), ("D23", "memoryless_output"),
)}
SEEDS = (1024, 2026, 31415)
PHASES = ("train", "val")
CACHE_NAMES = (
    "train_latent.npy", "train_P.npy", "val_latent.npy", "val_P.npy",
    "TRAIN_TRANSFORMS.npz", "Ccap_lambda0.01_COEFFICIENTS.npz", "P_FT_BASELINE.json",
)


def read(path):
    assert path.suffix == ".json", path
    return json.loads(path.read_text())


def metadata_digest(path):
    assert path.suffix in {".json", ".py"}, path
    return sha256(path.read_bytes()).hexdigest()


def split(rows):
    return {phase: [row for row in rows if row["phase"] == phase] for phase in PHASES}


def ids(rows):
    result = {row["record"] for row in rows}
    assert len(result) == len(rows)
    return result


def coverage(records, phases):
    return {
        "records": len(records),
        "overlap_current_TRAIN": len(records & ids(phases["train"])),
        "overlap_current_VAL": len(records & ids(phases["val"])),
    }


def contrastive(phase, manifest):
    path = OLD / f"prepared/{phase}_windows.json"
    rows = read(path)
    by_story = defaultdict(lambda: defaultdict(set))
    subjects = defaultdict(set)
    for row in rows:
        source = manifest[row["record_id"]]
        assert all(row[key] == source[key] for key in ("phase", "subject", "story"))
        by_story[row["story"]][row["start"]].add(row["record_id"])
        subjects[row["story"]].add(row["subject"])
    possible, guaranteed = set(), set()
    plans = possible_rows = 0
    for story, by_time in by_story.items():
        starts = sorted(by_time)
        blocks = len(starts) // 128
        if not blocks:
            continue
        size = len(starts) // blocks
        plans += blocks * len(subjects[story])
        for start in range(0, blocks * size, size):
            block = starts[start:start + size]
            exclusive = Counter(next(iter(by_time[t])) for t in block if len(by_time[t]) == 1)
            guaranteed.update(record for record, n in exclusive.items() if n > size - 128)
            for t in block:
                possible.update(by_time[t])
                possible_rows += len(by_time[t])
    return path, possible, guaranteed, {
        "phase_index_rows": len(rows), "possible_sampled_rows": possible_rows,
        "plans_per_epoch": plans, "presentations_per_epoch": plans * 128,
    }


def inventory():
    local_names = ("s03_incremental_continuity_", "s03_finetuned_", "s03_current_readout_",
                   "s03_train_diagnostics_", "s03_full_ft_", "s03_preprojection_memory_",
                   "s03_history_", "s03_conditional_", "s03_observation_state_",
                   "s03_readout_attribution_", "s03_physical_objective_", "s03_research_gap_")
    roots = [OLD, *RUNS.values(), SATA / "s03_output_ridge_20261106"]
    roots += sorted(path for path in HOME.iterdir()
                    if path.name.startswith(local_names) and path.is_dir())
    skip = re.compile(r"test|dtu|external|openneuro|ds004408|__pycache__", re.I)
    find = re.compile(r"(?i)(?<![a-z0-9])(?:oof|out[-_]?of[-_]?fold|cross[-_]?fit|fold\\d+|kfold)(?![a-z])")
    scanned = Counter()
    candidates = []
    for root in roots:
        assert root.is_dir()
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(name for name in dirs if not skip.search(name))
            scanned[str(root)] += 1
            candidates.extend(str(Path(directory) / name) for name in sorted(dirs + files)
                              if not skip.search(name) and find.search(name))
    return {"mode": "filename inventory only", "excluded_names_regex": skip.pattern,
            "candidate_names_regex": find.pattern, "directories_per_root": dict(scanned),
            "candidates": sorted(candidates)}


def main():
    prepared = OLD / "prepared/manifest_frozen.json"
    prep = read(prepared)
    norm = OLD / "prepared/train_only_normalization.npz"
    bp = read(BASE / "BASELINE_PROTOCOL.json")
    fp = read(BASE / "FINETUNE_PROTOCOL.json")
    intake = read(BASE / "INTAKE.json")
    final = read(BASE / "FINAL_CHECKPOINT_MANIFEST.json")
    envelopes = OLD / "envelope_readout_seed1024/manifest_frozen.json"
    env = read(envelopes)
    assert prep["normalization_fit_phase"] == "train_only"
    assert bp["sources"][str(norm)] == fp["sources"][str(norm)] == intake["normalization_sha256"]
    assert metadata_digest(prepared) == bp["sources"][str(prepared)] == fp["sources"][str(prepared)]
    assert metadata_digest(envelopes) == fp["sources"][str(envelopes)]
    training_sources = [BASE / "input/pub_01_s03_ss07_baseline.py",
                        BASE / "input/pub_01_s03_ss07_finetune.py",
                        OLD / "input/pub_01_s03_transformer_train_v2.py"]
    for path in training_sources:
        digest = metadata_digest(path)
        assert fp["sources"][str(path)] == digest
        if str(path) in bp["sources"]:
            assert bp["sources"][str(path)] == digest
    d8p = D8 / "PROTOCOL.json"
    d8 = read(d8p)
    d19p = RUNS["D19"] / "PREPARATION_METADATA.json"
    d19 = read(d19p)
    fam_path = BASE / "SOURCE_FAMILY_JOIN.json"
    families = split(read(fam_path)["records"])
    assert d19["provenance"][str(fam_path)] == metadata_digest(fam_path)
    assert d19["provenance"][str(d8p)] == metadata_digest(d8p)
    d8_source = D8 / "input/pub_01_s03_ss07_ft_readout.py"
    assert d19["provenance"][str(d8_source)] == metadata_digest(d8_source)
    assert read(RUNS["D19"] / "PROTOCOL.json")["metadata_sha256"] == metadata_digest(d19p)
    bindings = {"D19": str(d19p)}
    for label in ("D20", "D22", "D23"):
        path = RUNS[label] / "PROTOCOL.json"
        protocol = read(path)
        pins = protocol["d19_pins" if label == "D20" else "frozen_pins"]
        assert pins[str(d19p)] == metadata_digest(d19p)
        assert pins[str(RUNS["D19"] / "PROTOCOL.json")] == metadata_digest(RUNS["D19"] / "PROTOCOL.json")
        bindings[label] = str(path)
    phases = {phase: d19["phases"][phase]["records"] for phase in PHASES}
    cohort, contrast, groups = {}, {}, {}
    for phase in PHASES:
        rows = phases[phase]
        key = lambda row: (row["record"], row["subject"], row["story"], row["common_samples"])
        reference = [key(row) for row in rows]
        assert reference == [key(row) for row in split(prep["records"])[phase]]
        assert reference == [key(row) for row in split(env["records"])[phase]]
        assert reference == [key(row) for row in d8["records"][phase]]
        join = {row["record"]: row for row in families[phase]}
        assert set(join) == ids(rows)
        assert all(row["source_family"] == join[row["record"]]["source_family"] for row in rows)
        windows = sum(row["windows"] for row in rows)
        assert windows == sum(row["readout_windows"] for row in split(env["records"])[phase])
        assert windows == intake["head_windows"][phase] == env["phase_windows"][phase]
        cohort[phase] = dict(records=len(rows), subjects=len({r["subject"] for r in rows}),
                             stories=len({r["story"] for r in rows}), windows=windows,
                             prepared_points=sum(row["common_samples"] for row in rows),
                             Ccap_i_ge_4_windows=windows - 4 * len(rows))
        assert min(row["windows"] for row in rows) > 4
        path, possible, guaranteed, counts = contrastive(phase, prep["records"])
        possible_ids = {prep["records"][i]["record"] for i in possible}
        assert possible_ids == ids(rows)
        assert metadata_digest(path) == intake[f"{phase}_windows_sha256"]
        contrast[phase] = dict(index=str(path), possible_records=coverage(possible_ids, phases),
                               guaranteed_minimum_records=len(guaranteed), **counts)
        groups[phase] = defaultdict(list)
        for row in rows:
            groups[phase][row["source_family"]].append(row)
    assert len(phases["train"]) == 426 and len(phases["val"]) == 49
    assert not (ids(phases["train"]) & ids(phases["val"]))
    fitted = coverage(ids(phases["train"]), phases)
    shared = groups["train"].keys() & groups["val"].keys()
    group_sizes = [len(rows) for rows in groups["train"].values()]
    grouped = []
    for family in sorted(groups["train"].keys() | groups["val"].keys()):
        row = {"source_family": family}
        for phase in PHASES:
            records = groups[phase].get(family, [])
            row[phase] = dict(records=len(records), subjects=len({r["subject"] for r in records}),
                              windows=sum(r["windows"] for r in records))
        grouped.append(row)

    evidence, anchors = {}, {}
    for seed in SEEDS:
        pre = BASE / f"baseline/pretrain_seed{seed}"
        head = BASE / f"baseline/head_seed{seed}"
        ft = BASE / f"finetune/native_P_seed{seed}"
        pre_end, head_end, ft_end = [read(path / "completed.json") for path in (pre, head, ft)]
        resolved = read(ft / "resolved_config.json")
        head_cache = read(head / "cache_manifest.json")
        for phase in PHASES:
            cached = {record: row for record, row in head_cache["records"].items()
                      if row["phase"] == phase}
            assert set(cached) == ids(phases[phase])
            assert sum(row["windows"] for row in cached.values()) == cohort[phase]["windows"]
        encoder = final["encoders"][str(seed)]
        old_head = final["entries"][f"native_P:{seed}"]
        assert encoder["path"] == head_cache["encoder_source"] == str(pre / "best_validation.pt")
        assert encoder["sha256"] == head_cache["encoder_sha256"] == head_end["encoder_sha256"] == resolved["initial_encoder_sha256"]
        assert old_head["path"] == str(head / "best_validation.pt")
        assert old_head["sha256"] == resolved["initial_head_sha256"]
        assert pre_end["selected_epoch"] == encoder["selected_epoch"]
        assert pre_end["updates"] == 820 * pre_end["epochs"]
        train_windows = cohort["train"]["windows"]
        for end in (head_end, ft_end):
            assert end["target_window_presentations"] == train_windows * end["epochs"]
            assert end["updates"] == ((train_windows + 63) // 64) * end["epochs"]
        folder = D8 / f"seed{seed}"
        cp = d8["checkpoints"][str(seed)]
        assert cp["checkpoint"] == str(ft / "best_validation.pt")
        assert Path(cp["checkpoint"]).stat().st_size == cp["checkpoint_bytes"]
        assert cp["checkpoint_sha256"] == read(folder / "CACHE_COMPLETED.json")["checkpoint_sha"]
        assert cp["selected_epoch"] == ft_end["selected_epoch"] == 4
        assert cp["prior_extra_updates"] == ft_end["updates"]
        for phase, name in zip(PHASES, ("FT_TRAIN_CACHE.json", "FT_VAL_BRIDGE.json")):
            assert read(folder / name)["encoder_head_windows"] == cohort[phase]["windows"]
        for name in CACHE_NAMES:
            path = folder / name
            pin = d19["inputs"][str(path)]
            stat = path.stat()
            assert (stat.st_size, stat.st_mtime_ns) == (pin["bytes"], pin["mtime_ns"])
        metric = read(folder / "Ccap_lambda0.01_METRICS.json")
        assert metric["arm"] == "Ccap" and metric["lam"] == 0.01
        calibration = read(folder / "P_FT_BASELINE.json")["calibration"]
        anchor = {label: read(path / f"fit/seed{seed}_ANCHOR.json")["anchor"]
                  for label, path in RUNS.items()}
        assert all(value == anchor["D19"] for value in anchor.values())
        anchors[str(seed)] = {phase: {key: anchor["D19"][phase][key]
                                      for key in ("r", "MSE", "active_MSE", "active_points")}
                              for phase in PHASES}
        evidence[str(seed)] = dict(
            contrastive_selected_epoch=pre_end["selected_epoch"],
            contrastive_selected_prefix_updates=820 * (pre_end["selected_epoch"] + 1),
            head_epochs=head_end["epochs"], head_selected_epoch=head_end["selected_epoch"],
            head_target_window_presentations=head_end["target_window_presentations"],
            FT_epochs=ft_end["epochs"], FT_updates=ft_end["updates"],
            FT_target_window_presentations=ft_end["target_window_presentations"],
            FT_selected_epoch=cp["selected_epoch"], FT_checkpoint=cp["checkpoint"],
            FT_checkpoint_recorded_sha256=cp["checkpoint_sha256"],
            Ccap_lambda=metric["lam"], Ccap_dimension=metric["dimension"],
            FT_calibration_TRAIN_physical_points=calibration["TRAIN_physical_points"],
            D19_asset_stat_matches=list(CACHE_NAMES), D19_D20_D22_D23_anchor_values_equal=True,
        )
    anchor_mean = {phase: {key: mean(item[phase][key] for item in anchors.values())
                           for key in ("r", "MSE", "active_MSE")} for phase in PHASES}
    result = dict(
        scope="server2203 public TRAIN/VAL JSON, source provenance and path/stat only",
        scientific_array_reads=0, weight_reads=0, NN_calls=0, fits=0, downloads=0,
        old_test_payload_or_metrics_read=False, external_payload_or_metrics_read=False,
        paths=dict(prepared_manifest=str(prepared), target_manifest=str(envelopes),
                   intake=str(BASE / "INTAKE.json"), family_join=str(fam_path),
                   baseline_protocol=str(BASE / "BASELINE_PROTOCOL.json"),
                   finetune_protocol=str(BASE / "FINETUNE_PROTOCOL.json"),
                   checkpoint_manifest=str(BASE / "FINAL_CHECKPOINT_MANIFEST.json"),
                   D8_protocol=str(d8p), downstream_bindings=bindings,
                   training_sources=[str(path) for path in (*training_sources, d8_source)]),
        current_cohort=cohort, contrastive=contrast,
        contrastive_record_visit_log_present_in_receipts=False,
        contrastive_historical_example=intake["stochastic_coverage_example"],
        fit_record_coverage={name: fitted for name in
                             ("EEG_channel_norm", "native_head", "joint_FT", "FT_affine",
                              "Ccap_transforms", "Ccap_coefficients")},
        EEG_channel_norm_artifact=str(norm), EEG_channel_norm_recorded_sha256=intake["normalization_sha256"],
        seeds=evidence, anchor_by_seed=anchors, anchor_seed_mean=anchor_mean,
        source_family=dict(
            train_groups=len(groups["train"]), val_groups=len(groups["val"]), shared=sorted(shared),
            shared_train_records=sum(len(groups["train"][k]) for k in shared),
            shared_val_records=sum(len(groups["val"][k]) for k in shared),
            train_record_counts=dict(min=min(group_sizes), median=median(group_sizes), max=max(group_sizes)),
            train_groups_at_most_5=sum(n <= 5 for n in group_sizes),
            story_overlap=sorted({r["story"] for r in phases["train"]} & {r["story"] for r in phases["val"]}),
            subject_overlap=len({r["subject"] for r in phases["train"]} & {r["subject"] for r in phases["val"]}),
            groups=grouped,
        ),
        oof_inventory=inventory(),
    )
    json.dump(result, sys.stdout, indent=2, ensure_ascii=True, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
