"""Build a public metadata-only candidate for the complete source-family cross-fit."""
import argparse
import json
import os
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

HOME = Path("/home/fanyu/auditory_simulation_m6a/pub_01")
SATA = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01")
OLD = HOME / "s03_strong_baselines_20261015/transformer_available01"
BASE = HOME / "s03_incremental_continuity_20261101"
D19 = SATA / "s03_physical_stream_20261106"
ROOT = SATA / "s03_source_family_crossfit_20261106"
SEEDS = {1024: 9, 2026: 12, 31415: 11}
PHYSICAL_BATCH = 16384
META_EPOCHS = 20
META_ACTIVE_POINTS = 23703510


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n"


def write_once(path, body):
    if path.exists():
        if path.read_text() != body:
            raise ValueError(f"Frozen metadata differs: {path}")
        return
    with path.open("x") as stream:
        stream.write(body)


def batches(n, size):
    return (n + size - 1) // size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-server", action="store_true")
    args = parser.parse_args()
    d23 = SATA / "s03_memoryless_output_20261106"
    original_meta_config = d23 / "fit/seed1024_P_past_prediction/CONFIG.json"
    original_meta_protocol = d23 / "PROTOCOL.json"
    assert read(original_meta_config)["batch"] == PHYSICAL_BATCH
    assert read(original_meta_config)["epochs"] == META_EPOCHS
    assert read(original_meta_protocol)["formal_budget"]["optimizer_updates"] == 3*META_EPOCHS*batches(META_ACTIVE_POINTS, PHYSICAL_BATCH)
    inputs = {
        "prepared_manifest": OLD / "prepared/manifest_frozen.json",
        "train_window_index": OLD / "prepared/train_windows.json",
        "train_envelope_manifest": OLD / "envelope_readout_seed1024/manifest_frozen.json",
        "source_families": BASE / "SOURCE_FAMILY_JOIN.json",
        "D19_metadata": D19 / "PREPARATION_METADATA.json",
    }
    prepared = read(inputs["prepared_manifest"])["records"]
    envelopes = {row["record"]: row for row in read(inputs["train_envelope_manifest"])["records"]
                 if row["phase"] == "train"}
    family_map = {row["record"]: row["source_family"]
                  for row in read(inputs["source_families"])["records"] if row["phase"] == "train"}
    physical = read(inputs["D19_metadata"])["phases"]["train"]
    assert physical["active_points"] == META_ACTIVE_POINTS
    records = physical["records"]
    assert len(records) == len(envelopes) == len(family_map) == 426
    groups = defaultdict(list)
    for row in records:
        env = envelopes[row["record"]]
        family = family_map[row["record"]]
        assert row["source_family"] == family
        assert (row["offset"], row["windows"], row["common_samples"]) == (
            env["phase_offset"], env["readout_windows"], env["common_samples"])
        groups[family].append(row)
    families = [
        (family, sum(row["windows"] for row in rows))
        for family, rows in groups.items()
    ]
    folds = [dict(id=i, windows=0, records=0, families=[]) for i in range(3)]
    assignment = {}
    for family, windows in sorted(families, key=lambda item: (-item[1], item[0])):
        target = min(folds, key=lambda fold: (fold["windows"], fold["id"]))
        summary = dict(source_family=family, windows=windows, records=len(groups[family]))
        target["windows"] += windows
        target["records"] += summary["records"]
        target["families"].append(summary)
        assignment[family] = target["id"]
    train = []
    for row in records:
        family = row["source_family"]
        fold = assignment[family]
        train.append(dict(row, eeg_path=envelopes[row["record"]]["eeg_path"],
                          OOF_model_fold=fold, IS_model_fold=(fold + 1) % 3))
    record_map = {row["record"]: row for row in train}
    story_to_fold = {}
    pretrain = [defaultdict(lambda: {"starts": set(), "subjects": set(), "rows": 0}) for _ in range(3)]
    for row in read(inputs["train_window_index"]):
        manifest = prepared[row["record_id"]]
        record = record_map[manifest["record"]]
        assert row["phase"] == manifest["phase"] == record["phase"] == "train"
        assert row["story"] == manifest["story"] == record["story"]
        fold = record["OOF_model_fold"]
        assert story_to_fold.setdefault(row["story"], fold) == fold
        story = pretrain[fold][row["story"]]
        story["starts"].add(row["start"])
        story["subjects"].add(row["subject"])
        story["rows"] += 1
    held_plans = [
        sum((len(info["starts"]) // 128) * len(info["subjects"]) for info in block.values())
        for block in pretrain
    ]
    assert sum(held_plans) == 820
    jobs = []
    for fold in folds:
        f = fold["id"]
        fit = [row for row in train if row["OOF_model_fold"] != f]
        oof = [row for row in train if row["OOF_model_fold"] == f]
        isin = [row for row in train if row["IS_model_fold"] == f]
        fit_ids = {row["record"] for row in fit}
        assert not fit_ids.intersection(row["record"] for row in oof)
        assert {row["record"] for row in isin} <= fit_ids
        assert {row["source_family"] for row in oof}.isdisjoint(row["source_family"] for row in fit)
        fit_windows = sum(row["windows"] for row in fit)
        for seed, contrast_epochs in SEEDS.items():
            jobs.append(dict(
                seed=seed, model_fold=f, fit_records=len(fit), fit_windows=fit_windows,
                fit_raw_channel_normalization_samples=sum(row["common_samples"] for row in fit),
                affine_physical_points=sum(row["covered_stop"] - 1 for row in fit),
                contrastive=dict(epochs=contrast_epochs, batches_per_epoch=820-held_plans[f],
                                 updates=contrast_epochs*(820-held_plans[f]),
                                 EEG_audio_window_pairs=contrast_epochs*(820-held_plans[f])*128),
                head=dict(epochs=250, updates=250*batches(fit_windows, 64),
                          target_windows=250*fit_windows),
                FT=dict(epochs=5, updates=5*batches(fit_windows, 64), target_windows=5*fit_windows),
                Ccap=dict(lambd=0.01, fits=1, fit_windows=fit_windows-4*len(fit)),
                OOF_output_records=len(oof), OOF_output_windows=sum(row["windows"] for row in oof),
                IS_output_records=len(isin), IS_output_windows=sum(row["windows"] for row in isin),
            ))
    train_body = canonical({"record_count": len(train), "records": train})
    result = dict(
        stage="METADATA_CANDIDATE_V2", original_meta_config=str(original_meta_config),
        original_meta_protocol=str(original_meta_protocol), input_files={k: {"path": str(path), "sha256": digest(path)}
                                                  for k, path in inputs.items()},
        root=str(ROOT), train_manifest=str(ROOT / "TRAIN_RECORD_ASSIGNMENT.json"),
        train_manifest_sha256=sha256(train_body.encode("ascii")).hexdigest(),
        rule="sort whole source families by descending TRAIN windows, then family name; assign to least-loaded fold, then fold id",
        model_routing=dict(OOF="record's fold f", IS="(f+1)%3"),
        folds=folds, jobs=jobs,
        downstream=dict(arms=["B_IS", "S_IS", "B_OOF", "S_OOF"], seeds=list(SEEDS),
                        physical_batch=PHYSICAL_BATCH, epochs=META_EPOCHS,
                        batches_per_epoch=batches(META_ACTIVE_POINTS, PHYSICAL_BATCH),
                        last_batch=META_ACTIVE_POINTS%PHYSICAL_BATCH,
                        updates=12*META_EPOCHS*batches(META_ACTIVE_POINTS, PHYSICAL_BATCH),
                        backward_points=12*META_EPOCHS*META_ACTIVE_POINTS,
                        evaluation="common frozen full-TRAIN FT+Ccap anchor, original VAL49"),
        no_new_fits=True,
    )
    body = canonical(result)
    if args.write_server:
        ROOT.mkdir(exist_ok=True)
        assert ROOT.resolve() == ROOT and os.stat(ROOT).st_dev == os.stat(SATA).st_dev
        write_once(ROOT / "TRAIN_RECORD_ASSIGNMENT.json", train_body)
        write_once(ROOT / "FOLD_AND_BUDGET_CANDIDATE_V2.json", body)
    print(body, end="")


if __name__ == "__main__":
    main()
