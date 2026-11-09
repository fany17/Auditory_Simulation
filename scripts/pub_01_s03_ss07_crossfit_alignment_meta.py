"""Prepare the D25 fixed-checkpoint source alignment routing from public metadata."""
import argparse
import hashlib
import json
from pathlib import Path

SATA = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01")
D24 = SATA / "s03_source_family_crossfit_20261106"
D19 = SATA / "s03_physical_stream_20261106"
ROOT = SATA / "s03_crossfit_source_alignment_20261106"
SEEDS = (1024, 2026, 31415)
EPOCHS = (0, 19)
VARIANTS = ("B", "S")
FIRST_ACTIVE = 1344


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path, body):
    if path.exists():
        assert path.read_text() == body
        return
    with path.open("x") as stream:
        stream.write(body)


def serialized(item):
    return json.dumps(item, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-server", action="store_true")
    args = parser.parse_args()
    assignment_path = D24/"TRAIN_RECORD_ASSIGNMENT.json"
    physical_path = D19/"PREPARATION_METADATA.json"
    d24_gate_path = D24/"DOWNSTREAM_FORMAL_GATE_V06.json"
    d24_gate = load(d24_gate_path)
    assert d24_gate["passed"] and d24_gate["sources"][str(assignment_path)] == sha(assignment_path)
    assert d24_gate["sources"][str(physical_path)] == sha(physical_path)
    matrix_path = D24/"UPSTREAM_RECOVERY_TERMINAL.json"
    candidate_path = D24/"FOLD_AND_BUDGET_CANDIDATE_V2.json"
    envelope_path = Path(load(candidate_path)["input_files"]["train_envelope_manifest"]["path"])
    assert sha(envelope_path) == load(candidate_path)["input_files"]["train_envelope_manifest"]["sha256"]
    val_meta = load(physical_path)["phases"]["val"]
    val_envelopes = {r["record"]: r for r in load(envelope_path)["records"] if r["phase"] == "val"}
    family_fold = {}
    train = load(assignment_path)["records"]
    for row in train:
        family = row["source_family"]
        assert family_fold.setdefault(family, row["OOF_model_fold"]) == row["OOF_model_fold"]
    assert len(train) == 426 and len(family_fold) == 37
    rows = []
    block_counts = {}
    for block in val_meta["blocks"]:
        block_counts[block["record"]] = block_counts.get(block["record"], 0) + 1
    for row in val_meta["records"]:
        env = val_envelopes[row["record"]]
        assert row["common_samples"] == env["common_samples"] and row["windows"] == env["readout_windows"]
        family = row["source_family"]
        shared = family in family_fold
        routes = [family_fold[family]] if shared else [0, 1, 2]
        rows.append(dict(row, eeg_path=env["eeg_path"], folds=routes, shared_train_family=shared,
                         VAL_blocks=block_counts[row["record"]],
                         active_points=row["covered_stop"]-FIRST_ACTIVE))
    assert len(rows) == len(val_envelopes) == len({r["subject"] for r in rows}) == 49
    assert sum(r["windows"] for r in rows) == 30500
    assert sum(r["VAL_blocks"] for r in rows) == 752
    assert sum(r["active_points"] for r in rows) == val_meta["active_points"]
    fold_rows = []
    for fold in range(3):
        selected = [r for r in rows if fold in r["folds"]]
        assert all(family_fold[r["source_family"]] == fold for r in selected if r["shared_train_family"])
        fold_rows.append(dict(fold=fold, records=len(selected),
                              shared_records=sum(r["shared_train_family"] for r in selected),
                              windows=sum(r["windows"] for r in selected),
                              active_points=sum(r["active_points"] for r in selected),
                              blocks=sum(r["VAL_blocks"] for r in selected)))
    weights = []
    for seed in SEEDS:
        base = D24/"downstream_formal"/f"seed{seed}"
        for variant in VARIANTS:
            model = base/f"{variant}_OOF"/f"seed{seed}_P_past_prediction"
            history = load(model/"RESULT.json")["epochs"]
            assert [point["epoch"] for point in history] == list(range(-1, 20))
            for epoch in EPOCHS:
                path = model/"recovery"/f"epoch{epoch:02}.pt"
                weights.append(dict(seed=seed, variant=variant, epoch=epoch, path=str(path),
                                    bytes=path.stat().st_size, full_anchor_VAL_score=history[0]["val"]["r"],
                                    full_model_VAL_score=history[epoch+1]["val"]["r"],
                                    full_model_VAL_active_MSE=history[epoch+1]["val"]["active_MSE"]))
    manifest_body = serialized({"records": rows})
    pins = (assignment_path, physical_path, d24_gate_path, matrix_path, candidate_path, envelope_path)
    grouped = {family: dict(shared_train_family=family in family_fold,
                           eligible_folds=[family_fold[family]] if family in family_fold else [0,1,2],
                           records=sum(r["source_family"] == family for r in rows))
               for family in sorted({r["source_family"] for r in rows})}
    window_total = len(SEEDS)*sum(f["windows"] for f in fold_rows)
    physical_total = len(SEEDS)*len(VARIANTS)*len(EPOCHS)*sum(f["active_points"] for f in fold_rows)
    route = dict(candidate="D25 source-alignment, evaluation-only", official_VAL_records=49,
                 official_VAL_active_points=val_meta["active_points"],
                 official_VAL_blocks=752, source_groups=grouped, folds=fold_rows,
                 weights=weights, sources={str(path): sha(path) for path in pins},
                 routing_manifest=str(ROOT/"VAL_ROUTING_METADATA.json"),
                 routing_manifest_sha256=hashlib.sha256(manifest_body.encode("ascii")).hexdigest(),
                 budget=dict(upstream_encoder_fold_batches=9, upstream_encoder_windows=window_total,
                             logical_fold_readout_routes=36, downstream_active_points=physical_total,
                             full_VAL_equivalent_passes=physical_total/val_meta["active_points"],
                             new_fits=0, new_fit_statistics=0, old_test_and_external=0))
    assert window_total <= 274500 and physical_total <= 36*val_meta["active_points"]
    if args.write_server:
        ROOT.mkdir(exist_ok=True)
        assert ROOT.resolve() == ROOT
        write_once(ROOT/"VAL_ROUTING_METADATA.json", manifest_body)
        write_once(ROOT/"ROUTING_AND_BUDGET_CANDIDATE.json", serialized(route))
    print(serialized(route), end="")


if __name__ == "__main__":
    main()
