"""D19 source/metadata preparation. Standard library only, no array payloads."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import struct
import zipfile


BASE = Path("/home/fanyu/auditory_simulation_m6a/pub_01")
D8 = BASE / "s03_finetuned_readout_transfer_20261101"
ROOT = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_physical_stream_20261106")
LEGACY_METRIC = (
    BASE / "s03_current_readout_internal_audit_20261101/input"
    / "pub_01_s03_ss07_generalization_core.py"
)
TRAIN_MASK = BASE / "s03_full_ft_error_decomposition_20261102/SUPPORT_MASK.json"
SOURCE_FAMILIES = BASE / "s03_incremental_continuity_20261101/SOURCE_FAMILY_JOIN.json"
SEEDS = (1024, 2026, 31415)
ROLES = ("C_current", "H_past_eeg", "P_past_prediction")
PHASES = ("train", "val")
FRAME, HOP = 192, 96
NEAR, OLDER = 288, 960
BINS, BIN_POINTS, CHANNELS = 96, 4, 8
HISTORY = BINS * BIN_POINTS
WIDTH, CONTEXT_WIDTH = 32, 64
CANDIDATE = "D19_ORDERED_16HZ_V2"
CONTEXT_CHANNELS = {"C_current": CHANNELS, "H_past_eeg": CHANNELS, "P_past_prediction": 1}
PARAMETERS = {
    role: ((CHANNELS + 1) * WIDTH + (BINS * width + 1) * CONTEXT_WIDTH
           + (CONTEXT_WIDTH + 1) * WIDTH + (WIDTH + 1)**2)
    for role, width in CONTEXT_CHANNELS.items()
}
FIRST_ACTIVE = OLDER + HISTORY
BATCH = 16_384
EPOCHS = 20
FIXED_TRAIN = (0, 4, 9, 19)
EXPECTED = {"train": (426, 70, 252_454, 23_703_510, 6104),
            "val": (49, 49, 30_500, 2_866_799, 752)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def npy_header(path):
    """Read only the NPY header, then verify payload length via stat."""
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"Invalid NPY magic: {path}")
        version = stream.read(2)
        fmt = {b"\x01\x00": "<H", b"\x02\x00": "<I", b"\x03\x00": "<I"}[version]
        length, = struct.unpack(fmt, stream.read(struct.calcsize(fmt)))
        if length > 16_384:
            raise ValueError(f"Unexpected NPY header size: {path}")
        encoding = "utf-8" if version == b"\x03\x00" else "latin1"
        header = ast.literal_eval(stream.read(length).decode(encoding))
        offset = stream.tell()
    if header["descr"] not in ("<f4", "<f8") or header["fortran_order"]:
        raise ValueError(f"Unexpected NPY type or layout: {path}")
    entries = 1
    for dimension in header["shape"]:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"Invalid NPY shape: {path}")
        entries *= dimension
    stamp = identity(path)
    if offset + entries * int(header["descr"][-1]) != stamp["bytes"]:
        raise ValueError(f"NPY length mismatch: {path}")
    return dict(stamp, shape=list(header["shape"]), dtype=header["descr"],
                payload_offset=offset)


def covered_stop(windows):
    """Positive original Hann coverage is [1, covered_stop)."""
    return (windows - 1) * HOP + FRAME - 1


def raw_support(windows, point):
    """Union of whole 192-point raw windows with positive Hann weight."""
    first = max(0, (point - (FRAME - 2) + HOP - 1) // HOP)
    last = min(windows - 1, (point - 1) // HOP)
    if not 1 <= point < covered_stop(windows) or last < first:
        raise ValueError(f"No positive contributor at {point}")
    return first * HOP, last * HOP + FRAME


def history_support(windows, point, lag):
    return (raw_support(windows, point - lag - HISTORY + 1)[0],
            raw_support(windows, point - lag)[1])


def check_record_geometry(windows):
    # All hop residues at the earliest active hop and at both tail hops.
    # Interior support translates by HOP, with only a possible left clamp.
    stop = covered_stop(windows)
    # Probe each residue, including all history-left boundary transitions
    # and both right boundary hops.
    points = sorted(set(range(FIRST_ACTIVE, FIRST_ACTIVE + 3 * HOP))
                    | set(range(stop - 2 * HOP, stop)))
    max_future = 0
    for point in points:
        start, end = raw_support(windows, point)
        max_future = max(max_future, end - 1 - point)
        near_start, near_end = history_support(windows, point, NEAR)
        _, old_end = history_support(windows, point, OLDER)
        if not old_end <= near_start < near_end <= start:
            raise ValueError(f"Raw support intervals overlap at {point}")
    return len(points), max_future


def assets(seed):
    folder = D8 / f"seed{seed}"
    return {
        "folder": folder,
        "stats": folder / "TRAIN_TRANSFORMS.npz",
        "coefficient": folder / "Ccap_lambda0.01_COEFFICIENTS.npz",
        "calibration": folder / "P_FT_BASELINE.json",
        "anchor_metrics": folder / "Ccap_lambda0.01_METRICS.json",
        "cache_receipt": folder / "CACHE_COMPLETED.json",
    }


def inventory():
    protocol = read_json(D8 / "PROTOCOL.json")
    train_mask = read_json(TRAIN_MASK)["train"]
    families = {(row["phase"], row["record"]): row
                for row in read_json(SOURCE_FAMILIES)["records"]
                if row["phase"] in PHASES}
    if len(families) != 475:
        raise ValueError("Frozen source family mapping changed")
    phases = {}
    paths = {}
    for phase in PHASES:
        records = []
        offset = 0
        cases = 0
        max_future = 0
        for row in protocol["records"][phase]:
            if row["phase"] != phase or row["offset"] != offset:
                raise ValueError("Frozen phase/offset mismatch")
            family = families[(phase, row["record"])]
            if any(row[key] != family[key] for key in
                   ("subject", "story", "common_samples", "windows")):
                raise ValueError("Frozen source family join mismatch")
            windows = row["windows"]
            if (row["common_samples"] - FRAME) // HOP + 1 != windows:
                raise ValueError("Frozen frame count changed")
            offset += windows
            path = Path(row["envelope_path"])
            if not path.is_relative_to(BASE / "s06/sparrkulee_v3.1") or path.suffix != ".npy":
                raise ValueError("Unexpected continuous target path")
            if str(path) not in paths:
                paths[str(path)] = npy_header(path)
            shape = paths[str(path)]["shape"]
            if len(shape) != 1 and not (len(shape) == 2 and 1 in shape):
                raise ValueError("Unexpected envelope axes")
            if max(shape) < row["common_samples"]:
                raise ValueError("Envelope is shorter than frozen coverage")
            record = {key: row[key] for key in
                      ("record", "subject", "story", "phase", "common_samples",
                       "windows", "envelope_path", "offset")}
            record["source_family"] = family["source_family"]
            record["covered_stop"] = covered_stop(windows)
            checked, future = check_record_geometry(windows)
            cases += checked
            max_future = max(max_future, future)
            records.append(record)
        old_mask = protocol["mask"] if phase == "val" else train_mask
        if phase == "val" and (not old_mask["prediction_independent"] or old_mask["test_read"]):
            raise ValueError("Unexpected VAL mask provenance")
        if phase == "train" and not old_mask["target_only"]:
            raise ValueError("Unexpected TRAIN mask provenance")
        record_map = {row["record"]: row for row in records}
        blocks = []
        for block in old_mask["blocks"]:
            if not block["eligible"]:
                continue
            row = record_map[block["record"]]
            start = block["start"]
            stop = start + 3840
            samples = max(0, min(row["covered_stop"], stop) - max(1, start))
            if samples != block.get("samples", block.get("covered")):
                raise ValueError("Frozen score mask and positive Hann coverage differ")
            blocks.append(dict(record=row["record"], start=start, samples=samples))
        active = sum(row["covered_stop"] - FIRST_ACTIVE for row in records)
        got = (len(records), len({row["subject"] for row in records}), offset,
               active, len(blocks))
        if got != EXPECTED[phase]:
            raise ValueError(f"Phase geometry changed: {got}")
        phases[phase] = dict(records=records, blocks=blocks, active_points=active,
                             covered_points=active + (FIRST_ACTIVE - 1) * len(records),
                             coordinate_cases=cases, max_future_samples=max_future,
                             batches_per_pass=(active + BATCH - 1) // BATCH,
                             last_batch=active % BATCH)
    for seed in SEEDS:
        files = assets(seed)
        for phase in PHASES:
            windows = EXPECTED[phase][2]
            for suffix, shape in (("latent", [windows, 8, 192]), ("P", [windows, 192])):
                path = files["folder"] / f"{phase}_{suffix}.npy"
                header = npy_header(path)
                if header["shape"] != shape or header["dtype"] != "<f4":
                    raise ValueError(f"Unexpected D8 cache header: {path}")
                paths[str(path)] = header
        for name, names in (
            ("stats", {f"{key}_{i}.npy" for key in ("mean", "std", "constant") for i in range(3)}),
            ("coefficient", {"coef.npy", "bias.npy"}),
        ):
            path = files[name]
            with zipfile.ZipFile(path) as archive:
                if set(archive.namelist()) != names or len(archive.namelist()) != len(names):
                    raise ValueError(f"Unexpected NPZ members: {path}")
            paths[str(path)] = dict(identity(path), members=sorted(names))
        for name in ("calibration", "anchor_metrics", "cache_receipt"):
            paths[str(files[name])] = identity(files[name])
    provenance = [
        D8 / "PROTOCOL.json", TRAIN_MASK, SOURCE_FAMILIES, LEGACY_METRIC,
        D8 / "input/pub_01_s03_ss07_ft_readout.py",
        BASE / "s03_preprojection_memory_20261103/input/pub_01_s03_ss07_preprojection_data.py",
    ]
    source_pins = {str(path): digest(path) for path in provenance}
    train, val = phases["train"], phases["val"]
    training_points = 9 * EPOCHS * train["active_points"]
    fixed_points = 9 * len(FIXED_TRAIN) * train["active_points"]
    val_points = 9 * EPOCHS * val["active_points"]
    pair_points = 6 * val["active_points"]
    bytes_per_row = 2 * CHANNELS * 4 + 4 + 2 * 8
    table_bytes = sum((p["covered_points"] + len(p["records"])) * bytes_per_row
                      + sum(row["windows"] for row in p["records"]) * (FRAME // BIN_POINTS) * CHANNELS * 4
                      for p in phases.values())
    return dict(candidate=CANDIDATE, stage="SOURCE_METADATA_ONLY", frozen_for_fit=False, phases=phases,
                inputs=paths, provenance=source_pins, writer_root=str(ROOT.resolve()),
                writer_free_bytes=shutil.disk_usage(ROOT).free,
                geometry=dict(frame=FRAME, hop=HOP, first_active=FIRST_ACTIVE,
                              near=NEAR, older=OLDER, history_points=HISTORY,
                              history_bins=BINS, bin_points=BIN_POINTS),
                proposal=dict(seeds=SEEDS, roles=ROLES, epochs=EPOCHS, batch=BATCH,
                              fixed_TRAIN=FIXED_TRAIN, trainable_parameters=PARAMETERS,
                              engineering_current_stem_points=9 * 3 * 4096,
                              engineering_context_output_points=(9 * 3 + 6) * 4096,
                              engineering_backward_points=9 * 2 * 4096,
                              engineering_TRAIN_records=6, engineering_VAL_records=3 * EXPECTED["val"][0],
                              engineering_optimizer_updates=0,
                              engineering_TRAIN_cached_windows=3 * sum(row["windows"] for row in train["records"][:2]),
                              engineering_VAL_cached_windows=3 * EXPECTED["val"][2],
                              formal_cache_records=3 * sum(len(p["records"]) for p in phases.values()),
                              formal_cache_windows=3 * sum(row["windows"] for p in phases.values()
                                                           for row in p["records"]),
                              persistent_cuda_table_bytes_per_seed=table_bytes,
                              TRAIN_global_shuffle_bytes=train["active_points"] * 4,
                              batch_float32_context_bytes={k: BATCH * BINS * v * 4
                                                           for k, v in CONTEXT_CHANNELS.items()},
                              optimizer_updates=9 * EPOCHS * train["batches_per_pass"],
                              train_target_presentations=training_points,
                              fixed_TRAIN_forwards=fixed_points,
                              epoch_VAL_forwards=val_points,
                              selected_pair_current_stem_points_max=pair_points,
                              selected_pair_context_output_points_max=2 * pair_points,
                              current_stem_points_max=training_points + fixed_points + val_points + pair_points,
                              context_output_points_max=training_points + fixed_points + val_points + 2 * pair_points),
                scientific_array_payloads_read=0, new_encoder_head_windows=0,
                new_NN_forwards=0, new_backwards=0, optimizer_updates=0,
                formal_fits=0, goal_complete=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(inventory(), ensure_ascii=True, indent=2, allow_nan=False))
