from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any


CONTROL_STIMULI = {"welcome", "instructions1", "instructions2", "instructions2_clipped", "instructions3", "pleasePressSpace"}
MAX_TRAILING_UNANNOTATED_SECONDS = 5.0


def classify_stimulus(stimulus_id: str) -> tuple[str, str, str]:
    if stimulus_id.startswith("catalan-"):
        return "PASSAGE", "ca", "STIMULUS_ID_AND_EVENT_LEXICON"
    if stimulus_id.startswith("s") and "-ex" in stimulus_id:
        return "PASSAGE", "en", "BLOCK_DIRECTORY_AND_EVENT_LEXICON"
    if stimulus_id in CONTROL_STIMULI:
        return "CONTROL", "en", "CONTROL_TRANSCRIPT"
    return "UNKNOWN", "UNKNOWN", "UNRESOLVED"


def speaker_id_for(stimulus_id: str, role: str) -> str:
    if role != "PASSAGE":
        return ""
    if stimulus_id.startswith("s") and "-ex" in stimulus_id:
        return stimulus_id.split("-ex", maxsplit=1)[0]
    if stimulus_id.startswith("catalan-"):
        return stimulus_id
    return "UNKNOWN"


def deterministic_block_split(block_ids: list[str], seed: int) -> dict[str, str]:
    if len(block_ids) != 6:
        raise ValueError(f"expected six stimulus blocks, found {len(block_ids)}")
    shuffled = sorted(block_ids)
    random.Random(seed).shuffle(shuffled)
    return {
        **{block_id: "train" for block_id in shuffled[:4]},
        shuffled[4]: "validation",
        shuffled[5]: "test",
    }


def contiguous_groups(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    groups: list[list[dict[str, str]]] = []
    for row in rows:
        stimulus_id = row.get("ex_name", "").strip()
        if not stimulus_id:
            continue
        if groups and groups[-1][0]["ex_name"] == stimulus_id:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def connected_component_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in rows:
        union(f"recording:{row['recording_id']}", f"stimulus:{row['stimulus_id']}")
    component_sizes = Counter(find(node) for node in parent)
    return {
        "node_count": len(parent),
        "component_count": len(component_sizes),
        "largest_component_nodes": max(component_sizes.values(), default=0),
        "policy": "stimulus_id_plus_recording_id_must_not_cross_split",
        "feasible_for_three_splits": len(component_sizes) >= 3,
    }


def split_checks(rows: list[dict[str, Any]], embargo_seconds: float) -> dict[str, Any]:
    stimulus_splits: dict[str, set[str]] = defaultdict(set)
    block_splits: dict[str, set[str]] = defaultdict(set)
    language_splits: dict[str, set[str]] = defaultdict(set)
    recording_splits: dict[str, set[str]] = defaultdict(set)
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stimulus_splits[row["stimulus_id"]].add(row["split"])
        block_splits[row["block_id"]].add(row["split"])
        language_splits[row["language"]].add(row["split"])
        recording_splits[row["recording_id"]].add(row["split"])
        speaker_id = str(row.get("speaker_id") or "")
        if speaker_id:
            speaker_splits[speaker_id].add(row["split"])
        by_recording[row["recording_id"]].append(row)

    temporal_violations: list[str] = []
    for recording_id, recording_rows in by_recording.items():
        ordered = sorted(recording_rows, key=lambda item: float(item["audio_onset_seconds"]))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                gap = float(right["audio_onset_seconds"]) - float(left["audio_offset_seconds"])
                if gap >= embargo_seconds:
                    break
                if left["split"] != right["split"]:
                    temporal_violations.append(
                        f"{recording_id}:{left['segment_id']}/{left['split']}->{right['segment_id']}/{right['split']} gap={gap:.6f}"
                    )

    return {
        "stimulus_split_conflicts": sorted(key for key, values in stimulus_splits.items() if len(values) > 1),
        "block_split_conflicts": sorted(key for key, values in block_splits.items() if len(values) > 1),
        "language_split_coverage": {key: sorted(values) for key, values in sorted(language_splits.items())},
        "recordings_spanning_splits": sum(1 for values in recording_splits.values() if len(values) > 1),
        "speaker_split_conflicts": sorted(key for key, values in speaker_splits.items() if len(values) > 1),
        "temporal_embargo_seconds": embargo_seconds,
        "temporal_cross_split_violations": temporal_violations,
    }


def _read_timing_templates(path: Path) -> dict[str, list[dict[str, str]]]:
    templates: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            stimulus_id = (row.get("ex_name") or "").strip()
            if stimulus_id in {"", "ex_name"}:
                continue
            templates[stimulus_id].append(row)
    return dict(templates)


def _block_inventory(root: Path) -> tuple[dict[str, str], dict[str, Path], list[str]]:
    stimulus_to_block: dict[str, str] = {}
    stimulus_to_file: dict[str, Path] = {}
    errors: list[str] = []
    for block_dir in sorted((root / "stimuli" / "excerpts").glob("Block *")):
        try:
            block_id = f"block-{int(block_dir.name.split()[-1]):02d}"
        except ValueError:
            errors.append(f"unreadable block directory: {block_dir.name}")
            continue
        for path in sorted(block_dir.glob("*_normed.wav")):
            stimulus_id = path.stem.removesuffix("_normed")
            if stimulus_id in stimulus_to_block:
                errors.append(f"stimulus appears in multiple blocks: {stimulus_id}")
                continue
            stimulus_to_block[stimulus_id] = block_id
            stimulus_to_file[stimulus_id] = path
    return stimulus_to_block, stimulus_to_file, errors


def _audio_path(
    root: Path,
    stimulus_id: str,
    role: str,
    block_files: dict[str, Path],
) -> tuple[Path, str]:
    if stimulus_id.startswith("catalan-"):
        return root / "stimuli" / "excerpts" / "catalan_v2" / f"{stimulus_id}_normed.wav", "PROVISIONAL_CATALAN_V2"
    if role == "PASSAGE":
        return block_files[stimulus_id], "BUNDLED_BLOCK_AUDIO"
    return root / "stimuli" / f"{stimulus_id}_normed.wav", "BUNDLED_CONTROL_AUDIO"


def _timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()


def build_ds004703_manifests(
    root: Path,
    seed: int = 20260811,
    embargo_seconds: float = 2.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import soundfile  # type: ignore[import-untyped]

    root = root.resolve()
    templates = _read_timing_templates(root / "stimuli" / "stim-times.tsv")
    stimulus_to_block, block_files, errors = _block_inventory(root)
    block_ids = sorted(set(stimulus_to_block.values()))
    block_split = deterministic_block_split(block_ids, seed)
    audio_cache: dict[Path, dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []

    for event_path in sorted(root.rglob("*events.tsv")):
        relative_event = event_path.relative_to(root).as_posix()
        path_parts = event_path.relative_to(root).parts
        participant_id = path_parts[0]
        session_id = path_parts[1]
        recording_id = event_path.name.removesuffix("_events.tsv")
        with event_path.open("r", encoding="utf-8-sig", newline="") as handle:
            event_rows = list(csv.DictReader(handle, delimiter="\t"))
        for segment_index, segment_rows in enumerate(contiguous_groups(event_rows), start=1):
            stimulus_id = segment_rows[0]["ex_name"]
            role, language, language_source = classify_stimulus(stimulus_id)
            speaker_id = speaker_id_for(stimulus_id, role)
            block_id = stimulus_to_block.get(stimulus_id, "NOT_APPLICABLE")
            template = templates.get(stimulus_id, [])
            alignment_status = "PASS"
            alignment_spread = None
            audio_onset = None
            if len(segment_rows) != len(template):
                alignment_status = "EVENT_TEMPLATE_COUNT_MISMATCH"
            else:
                offsets: list[float] = []
                for event_row, template_row in zip(segment_rows, template):
                    fields_match = all(event_row.get(key) == template_row.get(key) for key in ("word", "pos", "phone"))
                    duration_match = abs(float(event_row["duration"]) - float(template_row["duration"])) <= 1e-9
                    if not fields_match or not duration_match:
                        alignment_status = "EVENT_TEMPLATE_CONTENT_MISMATCH"
                        break
                    offsets.append(float(event_row["onset"]) - float(template_row["t_start"]))
                if offsets and alignment_status == "PASS":
                    alignment_spread = max(offsets) - min(offsets)
                    audio_onset = median(offsets)
                    if alignment_spread > 1e-6:
                        alignment_status = "EVENT_TEMPLATE_OFFSET_DRIFT"

            try:
                audio_path, source_status = _audio_path(root, stimulus_id, role, block_files)
            except KeyError:
                audio_path = root / "MISSING"
                source_status = "MISSING_BLOCK_MAPPING"
            if audio_path not in audio_cache and audio_path.is_file():
                info = soundfile.info(str(audio_path))
                audio_cache[audio_path] = {
                    "audio_bytes": audio_path.stat().st_size,
                    "audio_modified_at_utc": _timestamp(audio_path),
                    "audio_sample_rate_hz": int(info.samplerate),
                    "audio_channels": int(info.channels),
                    "audio_frames": int(info.frames),
                    "audio_duration_seconds": float(info.duration),
                }
            audio_info = audio_cache.get(audio_path)
            if audio_info is None:
                errors.append(f"audio missing: {stimulus_id} -> {audio_path}")
                audio_info = {
                    "audio_bytes": 0,
                    "audio_modified_at_utc": "",
                    "audio_sample_rate_hz": 0,
                    "audio_channels": 0,
                    "audio_frames": 0,
                    "audio_duration_seconds": 0.0,
                }

            template_end = max(
                (float(row["t_start"]) + float(row["duration"]) for row in template),
                default=0.0,
            )
            timeline_margin = float(audio_info["audio_duration_seconds"]) - template_end
            analysis_eligible = (
                role == "PASSAGE"
                and language == "en"
                and alignment_status == "PASS"
                and -0.05 <= timeline_margin <= MAX_TRAILING_UNANNOTATED_SECONDS
            )
            if role == "PASSAGE" and block_id == "NOT_APPLICABLE":
                errors.append(f"passage lacks block mapping: {stimulus_id}")
                analysis_eligible = False
            row = {
                "segment_id": f"{recording_id}__seg-{segment_index:03d}",
                "participant_id": participant_id,
                "session_id": session_id,
                "recording_id": recording_id,
                "event_file": relative_event,
                "segment_index": segment_index,
                "stimulus_id": stimulus_id,
                "stimulus_role": role,
                "speaker_id": speaker_id,
                "language": language,
                "language_source": language_source,
                "block_id": block_id,
                "analysis_eligible": analysis_eligible,
                "exclusion_reason": "" if analysis_eligible else (
                    "CATALAN_AUDIO_PROVENANCE_CONFLICT" if language == "ca" else "NON_PASSAGE_OR_ALIGNMENT_FAILURE"
                ),
                "event_count": len(segment_rows),
                "event_onset_seconds": min(float(item["onset"]) for item in segment_rows),
                "event_offset_seconds": max(float(item["onset"]) + float(item["duration"]) for item in segment_rows),
                "audio_onset_seconds": audio_onset,
                "audio_offset_seconds": audio_onset + float(audio_info["audio_duration_seconds"]) if audio_onset is not None else None,
                "event_template_alignment": alignment_status,
                "alignment_offset_spread_seconds": alignment_spread,
                "template_end_seconds": template_end,
                "audio_timeline_margin_seconds": timeline_margin,
                "audio_file": audio_path.relative_to(root).as_posix() if audio_path.is_relative_to(root) else str(audio_path),
                "audio_source_status": source_status,
                **audio_info,
            }
            manifest.append(row)

    split_manifest: list[dict[str, Any]] = []
    for row in manifest:
        if not row["analysis_eligible"]:
            continue
        split_manifest.append(
            {
                **row,
                "sample_id": row["segment_id"],
                "subject_id": row["participant_id"],
                "start_sec": row["audio_onset_seconds"],
                "end_sec": row["audio_offset_seconds"],
                "split": block_split[row["block_id"]],
            }
        )

    cue_path = root / "stimuli" / "pleasePressSpace_normed.wav"
    cue_audio, cue_rate = soundfile.read(cue_path, dtype="float64")
    catalan_block_comparisons: list[dict[str, Any]] = []
    for stimulus_id, path in sorted(block_files.items()):
        if not stimulus_id.startswith("catalan-"):
            continue
        audio, rate = soundfile.read(path, dtype="float64")
        same_shape = audio.shape == cue_audio.shape and rate == cue_rate
        if same_shape:
            difference = audio - cue_audio
            rms_difference = float(np.sqrt(np.mean(difference * difference)))
            correlation = float(np.corrcoef(audio, cue_audio)[0, 1])
        else:
            rms_difference = None
            correlation = None
        catalan_block_comparisons.append(
            {
                "stimulus_id": stimulus_id,
                "block_file": path.relative_to(root).as_posix(),
                "same_shape_as_please_press_space": same_shape,
                "waveform_correlation_with_please_press_space": correlation,
                "waveform_rms_difference": rms_difference,
            }
        )

    checks = split_checks(split_manifest, embargo_seconds)
    component_report = connected_component_summary(split_manifest)
    language_counts = Counter(row["language"] for row in manifest if row["stimulus_role"] == "PASSAGE")
    eligible_language_counts = Counter(row["language"] for row in split_manifest)
    summary = {
        "task_id": "M6A-PUBLIC-001",
        "dataset_id": "ds004703",
        "dataset_version": "1.1.0",
        "integrity_policy": "NON_HASH_AUDIT",
        "segment_count": len(manifest),
        "analysis_eligible_segment_count": len(split_manifest),
        "recording_count": len({row["recording_id"] for row in manifest}),
        "stimulus_count": len({row["stimulus_id"] for row in manifest}),
        "passage_language_counts": dict(sorted(language_counts.items())),
        "eligible_language_counts": dict(sorted(eligible_language_counts.items())),
        "block_assignments": block_split,
        "original_grouping_connected_components": component_report,
        "refrozen_primary_policy": {
            "required_group_keys": ["stimulus_id", "block_id"],
            "recording_policy": "MAY_SPAN_SPLITS_WITH_NONOVERLAPPING_PASSAGE_WINDOWS_AND_TEMPORAL_EMBARGO",
            "language_policy": "EXPLICIT_MANIFEST_AND_SPLIT_COVERAGE_AUDIT",
            "speaker_policy": "ADVISORY_ONLY_SPEAKER_GENERALIZATION_NOT_CLAIMED",
            "catalan_status": "EXCLUDED_FROM_PRIMARY_BASELINE_PENDING_AUDIO_PROVENANCE_RESOLUTION",
        },
        "split_checks": checks,
        "metadata_conflicts": [
            "README and iEEG sidecars declare 6 blocks x 7 passages, but bundled Block directories contain 8 English passages plus 1 Catalan-labeled file each",
            "all six Catalan-labeled Block files are waveform-identical to pleasePressSpace_normed.wav while events templates span approximately 46 seconds",
            "catalan_v2 files match the events template duration but exact played-waveform provenance remains unresolved",
        ],
        "catalan_block_nonhash_waveform_comparisons": catalan_block_comparisons,
        "errors": sorted(set(errors)),
    }
    failures = (
        summary["errors"]
        or checks["stimulus_split_conflicts"]
        or checks["block_split_conflicts"]
        or checks["temporal_cross_split_violations"]
        or any(row["language"] == "UNKNOWN" for row in manifest if row["stimulus_role"] == "PASSAGE")
    )
    summary["manifest_status"] = "FAIL" if failures else "PASS_WITH_CATALAN_EXCLUSION_AND_SPLIT_REDESIGN"
    summary["g2_status"] = "PENDING_FULL_DATASET_AUDIT"
    return manifest, split_manifest, summary
