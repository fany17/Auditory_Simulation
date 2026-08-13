from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from m6a_public.embargo_gate import evaluate_final_embargo


FORBIDDEN_FIELD_NAMES = {
    "sha",
    "sha1",
    "sha256",
    "sha512",
    "md5",
    "checksum",
    "etag",
    "file_hash",
    "object_hash",
    "commit_hash",
    "commit_id",
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def find_forbidden_fields(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if (
                normalized in FORBIDDEN_FIELD_NAMES
                or normalized.endswith("_hash")
                or normalized.startswith("hash_")
                or normalized.endswith("_checksum")
            ):
                findings.append(f"{path}.{key}")
            findings.extend(find_forbidden_fields(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(find_forbidden_fields(child, f"{path}[{index}]"))
    return findings


def validate_task_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("task_id") != "M6A-PUBLIC-001":
        errors.append("task_id must be M6A-PUBLIC-001")
    if config.get("status") != "ACTIVE_EXECUTION":
        errors.append("status must be ACTIVE_EXECUTION")

    integrity = config.get("integrity_policy", {})
    if integrity.get("mode") != "NON_HASH_AUDIT":
        errors.append("integrity_policy.mode must be NON_HASH_AUDIT")
    if integrity.get("cryptographic_integrity_claim") is not False:
        errors.append("cryptographic_integrity_claim must be false")

    dataset = config.get("dataset", {})
    if dataset.get("dataset_id") != "ds004703":
        errors.append("dataset_id must be ds004703")
    if dataset.get("version") != "1.1.0":
        errors.append("dataset version must be 1.1.0")
    license_status = dataset.get("license_status")
    if dataset.get("download_allowed") and license_status != "ACCEPTED_WITH_STRICTER_README_BOUNDARY":
        errors.append("dataset download requires the stricter README boundary")
    boundaries = set(dataset.get("strict_use_boundaries", []))
    required_boundaries = {
        "NONCOMMERCIAL_ACADEMIC_RESEARCH_ONLY",
        "NO_REIDENTIFICATION",
        "NO_RAW_DATA_EXPORT_FROM_2203",
    }
    if not required_boundaries.issubset(boundaries):
        errors.append("dataset strict_use_boundaries are incomplete")
    if dataset.get("interactive_terms_stop") is not True:
        errors.append("interactive_terms_stop must be true")

    g2 = config.get("g2", {})
    if g2.get("status") != "G2_COORDINATOR_ACCEPTED_FOR_AUDIO_CONTEXT_GATE":
        errors.append("G2 must record coordinator acceptance for the audio-context gate")
    if g2.get("coordinator_review") != "ACCEPT" or g2.get("reviewed_on") != "2026-08-13":
        errors.append("G2 coordinator acceptance and review date are incomplete")
    if g2.get("whole_m6a_pass_claimed") is not False:
        errors.append("G2 acceptance cannot claim the whole M6A task passed")

    model = config.get("model", {})
    if model.get("model_id") != "facebook/wav2vec2-base":
        errors.append("first model must be facebook/wav2vec2-base")
    if model.get("trainable") is not False:
        errors.append("first model must remain frozen")
    if model.get("sampling_rate_hz") != 16000:
        errors.append("wav2vec2 input sampling rate must be 16000 Hz")
    if model.get("download_allowed") is not False:
        errors.append("model download must be closed after semantic cache validation")
    if model.get("cache_state") != "SEMANTICALLY_VALIDATED_REMOTE_ONLY":
        errors.append("model cache must be frozen as semantically validated and remote-only")
    if model.get("revision_label") != "main":
        errors.append("the sole model revision label must remain main")
    if model.get("revision_limitation") != "MUTABLE_MAIN_LABEL_NON_CRYPTOGRAPHIC_REPRODUCIBILITY_ONLY":
        errors.append("the mutable main revision limitation must be explicit")
    if model.get("source_endpoint") != "https://hf-mirror.com":
        errors.append("the audio-context node must use only the fixed 2203-accessible mirror")
    if model.get("source_endpoint_role") != "PUBLIC_HUGGING_FACE_ENDPOINT_MIRROR":
        errors.append("model source endpoint role is not auditable")
    if (
        model.get("source_endpoint_limitation")
        != "THIRD_PARTY_MIRROR_PLUS_MUTABLE_MAIN_AND_NO_HASH_POLICY_DO_NOT_PROVIDE_CRYPTOGRAPHIC_INTEGRITY_OR_IMMUTABLE_PROVENANCE"
    ):
        errors.append("mirror and mutable-main provenance limitation must be explicit")
    if model.get("remote_cache") != (
        "/home/fanyu/auditory_simulation_m6a/cache/huggingface/"
        "facebook_wav2vec2_base_main_20260813"
    ):
        errors.append("the model cache must remain at the dedicated remote-only path")
    input_gate = model.get("inference_input", {})
    expected_input_gate = {
        "source_sampling_rate_hz": 44100,
        "model_sampling_rate_hz": 16000,
        "channels": 1,
        "mono_policy": "REQUIRE_MONO_NO_IMPLICIT_DOWNMIX",
        "passage_policy": "ONE_ELIGIBLE_PASSAGE_PER_INFERENCE_CALL",
        "neighbor_audio_read_allowed": False,
        "batch_padding": "FORBIDDEN_PRIMARY_INFERENCE",
        "transformer_attention_scope": "GLOBAL_WITHIN_SINGLE_PASSAGE",
        "transformer_local_receptive_field_claimed": False,
    }
    if input_gate != expected_input_gate:
        errors.append("wav2vec2 passage-isolation input semantics are not frozen")

    split = config.get("split", {})
    required_groups = set(split.get("required_group_keys", []))
    if required_groups != {"stimulus_id", "block_id"}:
        errors.append("split must guard stimulus_id and block_id")
    if "language" not in set(split.get("stratification_keys", [])):
        errors.append("split must carry language as an explicit stratification key")
    if split.get("language_policy") != "EXPLICIT_MANIFEST_AND_SPLIT_COVERAGE_AUDIT":
        errors.append("split language policy must audit explicit split coverage")
    if split.get("recording_policy") != "MAY_SPAN_SPLITS_WITH_NONOVERLAPPING_PASSAGE_WINDOWS_AND_TEMPORAL_EMBARGO":
        errors.append("split recording policy must preserve within-recording temporal isolation")
    if split.get("original_recording_grouping_status") != "INFEASIBLE_SINGLE_CONNECTED_COMPONENT":
        errors.append("original recording grouping no-go must remain recorded")
    expected_block_assignments = {
        "block-01": "train",
        "block-02": "train",
        "block-03": "validation",
        "block-04": "test",
        "block-05": "train",
        "block-06": "train",
    }
    if split.get("block_assignments") != expected_block_assignments:
        errors.append("block assignments must match the reviewed deterministic ratio optimum")
    if split.get("assignment_method") != "DETERMINISTIC_GROUP_SIZE_RATIO_OPTIMIZATION":
        errors.append("split assignment must use deterministic group-size ratio optimization")
    if split.get("preliminary_minimum_embargo_seconds", 0) != 2.0:
        errors.append("preliminary minimum embargo must remain 2 seconds")
    if split.get("split_status") != "BASELINE_FINAL_COORDINATOR_ACCEPTED":
        errors.append("split must record baseline-final coordinator acceptance")
    if split.get("baseline_final") is not True:
        errors.append("baseline_final must be true after final-embargo coordinator acceptance")
    if split.get("final_embargo_seconds") != 2.0:
        errors.append("accepted final embargo must be 2 seconds")
    if split.get("final_embargo_candidate_seconds") != 2.0:
        errors.append("final embargo candidate must be 2 seconds")
    if (
        split.get("final_embargo_status")
        != "FINAL_EMBARGO_COORDINATOR_ACCEPTED"
    ):
        errors.append("final embargo must record coordinator acceptance")
    if split.get("primary_generalization_scope") != "WITHIN_SUBJECT_UNSEEN_STIMULUS_AND_BLOCK_ONLY":
        errors.append("primary generalization scope must remain within-subject unseen stimulus/block only")
    for claim_key in (
        "subject_heldout_claim_allowed",
        "speaker_heldout_claim_allowed",
        "cross_language_claim_allowed",
    ):
        if split.get(claim_key) is not False:
            errors.append(f"{claim_key} must be false at the preliminary split gate")
    if split.get("secondary_subject_generalization") is not False:
        errors.append("secondary subject generalization is not supported by the current split")
    embargo_report = evaluate_final_embargo(split.get("final_embargo_components_seconds", {}))
    if (
        embargo_report["status"] != "PASS"
        or embargo_report["baseline_final"] is not True
        or embargo_report.get("final_embargo_seconds") != 2.0
    ):
        errors.append("final embargo components must form an accepted baseline-final 2-second gate")
    if split.get("final_embargo_components_seconds", {}).get("filter_or_padding_edge_seconds") != 1.091796875:
        errors.append("split embargo must record the frozen neural filter/resampling edge")
    if split.get("final_embargo_components_seconds", {}).get("audio_cross_split_context_overlap_seconds") != 0.0:
        errors.append("measured audio cross-split input overlap must be zero for isolated passages")
    if (
        split.get("final_embargo_components_seconds", {}).get("audio_resampling_edge_seconds")
        != 0.0006349206349206349
    ):
        errors.append("split embargo must record the finite audio-resampling edge")
    if set(split.get("allowed_splits", [])) != {"train", "validation", "test"}:
        errors.append("allowed_splits must be train/validation/test")

    g3 = config.get("g3_single_recording", {})
    expected_g3 = {
        "status": "G3_SINGLE_RECORDING_COORDINATOR_ACCEPTED_ENGINEERING_ONLY",
        "coordinator_review": "ACCEPT",
        "reviewed_on": "2026-08-13",
        "candidate_report": "reports/g3_single_recording_candidate_20260813.json",
        "scientific_result_claimed": False,
        "config_path": "configs/m6a_g3_single_recording_candidate.json",
        "recording_id": "sub-SD012_ses-02_task-PassiveListen",
        "sample_id": "sub-SD012_ses-02_task-PassiveListen__seg-004",
        "edf_relative_path": (
            "sub-SD012/ses-02/ieeg/"
            "sub-SD012_ses-02_task-PassiveListen_ieeg.edf"
        ),
        "audio_relative_path": "stimuli/excerpts/Block 1/s4002b-ex01_normed.wav",
        "eligible_channel_count": 36,
        "real_neural_waveform_read_scope": (
            "ONE_SELECTED_RECORDING_ONE_PASSAGE_36_ELIGIBLE_CHANNELS_"
            "PLUS_FROZEN_FINITE_SUPPORT_ONLY"
        ),
        "other_recordings_allowed": False,
        "other_segments_allowed": False,
        "whole_dataset_neural_extraction_allowed": False,
        "formal_baseline_run_allowed": False,
        "scientific_result_claim_allowed": False,
        "exchange_candidate_creation_allowed": False,
    }
    if g3 != expected_g3:
        errors.append("G3 acceptance must remain exact, engineering-only and fail-closed")

    neural_target = config.get("neural_target", {})
    if neural_target.get("status") != "METHOD_FROZEN_AWAITING_EXECUTION_GATES":
        errors.append("neural target must remain method-frozen and awaiting execution gates")
    if neural_target.get("name") != "LINE_HARMONIC_EXCLUDED_MULTIBAND_HIGH_GAMMA_LOG_POWER":
        errors.append("neural target primary name is not frozen")
    if neural_target.get("method_candidate_path") != "configs/m6a_neural_target_method_candidate.json":
        errors.append("neural target method candidate path is not frozen")
    if (
        neural_target.get("method_candidate_schema_path")
        != "schemas/m6a_neural_target_method_candidate.schema.json"
    ):
        errors.append("neural target method candidate schema path is not frozen")
    if neural_target.get("primary_reference_policy") != "AS_RECORDED_SCALP_REFERENCE":
        errors.append("neural target reference must remain as recorded")
    if neural_target.get("sidecar_reference_value") != "scalp electrode, not included with data":
        errors.append("neural target must record the exact sidecar iEEGReference")
    if neural_target.get("sidecar_reference_recording_count") != 11:
        errors.append("neural target must record all 11 sidecar reference declarations")
    if neural_target.get("method_coordinator_review") != "ACCEPT":
        errors.append("neural target must record coordinator method-freeze acceptance")
    if neural_target.get("method_reviewed_on") != "2026-08-11":
        errors.append("neural target method review date is not frozen")
    if neural_target.get("observed_power_line_frequency_hz") != 60:
        errors.append("neural target must record the observed 60 Hz line frequency")
    if 120 not in neural_target.get("line_harmonics_inside_candidate_band_hz", []):
        errors.append("neural target must record the 120 Hz harmonic inside 70-150 Hz")
    if neural_target.get("neural_extraction_allowed") is not False:
        errors.append("neural extraction must remain blocked until target method is refrozen")
    if neural_target.get("resolution_status") != "METHOD_FROZEN":
        errors.append("neural target resolution must record method freeze")

    anatomy = config.get("anatomy_mapping", {})
    if anatomy.get("status") != "ANATOMY_MAPPING_NOT_READY":
        errors.append("anatomy mapping must remain explicitly not ready")
    if anatomy.get("region_summary_status") != "NOT_ESTIMABLE":
        errors.append("region summary must remain NOT_ESTIMABLE without audited mapping")
    if anatomy.get("contact_name_inference_allowed") is not False:
        errors.append("contact names cannot be used to infer brain regions")

    features = config.get("features", {})
    expected_features = {
        "protocol_status": "G4_PROTOCOL_CANDIDATE_AWAITING_COORDINATOR_REVIEW",
        "protocol_config_path": "configs/m6a_g4_protocol_candidate.json",
        "acoustic_baselines": [
            "amplitude_envelope",
            "log_mel_train_only_pca_20",
        ],
        "model_features": "wav2vec2_projected_plus_12_transformer_layers",
        "fit_transforms_on": "train_only_common_support_frames",
        "g4_execution_authorized": False,
    }
    if features != expected_features:
        errors.append("G4 feature protocol pointer and execution gate drifted")

    baseline = config.get("baseline", {})
    if baseline.get("protocol_status") != "G4_PROTOCOL_CANDIDATE_AWAITING_COORDINATOR_REVIEW":
        errors.append("baseline must remain a G4 protocol candidate")
    if baseline.get("primary") != "layerwise_ridge_encoding":
        errors.append("primary baseline must be layerwise_ridge_encoding")
    if (
        baseline.get("alpha_selection")
        != "validation_only_then_refit_train_plus_validation_with_locked_alpha"
        or baseline.get("transform_fit_partition")
        != "train_only_no_refit_after_validation"
        or baseline.get("test_evaluation_count") != 1
        or baseline.get("g4_execution_authorized") is not False
    ):
        errors.append("G4 ridge selection/refit/test-once gate drifted")
    if "region_summary" in baseline.get("secondary_metrics", []):
        errors.append("region_summary cannot be an ordinary metric before anatomy mapping")
    gated_metrics = {item.get("name"): item for item in baseline.get("gated_metrics", [])}
    if gated_metrics.get("region_summary", {}).get("status") != "NOT_ESTIMABLE":
        errors.append("region_summary must be a gated NOT_ESTIMABLE metric")

    nulls = config.get("nulls", {})
    expected_nulls = {
        "primary_smoke_null": (
            "stimulus_derangement_uniform_without_replacement_from_14833_test_derangements"
        ),
        "acoustic_secondary_diagnostic": (
            "within_passage_circular_shift_minimum_2_seconds"
        ),
        "wav2vec2_circular_shift_applicability": (
            "NOT_APPLICABLE_GLOBAL_WITHIN_PASSAGE_TRANSFORMER_CONTEXT"
        ),
        "smoke_permutations": 20,
        "formal_permutations": 1000,
        "multiple_comparison": "ONE_SIDED_MAX_STATISTIC_FIXED_FAMILIES",
        "smoke_significance_claim_allowed": False,
        "g4_execution_authorized": False,
    }
    if nulls != expected_nulls:
        errors.append("G4 primary null, max-statistic, or execution gate drifted")

    artifact = config.get("artifact", {})
    if artifact.get("internal_schema_path") != "schemas/m6a_public_internal_manifest.schema.json":
        errors.append("internal run manifest path is not frozen")
    if artifact.get("exchange_contract_status") != "REVISED_DRAFT_ACCEPTED_FOR_CANDIDATE_PREPARATION":
        errors.append("exchange contract review status must record revised DRAFT acceptance")
    if artifact.get("exchange_consumer_status") != "READY_WAITING_M6A_CANDIDATE":
        errors.append("exchange consumer must remain ready and waiting for a real M6A candidate")
    if artifact.get("consumer_cross_test_status") != "NOT_RUN":
        errors.append("consumer cross-test cannot run before a real candidate exists")
    if artifact.get("exchange_candidate_exists") is not False:
        errors.append("no exchange candidate exists before G2, target and embargo gates")
    if artifact.get("frozen_m6a_artifact_exists") is not False:
        errors.append("no frozen M6A artifact exists at G0-G2")

    resources = config.get("resources", {})
    if (
        resources.get("host_alias") != "server2203"
        or resources.get("remote_project_root")
        != "/home/fanyu/auditory_simulation_m6a"
        or resources.get("conda_environment") != "auditory_m6a_public_001"
        or resources.get("smoke_gpu_hours_limit") != 2
        or resources.get("continuous_gpu_hours_report_threshold") != 24
        or resources.get("storage_bytes_report_threshold") != 500_000_000_000
        or resources.get("minimum_free_bytes") != 500_000_000_000
    ):
        errors.append("G4 resource bounds or dedicated 2203 environment drifted")

    forbidden = find_forbidden_fields(config)
    if forbidden:
        errors.append("forbidden integrity fields: " + ", ".join(forbidden))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the M6A task gate without hashes.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    errors = validate_task_config(load_json(args.config))
    report = {"status": "PASS" if not errors else "FAIL", "errors": errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
