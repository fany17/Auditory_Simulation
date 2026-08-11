from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m6a_public.exchange_validator import validate_exchange_manifest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(
    (ROOT / "schemas" / "m6a_to_m6b_exchange_manifest_v1.schema.json").read_text(encoding="utf-8")
)


def revised_manifest() -> dict:
    timestamp = "2026-08-11T12:00:00Z"
    inventory = [
        {"path": "method/extract.py", "role": "METHOD_ENTRYPOINT", "bytes": 1, "modified_at_utc": timestamp, "availability": "INCLUDED_LOCAL"},
        {"path": "method/runtime.json", "role": "RUNTIME_SPEC", "bytes": 1, "modified_at_utc": timestamp, "availability": "INCLUDED_LOCAL"},
        {"path": "method/config.schema.json", "role": "EXTRACTION_CONFIG_SCHEMA", "bytes": 1, "modified_at_utc": timestamp, "availability": "INCLUDED_LOCAL"},
        {"path": "method/output.schema.json", "role": "EXTRACTION_OUTPUT_SCHEMA", "bytes": 1, "modified_at_utc": timestamp, "availability": "INCLUDED_LOCAL"},
        {"path": "canary/input.wav", "role": "CANARY_INPUT", "bytes": 1, "modified_at_utc": timestamp, "availability": "INCLUDED_LOCAL"},
        {"path": "canary/expected.npz", "role": "CANARY_EXPECTED_OUTPUT", "bytes": 1, "modified_at_utc": timestamp, "availability": "INCLUDED_LOCAL"},
        {"path": "remote_model/config.json", "role": "MODEL_CONFIG_SOURCE", "bytes": 1, "modified_at_utc": timestamp, "availability": "REMOTE_REFERENCE_ONLY"},
        {"path": "remote_model/model.safetensors", "role": "MODEL_WEIGHT_SOURCE", "bytes": 1, "modified_at_utc": timestamp, "availability": "REMOTE_REFERENCE_ONLY"},
    ]
    return {
        "identity": {
            "contract_version": "1-draft",
            "release_id": "fixture-revised-draft",
            "producer": "Auditory_Simulation/M6A-PUBLIC",
            "consumer": "STN_Decoding_Encoding/M6B-STN",
            "release_status": "REVISED_DRAFT_AWAITING_CONSUMER_REVIEW",
            "status_history": [
                {"status": "DRAFT_PROPOSED_BY_M6A", "recorded_at_utc": timestamp},
                {"status": "REVISED_DRAFT_AWAITING_CONSUMER_REVIEW", "recorded_at_utc": timestamp},
            ],
        },
        "acceptance": {
            "producer_review": "PASS",
            "consumer_cross_test": "PENDING",
            "coordinator_decision": "PENDING",
        },
        "public_dataset": {
            "dataset_id": "ds004703",
            "version": "1.1.0",
            "doi": "10.18112/openneuro.ds004703.v1.1.0",
            "license_boundary": "CC0 metadata plus stricter noncommercial and no-reidentification README boundary",
            "benchmark_role": "public benchmark only",
            "redistribution": "NO_RAW_DATA_OR_FULL_PUBLIC_FEATURE_PAYLOAD",
        },
        "model": {
            "model_id": "facebook/wav2vec2-base",
            "revision_label": "main resolved 2026-08-11",
            "revision_immutable": False,
            "resolved_at_utc": timestamp,
            "license": "Apache-2.0",
            "frozen": True,
            "revision_limitation": "Readable revision label is not a cryptographic fixation of model weights.",
            "software_versions": {"python": "3.11", "torch": "2.11", "transformers": "5.14", "numpy": "2.4"},
            "model_files": [
                {"file_name": "config.json", "inventory_path": "remote_model/config.json", "role": "MODEL_CONFIG", "bytes": 1, "modified_at_utc": timestamp, "cache_boundary": "remote model cache only", "included_in_exchange": False},
                {"file_name": "model.safetensors", "inventory_path": "remote_model/model.safetensors", "role": "MODEL_WEIGHTS", "bytes": 1, "modified_at_utc": timestamp, "cache_boundary": "remote model cache only", "included_in_exchange": False},
            ],
        },
        "audio_preprocessing": {
            "input_rate_hz": 16000,
            "channels": "mono",
            "amplitude_rule": "float waveform without peak renormalization",
            "resampling": "declared implementation and version required",
            "chunking": "no chunking for canary",
            "time_reference": "input sample zero",
            "edge_handling": "no padding for canary",
        },
        "layer_inventory": [
            {"layer_key": "projection", "ordinal": 0, "source": "feature projection", "frame_rate_hz": 50.0, "dtype": "float32", "feature_dim": 3, "time_axis_reference": "frame center"},
            {"layer_key": "encoder_00", "ordinal": 1, "source": "encoder layer 0", "frame_rate_hz": 50.0, "dtype": "float32", "feature_dim": 4, "time_axis_reference": "frame center"},
        ],
        "extraction_spec": {
            "entrypoint_file": "method/extract.py",
            "config_schema_file": "method/config.schema.json",
            "output_schema_file": "method/output.schema.json",
            "runtime_spec_file": "method/runtime.json",
            "output_layout": "layer keyed two-dimensional arrays",
            "batch_rule": "one audio item",
            "chunk_rule": "canary unchunked",
            "determinism_boundary": "declared runtime and numeric tolerance",
        },
        "transferable_transforms": [],
        "validation": {
            "evidence_profile": "DRAFT_EXPLICIT_LIMITATIONS",
            "split": "stimulus and recording grouped split pending G2",
            "leakage_checks": ["stimulus_id and block_id disjointness required; recording windows use temporal embargo"],
            "nulls": ["circular shift declared; not yet run"],
            "metrics": ["held-out Pearson r declared; not yet run"],
            "benchmark_summary": [
                {
                    "scope": "schema fixture only",
                    "dataset": {"dataset_id": "ds004703", "version": "1.1.0"},
                    "split": "not yet executed",
                    "target": "high gamma provisional",
                    "model_or_baseline": "layerwise ridge planned",
                    "metric": "held-out Pearson r",
                    "value": None,
                    "null_comparison": {"method": "circular shift", "status": "PENDING", "value": None, "permutations": None, "limitations": ["not yet run"]},
                    "status": "NOT_ESTIMABLE",
                    "limitations": ["no benchmark run exists at revised draft stage"],
                }
            ],
        },
        "known_failures": ["consumer cross-test not run"],
        "claims": {"can_claim": ["schema fixture validates"], "cannot_claim": ["contract frozen", "consumer cross-test passed"]},
        "method_package": {
            "package_version": "fixture-only",
            "entrypoint_file": "method/extract.py",
            "runtime_spec_file": "method/runtime.json",
            "files": ["method/extract.py", "method/runtime.json", "method/config.schema.json", "method/output.schema.json"],
        },
        "canary_fixture": {
            "synthetic": True,
            "input_file": "canary/input.wav",
            "input_format": "WAV_PCM16",
            "input_sample_rate_hz": 16000,
            "input_channels": 1,
            "input_sample_count": 16000,
            "expected_output_file": "canary/expected.npz",
            "expected_output_format": "NPZ",
            "expected_layer_order": ["projection", "encoder_00"],
            "expected_layers": [
                {"layer_key": "projection", "shape": [4, 3], "dtype": "float32", "frame_count": 4, "feature_dim": 3},
                {"layer_key": "encoder_00", "shape": [4, 4], "dtype": "float32", "frame_count": 4, "feature_dim": 4},
            ],
            "frame_time": {"unit": "seconds", "array_key": "frame_time_seconds", "shape": [4], "reference": "input sample zero"},
            "tolerance": {"absolute": 1e-5, "relative": 1e-4, "frame_time_seconds": 1e-7},
            "purpose": "NUMERIC_SHAPE_DTYPE_FRAME_TIME_LAYER_ORDER_CANARY",
        },
        "file_inventory": inventory,
    }


class ExchangeValidatorTests(unittest.TestCase):
    def test_revised_draft_fixture_passes(self) -> None:
        self.assertEqual(validate_exchange_manifest(revised_manifest(), SCHEMA), [])

    def test_additional_property_fails_closed(self) -> None:
        manifest = revised_manifest()
        manifest["unexpected"] = True
        self.assertTrue(any("schema" in item for item in validate_exchange_manifest(manifest, SCHEMA)))

    def test_empty_method_files_are_rejected(self) -> None:
        manifest = revised_manifest()
        manifest["method_package"]["files"] = []
        self.assertTrue(validate_exchange_manifest(manifest, SCHEMA))

    def test_empty_inventory_is_rejected(self) -> None:
        manifest = revised_manifest()
        manifest["file_inventory"] = []
        self.assertTrue(validate_exchange_manifest(manifest, SCHEMA))

    def test_method_file_must_be_in_inventory(self) -> None:
        manifest = revised_manifest()
        manifest["method_package"]["files"].append("method/missing.py")
        self.assertTrue(any("method package file" in item for item in validate_exchange_manifest(manifest, SCHEMA)))

    def test_duplicate_layer_key_is_rejected(self) -> None:
        manifest = revised_manifest()
        manifest["layer_inventory"][1]["layer_key"] = "projection"
        self.assertTrue(any("layer_key" in item for item in validate_exchange_manifest(manifest, SCHEMA)))

    def test_duplicate_layer_ordinal_is_rejected(self) -> None:
        manifest = revised_manifest()
        manifest["layer_inventory"][1]["ordinal"] = 0
        self.assertTrue(any("ordinal" in item for item in validate_exchange_manifest(manifest, SCHEMA)))

    def test_canary_files_must_be_in_inventory(self) -> None:
        manifest = revised_manifest()
        manifest["file_inventory"] = [
            item for item in manifest["file_inventory"] if item["role"] != "CANARY_EXPECTED_OUTPUT"
        ]
        self.assertTrue(any("canary_fixture.expected_output_file" in item for item in validate_exchange_manifest(manifest, SCHEMA)))

    def test_unsafe_inventory_path_is_rejected(self) -> None:
        manifest = revised_manifest()
        manifest["file_inventory"][0]["path"] = "../escape.py"
        self.assertTrue(any("unsafe inventory path" in item for item in validate_exchange_manifest(manifest, SCHEMA)))

    def test_illegal_transition_is_rejected(self) -> None:
        manifest = revised_manifest()
        manifest["identity"]["status_history"] = [
            manifest["identity"]["status_history"][0],
            {"status": "CANDIDATE_FOR_CROSS_TEST", "recorded_at_utc": "2026-08-11T12:00:00Z"},
        ]
        manifest["identity"]["release_status"] = "CANDIDATE_FOR_CROSS_TEST"
        manifest["validation"]["evidence_profile"] = "CANDIDATE_EVIDENCE"
        self.assertTrue(any("illegal release transition" in item for item in validate_exchange_manifest(manifest, SCHEMA)))

    def test_acceptance_state_is_controlled(self) -> None:
        manifest = revised_manifest()
        manifest["acceptance"]["consumer_cross_test"] = "NOT_RUN"
        self.assertTrue(validate_exchange_manifest(manifest, SCHEMA))

    def test_frozen_status_is_not_allowed_by_draft_schema(self) -> None:
        manifest = revised_manifest()
        manifest["identity"]["release_status"] = "FROZEN"
        self.assertTrue(validate_exchange_manifest(manifest, SCHEMA))

    def test_benchmark_item_requires_minimum_structure(self) -> None:
        manifest = revised_manifest()
        del manifest["validation"]["benchmark_summary"][0]["target"]
        self.assertTrue(validate_exchange_manifest(manifest, SCHEMA))

    def test_runtime_profile_is_nonempty_and_minimum(self) -> None:
        manifest = revised_manifest()
        del manifest["model"]["software_versions"]["numpy"]
        self.assertTrue(validate_exchange_manifest(manifest, SCHEMA))

    def test_nonimmutable_revision_requires_limitation(self) -> None:
        manifest = revised_manifest()
        del manifest["model"]["revision_limitation"]
        self.assertTrue(validate_exchange_manifest(manifest, SCHEMA))

    def test_model_files_must_link_to_remote_inventory(self) -> None:
        manifest = revised_manifest()
        manifest["model"]["model_files"][0]["inventory_path"] = "remote_model/missing.json"
        self.assertTrue(any("model file is not listed" in item for item in validate_exchange_manifest(manifest, SCHEMA)))

    def test_transform_must_link_to_inventory(self) -> None:
        manifest = revised_manifest()
        manifest["transferable_transforms"] = [{"name": "train_pca", "kind": "PCA", "fit_scope": "training split only", "source_split": "train", "input_layer": "projection", "output_dim": 2, "artifact_file": "transforms/pca.npz", "applicability": "public-fit transform only"}]
        self.assertTrue(any("transform artifact" in item for item in validate_exchange_manifest(manifest, SCHEMA)))

    def test_canary_shape_dtype_and_frame_time_are_cross_checked(self) -> None:
        for field, value in (("shape", [5, 3]), ("dtype", "float64")):
            manifest = revised_manifest()
            manifest["canary_fixture"]["expected_layers"][0][field] = value
            self.assertTrue(validate_exchange_manifest(manifest, SCHEMA))
        manifest = revised_manifest()
        manifest["canary_fixture"]["frame_time"]["shape"] = [5]
        self.assertTrue(validate_exchange_manifest(manifest, SCHEMA))

    def test_negative_canary_tolerance_is_rejected(self) -> None:
        manifest = revised_manifest()
        manifest["canary_fixture"]["tolerance"]["absolute"] = -1.0
        self.assertTrue(validate_exchange_manifest(manifest, SCHEMA))

    def test_candidate_requires_bundle_root(self) -> None:
        manifest = revised_manifest()
        manifest["identity"]["release_status"] = "CANDIDATE_FOR_CROSS_TEST"
        manifest["identity"]["status_history"].append({"status": "CANDIDATE_FOR_CROSS_TEST", "recorded_at_utc": "2026-08-11T12:00:00Z"})
        manifest["validation"]["evidence_profile"] = "CANDIDATE_EVIDENCE"
        self.assertTrue(any("requires bundle_root" in item for item in validate_exchange_manifest(manifest, SCHEMA)))

    def test_candidate_rejects_missing_and_duplicate_singleton_roles(self) -> None:
        manifest = revised_manifest()
        manifest["identity"]["release_status"] = "CANDIDATE_FOR_CROSS_TEST"
        manifest["identity"]["status_history"].append({"status": "CANDIDATE_FOR_CROSS_TEST", "recorded_at_utc": "2026-08-11T12:00:00Z"})
        manifest["validation"]["evidence_profile"] = "CANDIDATE_EVIDENCE"
        manifest["file_inventory"] = [
            item for item in manifest["file_inventory"] if item["role"] != "EXTRACTION_CONFIG_SCHEMA"
        ]
        errors = validate_exchange_manifest(manifest, SCHEMA)
        self.assertTrue(any("exactly one inventory item with role EXTRACTION_CONFIG_SCHEMA" in item for item in errors))

        manifest = revised_manifest()
        manifest["identity"]["release_status"] = "CANDIDATE_FOR_CROSS_TEST"
        manifest["identity"]["status_history"].append({"status": "CANDIDATE_FOR_CROSS_TEST", "recorded_at_utc": "2026-08-11T12:00:00Z"})
        manifest["validation"]["evidence_profile"] = "CANDIDATE_EVIDENCE"
        manifest["file_inventory"].append(
            {
                "path": "method/runtime-copy.json",
                "role": "RUNTIME_SPEC",
                "bytes": 1,
                "modified_at_utc": "2026-08-11T12:00:00Z",
                "availability": "INCLUDED_LOCAL",
            }
        )
        errors = validate_exchange_manifest(manifest, SCHEMA)
        self.assertTrue(any("exactly one inventory item with role RUNTIME_SPEC" in item for item in errors))

    def test_candidate_with_complete_local_files_passes(self) -> None:
        manifest = revised_manifest()
        manifest["identity"]["release_status"] = "CANDIDATE_FOR_CROSS_TEST"
        manifest["identity"]["status_history"].append({"status": "CANDIDATE_FOR_CROSS_TEST", "recorded_at_utc": "2026-08-11T12:00:00Z"})
        manifest["validation"]["evidence_profile"] = "CANDIDATE_EVIDENCE"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for item in manifest["file_inventory"]:
                target = root.joinpath(*item["path"].split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"x")
            self.assertEqual(validate_exchange_manifest(manifest, SCHEMA, root), [])


if __name__ == "__main__":
    unittest.main()
