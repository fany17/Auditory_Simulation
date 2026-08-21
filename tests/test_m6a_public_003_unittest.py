from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "m6a_public_003_temporal_benchmark.py"
CONFIG_PATH = ROOT / "configs" / "m6a_public_003.json"

try:
    import torch
except ImportError:  # pragma: no cover - runtime boundary is reported by the test runner
    torch = None


def load_module():
    spec = importlib.util.spec_from_file_location("m6a_public_003_temporal_benchmark", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch is not None, "PyTorch is required for model smoke tests")
class M6APublic003Tests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.specs = {spec.name: spec for spec in self.module.make_specs(self.config)}

    def test_three_group_scope_and_no_pretrained_dependency(self):
        self.assertEqual(
            {spec.family for spec in self.specs.values()},
            {"downsampling", "rf_growth", "explicit_change_branch"},
        )
        self.assertTrue(
            {
                "early_downsample", "late_downsample", "uniform_local", "exponential_growth",
                "delayed_growth", "parallel_multiscale", "rf_stride_coupled", "rf_dilation_decoupled",
                "event_baseline", "explicit_change", "ordinary_second_branch",
            } <= set(self.specs)
        )
        self.assertTrue(all("pretrained" not in spec.notes.lower() for spec in self.specs.values()))

    def test_rf_pair_is_exactly_matched_but_resolution_differs(self):
        coupled = self.module.final_rf_info(self.specs["rf_stride_coupled"], 512, 1.0)
        decoupled = self.module.final_rf_info(self.specs["rf_dilation_decoupled"], 512, 1.0)
        self.assertEqual(coupled[0], decoupled[0])
        self.assertEqual(coupled[0], 33)
        self.assertEqual(coupled[2], 4.0)
        self.assertEqual(decoupled[2], 1.0)

    def test_parameter_matching_for_primary_controls(self):
        def count(name: str) -> int:
            model = self.module.TemporalModel(self.specs[name], head_width=32)
            return sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(count("early_downsample"), count("late_downsample"))
        self.assertEqual(count("uniform_local"), count("exponential_growth"))
        self.assertEqual(count("uniform_local"), count("delayed_growth"))
        self.assertEqual(count("rf_stride_coupled"), count("rf_dilation_decoupled"))
        self.assertEqual(count("explicit_change"), count("ordinary_second_branch"))

    def test_change_branch_forward_shape(self):
        x = torch.randn(2, 1, 512)
        for name in ["event_baseline", "explicit_change", "ordinary_second_branch"]:
            model = self.module.TemporalModel(self.specs[name], head_width=32).eval()
            with torch.no_grad():
                output = model(x)
            self.assertEqual(output["features"].shape[0], 2)
            self.assertEqual(output["features"].shape[-1], 512)
            self.assertEqual(tuple(output["onset_reg"].shape), (2,))

    def test_rf_rows_include_required_schema_fields(self):
        rows = self.module.rf_rows_for_spec(self.specs["parallel_multiscale"], 512, 1.0)
        required = {
            "model", "layer", "kernel", "stride", "dilation", "jump_frame_step",
            "theoretical_RF_samples", "theoretical_RF_ms", "output_time_resolution_ms",
        }
        self.assertTrue(required <= set(rows[0]))
        self.assertGreaterEqual(len(rows), 5)


if __name__ == "__main__":
    unittest.main()
