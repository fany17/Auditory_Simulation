from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MODULE_PATH = SCRIPTS / "m6a_public_003_perturbation_eval.py"
CONFIG_PATH = ROOT / "configs" / "m6a_public_003.json"

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - runtime boundary is reported by the runner
    torch = None


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("m6a_public_003_perturbation_eval", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch is not None, "PyTorch is required for perturbation smoke tests")
class M6APublic003PerturbationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_exact_magnitude_grid_is_configured(self):
        self.assertEqual(self.config["supplementary_perturbation_magnitudes_ms"], [10.0, 20.0, 50.0])
        self.assertEqual(self.module.MAGNITUDES_MS, (10.0, 20.0, 50.0))
        self.assertEqual(set(self.module.METRIC_PROBES.values()), {"localization", "discrimination", "generalization"})

    def test_right_shift_moves_signal_and_onset_target(self):
        base = {
            "x": np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32),
            "onset_ms": np.asarray([1.0], dtype=np.float32),
            "onset_reg": np.asarray([1.0 / 6.0], dtype=np.float32),
        }
        shifted = self.module.right_shift_arrays(base, 2.0, 1.0)
        np.testing.assert_array_equal(shifted["x"], np.asarray([[0.0, 0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        self.assertEqual(float(shifted["onset_ms"][0]), 3.0)
        self.assertAlmostEqual(float(shifted["onset_reg"][0]), 0.5)

    def test_jitter_probe_isolates_only_jitter_magnitude(self):
        probe = self.module.isolated_jitter_arrays(self.config, 24, 1234, 50.0)
        self.assertEqual(set(np.unique(probe["jitter_ms"]).tolist()), {0.0, 50.0})
        self.assertEqual(set(np.unique(probe["rate_hz"]).tolist()), set(self.config["train_rates_hz"]))
        observed_phase = np.unique(probe["phase_rad"]).astype(float)
        allowed_phase = np.asarray([0.0, -0.25 * np.pi, 0.25 * np.pi, -0.5 * np.pi, 0.5 * np.pi])
        self.assertTrue(np.all(np.min(np.abs(observed_phase[:, None] - allowed_phase[None, :]), axis=1) < 1e-5))

    def test_summary_schema_keeps_all_metric_axes(self):
        rows = []
        for seed in [11, 22, 33]:
            for magnitude in self.module.MAGNITUDES_MS:
                for metric in self.module.SUPPLEMENT_METRICS:
                    rows.append({
                        "variant": "early_downsample",
                        "seed": seed,
                        "magnitude_ms": magnitude,
                        "probe": self.module.METRIC_PROBES[metric],
                        "metric": metric,
                        "value": 1.0,
                    })
        summary = self.module.supplement_summary_rows(rows)
        self.assertEqual(len(summary), 15)
        self.assertTrue({row["magnitude_ms"] for row in summary} == {10.0, 20.0, 50.0})
        self.assertTrue({row["metric"] for row in summary} == set(self.module.SUPPLEMENT_METRICS))
        self.assertTrue(all(row["n_seeds"] == 3 for row in summary))


if __name__ == "__main__":
    unittest.main()
