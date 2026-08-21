from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "m6a_public_003_temporal_benchmark.py"
CONFIG_PATH = ROOT / "configs" / "m6a_public_003.json"

torch = pytest.importorskip("torch")


def load_module():
    spec = importlib.util.spec_from_file_location("m6a_public_003_temporal_benchmark", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_required_three_group_specs_and_no_pretrained_dependency():
    module = load_module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    specs = module.make_specs(config)
    assert {spec.family for spec in specs} == {
        "downsampling",
        "rf_growth",
        "explicit_change_branch",
    }
    assert {spec.name for spec in specs} >= {
        "early_downsample",
        "late_downsample",
        "uniform_local",
        "exponential_growth",
        "delayed_growth",
        "parallel_multiscale",
        "rf_stride_coupled",
        "rf_dilation_decoupled",
        "event_baseline",
        "explicit_change",
        "ordinary_second_branch",
    }
    assert all("pretrained" not in spec.notes.lower() for spec in specs)


def test_rf_pair_is_exactly_matched_but_resolution_differs():
    module = load_module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    specs = {spec.name: spec for spec in module.make_specs(config)}
    coupled = module.final_rf_info(specs["rf_stride_coupled"], 512, 1.0)
    decoupled = module.final_rf_info(specs["rf_dilation_decoupled"], 512, 1.0)
    assert coupled[0] == decoupled[0] == 33
    assert coupled[2] == 4.0
    assert decoupled[2] == 1.0


def test_parameter_matching_for_primary_controls():
    module = load_module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    specs = {spec.name: spec for spec in module.make_specs(config)}

    def count(name: str) -> int:
        model = module.TemporalModel(specs[name], head_width=32)
        return sum(parameter.numel() for parameter in model.parameters())

    assert count("early_downsample") == count("late_downsample")
    assert count("uniform_local") == count("exponential_growth") == count("delayed_growth")
    assert count("rf_stride_coupled") == count("rf_dilation_decoupled")
    assert count("explicit_change") == count("ordinary_second_branch")


def test_change_branch_forward_shape_and_difference_input():
    module = load_module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    specs = {spec.name: spec for spec in module.make_specs(config)}
    x = torch.randn(2, 1, 512)
    for name in ["event_baseline", "explicit_change", "ordinary_second_branch"]:
        model = module.TemporalModel(specs[name], head_width=32).eval()
        with torch.no_grad():
            output = model(x)
        assert output["features"].shape[0] == 2
        assert output["features"].shape[-1] == 512
        assert output["onset_reg"].shape == (2,)


def test_rf_rows_include_required_schema_fields():
    module = load_module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    specs = module.make_specs(config)
    rows = module.rf_rows_for_spec(specs[5], 512, 1.0)
    required = {
        "model",
        "layer",
        "kernel",
        "stride",
        "dilation",
        "jump_frame_step",
        "theoretical_RF_samples",
        "theoretical_RF_ms",
        "output_time_resolution_ms",
    }
    assert required <= set(rows[0])
    assert len(rows) >= 5
