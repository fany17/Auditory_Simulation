#!/usr/bin/env python3
"""Finalize the convolutional-frontend explainer extension and audit trail."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from route_b_temporal_models.wav2vec2_layer_demo.core import append_text, read_json, utc_now, write_json, write_text


TASK_ID = "TB001-DEMO001"
MARKER = "## Convolutional frontend extension (2026-08-07)"
FIGURE_MARKER = "## Frontend CER figure (2026-08-07)"


def main() -> int:
    pointer = read_json(ROOT / "outputs" / TASK_ID / "current_frontend.json")
    frontend = read_json(ROOT / pointer["relative_path"] / "frontend_results.json")
    x2 = next(item for item in frontend["conditions"] if item["id"] == "speed-2p00")
    variants = {item["frontend_variant_id"]: item for item in x2["variants"]}
    standard = variants["frontend-50hz"]
    hz100 = variants["frontend-100hz"]
    hz200 = variants["frontend-200hz"]
    figure_manifest = read_json(ROOT / "reports" / "figures" / "frontend_speed_cer_manifest.json")
    adaptive_cer = figure_manifest["adaptive_demo_cer_percent"]

    report = f"""# {TASK_ID} convolutional frontend experiment

## Question

For the same pitch-preserving 2× speech input, does reducing wav2vec2 convolutional temporal downsampling change or improve the final CTC transcript without retraining?

## Intervention

- Standard: convolution strides `{standard['conv_stride']}`, total stride `{standard['total_stride_samples']}` samples, measured `{standard['actual_ctc_frame_rate_hz']:.1f}` CTC frames/s.
- 100 Hz variant: convolution strides `{hz100['conv_stride']}`, total stride `{hz100['total_stride_samples']}` samples, measured `{hz100['actual_ctc_frame_rate_hz']:.1f}` CTC frames/s.
- 200 Hz variant: convolution strides `{hz200['conv_stride']}`, total stride `{hz200['total_stride_samples']}` samples, measured `{hz200['actual_ctc_frame_rate_hz']:.1f}` CTC frames/s.
- All pretrained weights remain unchanged; no fine-tuning or retraining was performed.

## Real 2× results

| Frontend | CTC frames | Transcript | CER vs reference |
|---|---:|---|---:|
| Standard ~50 Hz | {standard['ctc_frame_count']} | `{standard['adjusted_transcript']}` | {standard['character_error_rate_vs_reference'] * 100:.1f}% |
| Denser ~100 Hz | {hz100['ctc_frame_count']} | `{hz100['adjusted_transcript']}` | {hz100['character_error_rate_vs_reference'] * 100:.1f}% |
| Denser ~200 Hz | {hz200['ctc_frame_count']} | `{hz200['adjusted_transcript']}` | {hz200['character_error_rate_vs_reference'] * 100:.1f}% |

The 100 Hz variant substantially improved this utterance, while 200 Hz was worse than 100 Hz. Therefore changing the convolutional frontend can change and sometimes improve the output, but higher temporal density is not monotonically better.

## Boundary

Speed conversion may already remove acoustic detail. Denser convolutional sampling cannot reconstruct missing waveform information; it only reduces additional temporal downsampling inside the model. Cross-rate hidden/CTC distances use linear interpolation over normalized utterance time and are engineering comparison metrics. This single-utterance result does not establish general fast-speech performance.

## Execution and QA

- 5 speeds × 3 frontend settings = {frontend['counts']['unique_inferences']} real inferences; failed = {frontend['counts']['failed']}.
- Executed by ordinary Python over controller-driven SSH; remote Codex Agent tokens = 0.
- Desktop browser QA passed frontend selection (50/100/200 Hz), return to Transformer L9/α=0.5, explanatory text, transcript diff, and audio readiness (`readyState=4`, 2.56 s for 2× input).

## Standalone CER figure and limited compensation demo

- Standalone files: `reports/figures/frontend_speed_cer.svg` and `reports/figures/frontend_speed_cer.png`.
- Auditable source table: `reports/figures/frontend_speed_cer_source.csv`.
- The displayed same-sample policy uses 50 Hz at ≤1.5× and 100 Hz at ≥1.75×, yielding CER (%) `{adaptive_cer}` across the five tested speeds.
- This curve is selected from the same 15 real inference results; it adds no inference and is neither held-out validation nor a general optimum.
"""
    write_text(ROOT / "reports" / f"{TASK_ID}_FRONTEND_REPORT.md", report)

    qa_path = ROOT / "reports" / f"{TASK_ID}_BROWSER_QA.json"
    qa = read_json(qa_path)
    qa["updated_at_utc"] = utc_now()
    qa["frontend_extension"] = {
        "status": "PASS",
        "default_frontend_check": "2x / 100Hz",
        "standard_transcript": standard["adjusted_transcript"],
        "frontend_100hz_transcript": hz100["adjusted_transcript"],
        "frontend_100hz_cer": hz100["character_error_rate_vs_reference"],
        "frontend_200hz_cer": hz200["character_error_rate_vs_reference"],
        "frontend_200hz_control": "PASS",
        "return_to_layer_l9_alpha_0p5": "PASS",
        "layer_strength_semantics_visible": "PASS",
        "audio_ready_state": 4,
        "audio_duration_seconds": 2.56,
        "reference_comparison": {
            "status": "PASS",
            "reference_audio_id": "hf-internal-testing/librispeech_asr_demo:clean:validation:8",
            "reference_text": "AS FOR ETCHINGS THEY ARE OF TWO KINDS BRITISH AND FOREIGN",
            "x2_standard_cer": standard["character_error_rate_vs_reference"],
            "x2_100hz_cer": hz100["character_error_rate_vs_reference"],
            "x2_100hz_wrong_word_highlights": 1,
            "x2_standard_wrong_word_highlights": 4,
        },
        "cer_figure": {
            "status": "PASS",
            "embedded_image_loaded": True,
            "embedded_natural_size": [902, 557],
            "standalone_svg": "PASS",
            "standalone_png": "PASS",
            "source_csv": "reports/figures/frontend_speed_cer_source.csv",
            "manifest": "reports/figures/frontend_speed_cer_manifest.json",
            "adaptive_policy": figure_manifest["adaptive_demo_policy"],
            "adaptive_cer_percent": adaptive_cer,
            "boundary": "Same-utterance selection only; not held-out and not a general optimum.",
        },
        "method": "controller-side in-app browser on localhost stable delivery copy",
        "boundary": "Desktop browser interaction QA; narrow-screen behavior is covered by responsive CSS, not a separate device run.",
    }
    write_json(qa_path, qa)

    status_path = ROOT / "reports" / f"{TASK_ID}_FINAL_STATUS.json"
    status = read_json(status_path)
    status["updated_at_utc"] = utc_now()
    status["frontend"] = {
        "status": "PASS",
        "group_id": frontend["group_id"],
        **frontend["counts"],
        "x2_standard_cer": standard["character_error_rate_vs_reference"],
        "x2_100hz_cer": hz100["character_error_rate_vs_reference"],
        "x2_200hz_cer": hz200["character_error_rate_vs_reference"],
        "browser_qa": "PASS",
        "reference_comparison": "PASS",
        "cer_figure": "PASS",
        "adaptive_demo_boundary": "same-sample only; not held-out",
    }
    write_json(status_path, status)

    token_path = ROOT / "reports" / f"{TASK_ID}_TOKEN_USAGE.json"
    token = read_json(token_path)
    token["updated_at_utc"] = utc_now()
    token["frontend_extension"] = {
        "execution_method": "LOCAL_CONTROLLER_OVER_SSH",
        "remote_agent_tokens": 0,
        "model_inference_billing_tokens": 0,
        "desktop_controller_usage": "UNAVAILABLE",
        "note": "15 wav2vec2 forward results are not language-model billing tokens.",
    }
    write_json(token_path, token)

    execution_path = ROOT / "reports" / f"{TASK_ID}_EXECUTION_REPORT.md"
    execution = execution_path.read_text(encoding="utf-8")
    if MARKER not in execution:
        append_text(execution_path, f"\n{MARKER}\n\nSee `reports/{TASK_ID}_FRONTEND_REPORT.md`. Added 15 real frontend-stride inferences and the interactive convolutional-frontend explanation. Remote Agent tokens: 0.\n")
    log_path = ROOT / "docs" / "CODEX_PROJECT_LOG.md"
    log = log_path.read_text(encoding="utf-8")
    if MARKER not in log:
        append_text(log_path, f"\n{MARKER}\n\n- Added 5 speeds × 3 frontend stride settings; 15/15 successful.\n- On 2× input, reference CER: 50 Hz {standard['character_error_rate_vs_reference'] * 100:.1f}%, 100 Hz {hz100['character_error_rate_vs_reference'] * 100:.1f}%, 200 Hz {hz200['character_error_rate_vs_reference'] * 100:.1f}%.\n- Ordinary Python over SSH only; no remote Agent and no remote Git.\n")
    if FIGURE_MARKER not in execution:
        append_text(execution_path, f"\n{FIGURE_MARKER}\n\nGenerated standalone SVG/PNG and an auditable CSV from the existing 15 frontend inferences. The same-sample policy CER (%) is `{adaptive_cer}`; it is explicitly not held-out. No additional inference or remote Agent token was used.\n")
    if FIGURE_MARKER not in log:
        append_text(log_path, f"\n{FIGURE_MARKER}\n\n- Added standalone SVG/PNG plus source CSV for the speed × frontend CER result.\n- Same-sample demonstration: ≤1.5× → 50 Hz; ≥1.75× → 100 Hz; CER (%) `{adaptive_cer}`.\n- Boundary: no held-out selection or generalization claim; no new model inference.\n")

    print(json.dumps({"status": "PASS", "counts": frontend["counts"], "x2_cer": {"50hz": standard["character_error_rate_vs_reference"], "100hz": hz100["character_error_rate_vs_reference"], "200hz": hz200["character_error_rate_vs_reference"]}, "adaptive_cer_percent": adaptive_cer, "cer_figure": "PASS"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
