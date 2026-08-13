from __future__ import annotations

import math
from typing import Any


REQUIRED_COMPONENTS = (
    "preliminary_minimum_embargo_seconds",
    "maximum_encoding_lag_seconds",
    "filter_or_padding_edge_seconds",
    "audio_cross_split_context_overlap_seconds",
    "audio_resampling_edge_seconds",
)


def evaluate_final_embargo(components: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    pending: list[str] = []
    numeric: dict[str, float] = {}
    for key in REQUIRED_COMPONENTS:
        value = components.get(key)
        if value is None:
            pending.append(key)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{key} must be a number or null")
            continue
        number = float(value)
        if not math.isfinite(number) or number < 0:
            errors.append(f"{key} must be finite and non-negative")
            continue
        numeric[key] = number

    if errors:
        return {
            "status": "FAIL",
            "baseline_final": False,
            "final_embargo_seconds": None,
            "pending_components": pending,
            "errors": errors,
        }
    if pending:
        return {
            "status": "PENDING_MEASUREMENT",
            "baseline_final": False,
            "final_embargo_seconds": None,
            "pending_components": pending,
            "errors": [],
        }
    return {
        "status": "PASS",
        "baseline_final": True,
        "final_embargo_seconds": max(numeric.values()),
        "pending_components": [],
        "errors": [],
    }


def evaluate_final_embargo_candidate(components: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a measured embargo without accepting it as baseline-final.

    A complete candidate can be submitted for coordinator review, but the
    baseline remains non-final until that independent review is recorded.
    """

    evaluated = evaluate_final_embargo(components)
    if evaluated["status"] != "PASS":
        return evaluated
    return {
        "status": "FINAL_EMBARGO_CANDIDATE_READY",
        "baseline_final": False,
        "final_embargo_seconds": None,
        "final_embargo_candidate_seconds": evaluated["final_embargo_seconds"],
        "pending_components": [],
        "errors": [],
    }
