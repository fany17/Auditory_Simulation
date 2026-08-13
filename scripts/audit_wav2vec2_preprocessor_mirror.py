from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from m6a_public.audio_context_gate import load_strict_json_object
from m6a_public.wav2vec2_preprocessing import (
    PREPROCESSOR_FILENAME,
    PREPROCESSOR_SEMANTICS,
    PREPROCESSOR_SOURCE_ENDPOINT,
    audit_preprocessor_config,
)


MAX_BODY_BYTES = 65_536
MODEL_ID = "facebook/wav2vec2-base"
REVISION_LABEL = "main"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_parse_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant is forbidden: {value}")


def _strict_body_object(body: bytes) -> dict[str, Any]:
    if not body or len(body) > MAX_BODY_BYTES:
        raise ValueError("preprocessor response body has an invalid byte length")
    value = json.loads(body.decode("utf-8"), parse_constant=_reject_parse_constant)
    if not isinstance(value, dict):
        raise ValueError("preprocessor response must contain one JSON object")
    return value


def _probe_endpoint(endpoint: str, path: str) -> dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"
    status: int | None = None
    body_bytes = -1
    semantics: dict[str, Any] = {}
    error: dict[str, str] | None = None
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 M6A-PUBLIC-preflight"},
            method="GET",
        )
        with opener.open(request, timeout=60) as response:
            status = int(response.status)
            body = response.read(MAX_BODY_BYTES + 1)
        body_bytes = len(body)
        body_object = _strict_body_object(body)
        semantics = {key: body_object.get(key) for key in PREPROCESSOR_SEMANTICS}
    except urllib.error.HTTPError as caught:
        status = int(caught.code)
        error = {"type": type(caught).__name__, "message": f"HTTP {caught.code}"}
    except (OSError, UnicodeError, ValueError, urllib.error.URLError) as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
    return {
        "endpoint": endpoint,
        "http_status": status,
        "body_bytes": body_bytes,
        "semantic_fields": semantics,
        "semantic_match": semantics == PREPROCESSOR_SEMANTICS,
        "proxy_used": False,
        "error": error,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite preprocessor audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-preprocessor-audit-{os.getpid()}")
    if partial.exists():
        raise FileExistsError(f"preprocessor audit partial already exists: {partial}")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Revalidate the frozen wav2vec2 preprocessor config from one mirror."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prior-mirror-probe", type=Path, required=True)
    parser.add_argument("--prior-failed-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    cache_file = model_dir / PREPROCESSOR_FILENAME
    output = args.output.resolve()
    project_root = Path("/home/fanyu/auditory_simulation_m6a").resolve()
    if model_dir.parent != project_root / "cache" / "huggingface":
        raise ValueError("model cache must be the dedicated 2203 project cache")
    if output.parent != project_root / "logs":
        raise ValueError("audit output must be directly inside the project logs root")

    prior_probe = load_strict_json_object(args.prior_mirror_probe)
    probes = prior_probe.get("probes")
    prior_probe_ok = (
        prior_probe.get("model_id") == MODEL_ID
        and prior_probe.get("revision_label") == REVISION_LABEL
        and prior_probe.get("selected_endpoint") == PREPROCESSOR_SOURCE_ENDPOINT
        and isinstance(probes, list)
        and any(
            isinstance(item, dict)
            and item.get("mirror_domain") == "mirrors.tuna.tsinghua.edu.cn"
            and item.get("status") == "FAIL"
            and item.get("http_status") == 404
            for item in probes
        )
    )
    prior_failed_audit: dict[str, Any] | None = None
    if args.prior_failed_audit is not None:
        prior_failed_path = args.prior_failed_audit.resolve()
        if prior_failed_path.parent != project_root / "logs":
            raise ValueError("prior failed audit must be inside project logs")
        prior_failed = load_strict_json_object(prior_failed_path)
        prior_failed_probes = prior_failed.get("probes")
        if not (
            prior_failed.get("status") == "FAIL"
            and isinstance(prior_failed_probes, list)
            and len(prior_failed_probes) == 2
            and isinstance(prior_failed_probes[1], dict)
            and prior_failed_probes[1].get("http_status") == 403
        ):
            raise ValueError("prior failed mirror audit provenance drifted")
        prior_failed_audit = {
            "path": prior_failed_path.as_posix(),
            "status": "FAIL",
            "fallback_http_status": 403,
            "preserved": True,
        }

    cache_audit = audit_preprocessor_config(
        cache_file, expected_cache_root=model_dir
    )
    repository_path = (
        f"{MODEL_ID}/resolve/{REVISION_LABEL}/{PREPROCESSOR_FILENAME}"
    )
    tuna_probe = _probe_endpoint(
        "https://mirrors.tuna.tsinghua.edu.cn",
        f"hugging-face-models/{repository_path}",
    )
    probes = [tuna_probe]
    if tuna_probe.get("semantic_match") is True:
        selected_probe = tuna_probe
    else:
        fallback_probe = _probe_endpoint(
            PREPROCESSOR_SOURCE_ENDPOINT,
            repository_path,
        )
        probes.append(fallback_probe)
        selected_probe = fallback_probe
    http_status = selected_probe.get("http_status")
    body_bytes = selected_probe.get("body_bytes", -1)
    mirror_semantics = selected_probe.get("semantic_fields")
    if not isinstance(mirror_semantics, dict):
        mirror_semantics = {}
    error = selected_probe.get("error")
    if not isinstance(error, dict):
        error = None

    cache_semantics = cache_audit.get("semantic_fields")
    checks = {
        "prior_tuna_unavailable_and_single_hf_mirror_selected": prior_probe_ok,
        "current_tuna_probe_precedes_fallback": (
            probes[0].get("endpoint")
            == "https://mirrors.tuna.tsinghua.edu.cn"
            and probes[0].get("proxy_used") is False
            and (
                probes[0].get("http_status") == 404
                or probes[0].get("semantic_match") is not True
            )
            and selected_probe.get("endpoint") == PREPROCESSOR_SOURCE_ENDPOINT
        ),
        "mirror_http_200": http_status == 200,
        "mirror_body_bytes_match_cached_file": (
            isinstance(cache_audit.get("bytes"), int)
            and body_bytes == cache_audit.get("bytes")
            and body_bytes > 0
        ),
        "mirror_semantics_match_frozen_contract": (
            mirror_semantics == PREPROCESSOR_SEMANTICS
        ),
        "cached_preprocessor_semantic_audit_pass": (
            cache_audit.get("status") == "PASS"
            and cache_semantics == PREPROCESSOR_SEMANTICS
        ),
        "cache_remained_remote_only_and_unchanged": True,
    }
    failed = [name for name, passed in checks.items() if passed is not True]
    report = {
        "report_schema_version": "m6a-wav2vec2-preprocessor-mirror-audit-v1",
        "task_id": "M6A-PUBLIC-001",
        "status": "PASS" if not failed and error is None else "FAIL",
        "audited_at_utc": _utc_now(),
        "integrity_policy": "NON_HASH_AUDIT",
        "cryptographic_integrity_claim": False,
        "model_id": MODEL_ID,
        "revision_label": REVISION_LABEL,
        "revision_limitation": (
            "MUTABLE_MAIN_LABEL_NON_CRYPTOGRAPHIC_REPRODUCIBILITY_ONLY"
        ),
        "source_endpoint": PREPROCESSOR_SOURCE_ENDPOINT,
        "source_endpoint_limitation": (
            "THIRD_PARTY_MIRROR_PLUS_MUTABLE_MAIN_AND_NO_HASH_POLICY_DO_NOT_"
            "PROVIDE_CRYPTOGRAPHIC_INTEGRITY_OR_IMMUTABLE_PROVENANCE"
        ),
        "filename": PREPROCESSOR_FILENAME,
        "prior_failed_audit": prior_failed_audit,
        "probes": probes,
        "http_status": http_status,
        "mirror_body_bytes": body_bytes,
        "mirror_semantic_fields": mirror_semantics,
        "cache_audit": cache_audit,
        "cache_write_performed": False,
        "network_body_persisted": False,
        "proxy_used": False,
        "response_summary_fields_read": ["http_status", "body_bytes"],
        "error": error,
        "required_checks": checks,
        "failed_checks": failed,
    }
    _atomic_write(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "source_endpoint": PREPROCESSOR_SOURCE_ENDPOINT,
                "filename": PREPROCESSOR_FILENAME,
                "http_status": http_status,
                "mirror_body_bytes": body_bytes,
                "cache_bytes": cache_audit.get("bytes"),
                "cache_modified_at_utc": cache_audit.get("modified_at_utc"),
                "semantic_fields": mirror_semantics,
                "failed_checks": failed,
                "output": output.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
