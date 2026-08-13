from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe one small model config body without response metadata.")
    parser.add_argument("url")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    parsed = urllib.parse.urlparse(args.url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("probe URL must use HTTPS")
    request = urllib.request.Request(
        args.url, headers={"User-Agent": "Auditory-Simulation-M6A/1"}
    )
    report: dict[str, object] = {
        "label": args.label,
        "mirror_domain": parsed.hostname,
        "model_id": "facebook/wav2vec2-base",
        "revision_label": "main",
        "revision_limitation": "MUTABLE_MAIN_LABEL_NON_CRYPTOGRAPHIC_REPRODUCIBILITY_ONLY",
        "integrity_policy": "NON_HASH_AUDIT",
    }
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"unexpected HTTP status: {response.status}")
            body = response.read(1024 * 1024)
        payload = json.loads(body)
        if not isinstance(payload, dict) or payload.get("model_type") != "wav2vec2":
            raise ValueError("config body is not facebook/wav2vec2-base semantics")
        report.update(
            {
                "status": "PASS",
                "http_status": 200,
                "body_bytes": len(body),
                "json_readable": True,
                "model_type": "wav2vec2",
                "architectures": payload.get("architectures"),
                "hidden_size": payload.get("hidden_size"),
                "num_hidden_layers": payload.get("num_hidden_layers"),
                "conv_dim": payload.get("conv_dim"),
                "conv_kernel": payload.get("conv_kernel"),
                "conv_stride": payload.get("conv_stride"),
            }
        )
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        report.update(
            {
                "status": "FAIL",
                "json_readable": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
