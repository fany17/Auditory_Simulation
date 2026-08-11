from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


S3_ENDPOINT = "https://s3.amazonaws.com/openneuro.org"
XML_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def request_page(prefix: str, continuation: str | None = None) -> bytes:
    query = {"list-type": "2", "prefix": prefix}
    if continuation:
        query["continuation-token"] = continuation
    url = S3_ENDPOINT + "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(url, headers={"User-Agent": "M6A-PUBLIC-non-hash-inventory/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def iter_objects(prefix: str) -> Iterator[dict[str, object]]:
    continuation: str | None = None
    while True:
        root = ET.fromstring(request_page(prefix, continuation))
        for item in root.findall("s3:Contents", XML_NAMESPACE):
            key = item.findtext("s3:Key", default="", namespaces=XML_NAMESPACE)
            if not key or key.endswith("/"):
                continue
            yield {
                "key": key,
                "bytes": int(item.findtext("s3:Size", default="0", namespaces=XML_NAMESPACE)),
                "modified_at_utc": item.findtext(
                    "s3:LastModified", default="", namespaces=XML_NAMESPACE
                ),
            }
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=XML_NAMESPACE)
        if truncated.lower() != "true":
            return
        continuation = root.findtext("s3:NextContinuationToken", default="", namespaces=XML_NAMESPACE)
        if not continuation:
            raise RuntimeError("S3 listing is truncated but has no continuation token")


def direct_url(key: str) -> str:
    return f"{S3_ENDPOINT}/{urllib.parse.quote(key, safe='/')}"


def main() -> int:
    parser = argparse.ArgumentParser(description="List public OpenNeuro S3 objects without hashes.")
    parser.add_argument("dataset_id")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    prefix = args.dataset_id.rstrip("/") + "/"
    objects = sorted(iter_objects(prefix), key=lambda item: str(item["key"]))
    rows = [
        {
            "path": str(item["key"])[len(prefix) :],
            "bytes": item["bytes"],
            "modified_at_utc": item["modified_at_utc"],
            "source_url": direct_url(str(item["key"])),
        }
        for item in objects
    ]

    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    with args.inventory.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "modified_at_utc", "source_url"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_id": args.dataset_id,
        "listed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": S3_ENDPOINT,
        "object_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "integrity_policy": "NON_HASH_AUDIT",
        "evidence_fields": ["path", "bytes", "modified_at_utc"],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
