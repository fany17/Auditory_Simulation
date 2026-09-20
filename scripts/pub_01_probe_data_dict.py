"""One independent public re-fetch for a reproducibly unreadable data_dict.

Preserve both copies; directly compare byte streams without hashes, then decode
using the same fail-closed numeric loader. This does not overwrite canonical raw.
"""
import csv
import json
from pathlib import Path
from pub_01_s06_download import download
from pub_01_s06_format_qc import inspect, ROOT


def main():
    inventory = ROOT / 'pub_01/s01/audit_20260919_v1/sparrkulee_official_inventory.csv'
    with inventory.open() as f:
        row = next(r for r in csv.DictReader(f) if r['relative_path'] == 'derivatives/preprocessed_stimuli/podcast_35-1.data_dict')
    out = ROOT / 'pub_01/s06/podcast_35_1_independent_refetch_20260919'
    out.mkdir(exist_ok=False)
    result = download(row, out)
    original = ROOT / 'pub_01/s06/sparrkulee_v3.1/raw' / row['relative_path']
    second = out / row['relative_path']
    if result['status'] != 'DOWNLOADED_SIZE_MATCH':
        raise RuntimeError(str(result))
    identical = True
    with original.open('rb') as a, second.open('rb') as b:
        while x := a.read(1048576):
            if x != b.read(len(x)):
                identical = False
                break
        if b.read(1):
            identical = False
    failures = {}
    for label, path in [('original', original), ('independent_refetch', second)]:
        try:
            inspect(path, out)
            failures[label] = 'READABLE'
        except Exception as exc:
            failures[label] = type(exc).__name__ + ': ' + str(exc)
    report = dict(source_url=row['source_url'], official_bytes=int(row['bytes']),
                  original_server_path=str(original), second_server_path=str(second),
                  direct_byte_streams_identical=identical, decoding=failures,
                  status='HOLD' if any(x != 'READABLE' for x in failures.values()) else 'PASS',
                  inference='If identical and both truncated, current served object is unreadable; do not blame a local interrupted transfer',
                  originals_preserved=True, hash_audit=False)
    (out / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
