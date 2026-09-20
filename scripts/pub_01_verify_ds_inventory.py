"""Independent live source/local file-size reconciliation on server2203.

No signal files are modified and no checksum fields are inspected or exported.
This is download evidence only, not format or scientific validation.
"""
import argparse
import csv
import json
from pathlib import Path
import platform
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path('/home/fanyu/auditory_simulation_m6a')
ENDPOINT = 'https://s3.amazonaws.com/openneuro.org'
NS = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux' or not ROOT.is_dir():
        raise RuntimeError('Run only in the server project')
    output = Path(args.output).resolve()
    if ROOT not in output.parents:
        raise ValueError('Output outside server project')
    output.mkdir(parents=True, exist_ok=False)
    base = ROOT / 'data/ds004703/v1.1.0'
    rows = []
    continuation = None
    while True:
        params = {'list-type': '2', 'prefix': 'ds004703/'}
        if continuation:
            params['continuation-token'] = continuation
        request = urllib.request.Request(ENDPOINT + '?' + urllib.parse.urlencode(params))
        with urllib.request.urlopen(request, timeout=90) as response:
            document = ET.fromstring(response.read())
        for item in document.findall('s3:Contents', NS):
            key = item.findtext('s3:Key', namespaces=NS)
            if not key or key.endswith('/'):
                continue
            relative = key.removeprefix('ds004703/')
            path = (base / relative).resolve()
            if base not in path.parents:
                raise ValueError('Unexpected source path')
            expected = int(item.findtext('s3:Size', namespaces=NS))
            actual = path.stat().st_size if path.is_file() else None
            category = ('anatomy' if '/anat/' in relative else
                        'audio' if relative.endswith('.wav') else
                        'neural' if relative.endswith('.edf') else 'metadata_or_other')
            rows.append(dict(path=relative, category=category, expected_bytes=expected,
                             actual_bytes=actual, status='MISSING' if actual is None else
                             'SIZE_MATCH' if actual == expected else 'SIZE_MISMATCH'))
        if document.findtext('s3:IsTruncated', namespaces=NS) != 'true':
            break
        continuation = document.findtext('s3:NextContinuationToken', namespaces=NS)
        if not continuation:
            raise RuntimeError('Missing pagination token')
    if not rows:
        raise RuntimeError('Empty source listing')
    with (output / 'inventory_reconciliation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(source=ENDPOINT + '/ds004703/',
                   scope='Live public S3 listing vs existing v1.1.0 directory; not immutable-version proof',
                   expected_files=len(rows), expected_bytes=sum(r['expected_bytes'] for r in rows),
                   status_counts={s: sum(r['status'] == s for r in rows)
                                  for s in ['SIZE_MATCH', 'SIZE_MISMATCH', 'MISSING']},
                   categories={c: {s: sum(r['category'] == c and r['status'] == s for r in rows)
                                    for s in ['SIZE_MATCH', 'SIZE_MISMATCH', 'MISSING']}
                               for c in ['anatomy', 'audio', 'neural', 'metadata_or_other']},
                   limitations=['Size equality is not content integrity or format readability.',
                                'Live listing may differ from the immutable dataset version.'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
