"""Read-only remaining 96 ds004703 formats; never execute source/bytecode."""
import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path
import zipfile
from xml.etree import ElementTree
import numpy as np
import nibabel as nib

ROOT=Path('/home/fanyu/auditory_simulation_m6a')


def inspect(path):
    name=path.name
    if name.endswith('.nii.gz') or name.endswith('.nii'):
        obj=nib.load(str(path)); shape=obj.shape
        if not shape or any(n<=0 for n in shape):raise ValueError('Invalid NIfTI shape')
        nonfinite=0; count=0
        # Read each first-axis slab through the decoder; no reconstruction/viewing.
        for i in range(shape[0]):
            block=np.asanyarray(obj.dataobj[i:i+1,...])
            nonfinite+=int((~np.isfinite(block)).sum());count+=block.size
        return dict(format='NIFTI',status='PASS_FORMAT' if nonfinite==0 else 'HOLD_NONFINITE',
                    shape=json.dumps(shape),samples=count,nonfinite=nonfinite,
                    scope='All voxel values decoded; no reconstruction, visualization or identity inference; not benchmark input')
    if path.suffix in ['.csv','.tsv']:
        with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.reader(f,delimiter='\t' if path.suffix=='.tsv' else ','))
        widths=Counter(map(len,rows))
        return dict(format=path.suffix[1:].upper(),status='PASS_FORMAT',rows=len(rows),width_counts=json.dumps(dict(widths)),scope='Decoded table; semantic/rectangular requirements not assumed')
    if path.suffix in ['.json','.ipynb']:
        obj=json.loads(path.read_text(encoding='utf-8-sig'))
        return dict(format='NOTEBOOK_JSON' if path.suffix=='.ipynb' else 'JSON',status='PASS_FORMAT',scope='Parsed JSON; notebook cells NOT executed',top_level=type(obj).__name__)
    if path.suffix=='.docx':
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                payload=z.read(info)
                if info.filename.endswith('.xml'):ElementTree.fromstring(payload)
        return dict(format='DOCX',status='PASS_FORMAT',scope='All archive entries decoded, XML parsed')
    if path.suffix=='.pyc':
        data=path.read_bytes()
        return dict(format='LEGACY_BYTECODE',status='EXCLUDED_NOT_EXECUTED',bytes_read=len(data),
                    scope='Opaque bytes read only; no marshal/unpickle/import/execution; not evidence of runtime compatibility')
    text=path.read_text(encoding='utf-8-sig')
    if path.suffix=='.py':
        try:ast.parse(text);parse='CURRENT_PYTHON_AST_PASS'
        except SyntaxError as exc:parse='LEGACY_OR_INVALID_SYNTAX:'+str(exc)
        return dict(format='PYTHON_SOURCE',status='PASS_TEXT_ONLY',static_parse=parse,scope='Never imported/executed; current runtime compatibility not asserted')
    return dict(format='TEXT',status='PASS_TEXT_ONLY',characters=len(text),scope='UTF-8 decode only; not code execution or semantic validation')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output).resolve()
    if ROOT not in out.parents:raise ValueError('Server only')
    with (ROOT/'pub_01/review/ds_inventory_20260919_v1/inventory_reconciliation.csv').open() as f:
        inventory=list(csv.DictReader(f))
    pending=[r for r in inventory if r['category'] not in ['audio','neural']]
    assert len(pending)==96
    results=[]
    for item in pending:
        path=ROOT/'data/ds004703/v1.1.0'/item['path']
        row=dict(relative_path=item['path'],category=item['category'],expected_bytes=item['expected_bytes'])
        try:
            if path.stat().st_size!=int(item['expected_bytes']):raise ValueError('Size mismatch')
            row.update(inspect(path))
        except Exception as exc:row.update(status='HOLD',error=type(exc).__name__+': '+str(exc))
        results.append(row)
    out.mkdir(parents=True,exist_ok=False)
    fields=list(dict.fromkeys(k for r in results for k in r))
    with (out/'remaining_format_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
    summary=dict(expected=96,checked=len(results),formats=dict(Counter(r.get('format','FAILED') for r in results)),
                 statuses=dict(Counter(r['status'] for r in results)),signal_files_previously_checked=281,
                 full_inventory=377,scope='Remaining-format check complements 281 all-sample EDF/WAV QC; anatomy not used in benchmark')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
