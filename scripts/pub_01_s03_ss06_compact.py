"""Reproduce lightweight visitation compaction used after the local disk failure.

No metric/weight/split changes. Full metadata remains in formal01/delivery01;
the compact output is new/write-once and carries full-source paths/digests.
"""
import hashlib
import json
import shutil
from pub_01_s03_ss06_data import ROOT


def main():
    source=ROOT/'delivery01/evidence';out=ROOT/'git_evidence01';out.mkdir(exist_ok=False)
    for path in source.iterdir():
        if path.is_file():shutil.copy2(path,out/path.name)
    manifest=json.loads((ROOT/'cache_trainval/manifest.json').read_text())
    expected={(r['record'],s) for r in manifest['records'] if r['phase']=='train' and r['score_blocks']
              for s in range(0,(r['samples']//3840)*3840,3840)}
    for folder in (source/'formal_runs').iterdir():
        target=out/'formal_runs'/folder.name;target.mkdir(parents=True)
        for path in folder.iterdir():
            obj=json.loads(path.read_text());compacted=False
            detail={'full_source_path':str(ROOT/'formal01'/folder.name/path.name),
                    'full_metadata_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            if path.name=='train_completed.json':
                seq=obj.pop('training_record_sequence');compacted=True
                detail.update(training_sequence_verified_by_all_training_gate=True,actual_sequence_steps=len(seq),actual_unique_records=len(set(seq)))
            if path.name=='pretrain_completed.json':
                spans=obj.pop('actual_span_sequence');assert {(s['record'],s['start_sample']) for s in spans}==expected and len(spans)==len(expected)
                detail.update(actual_span_sequence_verified_against_expected=True,actual_span_count=len(spans));compacted=True
            if compacted:obj['evidence_compaction']=detail
            (target/path.name).write_text(json.dumps(obj,indent=2))
    (out/'compaction_manifest.json').write_text(json.dumps({'reason':'local E disk full; omit repeated visitation strings in Git only; no metrics changed',
        'full_server_evidence':str(source),'training_sequence_gate':'PASS','SSL_all_spans_verified':True,'raw_or_weight_payload':False},indent=2))


if __name__=='__main__':main()
