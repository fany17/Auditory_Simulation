"""Allowlisted light run02 delivery; never export tensors/signals/checkpoints."""
import json
import shutil
from pub_01_s03_ss06_matched import ROOT,read,dump,sha


def main():
    assert read(ROOT/'review01/REVIEW.json')['technical']=='PASS'
    out=ROOT/'delivery01';out.mkdir(exist_ok=False)
    names=['INTAKE.json','ENGINEERING_CHECK.json','FORMAL_PROTOCOL.json','P_NATIVE_AMENDMENT.json',
        'P_NATIVE_AMENDMENT_02.json','SOURCE_COMMIT.json','P_NATIVE_SOURCE_COMMIT.json',
        'ALL_TRAINING_REVIEW.json','ASSESSMENT_OPENED.json','ACOUSTIC_SUPPORT.json','OBSERVER_SOURCE_COMMIT.json','audit01/completed.json',
        'audit01/RESULTS.csv','audit01/per_record.csv','audit01/per_subject.csv',
        'review01/REVIEW.json','review01/SUMMARY.csv','review01/training_ledger.csv','review01/primary_contrasts.csv']
    names += [str(p.relative_to(ROOT)) for p in (ROOT/'audit01').glob('*_blocks.csv')]
    names += [str(p.relative_to(ROOT)) for p in (ROOT/'figures').glob('*.png')]
    for name in names:
        source=ROOT/name;assert source.suffix in ('.json','.csv','.png')
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    for folder in (ROOT/'formal01').iterdir():
        if not folder.is_dir():continue
        d=read(folder/'completed.json');sequence=d.pop('actual_sequence',None);indices=d.pop('actual_window_index_sequence',None)
        if sequence is not None:
            d['actual_visitation_summary']={'rows':len(sequence),'unique_records':len({r['record'] for r in sequence}),
                'repeat_anchor_windows':sum(r.get('repeat_anchor',False) for r in sequence),
                'first':sequence[0],'last':sequence[-1]}
        if indices is not None:d['actual_window_index_summary']={'count':len(indices),'unique':len(set(indices)),
            'min':min(indices),'max':max(indices),'sequential':indices==list(range(len(indices)))}
        d['canonical_full_metadata_path']=str(folder/'completed.json');d['canonical_full_metadata_sha256']=sha(folder/'completed.json')
        dest=out/'formal_compact'/folder.name;dest.mkdir(parents=True);dump(dest/'completed.json',d)
        shutil.copyfile(folder/'history.json',dest/'history.json')
    files=list(out.rglob('*'));bytes_=sum(p.stat().st_size for p in files if p.is_file())
    dump(out/'DELIVERY.json',{'canonical_server_root':str(ROOT),'light_bytes_before_manifest':bytes_,
        'contains_only':'aggregate CSV/JSON, stage history/compact visitation metadata, static aggregate plots',
        'tensor_signal_checkpoint_audio_or_normalizer_files':False,'no_raw_prediction_arrays':True,
        'full_visitation_sequences_retained_on_server':True,'files':[str(p.relative_to(out)) for p in files if p.is_file()]})
    print(json.dumps({'delivery':str(out),'bytes':bytes_}),flush=True)

if __name__=='__main__':main()
