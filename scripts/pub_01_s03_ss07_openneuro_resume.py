"""Transport-only handoff for our own preprocessing job; retain all evidence."""
import argparse,csv,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
from pub_01_s03_ss07_openneuro_transport import ROOT
import pub_01_s03_ss07_openneuro_adapter_v2 as a


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def bridge():
    assert os.environ['OPENBLAS_NUM_THREADS']==os.environ['OMP_NUM_THREADS']==os.environ['MKL_NUM_THREADS']=='1'
    records=[json.loads(line) for line in (ROOT/'EEG_PREPARE_RECORDS.jsonl').read_text().splitlines() if line]
    reference=next(r for r in records if r['record']=='sub-001_task-listening_run-01_eeg')
    assert sha(reference['processed_path'])==reference['processed_sha256']
    data=json.loads((ROOT/'HEADERS_AND_SIZES_V2.json').read_text());path=reference['path']
    head=next(r for r in data['headers'] if r['path']==path[:-4]+'.vhdr')
    table=ROOT/'metadata/channel_tables'/f"{reference['channel_table_git_blob']}.tsv"
    good=[r['name'] for r in csv.DictReader(table.open(),delimiter='\t') if r['status']=='good']
    matrix,_=a.geometry(good);_,_,amplitude=a.source_normalization()
    target=np.load(ROOT/'targets/audio01.npy');payload=(ROOT/'canary_raw'/Path(path).name).read_bytes()
    x,cal=a.prepare_record(payload,head,good,matrix,len(target),amplitude)
    old=np.load(reference['processed_path']);difference=float(abs(x-old).max())
    assert difference<=1e-6
    dump('PREPROCESSING_BLAS1_BRIDGE.json',dict(passed=True,source_sha256=sha(a.__file__),
        first_record=reference['record'],FP32_output_max_abs=difference,exact_array_equal=bool(np.array_equal(x,old)),
        calibration=cal,NN_forwards=0,reference_sha256=reference['processed_sha256']))
    print('BLAS1_BRIDGE',difference,flush=True)


def handoff():
    assert json.loads((ROOT/'PREPROCESSING_BLAS1_BRIDGE.json').read_text())['passed']
    protocol=json.loads((ROOT/'PREPROCESSING_PROTOCOL.json').read_text())
    for path,digest in protocol['source_pins'].items():assert sha(path)==digest,path
    metadata=json.loads((ROOT/'HEADERS_AND_SIZES_V2.json').read_text())
    table_hashes={row['git_blob_sha'] for row in metadata['channel_table_mapping']}
    assert len(table_hashes)==119
    for digest in table_hashes:
        data=(ROOT/'metadata/channel_tables'/f'{digest}.tsv').read_bytes()
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==digest
    audio=json.loads((ROOT/'AUDIO_PREPARE_COMPLETED.json').read_text())['audios']
    assert len(audio)==20
    for row in audio:
        p=ROOT/'targets'/Path(row['path']).with_suffix('.npy').name
        assert sha(p)==row['target_sha256']
    dump('PREPROCESSING_REUSE_INPUT_CHECK.json',dict(passed=True,old_protocol_sha256=sha(ROOT/'PREPROCESSING_PROTOCOL.json'),
        original_source_pins_verified=len(protocol['source_pins']),channel_TSV_Git_blobs_verified=119,target_NPY_original_SHA_verified=20,NN_forwards=0))
    launch=json.loads((ROOT/'EEG_PREPARE_LAUNCH.json').read_text());pid=launch['pid']
    assert pid==1379762 and launch['UID']==os.getuid()==1012 and launch['start_ticks']=='753013869'
    proc=Path(f'/proc/{pid}')
    assert proc.exists()
    assert (proc/'stat').read_text().split()[21]==launch['start_ticks']
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert command==launch['argv'] and 'pub_01_s03_ss07_openneuro_prepare.py run' in command
    assert proc.stat().st_uid==os.getuid()
    children=[]
    for thread in (proc/'task').iterdir():
        p=thread/'children'
        if p.exists():children.extend(p.read_text().split())
    assert not children
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':break
        time.sleep(.2)
    assert not proc.exists() or (proc/'stat').read_text().split()[2]=='Z'
    lines=(ROOT/'EEG_PREPARE_RECORDS.jsonl').read_text().splitlines();records=[];last_truncated=None
    for index,line in enumerate(lines):
        try:records.append(json.loads(line))
        except json.JSONDecodeError:
            assert index==len(lines)-1,'Only interrupted trailing JSONL line may be incomplete'
            last_truncated=index
    assert len({r['record'] for r in records})==len(records)
    valid=[]
    catalog={r['path']:r for r in json.loads((ROOT/'ANNEX_CATALOG_V2.json').read_text())['objects']}
    layouts={r['record']:r for r in json.loads((ROOT/'PROVISIONAL_DATA_MASK.json').read_text())['rows']}
    for row in records:
        p=Path(row['processed_path']);assert p.resolve().is_relative_to((ROOT/'processed_EEG').resolve())
        assert p.stat().st_size==row['processed_bytes'] and sha(p)==row['processed_sha256']
        x=np.load(p,allow_pickle=False)
        assert x.shape==(64,row['mask']['length']) and x.dtype==np.float32 and x.flags.c_contiguous and np.isfinite(x).all()
        assert row['source_sha256']==catalog[row['path']]['payload_sha256']
        assert row['channel_table_git_blob'] in table_hashes
        expected=layouts[row['record']]
        assert all(row['mask'][key]==expected[key] for key in ['length','windows','covered','blocks','tail'])
        assert row['calibration']['A_source']==protocol['source_amplitude']
        valid.append(row)
    recognized={Path(row['processed_path']).resolve() for row in valid}
    backup=ROOT/'preprocessing_transport_handoff_backup';backup.mkdir(exist_ok=False)
    moved=[]
    for p in (ROOT/'processed_EEG').glob('*.npy'):
        resolved=p.resolve();assert resolved.is_relative_to((ROOT/'processed_EEG').resolve())
        if resolved not in recognized:
            target=backup/p.name;assert target.resolve().is_relative_to(ROOT.resolve()) and not target.exists()
            p.rename(target);moved.append(dict(source=str(p),backup=str(target),bytes=target.stat().st_size))
    dump('PREPROCESSING_TRANSPORT_HANDOFF.json',dict(time=time.time(),old_pid=pid,old_start_ticks=launch['start_ticks'],
        termination='Only verified own v1 transport job; all source/log/protocol and verified outputs retained',
        reason='Measured2MiB9.360s/8MiB40.028s, four streams forecast exceeds4h; switch to8streams/BLAS1, same scientific adapter',
        old_processed_verified=len(valid),old_record_rows=valid,truncated_final_JSONL_line=last_truncated,
        unregistered_outputs_moved_to_backup=moved,no_deletion=True,new_NN=0))
    print('HANDOFF_VERIFIED_REUSABLE',len(valid),'BACKUP',len(moved),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['bridge','handoff']);args=p.parse_args()
    bridge() if args.mode=='bridge' else handoff()
