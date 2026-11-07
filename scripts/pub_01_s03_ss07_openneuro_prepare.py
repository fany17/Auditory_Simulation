"""One predeclared shared amplitude-calibrated preparation, 2203 only; no NN."""
import argparse,concurrent.futures as futures,csv,hashlib,json,os,resource,shutil,threading,time
from pathlib import Path
import numpy as np
import pub_01_s03_ss07_openneuro_adapter as legacy
import pub_01_s03_ss07_openneuro_adapter_v2 as adapter
import pub_01_s03_ss07_openneuro_metrics as metric
import pub_01_s03_ss07_openneuro_stream as stream
from pub_01_s03_ss07_openneuro_transport import ROOT

BASE=ROOT.parent
WRITE_LOCK=threading.Lock()
START=time.time()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def guard(full=False):
    assert time.time()-START<4*3600,'4h stage cap'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3,'8GiB process RSS cap'
    assert shutil.disk_usage(ROOT).free>=20*1024**3,'20GiB reserve'
    if full:
        size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
        arrays=sum(p.stat().st_size for directory in list(BASE.glob('s03_*_20261101'))+[ROOT] for p in directory.rglob('*.npy'))
        assert size<=3*1024**3 and arrays<=28*1024**3
        return dict(root_bytes=size,all_SS07_NPY_bytes=arrays,free_bytes=shutil.disk_usage(ROOT).free)


def freeze():
    guard(True)
    assert not (ROOT/'PREPROCESSING_PROTOCOL.json').exists()
    assert read(ROOT/'HUMAN_AUTHORIZATION.json')['status']=='AUTHORIZED_BY_HUMAN_DATA_SCOPE'
    decision=read(ROOT/'COORDINATOR_GAIN_CALIBRATION_DECISION_20261102.json')
    check=read(ROOT/'ADAPTER_V2_CALIBRATION_CHECK_V2.json')
    assert check['passed'] and check['raw_constant_DC_before_filter_rejected']
    assert check['source_sha256']==sha(adapter.__file__)
    stats=read(ROOT/'STATISTICS_SYNTHETIC_CHECK_V2.json')
    assert stats['passed'] and stats['paired_block_identity_mismatch_rejected'] and stats['source_sha256']==sha(metric.__file__)
    layout=read(ROOT/'PROVISIONAL_DATA_MASK.json')
    assert layout['prediction_independent'] and layout['records']==380 and layout['windows']==45429
    assert read(ROOT/'AUDIO_PREPARE_COMPLETED.json')['passed']
    pins=[Path(__file__),Path(adapter.__file__),Path(legacy.__file__),Path(metric.__file__),Path(stream.__file__),
        ROOT/'input/pub_01_s03_ss07_openneuro_transport.py',ROOT/'PLAN.md',ROOT/'AMENDMENT_GAIN_CALIBRATION.md',
        ROOT/'COORDINATOR_GAIN_CALIBRATION_DECISION_20261102.json',ROOT/'HUMAN_AUTHORIZATION.json',
        ROOT/'ADAPTER_V2_CALIBRATION_CHECK_V2.json',ROOT/'STATISTICS_SYNTHETIC_CHECK_V2.json',
        ROOT/'AUDIO_PREPARE_COMPLETED.json',ROOT/'HEADERS_AND_SIZES_V2.json',ROOT/'ANNEX_CATALOG_V2.json',
        ROOT/'PROVISIONAL_DATA_MASK.json',adapter.NORM]
    _,_,a=adapter.source_normalization()
    dump('PREPROCESSING_PROTOCOL.json',dict(frozen_before_bulk_preprocessing=True,
        source_pins={str(p):sha(p) for p in pins},records=380,network_threads=4,BLAS_threads=2,
        stage_cap_seconds=14400,per_object_cap_seconds=1800,raw_body_persistence='Only first existing canary',
        chunk_bytes=2*1024**2,source_amplitude=a,scale='Single full-common-record label-free scalar gain',
        physical_unit_status='UNVERIFIED',offline_transductive=True,NN_forwards=0,
        final_prediction_freeze_still_required=True,resource=guard(True)))
    print('PREPROCESSING_FROZEN_NO_NN',flush=True)


def run():
    protocol=read(ROOT/'PREPROCESSING_PROTOCOL.json')
    for path,digest in protocol['source_pins'].items():assert sha(path)==digest,path
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ[key]=='2'
    assert not (ROOT/'EEG_PREPARE_STARTED.json').exists()
    (ROOT/'processed_EEG').mkdir(exist_ok=True)
    dump('EEG_PREPARE_STARTED.json',dict(pid=os.getpid(),time=START,protocol_sha256=sha(ROOT/'PREPROCESSING_PROTOCOL.json'),NN_forwards=0))
    catalog=read(ROOT/'ANNEX_CATALOG_V2.json')['objects']
    objects=sorted([r for r in catalog if r['path'].endswith('.eeg')],key=lambda r:r['path'])
    headers=read(ROOT/'HEADERS_AND_SIZES_V2.json')
    hm={r['path']:r for r in headers['headers']}
    cm={r['path']:r['git_blob_sha'] for r in headers['channel_table_mapping']}
    layouts={r['record']:r for r in read(ROOT/'PROVISIONAL_DATA_MASK.json')['rows']}
    targets={f'{i:02d}':np.load(ROOT/'targets'/f'audio{i:02d}.npy',allow_pickle=False) for i in range(1,21)}
    operators={}
    for digest in set(cm.values()):
        rows=list(csv.DictReader((ROOT/'metadata/channel_tables'/f'{digest}.tsv').open(),delimiter='\t'))
        assert tuple(r['name'] for r in rows)==adapter.SOURCE_NAMES
        assert all(r['status'] in ('good','bad') for r in rows)
        good=[r['name'] for r in rows if r['status']=='good']
        matrix,details=adapter.geometry(good);operators[digest]=(good,matrix,details)
    assert len(operators)==119
    dump('SPATIAL_OPERATORS_SUMMARY.json',{key:values[2] for key,values in operators.items()})
    def one(row):
        guard();start=time.time()
        base=row['path'][:-len('_eeg.eeg')]
        digest=cm[base+'_channels.tsv'];good,matrix,details=operators[digest]
        head=hm[row['path'][:-4]+'.vhdr']
        record=Path(row['path']).stem;run=base.split('_run-')[-1];target=targets[run]
        payload=stream.obtain_memory(row,START)
        x,calibration=adapter.prepare_record(payload,head,good,matrix,len(target),protocol['source_amplitude'])
        del payload
        expected=layouts[record];actual=metric.mechanical_mask(target[:x.shape[1]])
        assert all(actual[k]==expected[k] for k in ['length','windows','covered','blocks','tail'])
        out=ROOT/'processed_EEG'/f'{record}.npy'
        assert x.shape==(64,expected['length']) and x.flags.c_contiguous and np.isfinite(x).all()
        with WRITE_LOCK:
            guard()
            with out.open('xb') as f:np.save(f,x,allow_pickle=False)
        result=dict(path=row['path'],record=record,subject=base.split('/')[0],run=run,
            source_sha256=row['payload_sha256'],source_bytes=row['declared_bytes'],
            processed_path=str(out),processed_bytes=out.stat().st_size,processed_sha256=sha(out),
            channel_table_git_blob=digest,good_channels=len(good),calibration=calibration,
            processed_channel_std_min_median_max=[float(x.std(1,dtype='f8').min()),float(np.median(x.std(1,dtype='f8'))),float(x.std(1,dtype='f8').max())],
            wall_seconds=time.time()-start,mask=actual)
        return result
    successes=[];failures=[]
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(one,row):row for row in objects}
        for completed,future in enumerate(futures.as_completed(pending),1):
            row=pending[future]
            try:
                value=future.result();successes.append(value)
                with (ROOT/'EEG_PREPARE_RECORDS.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
                print(json.dumps(dict(stage='EEG_PREPARED',done=completed,total=380,record=value['record'],failures=len(failures),wall_seconds=time.time()-START)),flush=True)
            except Exception as error:
                failure=dict(path=row['path'],error=repr(error));failures.append(failure)
                with (ROOT/'EEG_PREPARE_RECORD_FAILURES.jsonl').open('a') as f:f.write(json.dumps(failure)+'\n')
                print(json.dumps(dict(stage='RECORD_FAILURE',done=completed,**failure)),flush=True)
            if completed%50==0:guard(True)
    result=dict(passed=not failures and len(successes)==380,records=len(successes),failures=failures,
        successful_record_ids=sorted(r['record'] for r in successes),eligible_blocks=sum(sum(b['eligible'] for b in r['mask']['blocks']) for r in successes),
        windows=sum(r['mask']['windows'] for r in successes),wall_seconds=time.time()-START,
        CPU_seconds=time.process_time(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource=guard(True),NN_forwards=0,new_fits=0,physical_unit_status='UNVERIFIED',offline_gain_calibrated=True,
        next='Final model/source/actual mask/statistics freeze and coordinator review before any prediction')
    dump('EEG_PREPARE_RESULT.json',result)
    if result['passed']:dump('EEG_PREPARE_COMPLETED.json',result)
    else:raise RuntimeError('Preprocessing has recorded failures; inspect affected objects without changing scientific transform')
    print(json.dumps(dict(stage='COMPLETED',**result)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','run']);args=parser.parse_args()
    try:freeze() if args.mode=='freeze' else run()
    except BaseException:
        import traceback
        name='PREPROCESSING_FREEZE_FAILURE.json' if args.mode=='freeze' else 'EEG_PREPARE_FAILURE.json'
        if not (ROOT/name).exists():dump(name,dict(time=time.time(),traceback=traceback.format_exc()))
        raise
