"""SS07 ds004408 pinned catalog + text header acquisition on2203; no model/audio/EEG body."""
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.request

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_openneuro_confirmation_20261102')
COMMIT='b2d91784a90c18e1d76bd77dcb2b29b0ce27a0be'
RAW='https://raw.githubusercontent.com/OpenNeuroDatasets/ds004408/'+COMMIT+'/'
S3='https://s3.amazonaws.com/openneuro.org/ds004408/'
START=time.time()
LOCK=threading.Lock()


def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def status(**value):
    p=ROOT/'HEADER_STATUS.writing'
    p.write_text(json.dumps(dict(time=time.time(),elapsed=time.time()-START,**value)))
    os.replace(p,ROOT/'HEADER_STATUS.json')
    print(json.dumps(value),flush=True)


def fetch(url,limit=2*1024**2):
    for attempt in range(1,4):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Auditory-Simulation-public-research/1.0'})
            with urllib.request.urlopen(req,timeout=40) as f:
                data=f.read(limit+1)
                assert len(data)<=limit,'Unexpected response size'
                return data
        except Exception as e:
            with LOCK:
                with (ROOT/'TRANSPORT_EVENTS.jsonl').open('a') as f:
                    f.write(json.dumps(dict(time=time.time(),url=url,attempt=attempt,error=repr(e)))+'\n')
            if attempt==3:raise
            time.sleep(.5*attempt)


def immutable(path,data):
    if path.exists():
        assert path.read_bytes()==data,'Existing immutable metadata differs'
    else:
        with path.open('xb') as f:f.write(data)


def pointer(row):
    data=fetch(RAW+row['path'],4096)
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert blob==row['sha']
    m=re.search(rb'SHA256E-s(\d+)--([a-f0-9]{64})(\.[a-zA-Z0-9]+)',data)
    assert m, 'Unsupported annex key; do not infer bytes'
    immutable(ROOT/'metadata/annex_pointers'/str(row['sha']+'.txt'),data)
    return dict(path=row['path'],git_blob_sha=row['sha'],declared_bytes=int(m[1]),
                payload_sha256=m[2].decode(),public_url=S3+row['path'])


def text_object(row):
    assert row['path'].endswith(('.vhdr','.vmrk')) and row['declared_bytes']<65536
    payload=fetch(row['public_url'],65536)
    assert len(payload)==row['declared_bytes']
    assert hashlib.sha256(payload).hexdigest()==row['payload_sha256']
    immutable(ROOT/'metadata/payload_headers'/Path(row['path']).name,payload)
    text=payload.decode('utf-8-sig')
    if row['path'].endswith('.vmrk'):
        return dict(path=row['path'],marker_count=len(re.findall(r'^Mk\d+=',text,re.M)),
                    bytes=len(payload),sha256=row['payload_sha256'])
    channels=[]
    for name,res,unit in re.findall(r'^Ch\d+=([^,]+),[^,]*,([^,]+),([^,\n]*)',text,re.M):
        channels.append(dict(name=name,resolution=float(res),unit=unit))
    def field(key):
        match=re.search(r'^'+re.escape(key)+r'=(.*)$',text,re.M)
        assert match,key
        return match[1].strip()
    return dict(path=row['path'],channels=channels,channel_count=int(field('NumberOfChannels')),
        sampling_interval_us=float(field('SamplingInterval')),
        data_format=field('DataFormat'),orientation=field('DataOrientation'),
        binary_format=field('BinaryFormat'),data_file=field('DataFile'),
        marker_file=field('MarkerFile'),exporter_pybv075='pybv 0.7.5' in text,
        bytes=len(payload),sha256=row['payload_sha256'])


def channel_table(row):
    payload=fetch(RAW+row['path'],65536)
    blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    assert blob==row['sha']
    immutable(ROOT/'metadata/channel_tables'/str(row['sha']+'.tsv'),payload)
    return dict(path=row['path'],git_blob_sha=row['sha'],bytes=len(payload))


def run():
    assert not (ROOT/'HEADER_ACQUISITION_STARTED.json').exists()
    for folder in ('annex_pointers','payload_headers','channel_tables'):
        (ROOT/'metadata'/folder).mkdir(exist_ok=False)
    auth=json.loads((ROOT/'HUMAN_AUTHORIZATION.json').read_text())
    assert auth['status']=='AUTHORIZED_BY_HUMAN_DATA_SCOPE' and auth['dataset']['commit']==COMMIT
    dump(ROOT/'HEADER_ACQUISITION_STARTED.json',dict(pid=os.getpid(),time=time.time(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),human_scope_sha256=hashlib.sha256((ROOT/'HUMAN_AUTHORIZATION.json').read_bytes()).hexdigest(),NN_forwards=0))
    tree=json.loads((ROOT/'metadata/PINNED_TREE.json').read_text())['tree']
    annex=[r for r in tree if r['mode']=='120000' and r['path'].endswith(('.eeg','.vhdr','.vmrk','.wav'))]
    assert len(annex)==1160
    catalog=[];failures=[]
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        mapping={pool.submit(pointer,r):r for r in annex}
        for i,f in enumerate(futures.as_completed(mapping),1):
            try:catalog.append(f.result())
            except Exception as e:failures.append(dict(stage='pointer',path=mapping[f]['path'],error=repr(e)))
            if i%50==0:status(stage='pinned_annex_catalog',done=i,total=1160,failures=len(failures))
    catalog.sort(key=lambda r:r['path'])
    dump(ROOT/'ANNEX_CATALOG.json',dict(commit=COMMIT,objects=catalog,failures=failures,NN_forwards=0))
    assert len(catalog)==1160 and not failures,'Partial metadata preserved; do not advance'
    headers=[]
    wanted=[r for r in catalog if r['path'].endswith(('.vhdr','.vmrk'))]
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        mapping={pool.submit(text_object,r):r for r in wanted}
        for i,f in enumerate(futures.as_completed(mapping),1):
            try:headers.append(f.result())
            except Exception as e:failures.append(dict(stage='header',path=mapping[f]['path'],error=repr(e)))
            if i%50==0:status(stage='verified_text_headers',done=i,total=760,failures=len(failures))
    # One verified copy per119 distinct TSV blobs, preserving all380 path mappings.
    all_tsv=[r for r in tree if r['path'].endswith('_channels.tsv')]
    unique={r['sha']:r for r in all_tsv}
    tables=[]
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        mapping={pool.submit(channel_table,r):r for r in unique.values()}
        for f in futures.as_completed(mapping):
            try:tables.append(f.result())
            except Exception as e:failures.append(dict(stage='channel_table',path=mapping[f]['path'],error=repr(e)))
    vh=[r for r in headers if r['path'].endswith('.vhdr')]
    vm=[r for r in headers if r['path'].endswith('.vmrk')]
    types={}
    for suffix in ('.eeg','.wav','.vhdr','.vmrk'):
        rr=[r for r in catalog if r['path'].endswith(suffix)]
        types[suffix]=dict(count=len(rr),declared_bytes=sum(r['declared_bytes'] for r in rr),maximum_object_bytes=max(r['declared_bytes'] for r in rr))
    result=dict(commit=COMMIT,headers=sorted(headers,key=lambda r:r['path']),
        channel_tables=tables,channel_table_mapping=[dict(path=r['path'],git_blob_sha=r['sha']) for r in all_tsv],
        types=types,failures=failures,no_EEG_audio_body=True,NN_forwards=0,
        wall_seconds=time.time()-START,root_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
        free_disk_bytes=shutil.disk_usage(ROOT).free)
    dump(ROOT/'HEADERS_AND_SIZES.json',result)
    assert len(vh)==380 and len(vm)==380 and len(tables)==119 and not failures
    assert result['root_bytes']<=3*1024**3 and result['free_disk_bytes']>=20*1024**3
    status(stage='COMPLETED',headers=760,channel_tables=119,types=types,NN_forwards=0)
    dump(ROOT/'HEADER_ACQUISITION_COMPLETED.json',dict(wall_seconds=time.time()-START,headers=760,channel_tables=119,types=types,NN_forwards=0))


if __name__=='__main__':
    try:run()
    except BaseException:
        import traceback
        if not (ROOT/'HEADER_ACQUISITION_FAILURE.json').exists():
            dump(ROOT/'HEADER_ACQUISITION_FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
