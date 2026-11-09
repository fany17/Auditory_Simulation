"""Bounded SHA-verified public S3 Range transport. No models or data processing."""
import hashlib,json,os,re,shutil,threading,time,urllib.parse,urllib.request
from pathlib import Path
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_openneuro_confirmation_20261102')
LOCK=threading.Lock()
CHUNK=1024*1024


def event(value):
    with LOCK:
        with (ROOT/'RANGE_TRANSPORT_EVENTS.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(),pid=os.getpid(),**value))+'\n')


def obtain_verified(row,folder):
    path=folder/Path(row['path']).name
    if path.exists():
        data=path.read_bytes()
        assert len(data)==row['declared_bytes'] and hashlib.sha256(data).hexdigest()==row['payload_sha256']
        return data
    expected=row['declared_bytes']
    assert expected<=128*1024**2
    assert shutil.disk_usage(ROOT).free-expected>=20*1024**3
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())+expected<=3*1024**3
    output=bytearray(expected)
    version=None;start=time.time()
    for first in range(0,expected,CHUNK):
        last=min(expected,first+CHUNK)-1
        for attempt in range(1,4):
            try:
                url=row['public_url']
                if version is not None:url+='?'+urllib.parse.urlencode({'versionId':version})
                request=urllib.request.Request(url,headers={'Range':f'bytes={first}-{last}','User-Agent':'Auditory-Simulation-public-research/1.0'})
                with urllib.request.urlopen(request,timeout=30) as response:
                    assert response.status==206,'S3 Range must be honored'
                    assert response.headers.get('Content-Range')==f'bytes {first}-{last}/{expected}'
                    current_version=response.headers.get('x-amz-version-id')
                    assert current_version and (version is None or current_version==version)
                    data=response.read(last-first+2)
                assert len(data)==last-first+1
                version=current_version
                output[first:last+1]=data
                event(dict(path=row['path'],first=first,last=last,total=expected,status='RANGE_VERIFIED',version=version))
                break
            except Exception as e:
                event(dict(path=row['path'],first=first,last=last,attempt=attempt,status='RANGE_ERROR',error=repr(e)))
                if attempt==3:raise
                time.sleep(.5*attempt)
        assert time.time()-start<1800,'Per-object30min transport cap'
    digest=hashlib.sha256(output).hexdigest()
    assert digest==row['payload_sha256'],'Pinned full object digest failed'
    with path.open('xb') as f:f.write(output)
    event(dict(path=row['path'],total=expected,status='FULL_OBJECT_SHA_VERIFIED',sha256=digest,version=version))
    return output
