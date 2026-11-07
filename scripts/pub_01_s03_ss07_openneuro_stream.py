"""Pinned public EEG transport to bounded memory, never a full raw archive on disk."""
import hashlib,time,urllib.parse,urllib.request
from pathlib import Path
from pub_01_s03_ss07_openneuro_transport import ROOT,event


def obtain_memory(row,stage_started):
    retained=ROOT/'canary_raw'/Path(row['path']).name
    if retained.exists():
        data=retained.read_bytes()
        assert len(data)==row['declared_bytes'] and hashlib.sha256(data).hexdigest()==row['payload_sha256']
        event(dict(path=row['path'],status='MEMORY_REUSED_VERIFIED_CANARY'))
        return data
    expected=row['declared_bytes'];assert 0<expected<=128*1024**2
    output=bytearray(expected);version=None;started=time.time()
    chunk=2*1024**2
    for first in range(0,expected,chunk):
        assert time.time()-stage_started<4*3600,'4h preprocessing stage cap'
        assert time.time()-started<1800,'30min object transport cap'
        last=min(expected,first+chunk)-1
        for attempt in range(1,4):
            try:
                url=row['public_url']
                if version is not None:url+='?'+urllib.parse.urlencode({'versionId':version})
                request=urllib.request.Request(url,headers={'Range':f'bytes={first}-{last}',
                    'User-Agent':'Auditory-Simulation-public-research/1.0'})
                with urllib.request.urlopen(request,timeout=30) as response:
                    assert response.status==206
                    assert response.headers.get('Content-Range')==f'bytes {first}-{last}/{expected}'
                    current=response.headers.get('x-amz-version-id')
                    assert current and (version is None or current==version)
                    data=response.read(last-first+2)
                assert len(data)==last-first+1
                version=current;output[first:last+1]=data
                event(dict(path=row['path'],first=first,last=last,total=expected,
                    status='MEMORY_RANGE_VERIFIED',version=version))
                break
            except Exception as error:
                event(dict(path=row['path'],first=first,last=last,attempt=attempt,
                    status='MEMORY_RANGE_ERROR',error=repr(error)))
                if attempt==3:raise
                time.sleep(.5*attempt)
    digest=hashlib.sha256(output).hexdigest()
    assert digest==row['payload_sha256'],'Pinned complete EEG digest failed'
    event(dict(path=row['path'],total=expected,status='MEMORY_FULL_SHA_VERIFIED',sha256=digest,version=version))
    return output
