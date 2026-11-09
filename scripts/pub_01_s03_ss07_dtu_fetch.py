"""Coordinator-approved two-file DTU intake; execute ONLY server2203.

No models, fitting, test performance, raw EEG.zip or AUDIO.zip download.
All partial/failed evidence retained. No recursive or permanent deletion.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time
import urllib.request
import zipfile

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_continuity_20261101')
DEST=ROOT/'external_dtu'
ALLOWED={'DATA_preproc.zip':(1838947136,'1c1ca1390ad99ff41c33f36ca67f8418'),
    'preproc_data.m':(6252,'7723ddc5334f5aa1f473143251fdcffd')}


def dump(name,value):
    with (DEST/name).open('x') as f:json.dump(value,f,indent=2)


def status(value):
    temp=DEST/'download_status.writing';temp.write_text(json.dumps(value,indent=2));os.replace(temp,DEST/'download_status.json')


def guard(extra):
    actual=sum(p.stat().st_size for p in (ROOT/'baseline').rglob('*_latent.npy'))
    # Five own-seed caches, as frozen before any external expansion.
    pending=max(0,5*(252454+30500)*8*192*4-actual)
    free=shutil.disk_usage(DEST).free
    if free-extra-pending<20*1024**3:raise RuntimeError('reserve20GiB including pending original cache violated')
    return {'free_bytes':free,'pending_baseline_cache_bytes':pending,'planned_extra_bytes':extra}


def main():
    assert ROOT.is_dir() and str(Path.cwd()).startswith('/'), 'server2203 only'
    api=json.load(urllib.request.urlopen('https://zenodo.org/api/records/1199011',timeout=30))
    assert api['metadata']['license']['id']=='cc-by-nc-4.0'
    files={f['key']:f for f in api['files']};probe=json.loads((DEST/'ZIP_DIRECTORY_PROBE.json').read_text())
    assert probe['range_supported'];unpacked=probe['uncompressed_total_bytes']
    peak=sum(s for s,_ in ALLOWED.values())+unpacked
    assert peak<=12*1024**3
    resources=guard(peak)
    dump('INTAKE_AUTHORIZATION.json',{'scope':'SS07 coordinator research extension under human full-public-EEG goal',
        'coordinator_chat':'01a134e4-2036-7e02-81e9-1289a07606f2','human_named_dataset':False,
        'allowed':ALLOWED,'license':'CC-BY-NC-4.0; noncommercial research/attribution',
        'source':'https://zenodo.org/records/1199011','zip_plus_extraction_peak_bytes':peak,
        'resource_check':resources,'models_or_performance_authorized_here':False,'time':time.time()})
    for name,(expected,md5) in ALLOWED.items():
        meta=files[name];assert meta['size']==expected and meta['checksum']=='md5:'+md5
        path=DEST/name;partial=DEST/(name+'.part');assert not path.exists()
        count=0;digest=hashlib.md5();started=time.time()
        with partial.open('xb') as output,urllib.request.urlopen(meta['links']['self'],timeout=90) as response:
            while True:
                chunk=response.read(8*1024**2)
                if not chunk:break
                count+=len(chunk);assert count<=expected;output.write(chunk);digest.update(chunk)
                guard(unpacked)
                status({'file':name,'bytes':count,'expected':expected,'elapsed_seconds':time.time()-started,'state':'DOWNLOADING'})
            output.flush();os.fsync(output.fileno())
        assert count==expected and digest.hexdigest()==md5,'official identity mismatch; preserve partial'
        assert not path.exists();partial.rename(path)
        dump(name+'.verified.json',{'bytes':count,'official_md5':md5,'verified':True,'seconds':time.time()-started})
    target=DEST/'extracted';target.mkdir(exist_ok=False);base=target.resolve()
    with zipfile.ZipFile(DEST/'DATA_preproc.zip') as archive:
        infos=archive.infolist();assert sum(i.file_size for i in infos)==unpacked
        assert [i.filename for i in infos]==[m['name'] for m in probe['members']]
        for info in infos:
            path=(base/info.filename).resolve()
            assert path.is_relative_to(base) and not stat.S_ISLNK(info.external_attr>>16)
            assert not info.is_dir() and path.parent==base and info.filename.endswith('_data_preproc.mat')
            guard(info.file_size)
            with archive.open(info) as source,path.open('xb') as output:shutil.copyfileobj(source,output,8*1024**2)
            assert path.stat().st_size==info.file_size
    dump('DOWNLOAD_EXTRACT_COMPLETED.json',{'members':len(infos),'verified_official_checksums':True,
        'archive_retained':True,'peak_budget_bytes':peak,'free_bytes':shutil.disk_usage(DEST).free,
        'neural_or_audio_arrays_returned_local':False,'model_performance_computed':False,'time':time.time()})
    status({'state':'DOWNLOAD_EXTRACT_COMPLETED','members':len(infos),'time':time.time()})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status({'state':'FAILED_PRESERVED','error':repr(exc),'time':time.time()});raise
