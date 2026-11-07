"""Header attempt02: use pinned Git metadata tar, same payload checks, no neural/audio body."""
import concurrent.futures as futures
import hashlib,io,json,os,re,tarfile,time
from pathlib import Path
import pub_01_s03_ss07_openneuro_headers as first
ROOT=first.ROOT
START=time.time()


def run():
    assert not (ROOT/'HEADER_ACQUISITION_V2_STARTED.json').exists()
    first.dump(ROOT/'HEADER_ACQUISITION_V2_STARTED.json',dict(pid=os.getpid(),time=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),shared_source_sha256=hashlib.sha256(Path(first.__file__).read_bytes()).hexdigest(),no_model_prediction=True))
    tree=json.loads((ROOT/'metadata/PINNED_TREE.json').read_text())['tree']
    mapping={r['path']:r for r in tree if r['type']=='blob'}
    archive=ROOT/'metadata/PINNED_REPOSITORY_METADATA.tar.gz'
    catalog=[];table_rows=[];table_map=[];verified=0
    with tarfile.open(archive,mode='r:gz') as tf:
        for member in tf.getmembers():
            if member.isdir():continue
            path=member.name.split('/',1)[1]
            row=mapping[path]
            if member.issym():
                data=member.linkname.encode()
            else:
                assert member.isfile()
                data=tf.extractfile(member).read()
            blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert blob==row['sha'],path
            verified+=1
            if member.issym() and path.endswith(('.eeg','.vhdr','.vmrk','.wav')):
                m=re.search(rb'SHA256E-s(\d+)--([a-f0-9]{64})(\.[A-Za-z0-9]+)',data);assert m
                catalog.append(dict(path=path,git_blob_sha=row['sha'],declared_bytes=int(m[1]),
                    payload_sha256=m[2].decode(),public_url=first.S3+path))
            elif path.endswith('_channels.tsv'):
                target=ROOT/'metadata/channel_tables'/str(row['sha']+'.tsv')
                first.immutable(target,data)
                table_map.append(dict(path=path,git_blob_sha=row['sha']))
                table_rows.append(dict(path=path,git_blob_sha=row['sha'],bytes=len(data)))
            elif path in ('README','dataset_description.json','participants.tsv'):
                first.immutable(ROOT/'metadata'/path,data)
    catalog.sort(key=lambda r:r['path'])
    assert len(catalog)==1160 and len(table_map)==380
    first.dump(ROOT/'ANNEX_CATALOG_V2.json',dict(commit=first.COMMIT,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        all_git_blob_entries_verified=verified,objects=catalog))
    first.status(stage='V2_PINNED_CATALOG_VERIFIED',objects=1160,git_blobs=verified)
    wanted=[r for r in catalog if r['path'].endswith(('.vhdr','.vmrk'))]
    headers=[];failures=[]
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending={pool.submit(first.text_object,r):r for r in wanted}
        for i,f in enumerate(futures.as_completed(pending),1):
            try:headers.append(f.result())
            except Exception as e:failures.append(dict(path=pending[f]['path'],error=repr(e)))
            if i%50==0:first.status(stage='V2_VERIFIED_TEXT_HEADERS',done=i,total=760,failures=len(failures))
    types={}
    for suffix in ('.eeg','.wav','.vhdr','.vmrk'):
        rr=[r for r in catalog if r['path'].endswith(suffix)]
        types[suffix]=dict(count=len(rr),declared_bytes=sum(r['declared_bytes'] for r in rr),
            maximum_object_bytes=max(r['declared_bytes'] for r in rr))
    vh=[r for r in headers if r['path'].endswith('.vhdr')]
    vm=[r for r in headers if r['path'].endswith('.vmrk')]
    summary=dict(types=types,headers=len(headers),failures=failures,
        channel_tables_unique=len({r['git_blob_sha'] for r in table_rows}),
        format_variants=sorted({r['binary_format'] for r in vh}),
        orientation_variants=sorted({r['orientation'] for r in vh}),
        channel_count_variants=sorted({r['channel_count'] for r in vh}),
        sampling_interval_us=sorted({r['sampling_interval_us'] for r in vh}),
        resolution_unit_variants=sorted({(c['resolution'],c['unit']) for r in vh for c in r['channels']}),
        marker_counts=sorted({r['marker_count'] for r in vm}),
        all_channel_names_A1_D32=all([c['name'] for c in r['channels']]==[letter+str(i) for letter in 'ABCD' for i in range(1,33)] for r in vh),
        projected_full_EEG_float32_64Hz_64channel_bytes_upper=types['.eeg']['declared_bytes']//16+380*128,
        no_EEG_audio_body=True,NN_forwards=0,wall_seconds=time.time()-START)
    first.dump(ROOT/'HEADERS_AND_SIZES_V2.json',dict(summary=summary,headers=headers,
        channel_tables=table_rows,channel_table_mapping=table_map))
    first.dump(ROOT/'HEADER_SUMMARY.json',summary)
    assert len(vh)==380 and len(vm)==380 and summary['channel_tables_unique']==119 and not failures
    first.status(stage='V2_COMPLETED',headers=760,channel_tables=119,types=types,NN_forwards=0)
    first.dump(ROOT/'HEADER_ACQUISITION_V2_COMPLETED.json',summary)


if __name__=='__main__':
    try:run()
    except BaseException:
        import traceback
        first.dump(ROOT/'HEADER_ACQUISITION_V2_FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
