"""D18 metadata/source freeze then0NN finalize. Does not load array payloads."""
import argparse,copy,random
from pub_01_s03_ss07_preprojection_runtime import *
def storage_schema_check(old,family):
    """JSON-only size stress case, NOT scientific results; no array/model read."""
    sample=read(D16/'fits/H_full_seed1024/SELECTED_METRICS.json')
    support=read(BASE/'s03_full_ft_error_decomposition_20261102/SUPPORT_MASK.json')
    generator=random.Random(20261103)
    def floats(value):
        if isinstance(value,float):return generator.uniform(-1e100,1e100)
        if isinstance(value,list):return [floats(x) for x in value]
        if isinstance(value,dict):return {k:floats(v) for k,v in value.items()}
        return value
    receipts=[]
    for phase in ('train','val'):
        value=copy.deepcopy(sample);records=[];segments=[]
        if phase=='train':blocks=[b for b in support['train']['blocks'] if b['eligible'] and b['stop']-b['start']==3840]
        else:blocks=[b for r in sample['records'] for b in r['segments']]
        for row in old['records'][phase]:
            meta={k:row[k] for k in ('record','subject','story')};meta['phase']=phase
            meta.update({k:family[(phase,row['record'])][k] for k in ('source_family','family_mapping_basis')})
            record=copy.deepcopy(sample['records'][0]);record.update(meta);record['windows']=row['windows'];record['segments']=[]
            record.update(window_prediction_mean=0.123,window_prediction_RMS=0.456)
            for block in (b for b in blocks if b['record']==row['record']):
                segment=copy.deepcopy(sample['segments'][0]);segment.update(meta);segment.update(start=block['start'],samples=block.get('covered',block.get('samples')))
                record['segments'].append(segment);segments.append(segment)
            records.append(record)
        value.update(records=records,segments=segments,people=[copy.deepcopy(sample['people'][0]) for _ in range({'train':70,'val':49}[phase])],boundary_tail_records=[dict(record=r['record'],**r['boundary_tail']) for r in records],source_family_descriptive=[dict(source_family=f,people=70,blocks=6104,r=0.123,scope='family-specific block->person macro, descriptive only') for f in sorted({r['source_family'] for r in records})],target_window_counts=dict(windows=252454,exact_raw_constant=0,exact_centered_zero=0,centered_norm_le_eps=12),negative_cosine=.123,maximum_window_abs=.456,maximum_fused_abs=.456,window_prediction_mean=.123,window_prediction_RMS=.456,new_prediction_disk_arrays=0,parameters_fixed=True)
        assert len(segments)=={'train':6104,'val':752}[phase]
        # Both duplicated segment representations have independent full-precision
        # exponent-valued floats. These synthetic bytes are NOT result metrics.
        value=floats(value);raw=json.dumps(value,separators=(',',':'),allow_nan=False).encode();packed=gzip.compress(raw,compresslevel=6,mtime=0)
        assert json.loads(gzip.decompress(packed))==value and len(packed)<METRIC_FILE_LIMIT
        receipts.append(dict(phase=phase,records=len(records),segments=len(segments),uncompressed_bytes=len(raw),gzip_bytes=len(packed),enforced_per_metric_file_bytes=METRIC_FILE_LIMIT,lossless_roundtrip=True))
    receipt=dict(passed=True,scope='Synthetic JSON serialization stress only; not data/model/metrics computation or a scientific result',cases=receipts,metric_file_count_upper=87,metric_bytes_upper=87*METRIC_FILE_LIMIT,other_structured_and_source_bytes_upper=32*1024**2,new_array_values_read=False,new_NN=0,new_backward=0)
    dump(ROOT/'STORAGE_SCHEMA_CHECK.json',receipt)
    return receipt

def metadata_prepare():
    start=time.time();cpu=time.process_time();guard();assert not (ROOT/'ENGINEERING_PROTOCOL.json').exists()
    old=read(D8/'PROTOCOL.json');d16=read(D16/'PROTOCOL.json')
    assert sha(D16/'PROTOCOL.json')=='56f4be8214ec7b5402b1c6ea3e4ff2154f082150b95c30538743bd1fa170b3a2'
    assert read(ROOT/'COORDINATOR_REFERENCE_SELFCHECK.json')['status']=='PASS'
    pins=dict(d16['source_pins']);pins.update(old['source_pins'])
    files=sorted((ROOT/'input').glob('pub_01_s03_ss07_preprojection_*.py'))
    expected={'model','data','metrics','runtime','prepare','engineering','train','queue','reference'}
    assert expected<={x.stem.removeprefix('pub_01_s03_ss07_preprojection_') for x in files}
    extra=[D8/'PROTOCOL.json',D16/'PROTOCOL.json',BASE/'s03_full_ft_error_decomposition_20261102/SUPPORT_MASK.json',BASE/'s03_incremental_continuity_20261101/SOURCE_FAMILY_JOIN.json',ROOT/'PLAN.md',ROOT/'COORDINATOR_DIRECTION.md',ROOT/'COORDINATOR_REFERENCE_SELFCHECK.json']+files
    for path in extra:pins[str(path)]=sha(path)
    assert sha(ROOT/'input/pub_01_s03_ss07_preprojection_reference.py')=='19eaa95bcd166ae5c35c0d30000e50b92089af32864ac0165d62a1448f401d18'
    for path,digest in pins.items():assert sha(path)==digest,path
    inputs=dict(old['inputs'])
    for seed in SEEDS:
        folder=D8/f'seed{seed}'
        for phase in ('train','val'):
            for kind in ('latent','P'):
                path=folder/f'{phase}_{kind}.npy';inputs[str(path)]=dict(d16['inputs'][str(path)],recheck_sha256=False)
        cp=Path(old['checkpoints'][str(seed)]['checkpoint']);s=cp.stat();inputs[str(cp)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=pins[str(cp)],recheck_sha256=True)
        for name in ('TRAIN_TRANSFORMS.npz','Ccap_lambda0.01_COEFFICIENTS.npz','P_FT_BASELINE.json'):
            path=folder/name;s=path.stat();inputs[str(path)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=pins[str(path)],recheck_sha256=True)
    rows=old['records']['train'];local=np.empty(252454,dtype='i4')
    family_rows=read(BASE/'s03_incremental_continuity_20261101/SOURCE_FAMILY_JOIN.json')['records']
    family={(r['phase'],r['record']):r for r in family_rows};assert len(family)==len(family_rows)
    for phase in ('train','val'):
        for row in old['records'][phase]:assert all(family[(phase,row['record'])][k]==row[k] for k in ('subject','story','common_samples','windows'))
        assert len({family[(phase,r['record'])]['source_family'] for r in old['records'][phase]})=={'train':37,'val':5}[phase]
    storage=storage_schema_check(old,family)
    pins[str(ROOT/'STORAGE_SCHEMA_CHECK.json')]=sha(ROOT/'STORAGE_SCHEMA_CHECK.json')
    for row in rows:
        o=row['offset'];w=row['windows'];local[o:o+w]=np.arange(w)
    assert int(np.sum(local<9))==3834
    orders=[];cases={}
    for seed in SEEDS:
        for epoch in range(20):
            order=permutation(seed,epoch,len(local));assert np.array_equal(np.sort(order),np.arange(len(local)))
            assert all(np.any(local[order[a:a+64]]>=9) for a in range(0,len(local),64))
            digest=hashlib.sha256(order.tobytes()).hexdigest()
            expected_digest=next(x['sha256'] for x in d16['permutations'] if x['seed']==seed and x['epoch']==epoch);assert digest==expected_digest
            orders.append(dict(seed=seed,epoch=epoch,sha256=digest))
        first=permutation(seed,0,len(local));cold=[r['offset']+i for r in rows[:8] for i in (0,8,9,10,11,12,13,14)]
        cases[str(seed)]=dict(shuffle0=first[:64].tolist(),shuffle1=first[64:128].tolist(),cold=cold)
    proto=dict(frozen=True,stage='ENGINEERING_ONLY_FORMAL_NOT_RUN',source_pins=pins,inputs=inputs,seeds=SEEDS,arms=ARMS,epochs=20,TRAIN_cases=cases,permutations=orders,objective='float64 centered negative cosine eps1e-6; all252454targets including3834fallback',optimizer=dict(algorithm='Adam',encoder_lr=1e-5,head_lr=.001,memory_and_prior_lr=.001,weight_decay=0.,gradient_clip=None),selection='VAL physical60s block->person macro,strict earliest greater including oldCcap -1; all20epochs',fallback='local<9 frozen oldCcap, retained loss denominator',positions='Fixed q0..191 and near slots-480..-97; same codes C/P and older replacement',history=dict(near=[5,3],older=[9,7],raw_support_end_le_current_start_minus96=True,E0_frozen_eval=True,current_Etheta_trainable=True),registered_parameters=dict(T_continue=3372617,H_history=3389513,C_current=3389513,P_prediction_history=3390153),engineering_caps=dict(current_prefix_and_reference=8192,current_head_and_reference=8192,frozen_prefix=16384,temporary_optimizer_updates=24,new_fits=0,all_other_component_operations_recorded=True),bridges=dict(same_latent_value=dict(abs=1e-10,rel=1e-9),same_latent_VJP=dict(abs=1e-9,rel=1e-8),same_raw_native_vs_zero_memory=dict(abs=1e-6,rel=0),raw_reencoded_vs_oldcache=dict(abs=1e-4,rel=1e-6),uses_only_TRAIN=True),offline_TRAIN_epochs=[-1,0,4,9,19],budget=dict(fits=12,epochs=240,optimizer_update_positions=946800,target_window_presentations=60588960,current_encoder_prefix_windows_upper=80392752,current_projection_windows_upper=80484252,current_head_windows_upper=80484252,frozen_encoder_prefix_windows_upper=59563425,attention_query_windows_upper=59473248,prior_projection_windows_upper=39588714),resources=dict(queue_wall_seconds=86400,max_workers=2,one_per_GPU=True,worker_RSS_bytes=48*1024**3,worker_GPU_allocated_bytes=32*1024**3,root_bytes=768*1024**2,inflight_reserve_bytes=64*1024**2,each_disk_free_bytes=20*1024**3,BLAS_threads=4,torch_threads=2),prediction_disk_arrays=False,new_dataset_or_weight_download=False,old_test_DTU_ds004408_access=False,goal_complete=False)
    previous=dict(production={},extra={},attempts=[])
    for failure in sorted((ROOT/'attempts').glob('engineering_attempt*/ENGINEERING_FAILURE.json')):
        evidence=read(failure);pins[str(failure)]=sha(failure)
        for key in ('counts_completed_rows','unfinished_model_invoked_component_counts'):
            for name,count in evidence[key].items():previous['production'][name]=previous['production'].get(name,0)+count
        for name,count in evidence['extra_counts'].items():previous['extra'][name]=previous['extra'].get(name,0)+count
        previous['attempts'].append(dict(path=str(failure),sha256=sha(failure),CPU_seconds=evidence.get('CPU_seconds'),wall_seconds=evidence.get('wall_seconds'),resource_timing_limitation='Attempt01 failed before timing receipt was implemented' if 'CPU_seconds' not in evidence else None))
    proto['prior_engineering_costs']=previous
    proto['storage_contract']=storage
    verify(proto);dump(ROOT/'ENGINEERING_PROTOCOL.json',proto)
    dump(ROOT/'METADATA_PREPARED.json',dict(passed=True,engineering_protocol_sha256=sha(ROOT/'ENGINEERING_PROTOCOL.json'),source_pins=len(pins),input_identities=len(inputs),all60_permutations_match_D16=True,all_batches_have_nonfallback=True,new_encoder_head_forwards=0,scientific_array_values_read=False,new_fits=0,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(passed=True,engineering_protocol_sha256=sha(ROOT/'ENGINEERING_PROTOCOL.json'))))
def finalize():
    start=time.time();cpu=time.process_time();assert not (ROOT/'PROTOCOL.json').exists()
    p=read(ROOT/'ENGINEERING_PROTOCOL.json');verify(p);e=read(ROOT/'ENGINEERING_CHECK.json');assert e['passed'] and e['engineering_protocol_sha256']==sha(ROOT/'ENGINEERING_PROTOCOL.json')
    p['stage']='FORMAL_READY_PENDING_COORDINATOR_SOURCE_REVIEW';p['engineering_protocol_sha256']=sha(ROOT/'ENGINEERING_PROTOCOL.json');p['source_pins'][str(ROOT/'ENGINEERING_PROTOCOL.json')]=sha(ROOT/'ENGINEERING_PROTOCOL.json');p['source_pins'][str(ROOT/'ENGINEERING_CHECK.json')]=sha(ROOT/'ENGINEERING_CHECK.json')
    # All87 metrics retained as compact lossless gzip, individually bounded.
    # Logs/receipts/source have32MiB provision;64MiB remains for in-flight writes.
    s=p['storage_contract']
    estimated=24*e['largest_parameter_checkpoint_bytes']+s['metric_bytes_upper']+s['other_structured_and_source_bytes_upper']+p['resources']['inflight_reserve_bytes']
    assert estimated<p['resources']['root_bytes'];p['storage_estimated_upper_bytes']=estimated
    p['engineering_measured_resource']=dict(peak_RSS_KiB=e['peak_RSS_KiB'],peak_GPU_allocated_bytes=e['peak_GPU_allocated_bytes'],CPU_seconds=e['CPU_seconds'],wall_seconds=e['wall_seconds'])
    dump(ROOT/'PROTOCOL.json',p);dump(ROOT/'FREEZE_RECORD.json',dict(passed=True,new_NN=0,new_backward=0,optimizer_updates=0,protocol_sha256=sha(ROOT/'PROTOCOL.json'),source_unchanged_from_engineering=True,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start));guard()
    print(json.dumps(dict(passed=True,protocol_sha256=sha(ROOT/'PROTOCOL.json'))))

def rebind_metadata():
    """Reuse validated permutations/size receipt after the narrow import repair."""
    a=ROOT/'attempts/engineering_attempt01';p=read(a/'ENGINEERING_PROTOCOL.json')
    assert not (ROOT/'ENGINEERING_PROTOCOL.json').exists()
    for path,digest in p['source_pins'].items():
        source=Path(path)
        archived=a/'input'/source.name if source.parent==ROOT/'input' else a/'STORAGE_SCHEMA_CHECK.json' if source==ROOT/'STORAGE_SCHEMA_CHECK.json' else source
        assert sha(archived)==digest,path
    for source in sorted((ROOT/'input').glob('pub_01_s03_ss07_preprojection_*.py')):p['source_pins'][str(source)]=sha(source)
    assert sha(ROOT/'input/pub_01_s03_ss07_preprojection_reference.py')=='19eaa95bcd166ae5c35c0d30000e50b92089af32864ac0165d62a1448f401d18'
    with (ROOT/'STORAGE_SCHEMA_CHECK.json').open('xb') as f:f.write((a/'STORAGE_SCHEMA_CHECK.json').read_bytes())
    failure=a/'ENGINEERING_FAILURE.json';e=read(failure);counts=dict(e['counts_completed_rows'])
    for key,count in e['unfinished_model_invoked_component_counts'].items():counts[key]=counts.get(key,0)+count
    p['source_pins'][str(failure)]=sha(failure)
    p['prior_engineering_costs']=dict(production=counts,extra=e['extra_counts'],attempts=[dict(path=str(failure),sha256=sha(failure),CPU_seconds=None,wall_seconds=None,resource_timing_limitation='Attempt01 failed before timing receipt was implemented; do not infer zero cost')])
    p['metadata_rebound_from_sha256']=sha(a/'ENGINEERING_PROTOCOL.json');verify(p);dump(ROOT/'ENGINEERING_PROTOCOL.json',p)
    dump(ROOT/'METADATA_PREPARED.json',dict(passed=True,engineering_protocol_sha256=sha(ROOT/'ENGINEERING_PROTOCOL.json'),reuses_validated_metadata=str(a/'METADATA_PREPARED.json'),new_permutations=0,new_size_cases=0,new_encoder_head_forwards=0,scientific_array_values_read=False,new_fits=0))
    print(json.dumps(dict(passed=True,engineering_protocol_sha256=sha(ROOT/'ENGINEERING_PROTOCOL.json'))))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('metadata','rebind','finalize'));args=parser.parse_args()
    {'metadata':metadata_prepare,'rebind':rebind_metadata,'finalize':finalize}[args.action]()
