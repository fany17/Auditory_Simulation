"""D18 metadata/source freeze then0NN finalize. Does not load array payloads."""
import argparse
from pub_01_s03_ss07_preprojection_runtime import *
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
    verify(proto);dump(ROOT/'ENGINEERING_PROTOCOL.json',proto)
    dump(ROOT/'METADATA_PREPARED.json',dict(passed=True,engineering_protocol_sha256=sha(ROOT/'ENGINEERING_PROTOCOL.json'),source_pins=len(pins),input_identities=len(inputs),all60_permutations_match_D16=True,all_batches_have_nonfallback=True,new_encoder_head_forwards=0,scientific_array_values_read=False,new_fits=0,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(passed=True,engineering_protocol_sha256=sha(ROOT/'ENGINEERING_PROTOCOL.json'))))
def finalize():
    start=time.time();cpu=time.process_time();assert not (ROOT/'PROTOCOL.json').exists()
    p=read(ROOT/'ENGINEERING_PROTOCOL.json');verify(p);e=read(ROOT/'ENGINEERING_CHECK.json');assert e['passed'] and e['engineering_protocol_sha256']==sha(ROOT/'ENGINEERING_PROTOCOL.json')
    p['stage']='FORMAL_READY_PENDING_COORDINATOR_SOURCE_REVIEW';p['engineering_protocol_sha256']=sha(ROOT/'ENGINEERING_PROTOCOL.json');p['source_pins'][str(ROOT/'ENGINEERING_PROTOCOL.json')]=sha(ROOT/'ENGINEERING_PROTOCOL.json');p['source_pins'][str(ROOT/'ENGINEERING_CHECK.json')]=sha(ROOT/'ENGINEERING_CHECK.json')
    # Bound actual size; include oldD16 observed structured-output footprint with
    # conservative scaling,24 mutable parameter checkpoints,64MiB write reserve.
    estimated=24*e['largest_parameter_checkpoint_bytes']+192*1024**2+64*1024**2
    assert estimated<p['resources']['root_bytes'];p['storage_estimated_upper_bytes']=estimated
    p['engineering_measured_resource']=dict(peak_RSS_KiB=e['peak_RSS_KiB'],peak_GPU_allocated_bytes=e['peak_GPU_allocated_bytes'],CPU_seconds=e['CPU_seconds'],wall_seconds=e['wall_seconds'])
    dump(ROOT/'PROTOCOL.json',p);dump(ROOT/'FREEZE_RECORD.json',dict(passed=True,new_NN=0,new_backward=0,optimizer_updates=0,protocol_sha256=sha(ROOT/'PROTOCOL.json'),source_unchanged_from_engineering=True,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start));guard()
    print(json.dumps(dict(passed=True,protocol_sha256=sha(ROOT/'PROTOCOL.json'))))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('metadata','finalize'));args=parser.parse_args()
    metadata_prepare() if args.action=='metadata' else finalize()
