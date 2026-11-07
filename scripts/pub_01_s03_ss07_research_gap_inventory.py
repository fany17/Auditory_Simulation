"""2203-only read-only TRAIN/VAL inventory: JSON/stat/NPY headers, no array values."""
import csv,json,hashlib,shutil,time
from pathlib import Path
from collections import Counter
import numpy as np

B=Path('/home/fanyu/auditory_simulation_m6a/pub_01')
P=B/'s03_incremental_continuity_20261101';D=B/'s03_finetuned_readout_transfer_20261101'
OUT=B/'s03_research_gap_20261102';START=time.time()
def read(p):return json.loads(p.read_text())
def header(p):
    with p.open('rb') as f:
        version=np.lib.format.read_magic(f)
        shape,order,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
        offset=f.tell()
    payload=int(np.prod(shape))*dtype.itemsize;assert p.stat().st_size==offset+payload
    return dict(path=str(p),shape=list(shape),dtype=str(dtype),fortran_order=order,header_bytes=offset,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
def dump(name,j):
    with (OUT/name).open('x') as f:json.dump(j,f,indent=2,allow_nan=False)

assert not OUT.exists(),'Exclusive new metadata output; never overwrite'
OUT.mkdir()
protocol=read(D/'PROTOCOL.json');rows=protocol['records']
families={x['record']:x for x in read(P/'SOURCE_FAMILY_JOIN.json')['records'] if x['phase'] in ['train','val']}
outrows=[];deps={};supports={};targets={};assets=[]
for phase in ['train','val']:
    rr=rows[phase];offset=0
    for x in rr:
        assert x['phase']==phase and x['offset']==offset and x['windows']==(x['common_samples']-192)//96+1
        offset+=x['windows'];f=families[x['record']];assert f['subject']==x['subject'] and f['story']==x['story']
        outrows.append(dict(phase=phase,record=x['record'],subject=x['subject'],story=x['story'],source_family=f['source_family'],family_mapping_basis=f['family_mapping_basis'],windows=x['windows'],offset=x['offset'],common_samples=x['common_samples']))
        p=Path(x['envelope_path'])
        if str(p) not in targets:targets[str(p)]=header(p)
        assert int(np.prod(targets[str(p)]['shape']))>=x['common_samples']
    deps[phase]=dict(records=len(rr),windows=offset,people=len({x['subject'] for x in rr}),stories=sorted({x['story'] for x in rr}),
        families=sorted({families[x['record']]['source_family'] for x in rr}),records_per_person=dict(Counter(x['subject'] for x in rr)),
        records_per_story=dict(Counter(x['story'] for x in rr)),records_per_family=dict(Counter(families[x['record']]['source_family'] for x in rr)))
    # Geometry examples only, not a proposed model-selection grid.
    supports[phase]=[]
    for k in [1,2,10]:
        first=max(4,2*k+1)
        supports[phase].append(dict(history_tokens=k,history_duration_seconds=3*k,first_local_window=first,
            residual_eligible_windows=sum(max(0,x['windows']-first) for x in rr),cold_fallback_windows=sum(min(first,x['windows']) for x in rr),
            new_materialized_history_bytes_if_copied=offset*k*8*192*4,global64_history_float32_input_bytes=64*k*8*192*4))
    for seed in [1024,2026,31415]:
        for suffix,shape in [('latent',[offset,8,192]),('P',[offset,192])]:
            j=header(D/f'seed{seed}/{phase}_{suffix}.npy');assert j['shape']==shape and j['dtype']=='float32' and not j['fortran_order'];assets.append(j)
    y=header(B/f's03_strong_baselines_20261015/transformer_available01/envelope_readout_seed1024/{phase}_envelope.npy')
    assert y['shape'][0]==offset;assets.append(y)
for key in ['subject','story','source_family']:
    a={x[key] for x in outrows if x['phase']=='train'};b={x[key] for x in outrows if x['phase']=='val'}
    deps[key+'_overlap']=dict(intersection=sorted(a&b),val_not_in_train=sorted(b-a))
deps['family_mapping_basis']=dict(Counter(x['family_mapping_basis'] for x in outrows))
deps['independence_warning']='Repeated TRAIN/VAL development; shared person/story/source families. Mapping is metadata provenance, not waveform identity audit.'
params={'round01':{},'warm':{},'frozen_FT_anchor':{}}
for name in ['m1_h16','m2_h16','m3_h16','m4_h16','m5_h16','m4_current_h16','static_m4_h16','output_prior_h16','static_alignment']:
    values=[]
    for seed in [1024,2026,31415]:
        p=P/f'full/{name}_seed{seed}/completed.json'
        if p.exists():values.append(dict(seed=seed,**read(p)['gradient_parameters'],manifest=str(p)))
    params['round01'][name]=values
for mode in ['frozen_p','frozen_current','frozen_past','joint_p','joint_current','joint_past']:
    values=[]
    for seed in [1024,2026,31415]:
        p=B/f's03_incremental_warm_20261101/fits/{mode}_seed{seed}/completed.json';j=read(p)
        values.append(dict(seed=seed,trainable_parameters=j['trainable_parameters'],manifest=str(p)))
    params['warm'][mode]=values
for seed in [1024,2026,31415]:
    p=P/f'finetune/native_P_seed{seed}/completed.json';j=read(p)
    params['frozen_FT_anchor'][str(seed)]=dict(selected_epoch=j['selected_epoch'],original_FT_trainable_parameters=j['registered_trainable_parameters'],
        checkpoint_path=protocol['checkpoints'][str(seed)]['checkpoint'],checkpoint_sha256=protocol['checkpoints'][str(seed)]['checkpoint_sha256'],
        cache_completed=read(D/f'seed{seed}/CACHE_COMPLETED.json'),future_frozen_encoder_head_trainable_parameters=0,existing_Ccap_coefficients_including_bias=626880)
free=shutil.disk_usage(D).free
audit=dict(scope='Existing FT TRAIN/VAL JSON/stat/NPY headers only; no array payload values, scores, model construction or forward',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),assets=assets,distinct_target_headers=list(targets.values()),
    all_headers_and_offsets_passed=True,strict_history=dict(window='[96i,96i+192) at64Hz',fused_current_union_earliest='96i-96 (interior)',
        token_indices='i-3, i-5, ..., i-(2K+1), chronological order reversed for input',
        raw_support_union='[96i-96-192K, 96i-96)',required_i='max(4,2K+1)',
        record_phase_boundary='Never use global offset subtraction without same-record checks; leading windows keep frozen anchor'),
    geometry_examples_not_hyperparameter_grid=supports,all_FT_Z_P_existing_bytes=sum(x['bytes'] for x in assets if '/seed' in x['path']),
    current_home_free_bytes=free,home_reserve_bytes=20*1024**3,free_above_reserve=free-20*1024**3,
    recommended_new_history_cache_bytes=0,resource_note='Index existing memmaps by batch; copied K-token full dataset could exceed free/cumulative caps. No new compute budget authorized.',
    new_NN=0,new_fits=0,array_values_read=False,new_external_access=False,wall_seconds=time.time()-START)
dump('FT_CACHE_METADATA_AUDIT.json',audit);dump('DEPENDENCY_COUNTS.json',deps);dump('PARAMETER_AUDIT.json',params)
with (OUT/'DEPENDENCY_RECORDS.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(outrows[0]));w.writeheader();w.writerows(outrows)
print(json.dumps(dict(passed=True,records=len(outrows),cache_headers=len(assets),unique_target_headers=len(targets),home_free_bytes=free,
    dependencies={p:{k:v for k,v in deps[p].items() if k in ['records','windows','people','stories','families']} for p in ['train','val']},new_NN=0,new_fits=0)))
