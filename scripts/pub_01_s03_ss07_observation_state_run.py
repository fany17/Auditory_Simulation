"""D11 fixed15 ridge +6 nofit interventions; original TRAIN/VAL, server2203."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import traceback
import numpy as np
import pub_01_s03_ss07_observation_state_core as core

BASE = Path('/home/fanyu/auditory_simulation_m6a/pub_01')
ROOT = BASE / 's03_observation_state_probe_20261101'
D8 = BASE / 's03_finetuned_readout_transfer_20261101'
OLD = BASE / 's03_strong_baselines_20261015/transformer_available01'
METRIC_PATH = BASE / 's03_current_readout_internal_audit_20261101/input'
sys.path.append(str(METRIC_PATH))
import pub_01_s03_ss07_generalization_core as metric

SEEDS = (1024, 2026, 31415)
START, CPU = time.time(), time.process_time()
COST = []
COUNTS = {}
SCORE_ROLES = ('Ccap',) + core.ARMS + ('PastOlder', 'PastNeutral')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    assert path.is_relative_to(ROOT)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def save(path, **arrays):
    path = Path(path)
    assert path.is_relative_to(ROOT)
    with path.open('xb') as stream:
        np.savez(stream, **arrays)
    guard()


def status(**value):
    temp = ROOT / 'STATUS.writing'
    temp.write_text(json.dumps(dict(time=time.time(), elapsed=time.time()-START, **value)))
    os.replace(temp, ROOT / 'STATUS.json')
    print(json.dumps(value), flush=True)


def account(stage, t, **value):
    COST.append(dict(stage=stage, wall_seconds=time.time()-t[0],
                     CPU_seconds=time.process_time()-t[1], **value))


def clock():
    return time.time(), time.process_time()


def all_npy():
    return sum(p.stat().st_size for root in BASE.glob('s03_*_20261101')
               for p in root.rglob('*.npy'))


def guard():
    assert time.time()-START < 3600, '60 minute wall cap'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 8*1024**3
    assert shutil.disk_usage(ROOT).free >= 20*1024**3
    files = [p for p in ROOT.rglob('*') if p.is_file()]
    assert sum(p.stat().st_size for p in files) <= 256*1024**2
    assert sum(p.stat().st_size for p in files if p.suffix in ('.npy','.npz')) <= 128*1024**2
    assert all_npy() <= 25*1024**3


def assert_sources(p):
    for name, expected in p['source_pins'].items():
        assert sha(name) == expected, name
    for name, expected in p['inputs'].items():
        st = Path(name).stat()
        assert st.st_size == expected['bytes'] and st.st_mtime_ns == expected['mtime_ns'], name


def prepare():
    assert not (ROOT/'PROTOCOL.json').exists()
    approval = read(ROOT/'COORDINATOR_PLAN_REVIEW.json')
    assert approval['status'] == 'PLAN_APPROVED'
    assert approval['plan_sha256'] == sha(ROOT/'PLAN.md')
    inv = read(ROOT/'INPUT_INVENTORY.json')
    d8 = read(D8/'PROTOCOL.json')
    assert inv['counts']['train']['fit_windows'] == 234988
    assert inv['counts']['val']['fit_windows'] == 28491
    checks = core.synthetic_check()
    checks['metric'] = metric.math_check()
    for windows in (42, 50, 409, 566, 900):
        for i in range(core.BURN, windows):
            for role, (a, b) in core.intervals(i).items():
                assert 0 <= a < b <= 96*(windows+1)
            assert core.intervals(i)['PastCov'][1] <= 96*(i-1)
        # Every contributor to warm primary blocks is post-burn-in.
        h = np.hanning(192)
        for start in (7680, 11520):
            if start+3840 > 96*(windows+1):
                continue
            for j in range(windows):
                coords = 96*j+np.flatnonzero(h>1e-8)
                if np.any((coords>=start)&(coords<start+3840)):
                    assert j >= core.BURN
    checks.update(warm_first_original_grid_start=7680, cold_first_block_start=0)
    dump(ROOT/'MATH_CHECK.json', checks)
    pins = dict(inv['source_pins'])
    for path in (Path(__file__), Path(core.__file__), Path(metric.__file__),
                 ROOT/'PLAN.md', ROOT/'COORDINATOR_PLAN_REVIEW.json',
                 ROOT/'COORDINATOR_NUMERICAL_REVIEW_SCOPE.json',
                 ROOT/'INPUT_INVENTORY.json', ROOT/'MATH_CHECK.json'):
        pins[str(path)] = sha(path)
    p = dict(frozen=True, seeds=SEEDS, arms=core.ARMS, cache_roles=core.CACHE_ROLES,
             score_roles=SCORE_ROLES, scores=24, new_ridge_fits=15,
             interventions=6, lambda_value=.01, burn=core.BURN,
             records=d8['records'], mask=d8['mask'], inputs=inv['inputs'],
             source_pins=pins, budget=inv['budget'],
             covariance_floor=1e-8, variance_relative_roundoff=1e-10,
             warm_block_minimum_start=4032, warm_first_grid_start=7680,
             cold_block_start=0, current_PCA32_TRAIN_support='D8_i>=4',
             new_factor_statistics_TRAIN_support='i>=41',
             NN_forwards=0, NN_updates=0, test_DTU=False, new_download=False,
             scientific_status='REPEATED_KNOWN_PERSON_TRAIN_VAL_DEVELOPMENT_ONLY',
             automatic_retry=False, parent='PUB-01-S03_SS07', patients='FORBIDDEN')
    assert_sources(p)
    guard()
    dump(ROOT/'PROTOCOL.json', p)
    print('D11_FROZEN_NOT_FITTED', flush=True)


def load_eeg(row, norm, prefix=False):
    raw = np.load(row['eeg_path'], mmap_mode='r')
    assert raw.shape[0] == 64
    n = 1920 if prefix else 96*(row['windows']+1)
    eeg = raw[:, :n].T
    result = ((np.asarray(eeg, np.float32)-norm['mean'])/norm['std']).astype(np.float32)
    assert result.shape == (n,64) and np.isfinite(result).all()
    return result.astype(np.float64)


def prefix_references(rows, norm):
    references, membership = {}, {}
    for j, row in enumerate(rows):
        k = core.covariance(load_eeg(row, norm, prefix=True))
        references[row['record']] = k
        membership.setdefault(row['subject'], []).append(row['record'])
        guard()
    totals = {person:sum(references[key] for key in keys) for person,keys in membership.items()}
    for person, keys in membership.items():
        assert len(keys) >= 2 and len(set(keys)) == len(keys)
    dump(ROOT/'PERSON_REFERENCE_PROVENANCE.json',
         dict(TRAIN_records_by_subject=membership,
              TRAIN_rule='equal-other-record covariance mean; leave current record out',
              VAL_rule='equal-all-same-person TRAIN record covariance mean',
              no_VAL_reference_fit=True))
    return references, membership, totals


def person_cov(row, phase, references, membership, totals):
    person = row['subject']
    if phase == 'train':
        keys = membership[person]
        assert row['record'] in keys
        return (totals[person]-references[row['record']])/(len(keys)-1)
    assert phase == 'val' and row['record'] not in references
    return totals[person]/len(membership[person])


def states(p):
    norm = np.load(OLD/'prepared/train_only_normalization.npz', allow_pickle=False)
    t = clock()
    references, membership, totals = prefix_references(p['records']['train'], norm)
    account('TRAIN_static_covariance_references', t, records=426)
    sums=np.zeros(2080); squares=sums.copy(); gram=np.zeros((2080,2080)); count=0
    t=clock()
    for j,row in enumerate(p['records']['train']):
        x=load_eeg(row,norm)
        parts=core.record_descriptors(x,row['windows'],
                 person_cov(row,'train',references,membership,totals),COUNTS)
        ss,tt,gg,nn=core.pooled_descriptor_moments(parts,row['windows']-core.BURN)
        sums+=ss; squares+=tt; gram+=gg; count+=nn
        if j%20==0: status(stage='TRAIN_state_moments',records=j+1,total=426)
        guard()
    assert count==4*234988
    st,pc=core.pca_from_moments(sums,squares,gram,count,8)
    save(ROOT/'STATE_TRAIN_TRANSFORMS.npz',sums=sums,squares=squares,gram=gram,
         n=np.array(count),basis=pc['basis'],eigenvalues=pc['eigenvalues'],**st)
    account('TRAIN_shared_descriptor_statistics_PCA8',t,pooled_windows=count,
            positive_components=pc['positive_components'])
    del gram, gg
    t=clock(); direct_checks=[]
    for phase,rows in p['records'].items():
        n=sum(row['windows'] for row in rows)
        caches={}
        for role in core.CACHE_ROLES:
            path=ROOT/f'{phase}_{role}.npy'
            assert not path.exists()
            caches[role]=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,8))
            caches[role][:]=0.
        for j,row in enumerate(rows):
            eeg=load_eeg(row,norm)
            parts=core.record_descriptors(eeg,row['windows'],
                    person_cov(row,phase,references,membership,totals),COUNTS)
            o,w=row['offset'],row['windows']
            if j in (0,len(rows)-1):
                for i in (core.BURN,w-1):
                    for role,(a,b) in core.intervals(i).items():
                        expected=core.descriptor(core.covariance(eeg[a:b]))
                        actual=parts[role] if parts[role].ndim==1 else parts[role][i-core.BURN]
                        difference=float(abs(expected-actual).max())
                        assert difference<=1e-8, 'Actual rolling/direct descriptor bridge'
                        direct_checks.append(dict(phase=phase,record=row['record'],window=i,
                            role=role,max_absolute=difference,tolerance=1e-8))
            for role,cache in caches.items():
                projected=core.project(parts[role],st,pc)
                cache[o+core.BURN:o+w]=projected
            if j%20==0: status(stage='PCA8_float32_cache',phase=phase,records=j+1)
            guard()
        for cache in caches.values():cache.flush()
        caches.clear()
    account('TRAIN_VAL_shared_state_cache',t,cache_files=10,payload_bytes=45272640)
    dump(ROOT/'STATE_NUMERICS.json',dict(counts=COUNTS,actual_direct_checks=direct_checks,PCA_eigenvalues=pc['eigenvalues'].tolist(),
                                       fit_phase='train_only',cache_quantization='float32 then FP64 for every downstream calculation'))


def load_states(phase):
    return {role:np.load(ROOT/f'{phase}_{role}.npy',mmap_mode='r') for role in core.CACHE_ROLES}


def frozen(seed):
    folder=D8/f'seed{seed}'
    z={phase:np.load(folder/f'{phase}_latent.npy',mmap_mode='r') for phase in ('train','val')}
    pp={phase:np.load(folder/f'{phase}_P.npy',mmap_mode='r') for phase in ('train','val')}
    f=np.load(folder/'TRAIN_TRANSFORMS.npz',allow_pickle=False)
    stats=[{k:f[f'{k}_{j}'] for k in ('mean','std','constant')} for j in range(3)]
    cal=read(folder/'P_FT_BASELINE.json')['calibration']
    co=np.load(folder/'Ccap_lambda0.01_COEFFICIENTS.npz',allow_pickle=False)
    assert co['coef'].shape==(3264,192) and cal['a']>0
    return z,pp,stats,cal,co


def anchor(row,z,pp,stats,cal,co):
    o,w=row['offset'],row['windows']
    baseline=cal['a']*np.asarray(pp[o:o+w],np.float64)+cal['b']
    zz=core.normalize(np.asarray(z[o+4:o+w],np.float64).reshape(w-4,1536),stats[0])
    pr=core.normalize(baseline[4:],stats[1])
    square=core.normalize(zz*zz,stats[2])
    x=np.concatenate((zz/np.sqrt(1536),pr/np.sqrt(192),square/np.sqrt(1536)),1)
    baseline[4:]+=x@co['coef']+co['bias']
    return baseline


def score_rows(predictions,rows,p):
    rr=[]
    for row,pred in zip(rows,predictions):
        target=np.load(row['envelope_path'],mmap_mode='r')
        blocks=[b for b in p['mask']['blocks'] if b['record']==row['record']]
        rr.append(metric.measure(pred,target,row,blocks))
        guard()
    result=metric.aggregate(rr)
    assert len(result['people'])==49 and len(result['segments'])==752
    subsets={}
    for key,condition in (('warm',lambda b:b['start']>=4032),
                          ('cold',lambda b:b['start']==0)):
        copied=[dict(r,segments=[b for b in r['segments'] if condition(b)]) for r in rr]
        sub=metric.aggregate(copied)
        # Only segment/person quantities belong to these fixed physical masks.
        # Whole-record window/overlap auxiliaries are deliberately omitted.
        subsets[key]=dict(people=sub['people'],segments=sub['segments'],
            phases={'val':{k:sub['phases']['val'][k] for k in
                    ('r','MSE','weighted_single_MSE','disagreement','people','blocks','records')}})
    return dict(full=result,subsets=subsets)


def bridge(seed,p):
    z,pp,stats,cal,co=frozen(seed)
    rows=p['records']['val']
    result=score_rows([anchor(row,z['val'],pp['val'],stats,cal,co) for row in rows],rows,p)
    old=read(D8/f'seed{seed}/Ccap_lambda0.01_METRICS.json')['metrics']
    delta=max(abs(a['r']-b['r']) for a,b in zip(result['full']['segments'],old['segments']))
    assert delta<=1e-10
    dump(ROOT/f'ANCHOR_CHECK_seed{seed}.json',dict(passed=True,block_r_max_abs=delta,
                                                 metrics=result,no_new_encoder=True))
    return result


def q_values(row,z,stats,pc):
    o,w=row['offset'],row['windows']
    return core.normalize(np.asarray(z[o+core.BURN:o+w],np.float64).reshape(-1,1536),stats[0])@pc['basis']


def h_values(row,caches,role,st):
    o,w=row['offset'],row['windows']
    return core.normalize(np.asarray(caches[role][o+core.BURN:o+w],np.float64),st)


def seed_fit(seed,p,anchor_metrics):
    folder=ROOT/f'seed{seed}'
    folder.mkdir(exist_ok=False)
    z,pp,stats,cal,co=frozen(seed)
    caches={phase:load_states(phase) for phase in ('train','val')}
    t=clock()
    with np.load(D8/f'seed{seed}/TRAIN_MOMENTS.npz',allow_pickle=False) as old:
        assert int(old['n'])==250750
        g=old['G']; cov=g[:1536,:1536]*1536
    pc=core.principal(cov,32)
    del g,cov
    n=234988
    qs=np.zeros(32); qt=qs.copy(); hs=np.zeros(8); ht=hs.copy()
    for row in p['records']['train']:
        q=q_values(row,z['train'],stats,pc)
        qs+=q.sum(0);qt+=(q*q).sum(0)
        o,w=row['offset'],row['windows']
        for role in core.STATE_ROLES:
            h=np.asarray(caches['train'][role][o+core.BURN:o+w],np.float64)
            hs+=h.sum(0);ht+=(h*h).sum(0)
        guard()
    qst=core.stats_finish(qs,qt,n)
    hst=core.stats_finish(hs,ht,4*n)
    ps=np.zeros(256);pt=ps.copy()
    for row in p['records']['train']:
        q=core.normalize(q_values(row,z['train'],stats,pc),qst)
        for role in core.STATE_ROLES:
            h=h_values(row,caches['train'],role,hst)
            products=(q[:,:,None]*h[:,None,:]).reshape(len(q),256)
            ps+=products.sum(0);pt+=(products*products).sum(0)
        guard()
    pst=core.stats_finish(ps,pt,4*n)
    save(folder/'TRAIN_TRANSFORMS.npz',current_basis=pc['basis'],current_eigenvalues=pc['eigenvalues'],
         **{f'{a}_{key}':value for a,s in [('q',qst),('h',hst),('product',pst)] for key,value in s.items()})
    account('TRAIN_q_h_product_statistics',t,seed=seed,windows=n)
    moments={}
    for role in core.ARMS:
        d=32 if role=='Add' else 296
        moments[role]=dict(G=np.zeros((d,d)),H=np.zeros((d,192)),xs=np.zeros(d),rs=np.zeros(192))
    yy=np.load(OLD/'envelope_readout_seed1024/train_envelope.npy',mmap_mode='r')
    t=clock()
    for j,row in enumerate(p['records']['train']):
        o,w=row['offset'],row['windows']
        residual=np.asarray(yy[o+core.BURN:o+w,0],np.float64)-anchor(row,z['train'],pp['train'],stats,cal,co)[core.BURN:]
        q=core.normalize(q_values(row,z['train'],stats,pc),qst)
        for role,m in moments.items():
            h=None if role=='Add' else h_values(row,caches['train'],role,hst)
            x=core.features(q,h,pst)
            m['G']+=x.T@x;m['H']+=x.T@residual;m['xs']+=x.sum(0);m['rs']+=residual.sum(0)
        if j%40==0:status(stage='TRAIN_residual_Gram',seed=seed,records=j+1)
        guard()
    coefficients={}
    for role,m in moments.items():
        xm,rm=m['xs']/n,m['rs']/n
        G=m['G']/n-np.outer(xm,xm);H=m['H']/n-np.outer(xm,rm)
        B,b,proof=core.ridge(G,H,xm,rm)
        save(folder/f'{role}_COEFFICIENTS.npz',coef=B,bias=b)
        save(folder/f'{role}_TRAIN_MOMENTS.npz',G=G,H=H,x_mean=xm,residual_mean=rm,n=np.array(n))
        coefficients[role]=(B,b)
        dump(folder/f'{role}_NORMAL_CHECK.json',proof)
    account('TRAIN_15_fixed_ridge_per_seed5',t,seed=seed,fits=5,NN_updates=0)
    t=clock();byrole={role:[] for role in SCORE_ROLES}
    intervention_checks=[]
    for row in p['records']['val']:
        baseline=anchor(row,z['val'],pp['val'],stats,cal,co)
        q=core.normalize(q_values(row,z['val'],stats,pc),qst)
        byrole['Ccap'].append(baseline)
        current_before=q.copy()
        for role in core.ARMS+('PastOlder','PastNeutral'):
            model_role='PastCov' if role.startswith('Past') else role
            if role=='Add': h=None
            elif role=='PastNeutral': h=np.zeros((len(q),8))
            else:
                state_role='OlderCov' if role=='PastOlder' else role
                h=h_values(row,caches['val'],state_role,hst)
            x=core.features(q,h,pst);B,b=coefficients[model_role]
            pred=baseline.copy();pred[core.BURN:]+=x@B+b
            assert np.array_equal(pred[:core.BURN],baseline[:core.BURN])
            assert np.array_equal(q,current_before)
            byrole[role].append(pred)
        intervention_checks.append(dict(record=row['record'],current_q_unchanged=True,
                                       baseline_and_cold_unchanged=True))
        guard()
    allmetrics={}
    for role,predictions in byrole.items():
        if role=='Ccap':
            result=anchor_metrics
        else:
            result=score_rows(predictions,p['records']['val'],p)
        full=result['full'];v=full['phases']['val']
        for a,b in zip(result['subsets']['cold']['segments'],anchor_metrics['subsets']['cold']['segments']):
            assert a['r']==b['r'] and a['MSE']==b['MSE']
        dump(folder/f'{role}_METRICS.json',dict(seed=seed,role=role,lambda_value=.01,
                                               **result))
        allmetrics[role]=dict(**v,
             Hann_identity_max_abs=full['Hann_identity_max_abs'],
             warm_r=result['subsets']['warm']['phases']['val']['r'],
             cold_r=result['subsets']['cold']['phases']['val']['r'])
    dump(folder/'INTERVENTION_CHECK.json',dict(passed=True,records=intervention_checks))
    account('VAL_8_fixed_roles',t,seed=seed,roles=8,windows=30500,people=49,blocks=752)
    dump(folder/'COMPLETED.json',dict(seed=seed,new_fits=5,interventions=2,results=allmetrics))
    return allmetrics


def run():
    p=read(ROOT/'PROTOCOL.json')
    assert p['frozen'] and p['new_ridge_fits']==15 and p['scores']==24
    assert_sources(p);guard()
    assert not (ROOT/'STARTED.json').exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60min cap')))
    signal.alarm(3600)
    dump(ROOT/'STARTED.json',dict(pid=os.getpid(),time=time.time(),source_sha=sha(__file__),
                                 protocol_sha=sha(ROOT/'PROTOCOL.json')))
    t=clock()
    bridges={seed:bridge(seed,p) for seed in SEEDS}
    account('Ccap_full_VAL_bridge_before_state_fit',t,seeds=3)
    status(stage='ANCHOR_BRIDGES_PASSED',new_fits=0)
    states(p)
    results={}
    for seed in SEEDS:
        status(stage='seed_ridge',seed=seed)
        results[str(seed)]=seed_fit(seed,p,bridges[seed])
    assert_sources(p);guard()
    pairs=[('PastCov','Ccap'),('PastCov','CurrentCov'),('PastCov','RecordPrefix'),
           ('PastCov','PersonTRAIN'),('PastCov','PastOlder'),('PastCov','PastNeutral')]
    contrasts={}
    for a,b in pairs:
        deltas=[results[str(seed)][a]['r']-results[str(seed)][b]['r'] for seed in SEEDS]
        contrasts[f'{a}-{b}']=dict(seed_differences=deltas,mean=float(np.mean(deltas)),
                                  all3positive=all(v>0 for v in deltas))
    dump(ROOT/'SUMMARY.json',dict(results=results,primary_contrasts=contrasts,
         condition_met=all(v['all3positive'] for v in contrasts.values()),
         claim_scope='selected8D_second_order_state_conditional_information_only',
         scientific_status=p['scientific_status'],goal_complete=False))
    dump(ROOT/'COST.json',dict(stages=COST,wall_seconds=time.time()-START,
         CPU_seconds=time.process_time()-CPU,new_ridge_fits=15,
         nofit_interventions=6,VAL_role_seed_scores=24,NN_forwards=0,NN_updates=0))
    dump(ROOT/'COMPLETED.json',dict(new_ridge_fits=15,wall_seconds=time.time()-START,
         CPU_seconds=time.process_time()-CPU,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
         all_SS07_NPY_bytes=all_npy(),new_root_bytes=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file()),
         free_disk_bytes=shutil.disk_usage(ROOT).free,test_DTU=False,goal_complete=False))
    status(stage='COMPLETED',new_fits=15,goal_complete=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','run'))
    args=parser.parse_args()
    try:
        prepare() if args.action=='prepare' else run()
    except BaseException:
        if not (ROOT/'FAILURE.json').exists():
            dump(ROOT/'FAILURE.json',dict(action=args.action,time=time.time(),
                 traceback=traceback.format_exc(),automatic_retry=False))
        raise
