"""Predeclared D14 verification; execute on 2203 after all nine fits finish.

Manual functional GRU and pairwise Hann code from the coordinator's earlier
audit, never production imports. Check selected, last19, and selected H older;
same selected/last state reuses one prediction. CPU only, no new fits.
"""
import hashlib,json,time,resource
from pathlib import Path
import numpy as np
import torch
import pub_01_s03_ss07_full_ft_history_independent_review as a

ROOT=a.BASE/'s03_full_ft_order_control_20261102'
OUT=ROOT/'coordinator_review'
PIN='83b05eafcb126952d48750930106b5fb04c56d49f82d2b0128ba7d2c813d0172'

def state_hash(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def audit_offline(folder):
    for name in ['OLD_LAST19_TRAIN_REFERENCE.json']+[f'OFFLINE_TRAIN_epoch{x}.json' for x in [-1,0,4,9,19]]:
        d=a.read(folder/name);assert d['parameters_fixed'] and d['windows']==252454
        assert len(d['people'])==70 and len(d['blocks'])==6104
        assert np.isfinite(d['window_MSE']) and d['window_MSE']>=0
        for person in d['people']:
            bb=[b for b in d['blocks'] if b['subject']==person['subject']]
            assert person['blocks']==len(bb)
            for k in ['r','MSE']:
                assert abs(np.mean([b[k] for b in bb])-person[k])<1e-12
        for k in ['r','MSE']:
            assert abs(np.mean([p[k] for p in d['people']])-d[f'physical60s_{k}'])<1e-12
    return 6

def run():
    started,cpu=time.time(),time.process_time();torch.set_num_threads(4)
    assert a.sha(ROOT/'PROTOCOL.json')==PIN
    done=a.read(ROOT/'QUEUE_COMPLETED.json');assert done['formal_residual_NN_windows']==61710828
    assert done['total_updates']==710100 and len(done['workers'])==9
    assert all(x['returncode']==0 for x in done['workers'])
    p=a.read(ROOT/'PROTOCOL.json')
    for path,digest in p['source_pins'].items():assert a.sha(path)==digest,path
    OUT.mkdir(exist_ok=False)
    a.write(OUT/'STARTED.json',dict(protocol_sha256=PIN,source_sha256=a.sha(__file__),
        helper_sha256=a.sha(a.__file__),device='CPU',threads=4,max_NN_windows=21*30059,
        evaluations='All9 selected, all9 last19 (reuse if equal state), selected H older3',
        new_encoder_head=0,optimizer_updates=0,learning_model_fits=0,tolerances=a.TOLERANCES))
    old=a.read(a.D8/'PROTOCOL.json');rows=old['records']['val'];mask=old['mask']['blocks']
    summaries=[];offline_files=0;old_crosschecks=0;permutation_logs=0;reused=0
    for seed in a.SEEDS:
        z,anchor,prior=a.inputs(seed,rows)
        expected_orders={x['epoch']:x['sha256'] for x in p['permutations'] if x['seed']==seed}
        assert len(expected_orders)==20
        for epoch,digest in expected_orders.items():
            order=np.random.default_rng(np.random.SeedSequence([seed,20261102,epoch])).permutation(252454)
            assert np.array_equal(np.sort(order),np.arange(252454))
            assert hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()==digest
        for arm in a.ARMS:
            folder=ROOT/f'fits/{arm}_seed{seed}';completed=a.read(folder/'COMPLETED.json')
            hist=[json.loads(l) for l in (folder/'HISTORY.jsonl').read_text().splitlines()]
            orders=[json.loads(l) for l in (folder/'PERMUTATIONS.jsonl').read_text().splitlines()]
            assert len(hist)==len(orders)==20 and completed['updates']==78900
            for epoch,(record,order) in enumerate(zip(hist,orders)):
                assert record['epoch']==order['epoch']==epoch
                assert record['updates']==3945*(epoch+1)
                assert order['sha256']==expected_orders[epoch] and order['fallback_presentations']==3834
                assert record['effective_TRAIN_windows']==248620 and record['full_target_windows']==252454
                permutation_logs+=1
            incumbent=a.read(folder/'INCUMBENT.json')['phases']['val']['r']
            selected=int(np.argmax([incumbent]+[h['validation']['r'] for h in hist]))-1
            assert selected==completed['selected_epoch']
            ss=torch.load(folder/'best_validation.pt',map_location='cpu',weights_only=True)
            ll=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
            for cp in [ss,ll]:assert cp['seed']==seed and cp['arm']==arm
            assert ss['epoch']==selected and ll['epoch']==19
            pred=a.predictions(ss['model'],arm,rows,z,anchor,prior)
            metrics=a.score(pred,rows,mask)
            a.compare_metrics(f'{arm}/{seed}/selected',metrics,a.read(folder/'SELECTED_METRICS.json'))
            a.write(OUT/f'{arm}_seed{seed}_selected.json',metrics)
            if state_hash(ss['model'])==state_hash(ll['model']):
                last_metrics=metrics;reused+=1
            else:
                last_pred=a.predictions(ll['model'],arm,rows,z,anchor,prior)
                last_metrics=a.score(last_pred,rows,mask);del last_pred
            a.compare(f'{arm}/{seed}/last19',last_metrics['phases']['val'],hist[-1]['validation'])
            a.write(OUT/f'{arm}_seed{seed}_last19.json',last_metrics)
            if arm=='H_full':
                older=a.predictions(ss['model'],arm,rows,z,anchor,prior,older=True)
                om=a.score(older,rows,mask)
                a.compare_metrics(f'{arm}/{seed}/older',om,a.read(folder/'OLDER_METRICS.json'))
                a.write(OUT/f'{arm}_seed{seed}_older.json',om);del older
            offline_files+=audit_offline(folder)
            d13=a.read(a.BASE/f's03_full_ft_error_decomposition_20261102/results/seed{seed}/SUMMARY.json')
            reference=a.read(folder/'OLD_LAST19_TRAIN_REFERENCE.json')
            oldmetric=next(x for x in d13 if x['phase']=='train' and x['arm']==arm)
            for k,new in [('r_new','physical60s_r'),('MSE_new','physical60s_MSE')]:
                assert abs(oldmetric[k]-reference[new])<2e-7;old_crosschecks+=1
            summaries.append(dict(seed=seed,arm=arm,selected_epoch=selected,
                selected_r=metrics['phases']['val']['r'],last19_r=last_metrics['phases']['val']['r'],
                selected_state_sha256=state_hash(ss['model']),last_state_sha256=state_hash(ll['model'])))
            print(json.dumps(summaries[-1]),flush=True)
            del ss,ll,pred,metrics,last_metrics
        del z,anchor,prior
    assert a.COUNT['residual_NN_window_forwards']==(21-reused)*30059
    result=dict(passed=not a.DIFFERENCES,source_sha256=a.sha(__file__),helper_sha256=a.sha(a.__file__),
        protocol_sha256=PIN,checks=a.CHECKS,failures=a.DIFFERENCES,summaries=summaries,
        all180_actual_permutation_logs_checked=permutation_logs,offline_TRAIN_aggregation_files=offline_files,
        old_last_TRAIN_D13_crosschecks=old_crosschecks,selected_equals_last_reused=reused,
        cost=dict(**a.COUNT,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-started,
                  peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,GPU_bytes=0),
        limits=['No second complete TRAIN neural forward for offline window MSE',
                'No replay of unretained intermediate epochs','Original repeatedly viewed VAL, not independent confirmation'],
        goal_complete=False)
    a.write(OUT/'REVIEW.json',result);assert result['passed'],a.DIFFERENCES
if __name__=='__main__':run()
