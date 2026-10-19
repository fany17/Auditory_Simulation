"""Post-execution review, raw-input CPU reconstruction and grouped paired statistics."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from pub_01_s03_ss05_data import ROOT,dump,split,oriented
from pub_01_s03_ss05_finite_history import FiniteHistoryDecoder
from pub_01_s03_ss05_train import digest,check_sources


def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def save(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def correlation(x,y):
    a=x.astype(np.float64).reshape(-1).copy();b=y.astype(np.float64).reshape(-1).copy();a-=a.mean();b-=b.mean()
    return float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))


def raw_windows(record,indices,scale):
    eeg=oriented(record)
    return np.stack([((np.asarray(eeg[int(i)*192:(int(i)+1)*192],dtype=np.float32)-scale['mean'])/scale['std']).astype(np.float32) for i in indices])


def independent_cpu(model,record,all_records,scale,lag,expected):
    # First60s query block, window40..59; rawinput not the new normalized cache.
    q=np.arange(40,60);x=raw_windows(record,np.arange(60),scale)
    with torch.no_grad():
        local,tokens=model.encode(torch.tensor(x)[None]);local=local[:,40:60];current=tokens[:,40:60]
        recent=tokens[:,torch.tensor(q[:,None]+np.arange(-20,0))[None][0]]
        candidates=[rid for rid,r in all_records.items() if rid!=record['record_id'] and r['window_count']>=20 and r['subject']!=record['subject'] and r['source_family']!=record['source_family']]
        if not candidates:candidates=[rid for rid,r in all_records.items() if rid!=record['record_id'] and r['window_count']>=20]
        donor=all_records[sorted(candidates)[record['record_id']%len(candidates)]]
        starts=(q-20)%(donor['window_count']-20+1);idx=starts[:,None]+np.arange(20)
        unique,inverse=np.unique(idx,return_inverse=True);wx=raw_windows(donor,unique,scale)
        _,wt=model.encode(torch.tensor(wx)[None]);wrong=wt[:,torch.tensor(inverse).reshape(20,20)]
        prediction={'true':model.queries(local,current,recent).numpy(),
            'reset':model.queries(local,current,recent,reset=True).numpy(),
            'wrong_record':model.queries(local,current,wrong).numpy()}
    env=np.load(record['envelope_path'],mmap_mode='r').reshape(-1);truth=np.asarray(env[40*192-lag:60*192-lag],dtype=np.float32)
    return [abs(correlation(value,truth)-float(expected[mode]['pearson_r'])) for mode,value in prediction.items()]


def grouped_bootstrap(record_deltas,records,subjects,families,draws=10000):
    si={s:i for i,s in enumerate(subjects)};fi={f:i for i,f in enumerate(families)}
    sums=np.zeros((len(subjects),len(families)));counts=np.zeros_like(sums)
    for rid,value in record_deltas.items():
        r=records[rid];i,j=si[r['subject']],fi[r['source_family']];sums[i,j]+=value;counts[i,j]+=1
    person=(sums.sum(1)/counts.sum(1));rng=np.random.default_rng(20261019)
    bsubject=person[rng.integers(0,len(person),size=(draws,len(person)))].mean(1)
    crossed=[]
    while sum(len(v) for v in crossed)<draws:
        remaining=draws-sum(len(v) for v in crossed)
        ws=rng.multinomial(len(subjects),np.ones(len(subjects))/len(subjects),size=remaining).T
        wf=rng.multinomial(len(families),np.ones(len(families))/len(families),size=remaining).T
        den=counts@wf;values=np.divide(sums@wf,den,out=np.zeros_like(den,dtype=float),where=den>0)
        weights=ws*(den>0);support=weights.sum(0);valid=support>0
        crossed.append(((values[:,valid]*weights[:,valid]).sum(0)/support[valid]))
    bcross=np.concatenate(crossed)[:draws]
    tail=.05/(2*6)*100
    return {'mean_delta_r':float(person.mean()),'subjects':len(subjects),'families':len(families),'persons_positive':int((person>0).sum()),
        'subject_CI95_low':float(np.percentile(bsubject,2.5)),'subject_CI95_high':float(np.percentile(bsubject,97.5)),
        'crossed_CI95_low':float(np.percentile(bcross,2.5)),'crossed_CI95_high':float(np.percentile(bcross,97.5)),
        'crossed_family95_low':float(np.percentile(bcross,tail)),'crossed_family95_high':float(np.percentile(bcross,100-tail))}


def main():
    status=json.loads((ROOT/'formal_status.json').read_text());assert status['stage']=='ACTUAL_RESULTS_COMPLETE'
    out=ROOT/'review01';out.mkdir(exist_ok=False);protocol=json.loads((ROOT/'FORMAL_PROTOCOL.json').read_text());check_sources(protocol)
    metadata=split();fit=json.loads((ROOT/'normalization_contract.json').read_text());train_ids={r['record'] for r in metadata['records'] if r['phase']=='train'}
    hold_ids={r['record'] for r in metadata['records'] if r['phase']=='holdout'}
    assert set(fit['fit_record_ids'])==train_ids and not train_ids&hold_ids and not fit['old_normalization_used']
    assert digest(ROOT/'newtrain_normalization.npz')==protocol['input_contract_sha256']['normalizer']
    cache=json.loads((ROOT/'cache_holdout/manifest.json').read_text());records={r['record_id']:r for r in cache['records']};subjects=sorted({s['subject'] for s in cache['segments']});families=sorted({s['source_family'] for s in cache['segments']})
    assert len(records)==47 and len(subjects)==21 and len(families)==7 and not cache['excluded_records']
    assert {r['record'] for r in records.values()}==hold_ids
    assert protocol['frozen_utc_epoch']<json.loads((ROOT/'holdout_opened.json').read_text())['utc_epoch']
    assert (ROOT/'cache_holdout/completed.json').stat().st_mtime>max((ROOT/'formal01'/f'{v}_seed{s}'/'train_completed.json').stat().st_mtime for v in protocol['variants'] for s in protocol['seeds'])
    scale=np.load(ROOT/'newtrain_normalization.npz');torch.set_num_threads(4);results=[];all_record=[];all_person=[];train_results=[];errors=[];cpu_errors=[]
    for variant in protocol['variants']:
        for seed in protocol['seeds']:
            run=ROOT/'formal01'/f'{variant}_seed{seed}';test=run/'holdout01';train=json.loads((run/'train_completed.json').read_text());history=json.loads((run/'history.json').read_text());result=json.loads((test/'result.json').read_text())
            selected=max(history,key=lambda r:r['validation_record_then_subject_macro_60s_r'])
            assert train['optimizer_steps']==2000 and train['best_step']==selected['step']==result['selected_step']
            assert [r['step'] for r in history]==list(range(200,2001,200)) and not train['holdout_access'] and result['test_evaluations']==1
            assert train['protocol_sha256']==result['protocol_sha256']==digest(ROOT/'FORMAL_PROTOCOL.json')
            assert all(value is True or value is None for value in result['canary'].values())
            segments=read(test/'per_segment.csv');person_rows=read(test/'per_subject.csv');record_rows=read(test/'per_record.csv')
            expected_keys={(s['record_id'],s['start_window']) for s in cache['segments']}
            for condition in protocol['history_conditions']:
                s=[r for r in segments if r['condition']==condition];rr=[r for r in record_rows if r['condition']==condition];pp=[r for r in person_rows if r['condition']==condition]
                assert len(s)==len(cache['segments']) and {(int(r['record_id']),int(r['start_window'])) for r in s}==expected_keys
                assert {int(r['record_id']) for r in rr}==set(records) and {r['subject'] for r in pp}==set(subjects)
                for record in rr:
                    values=[float(r['pearson_r']) for r in s if r['record_id']==record['record_id']];errors.append(abs(np.mean(values)-float(record['pearson_r'])))
                for person in pp:
                    values=[float(r['pearson_r']) for r in rr if r['subject']==person['subject']];errors.append(abs(np.mean(values)-float(person['pearson_r'])))
                macro=float(np.mean([float(r['pearson_r']) for r in pp]));errors.append(abs(macro-result['conditions'][condition]['subject_macro_r']))
                results.append({'variant':variant,'seed':seed,'condition':condition,'subjects':len(subjects),'records':len(records),'source_families':len(families),
                    'pearson_r':macro,'optimizer_steps':2000,'selected_step':train['best_step'],'lag_samples':protocol['lag_samples'],'status':'ACTUAL_HOLDOUT_TEST'})
            all_record.extend([dict(variant=variant,seed=seed,**r) for r in record_rows]);all_person.extend([dict(variant=variant,seed=seed,**r) for r in person_rows])
            train_results.append({'variant':variant,'seed':seed,'seconds':train['seconds'],'best_validation_r':train['best_validation_r'],'best_step':train['best_step']})
            checkpoint=torch.load(run/'best_validation.pt',map_location='cpu',weights_only=True);model=FiniteHistoryDecoder(variant);model.load_state_dict(checkpoint['model']);model.eval()
            for rid in (min(records),max(records)):
                expected={m:next(r for r in segments if int(r['record_id'])==rid and int(r['start_window'])==40 and r['condition']==m) for m in ('true','reset','wrong_record')}
                cpu_errors.extend(independent_cpu(model,records[rid],records,scale,protocol['lag_samples'],expected))
    assert max(errors)<1e-10 and max(cpu_errors)<1e-4
    summary=[]
    for variant in protocol['variants']:
        for condition in protocol['history_conditions']:
            vals=[r['pearson_r'] for r in results if r['variant']==variant and r['condition']==condition]
            summary.append({'variant':variant,'condition':condition,'mean_r':float(np.mean(vals)),'seed_SD':float(np.std(vals,ddof=1))})
    def means(v,c):return {rid:float(np.mean([float(r['pearson_r']) for r in all_record if r['variant']==v and r['condition']==c and int(r['record_id'])==rid])) for rid in records}
    primary=[('assoc_no_pred_minus_state','assoc_no_pred','true','state','true'),('assoc_no_pred_minus_gru','assoc_no_pred','true','gru','true'),
        ('full_minus_assoc_no_pred','full','true','assoc_no_pred','true'),('full_minus_gru','full','true','gru','true'),
        ('assoc_true_minus_wrong_record','assoc_no_pred','true','assoc_no_pred','wrong_record'),('full_true_minus_wrong_record','full','true','full','wrong_record')]
    contrasts=[];secondary=[]
    for name,a,ac,b,bc in primary:
        aa,bb=means(a,ac),means(b,bc);stat=grouped_bootstrap({rid:aa[rid]-bb[rid] for rid in records},records,subjects,families)
        contrasts.append(dict(contrast=name,primary=True,support_both_subject_and_crossed95=bool(stat['subject_CI95_low']>0 and stat['crossed_family95_low']>0),**stat))
    for variant in protocol['variants']:
        a=means(variant,'true')
        for condition in ('reset','wrong_time','shuffled','replace_high_similarity','replace_low_similarity'):
            b=means(variant,condition);stat=grouped_bootstrap({rid:a[rid]-b[rid] for rid in records},records,subjects,families)
            secondary.append(dict(variant=variant,contrast='true_minus_'+condition,primary=False,**stat))
    save(out/'RESULTS.csv',results);save(out/'SUMMARY.csv',summary);save(out/'per_record_all_seeds.csv',all_record);save(out/'per_subject_all_seeds.csv',all_person)
    save(out/'primary_contrasts.csv',contrasts);save(out/'history_deltas.csv',secondary);save(out/'training_summary.csv',train_results)
    decision={r['contrast']:r['support_both_subject_and_crossed95'] for r in contrasts}
    review={'technical':'PASS','acceptance':'PASS_WITH_LIMITATION','parent_consistency':'PASS_SS05_ONLY_PUBLIC_INTERNAL_HOLDOUT',
        'same_agent_separate_pass':True,'third_party_review':False,'runs':15,'test_conditions':len(protocol['history_conditions']),
        'maximum_aggregation_error':max(errors),'raw_EEG_CPU_60s_correlations_checked':len(cpu_errors),'maximum_raw_CPU_r_error':max(cpu_errors),
        'newtrain_only_normalization_verified':True,'holdout_family_disjoint':True,'old_exposed_test_payloads_used':False,'same_frozen_sources':True,
        'prospective_internal_not_external':True,'primary_support_flags':decision,
        'bootstrap':'10000 draws; first average seeds perrecord; record-equal/person-equal; subject CI and crossed independent subject/family multinomial weights; six primary contrasts Bonferroni99.1667% intervals',
        'limits':['7 heldout families/21 people; small grouped uncertainty','source previously used in SS03train','20window bounded context, not indefinite streaming',
            '3s centered-Pearson objective does not calibrate window mean/scale','full residual is not supervised forecasting error','active parameter/runtime differences disclosed',
            'EEG token similarity is not event relevance groundtruth','physical_sync UNKNOWN','no direct SS02/SS03 metric ranking']}
    dump(out/'review.json',review);print(json.dumps({'technical':'PASS','primary':contrasts,'summary_true':[r for r in summary if r['condition']=='true'],'max_CPU_error':max(cpu_errors)}),flush=True)


if __name__=='__main__':main()
