"""Independent post-execution score recomputation and original-model raw CPU pass.

This file is an observer, added after the training freeze; it never updates a
model or chooses a checkpoint, preprocessing, statistic or audit condition.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from m6a_public.continuity.execution import LocalDecoder,MatchedDense
from m6a_public.continuity.dense_state import PersistentStateDecoder
from m6a_public.continuity.predictive import PredictiveContinuityDecoder
from m6a_public.continuity.alignment_innovation import AdaptiveAlignmentDecoder,InnovationStateDecoder
from pub_01_s03_ss06_data import ROOT,LENGTH,WARMUP,Records,oriented,dump,read
from pub_01_s03_ss06_contract import verify_all_training,identities,digest


def table(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def number(value):return None if value in ('',None) else float(value)


def save(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def original_model(name):
    if name=='local':return LocalDecoder()
    if name.startswith('m1_'):return MatchedDense(name[3:])
    if name=='m2':return PersistentStateDecoder()
    if name in ('gru','m3_frozen','m3_finetune'):
        return PredictiveContinuityDecoder(mode={'gru':'no_pretrain','m3_frozen':'frozen','m3_finetune':'finetune'}[name])
    if name.startswith('m4_'):return AdaptiveAlignmentDecoder(mode=name[3:])
    return InnovationStateDecoder(mode={'m5_eeg':'eeg_only','m5_prior':'prior_only','m5_full':'full'}[name])


def raw_input(row,first,last,scale):
    x=oriented(row)
    return torch.from_numpy(((np.asarray(x[first+8:last+8],np.float64)-scale['mean'][:64])/scale['std'][:64]).astype(np.float32))[None]


def zero(tree):
    if isinstance(tree,torch.Tensor):return torch.zeros_like(tree)
    if isinstance(tree,dict):return {k:zero(v) for k,v in tree.items()}
    return tree


@torch.no_grad()
def raw_cpu_conditions(name,checkpoint,record,all_records):
    model=original_model(name).eval();model.load_state_dict(checkpoint['model'])
    scale=np.load(ROOT/'newtrain_EEG_target_normalization.npz')
    ordered=sorted(all_records,key=lambda r:r['record']);index=next(i for i,r in enumerate(ordered) if r['record']==record['record'])
    eligible=[r for r in ordered if r['record']!=record['record'] and r['subject']!=record['subject'] and r['source_family']!=record['source_family']]
    donor=eligible[index%len(eligible)]
    real=model(raw_input(record,0,WARMUP,scale)).state
    wrong=model(raw_input(donor,0,WARMUP,scale)).state
    reset={k:v if k in ('encoder','mode') else zero(v) for k,v in real.items()}
    mismatch={k:v if k in ('encoder','mode') else wrong[k] for k,v in real.items()}
    current=raw_input(record,WARMUP,WARMUP+LENGTH,scale)
    envelope=np.load(record['envelope_path'],mmap_mode='r').reshape(-1)
    target=(np.asarray(envelope[WARMUP:WARMUP+LENGTH],np.float64)-scale['mean'][-1])/scale['std'][-1]
    target=target.astype(np.float32).astype(np.float64);target-=target.mean()
    result={}
    for condition,state in (('history_true',real),('history_state_reset',reset),('history_state_wrong',mismatch)):
        p=model(current,state).prediction.numpy().reshape(-1).astype(np.float64)
        raw_y=(np.asarray(envelope[WARMUP:WARMUP+LENGTH],np.float64)-scale['mean'][-1])/scale['std'][-1]
        raw_y=raw_y.astype(np.float32).astype(np.float64)
        pc=p-p.mean();sd=float(np.sqrt(np.mean(pc**2)))
        result[condition]={'r':float(np.corrcoef(p,raw_y)[0,1]) if sd>1e-6 else None,'mse':float(np.mean((p-raw_y)**2))}
    return result


def bootstrap(deltas,metadata,primary_count=8,draws=10000):
    subjects=sorted({r['subject'] for r in metadata.values()});families=sorted({r['source_family'] for r in metadata.values()})
    si={s:i for i,s in enumerate(subjects)};fi={f:i for i,f in enumerate(families)}
    sums=np.zeros((len(subjects),len(families)));counts=np.zeros_like(sums)
    for rid,d in deltas.items():
        r=metadata[rid];i,j=si[r['subject']],fi[r['source_family']];sums[i,j]+=d;counts[i,j]+=1
    person=sums.sum(1)/counts.sum(1);rng=np.random.default_rng(20261023)
    subject=person[rng.integers(len(subjects),size=(draws,len(subjects)))].mean(1);crossed=[];length=0
    while length<draws:
        ws=rng.multinomial(len(subjects),np.ones(len(subjects))/len(subjects),size=draws-length).T
        wf=rng.multinomial(len(families),np.ones(len(families))/len(families),size=draws-length).T
        den=counts@wf;values=np.divide(sums@wf,den,out=np.zeros_like(den),where=den>0)
        weights=ws*(den>0);support=weights.sum(0);valid=support>0
        values=(values[:,valid]*weights[:,valid]).sum(0)/support[valid];crossed.append(values);length+=len(values)
    cross=np.concatenate(crossed)[:draws];tail=.05/(2*primary_count)*100
    return {'mean_delta_r':float(person.mean()),'subjects':len(subjects),'families':len(families),
        'subject_CI95_low':float(np.percentile(subject,2.5)),'subject_CI95_high':float(np.percentile(subject,97.5)),
        'crossed_CI95_low':float(np.percentile(cross,2.5)),'crossed_CI95_high':float(np.percentile(cross,97.5)),
        'crossed_family95_low':float(np.percentile(cross,tail)),'crossed_family95_high':float(np.percentile(cross,100-tail)),
        'positive_subjects':int((person>0).sum())}


def main():
    torch.set_num_threads(1);gate=verify_all_training(allow_audit=True)
    out=ROOT/'review01';out.mkdir(exist_ok=False);protocol=read(ROOT/'FORMAL_PROTOCOL.json')
    opened=read(ROOT/'audit_opened.json');assert opened['protocol_sha256']==digest(ROOT/'FORMAL_PROTOCOL.json')
    latest=max((ROOT/'formal01'/f'{v}_seed{s}'/'train_completed.json').stat().st_mtime for v in protocol['variants'] for s in protocol['seeds'])
    latest=max(latest,max((ROOT/'formal01'/f'pretrain_seed{s}'/'pretrain_completed.json').stat().st_mtime for s in protocol['seeds']))
    assert opened['epoch']>latest and protocol['freeze_epoch']<opened['epoch']
    split=read(ROOT/'SPLIT_PROVENANCE.json');fit=read(ROOT/'normalization_contract.json');data=Records('audit_holdout')
    metadata={r['record']:r for r in data.rows};subjects={r['subject'] for r in data.rows};families={r['source_family'] for r in data.rows}
    assert len(metadata)==41 and len(subjects)==36 and len(families)==5 and sum(r['score_blocks'] for r in data.rows)==538
    assert set(fit['fit_records'])=={r['record'] for r in split['records'] if r['phase']=='train'}
    all_records=[];all_persons=[];results=[];training=[];aggregate_errors=[];cpu_r_errors=[];cpu_mse_errors=[];cpu_nulls=0
    for variant in protocol['variants']:
        for seed in protocol['seeds']:
            folder=ROOT/'formal01'/f'{variant}_seed{seed}';test=folder/'audit01'
            result=read(test/'result.json');train=read(folder/'train_completed.json');checkpoint=torch.load(folder/'best_validation.pt',map_location='cpu',weights_only=True)
            assert result['audit_assessments']==1 and result['selected_step']==train['selected_step']==checkpoint['step']
            assert all(all(v is True for k,v in r.items() if k!='record') for r in result['history_canaries'])
            record_rows=table(test/'per_record.csv');person_rows=table(test/'per_subject.csv')
            for condition,expected in result['conditions'].items():
                segments=table(test/f'{condition}_segments.csv');rr=[r for r in record_rows if r['condition']==condition];pp=[r for r in person_rows if r['condition']==condition]
                assert {r['record'] for r in rr}==set(metadata) and {r['subject'] for r in pp}==subjects
                expected_keys={(r['record'],block) for r in data.rows for block in (range(r['score_blocks']) if not condition.startswith('history_') else (0,))}
                assert {(r['record'],int(r['block'])) for r in segments}==expected_keys
                for record in rr:
                    rows=[r for r in segments if r['record']==record['record']];valid=[number(r['r']) for r in rows if number(r['r']) is not None]
                    mean_r=float(np.mean(valid)) if valid else None
                    assert (mean_r is None)==(number(record['r']) is None)
                    if mean_r is not None:aggregate_errors.append(abs(mean_r-number(record['r'])))
                    aggregate_errors.append(abs(np.mean([float(r['mse']) for r in rows])-float(record['mse'])))
                for person in pp:
                    rr_person=[r for r in rr if r['subject']==person['subject']];valid=[number(r['r']) for r in rr_person if number(r['r']) is not None]
                    if valid:aggregate_errors.append(abs(np.mean(valid)-float(person['r'])))
                    else:assert number(person['r']) is None
                valid=[float(r['r']) for r in pp if number(r['r']) is not None]
                if valid:aggregate_errors.append(abs(np.mean(valid)-expected['r']))
                else:assert expected['r'] is None
                aggregate_errors.append(abs(np.mean([float(r['mse']) for r in pp])-expected['mse']))
                results.append({'variant':variant,'seed':seed,'condition':condition,'endpoint':expected['endpoint'],
                    'pearson_r':expected['r'],'MSE_normalized':expected['mse'],'undefined_r_blocks':expected['undefined_r_blocks'],
                    'blocks':expected['blocks'],'records':expected['records'],'subjects':expected['persons'],'source_families':5,
                    'optimizer_steps':train['steps'],'selected_step':train['selected_step'],'status':'ACTUAL_INTERNAL_EXPLORATORY_AUDIT'})
            all_records.extend(record_rows);all_persons.extend(person_rows)
            training.append({'variant':variant,'seed':seed,'steps':train['steps'],'samples':train['samples'],'selected_step':train['selected_step'],
                'seconds':train['seconds'],'train_step_seconds':train['train_step_seconds'],'device':train['device'],
                'peak_GPU_bytes':train['max_GPU_allocated_bytes'],'unique_training_records':train['unique_training_records'],**train['parameter_counts']})
            for record in (data.rows[0],data.rows[-1]):
                recomputed=raw_cpu_conditions(variant,checkpoint,record,data.rows)
                for condition,actual in recomputed.items():
                    expected=next(r for r in table(test/f'{condition}_segments.csv') if r['record']==record['record'])
                    er=number(expected['r']);assert (er is None)==(actual['r'] is None)
                    if er is not None:cpu_r_errors.append(abs(er-actual['r']))
                    else:cpu_nulls+=1
                    cpu_mse_errors.append(abs(float(expected['mse'])-actual['mse']))
    assert max(aggregate_errors)<1e-10 and max(cpu_r_errors)<1e-4 and max(cpu_mse_errors)<1e-4
    assert len(results)==243 and len(training)==39
    summary=[]
    for variant,condition in sorted({(r['variant'],r['condition']) for r in results}):
        rows=[r for r in results if r['variant']==variant and r['condition']==condition];valid=[r['pearson_r'] for r in rows if r['pearson_r'] is not None]
        summary.append({'variant':variant,'condition':condition,'mean_r':float(np.mean(valid)) if valid else None,
            'seed_SD':float(np.std(valid,ddof=1)) if len(valid)>1 else None,'mean_MSE_normalized':float(np.mean([r['MSE_normalized'] for r in rows])),
            'endpoint':rows[0]['endpoint'],'valid_r_seeds':len(valid),'blocks_per_seed':rows[0]['blocks']})
    def means(variant,condition):
        return {rid:float(np.mean([float(r['r']) for r in all_records if r['variant']==variant and r['condition']==condition and r['record']==rid])) for rid in metadata}
    pairs=[('m1_multi_minus_m1_long','m1_multi','true','m1_long','true'),('m2_minus_gru','m2','true','gru','true'),
        ('m2_minus_periodic_reset3s','m2','true','m2','periodic_reset3s'),('m3_finetune_minus_gru','m3_finetune','true','gru','true'),
        ('m3_finetune_minus_m3_frozen','m3_finetune','true','m3_frozen','true'),('m4_adaptive_minus_m4_fixed','m4_adaptive','true','m4_fixed','true'),
        ('m5_full_minus_m5_eeg','m5_full','true','m5_eeg','true'),('m5_full_minus_local_ar','m5_full','true','local','local_ar')]
    assert [r[0] for r in pairs]==protocol['primary_contrasts'];contrasts=[];hist=[]
    for name,a,ac,b,bc in pairs:
        aa,bb=means(a,ac),means(b,bc);stat=bootstrap({rid:aa[rid]-bb[rid] for rid in metadata},metadata)
        contrasts.append({'contrast':name,'primary':True,'positive_crossed_family95':stat['crossed_family95_low']>0,**stat})
    for variant in protocol['variants']:
        if variant=='m5_prior':continue
        true=means(variant,'history_true')
        for condition in ('state_reset','state_wrong','prefix_reset','prefix_wrong'):
            control=means(variant,'history_'+condition);stat=bootstrap({rid:true[rid]-control[rid] for rid in metadata},metadata)
            hist.append({'variant':variant,'contrast':'true_minus_'+condition,'primary':False,**stat})
    save(out/'RESULTS.csv',results);save(out/'SUMMARY.csv',summary);save(out/'per_record_all_seeds.csv',all_records);save(out/'per_subject_all_seeds.csv',all_persons)
    save(out/'training_summary.csv',training);save(out/'primary_contrasts.csv',contrasts);save(out/'history_deltas.csv',hist)
    review={'technical':'PASS','acceptance':'PASS_WITH_LIMITATION','parent_consistency':'PASS_SS06_ONLY',
        'same_agent_separate_pass':True,'third_party_review':False,'coordinator_review':'separate read-only source/split/runtime review; not an independent scientific signoff',
        'actual_downstream_runs':39,'actual_SSL_runs':3,'result_rows':len(results),'max_aggregation_error':max(aggregate_errors),
        'raw_input_original_CPU_r_checked':len(cpu_r_errors),'raw_input_nearconstant_r_null_checked':cpu_nulls,'raw_input_CPU_MSE_checked':len(cpu_mse_errors),
        'max_raw_CPU_r_error':max(cpu_r_errors),'max_raw_CPU_MSE_error':max(cpu_mse_errors),'all_training_identity_and_coverage_rechecked':True,
        'training_completed_before_audit_opening':True,'fit_ids_exact_newtrain':True,'audit_kind':'prospective INTERNAL / EXPLORATORY, historic training exposure',
        'primary_support_flags':{r['contrast']:r['positive_crossed_family95'] for r in contrasts},**identities()}
    dump(out/'review.json',review);print(json.dumps({'review':review,'true_summary':[r for r in summary if r['condition']=='true'],'primary':contrasts}),flush=True)


if __name__=='__main__':main()
