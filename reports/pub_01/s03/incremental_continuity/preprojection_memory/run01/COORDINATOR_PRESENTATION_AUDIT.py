"""Independent audit of final D18 saved-metric presentation; server2203 only."""
import csv,gzip,hashlib,json,math,time
from pathlib import Path
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_preprojection_memory_20261103')
OUT=ROOT/'report01'
ARMS=('H_history','C_current','T_continue','P_prediction_history')
SEEDS=(1024,2026,31415)
EPOCHS=(-1,0,4,9,19)

def audit():
    wall,cpu=time.time(),time.process_time()
    hashes={};checks=0;maximum_error=0.0
    def raw(path):
        data=path.read_bytes();hashes[str(path.relative_to(ROOT))]=hashlib.sha256(data).hexdigest();return data
    def read(path):
        data=raw(path)
        if path.name.endswith('.gz'):data=gzip.decompress(data)
        if path.suffix=='.jsonl':return [json.loads(line) for line in data.splitlines()]
        return json.loads(data)
    def table(name):return list(csv.DictReader(raw(OUT/name).decode().splitlines()))
    def close(a,b):
        nonlocal checks,maximum_error
        a,b=float(a),float(b);error=abs(a-b)
        assert math.isfinite(a) and math.isfinite(b) and error<=1e-10*max(1.,abs(b)),(a,b,error)
        checks+=1;maximum_error=max(maximum_error,error)
    def mean(values):
        values=list(values);assert values;return math.fsum(values)/len(values)
    summary=read(OUT/'SUMMARY.json');queue=read(ROOT/'QUEUE_COMPLETED.json');ledger=read(OUT/'COST_LEDGER.json')
    protocol=read(ROOT/'PROTOCOL.json')
    assert hashes['PROTOCOL.json']=='255ce2b95457d807e41775eaa208d40024dc5d1637c3a37b24edfec189c90632'
    assert summary['formal_complete_fits']==queue['formal_fits']==12
    assert summary['goal_complete'] is False and summary['no_fresh_test'] is True
    assert len(queue['workers'])==12 and all(x['returncode']==0 for x in queue['workers'])
    assert not (ROOT/'QUEUE_FAILURE.json').exists() and not (ROOT/'POSTPROCESS_FAILURE.json').exists()
    selected={(x['seed'],x['arm']):x for x in summary['rows']}
    csv_selected={(int(x['seed']),x['arm']):x for x in table('SELECTED_SUMMARY.csv')}
    fixed={(int(x['seed']),x['arm'],int(x['epoch'])):x for x in table('FIXED_TRAIN_MATCHED_VAL.csv')}
    trajectory={(int(x['seed']),x['arm'],int(x['epoch'])):x for x in table('ONLINE_TRAJECTORY_AND_VAL.csv')}
    assert len(selected)==len(csv_selected)==12 and len(fixed)==60 and len(trajectory)==240
    done={};metrics={};incumbents={};histories={}
    for seed in SEEDS:
        for arm in ARMS:
            key=(seed,arm);folder=ROOT/'fits'/f'{arm}_seed{seed}'
            d=read(folder/'COMPLETED.json');m=read(folder/'SELECTED_METRICS.json.gz')
            h=read(folder/'HISTORY.jsonl');inc=read(folder/'INCUMBENT.json.gz')
            assert d['seed']==seed and d['arm']==arm and d['optimizer_updates']==78900 and d['epochs']==20 and d['empty_positions']==0
            assert d['registered_trainable_parameters']==protocol['registered_parameters'][arm]
            assert not (folder/'FAILURE.json').exists()
            s=selected[key];assert s['selected_epoch']==d['selected_epoch']
            close(s['selected_VAL_r'],d['selected_validation_r']);close(s['selected_VAL_r'],m['phases']['val']['r'])
            close(s['anchor_VAL_r'],inc['phases']['val']['r']);close(s['last_VAL_r'],h[-1]['validation']['r'])
            close(s['selected_delta_vs_anchor'],m['phases']['val']['r']-inc['phases']['val']['r'])
            for field in ('MSE','window_MSE','overlap_MSE'):close(s['selected_VAL_'+field],m['phases']['val'][field])
            for field in ('window_prediction_RMS','maximum_window_abs','maximum_fused_abs'):
                dest='selected_VAL_prediction_RMS' if field=='window_prediction_RMS' else 'selected_VAL_'+field
                close(s[dest],m[field])
            assert s['selected_TRAIN_available']==(d['selected_epoch'] in EPOCHS)
            if not s['selected_TRAIN_available']:assert s['selected_TRAIN_r'] is None and s['selected_TRAIN_MSE'] is None
            for field,value in s.items():
                if value is None:assert csv_selected[key][field]==''
                elif isinstance(value,bool):assert csv_selected[key][field]==str(value)
                elif isinstance(value,(int,float)):close(csv_selected[key][field],value)
                else:assert csv_selected[key][field]==value
            for epoch in EPOCHS:
                train=read(folder/f'OFFLINE_TRAIN_epoch{epoch}.json.gz');tp=train['phases']['train']
                vp=inc['phases']['val'] if epoch==-1 else h[epoch]['validation'];row=fixed[(seed,arm,epoch)]
                for phase,value in (('TRAIN',tp),('VAL',vp)):
                    for field in ('r','MSE','window_MSE','overlap_MSE','people','blocks','windows'):close(row[phase+'_'+field],value[field])
                for dest,field in (('TRAIN_negative_cosine','negative_cosine'),('TRAIN_prediction_RMS','window_prediction_RMS'),('TRAIN_prediction_mean','window_prediction_mean'),('TRAIN_maximum_window_abs','maximum_window_abs'),('TRAIN_maximum_fused_abs','maximum_fused_abs')):close(row[dest],train[field])
                assert row['same_epoch_TRAIN_VAL']=='True'
                if epoch==d['selected_epoch']:close(s['selected_TRAIN_r'],tp['r']);close(s['selected_TRAIN_MSE'],tp['MSE'])
                if epoch==19:
                    close(s['last_TRAIN_r'],tp['r']);close(s['last_TRAIN_MSE'],tp['MSE']);close(s['last_TRAIN_negative_cosine'],train['negative_cosine'])
                if epoch==-1:
                    close(s['anchor_TRAIN_r'],tp['r']);close(s['anchor_TRAIN_negative_cosine'],train['negative_cosine'])
            for row in h:
                exported=trajectory[(seed,arm,row['epoch'])]
                for field in ('negative_cosine','window_MSE'):close(exported['online_trajectory_'+field],row['online_trajectory_'+field])
                for field in ('r','MSE','window_MSE','overlap_MSE'):close(exported['VAL_'+field],row['validation'][field])
                assert exported['online_is_fixed_model']=='False'
            done[key]=d;metrics[key]=m;incumbents[seed]=inc;histories[key]=h
        metrics[(seed,'H_older')]=read(ROOT/'fits'/f'H_history_seed{seed}'/'OLDER_METRICS.json.gz')
    paired={x['seed']:x for x in summary['paired_comparisons']};person=table('PERSON_PAIRED.csv')
    assert len(paired)==3 and len(person)==147
    gate_keys=['H_minus_'+a for a in ARMS[1:]]+['H_minus_anchor','H_near_minus_older']
    for seed in SEEDS:
        h=selected[(seed,'H_history')]['selected_VAL_r']
        expectations={'H_minus_'+a:h-selected[(seed,a)]['selected_VAL_r'] for a in ARMS[1:]}
        expectations.update(H_minus_anchor=h-incumbents[seed]['phases']['val']['r'],H_near_minus_older=h-metrics[(seed,'H_older')]['phases']['val']['r'])
        for field,value in expectations.items():close(paired[seed][field],value)
        persons={a:{p['subject']:p['r'] for p in metrics[(seed,a)]['people']} for a in (*ARMS,'H_older')}
        anchor={p['subject']:p['r'] for p in incumbents[seed]['people']}
        rows=[p for p in person if int(p['seed'])==seed];assert {p['subject'] for p in rows}==set(anchor)
        for row in rows:
            subject=row['subject'];hr=persons['H_history'][subject];close(row['H_r'],hr)
            for arm in ARMS[1:]:close(row['H_minus_'+arm],hr-persons[arm][subject])
            close(row['H_minus_anchor'],hr-anchor[subject]);close(row['H_near_minus_older'],hr-persons['H_older'][subject])
        for field,value in expectations.items():close(mean(float(row[field]) for row in rows),value)
    expected_gate=all(all(paired[seed][field]>0 for field in gate_keys) for seed in SEEDS)
    assert summary['scientific_all_seed_history_gate']==expected_gate==False
    means={x['arm']:x for x in summary['three_seed_means']}
    for arm in ARMS:
        for field in means[arm]:
            if field!='arm':close(means[arm][field],mean(selected[(seed,arm)][field] for seed in SEEDS))
    close(summary['mean_H_minus_C'],mean(paired[s]['H_minus_C_current'] for s in SEEDS))
    totals={key:sum(d['component_forward_counts'][key] for d in done.values()) for key in next(iter(done.values()))['component_forward_counts']}
    assert totals==queue['component_forward_counts']==summary['completed_matrix_components']==ledger['completed_matrix']['components']
    assert sum(d['optimizer_updates'] for d in done.values())==queue['total_optimizer_updates']==946800
    assert sum(d['target_window_presentations'] for d in done.values())==queue['target_window_presentations']==60588960
    close(ledger['completed_matrix']['worker_CPU_seconds'],math.fsum(d['CPU_seconds'] for d in done.values()))
    close(ledger['completed_matrix']['queue_global_wall_seconds'],queue['wall_seconds'])
    assert ledger['interrupted_formal_attempt01']==read(ROOT/'INTERRUPTED_FORMAL_COSTS.json')
    assert ledger['discarded_engineering']['extra_operations']['temporary_optimizer_updates']==28
    manifest=table('SOURCE_MANIFEST.csv')
    for row in manifest:
        p=Path(row['path']);assert p.is_relative_to(ROOT)
        data=p.read_bytes();assert len(data)==int(row['bytes']) and hashlib.sha256(data).hexdigest()==row['sha256']
    inventoried={row['file'] for row in table('REVIEW_COST_INVENTORY.csv')}
    late=[p.name for p in sorted(ROOT.glob('COORDINATOR_*.json')) if p.name not in inventoried]
    return dict(passed=True,scalar_comparisons=checks,maximum_absolute_error=maximum_error,
                source_manifest_entries_verified=len(manifest),selected_rows=12,fixed_TRAIN_rows=60,trajectory_rows=240,paired_people_rows=147,
                scientific_history_gate=False,three_seed_means={a:means[a]['selected_VAL_r'] for a in ARMS},
                later_coordinator_receipts_outside_original_cost_inventory=late,
                input_sha256=hashes,audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                audit_CPU_seconds=time.process_time()-cpu,audit_wall_seconds=time.time()-wall,
                audit_NN_forwards=0,audit_backwards=0,audit_optimizer_updates=0,
                scope='Saved-metric exports, paired people, summaries, fixed/online distinction, complete counts and original source manifest; no neural replay or new inference.',
                limitations='Repeated-development VAL. Original review-cost inventory is a rendering-time snapshot; later independent reviews must remain separately disclosed. No significance or fresh-test claim.',
                goal_complete=False)

if __name__=='__main__':print(json.dumps(audit(),allow_nan=False))
