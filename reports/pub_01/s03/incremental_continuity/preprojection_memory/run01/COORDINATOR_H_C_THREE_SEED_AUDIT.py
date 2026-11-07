from pathlib import Path
import gzip,hashlib,json,time
wall,cpu=time.time(),time.process_time()
root=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_preprojection_memory_20261103')
rows=[]
for seed in (1024,2026,31415):
    folders=[root/'fits'/f'{arm}_seed{seed}' for arm in ('H_history','C_current')]
    raw=[(p/'COMPLETED.json').read_bytes() for p in folders]
    values=[json.loads(x) for x in raw]
    for key in ('PERMUTATIONS.jsonl','INCUMBENT.json.gz'):
        assert (folders[0]/key).read_bytes()==(folders[1]/key).read_bytes()
    assert all(x['protocol_sha256']=='255ce2b95457d807e41775eaa208d40024dc5d1637c3a37b24edfec189c90632' and x['optimizer_updates']==78900 and x['registered_trainable_parameters']==3389513 for x in values)
    metrics=[json.loads(gzip.decompress((p/'SELECTED_METRICS.json.gz').read_bytes())) for p in folders]
    keys=('record','phase','subject','story','source_family','start','samples','family_mapping_basis')
    assert [[tuple(b[k] for k in keys) for b in m['segments']] for m in metrics][0]==[[tuple(b[k] for k in keys) for b in m['segments']] for m in metrics][1]
    rows.append(dict(seed=seed,H_r=values[0]['selected_validation_r'],C_r=values[1]['selected_validation_r'],H_minus_C=values[0]['selected_validation_r']-values[1]['selected_validation_r'],selected_epochs=[x['selected_epoch'] for x in values],completed_sha256=[hashlib.sha256(x).hexdigest() for x in raw],permutation_sha256=hashlib.sha256((folders[0]/'PERMUTATIONS.jsonl').read_bytes()).hexdigest(),incumbent_sha256=hashlib.sha256((folders[0]/'INCUMBENT.json.gz').read_bytes()).hexdigest(),selected_metric_sha256=[hashlib.sha256((p/'SELECTED_METRICS.json.gz').read_bytes()).hexdigest() for p in folders]))
print(json.dumps(dict(passed=True,scope='Three completed paired H/C fits; shared protocol, training order, incumbent and scoring coordinates. Selected repeated-development VAL only; T/P seed31415 pending; not a complete matrix or significance test.',rows=rows,mean_H_r=sum(x['H_r'] for x in rows)/3,mean_C_r=sum(x['C_r'] for x in rows)/3,mean_H_minus_C=sum(x['H_minus_C'] for x in rows)/3,H_gt_C_all_seeds=all(x['H_minus_C']>0 for x in rows),audit_CPU_seconds=time.process_time()-cpu,audit_wall_seconds=time.time()-wall,audit_NN_forwards=0,audit_backwards=0,audit_optimizer_updates=0,goal_complete=False)))
