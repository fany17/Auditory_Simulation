"""Export lightweight actual metrics and server path inventory after review."""
import csv
import json
from pathlib import Path

root=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261015')
work=root/'transformer_available01';contrast=work/'training_seed1024_attempt04';envelope=work/'envelope_readout_seed1024'
for run in (root/'hdps_m_available01/training_seed1024',contrast,envelope):
    review=json.loads((run/'review.json').read_text())
    if review['technical']!='PASS' or review['acceptance']!='PASS_WITH_LIMITATION':raise RuntimeError('unaccepted actual run')
fields=['dataset','protocol','method','split','subjects','candidate_n','seed','top1','top5','top10','accuracy_2afc','accuracy_5afc','pearson_r','r2','status','acceptance','protocol_candidate_n_expected','language_model_prior','blocker','provenance']
result_rows=[]
for name,run,method,key in [('contrastive',contrast,'Transformer_CLIP_single','accuracy_2afc'),('envelope',envelope,'Transformer_frozen_Conv32_envelope','pearson_r')]:
    result=json.loads((run/'result.json').read_text())
    for group in ('test_seen','test_heldout','weighted_seen_heldout_2to1'):
        row=dict.fromkeys(fields,'');row.update(dataset='SparrKULee',protocol='DERIVED_PUBLIC_AVAILABLE_PROTOCOL',method=method,split='author_fold4_'+group,
            seed=1024,status='COMPLETED',acceptance='PASS_WITH_LIMITATION',language_model_prior='none;fixed_XLSR19_acoustic_pretraining_at_contrastive_training',
            blocker='single_model_available_cohort_long_record_test_not_official_packet_or_ensemble',provenance='evidence/available_20261015/transformer/'+name+'/result.json')
        if group.startswith('weighted'):
            row['subjects']=sum(result['test'][g]['subjects'] for g in ('test_seen','test_heldout'))
            value=result['test']['weighted_2afc' if name=='contrastive' else 'weighted_pearson_r']
        else:row['subjects']=result['test'][group]['subjects'];value=result['test'][group][key]
        row['accuracy_2afc' if name=='contrastive' else 'pearson_r']=value
        if name=='contrastive':row.update(candidate_n=2,protocol_candidate_n_expected=2)
        result_rows.append(row)
with (work/'results_available_additions.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(result_rows)
inventory=[]
light={'result.json','review.json','history.json','test_per_subject.csv','test_per_segment.csv','excluded_segments.json','completed.json','test_summary.png','candidate_size_summary.png','candidate_trace.json','candidate_trace.csv','descriptive_errors.json','results_available_additions.csv','finalize_status.json','job_status.json'}
for p in sorted(root.rglob('*')):
    if not p.is_file():continue
    if p.suffix in ('.npy','.npz'):kind='SERVER_ONLY_ARRAY'
    elif p.suffix=='.pt':kind='SERVER_ONLY_WEIGHTS'
    elif p.suffix in ('.py','.json','.csv','.log','.sh'):kind='SOURCE_OR_STRUCTURED_METADATA'
    elif p.suffix=='.png':kind='STATIC_SUMMARY'
    else:kind='OTHER_SERVER_RUNTIME'
    inventory.append({'host':'server2203','path':str(p),'bytes':p.stat().st_size,'kind':kind,'local_return':'YES_LIGHTWEIGHT' if p.name in light else 'NO'})
with (work/'server_manifest_available.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['host','path','bytes','kind','local_return']);writer.writeheader();writer.writerows(inventory)
print(json.dumps({'actual_additional_result_rows':len(result_rows),'inventory_files':len(inventory),'accepted_actual_runs':3}))
