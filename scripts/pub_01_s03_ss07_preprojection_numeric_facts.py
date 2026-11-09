"""D18 final narrative tables from completed saved-metric presentation only.

Standard library, server2203; no neural/data/checkpoint loading. Not a new audit.
"""
import csv,hashlib,json,resource,statistics,time
from pathlib import Path
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_preprojection_memory_20261103')
OUT=ROOT/'report01';START=time.time();CPU=time.process_time();INPUTS=[]
ARMS=('H_history','C_current','T_continue','P_prediction_history')
LABELS=dict(zip(ARMS,('H真实历史','C当前信息','T继续训练','P预测历史')))
def source(path):
    raw=path.read_bytes();INPUTS.append(dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));return raw.decode()
def read(path):return json.loads(source(path))
def rows(path):return list(csv.DictReader(source(path).splitlines()))
def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def table(headers,values):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+['| '+' | '.join(str(x) for x in row)+' |' for row in values])
def main():
    assert (ROOT/'POSTPROCESS_COMPLETED.json').exists(),'Wait for one-shot full12fit presentation first'
    s=read(OUT/'SUMMARY.json');assert s['formal_complete_fits']==12
    fixed=rows(OUT/'FIXED_TRAIN_MATCHED_VAL.csv');pp=rows(OUT/'PERSON_PAIRED.csv');grad=rows(OUT/'FIRST_BATCH_GRADIENTS.csv');ledger=read(OUT/'COST_LEDGER.json')
    groupmeans=[]
    for arm in ARMS:
        for epoch in (-1,0,4,9,19):
            rr=[x for x in fixed if x['arm']==arm and int(x['epoch'])==epoch];assert len(rr)==3
            keys=('TRAIN_r','TRAIN_MSE','TRAIN_window_MSE','TRAIN_overlap_MSE','TRAIN_negative_cosine','TRAIN_prediction_RMS','VAL_r','VAL_MSE','VAL_window_MSE','VAL_overlap_MSE')
            groupmeans.append(dict(arm=arm,epoch=epoch,**{k:statistics.fmean(float(x[k]) for x in rr) for k in keys}))
    person_directions=[]
    for seed in (1024,2026,31415):
        rr=[x for x in pp if int(x['seed'])==seed];assert len(rr)==49
        for key in ('H_minus_C_current','H_minus_T_continue','H_minus_P_prediction_history','H_near_minus_older'):
            vv=[float(x[key]) for x in rr];person_directions.append(dict(seed=seed,comparison=key,people=49,positive=sum(x>0 for x in vv),negative=sum(x<0 for x in vv),equal=sum(x==0 for x in vv),scope='Descriptive same49participants, no additional independent n or test'))
    activity=[]
    for seed in (1024,2026,31415):
        for arm in ARMS:
            modules=('encoder','head') if arm=='T_continue' else ('encoder','head','memory_Q_weight','memory_K_weight','memory_V_weight')
            for module in modules:
                rr=[x for x in grad if int(x['seed'])==seed and x['arm']==arm and x['module']==module];assert len(rr)==20
                later=[x for x in rr if int(x['epoch'])>=1]
                activity.append(dict(seed=seed,arm=arm,module=module,epoch0_nonzero_scalars=int(rr[0]['nonzero_scalars']),later_sample_count=19,later_zero_samples=sum(int(x['nonzero_scalars'])==0 for x in later),later_minimum_L2=min(float(x['L2_norm']) for x in later),last_first_batch_L2=float(rr[-1]['L2_norm']),scope='Saved first batch only; not all updates'))
    partial=ledger['interrupted_formal_attempt01'];c=partial['C_exact_failure'];hb=partial['H_bounds'];completed=ledger['completed_matrix']['components']
    low,high=hb
    cumulative=dict(optimizer_updates_lower=946800+c['optimizer_updates']+low['optimizer_updates_bound'],optimizer_updates_upper=946800+c['optimizer_updates']+high['optimizer_updates_bound'],target_presentations_lower=60588960+partial['C_TRAIN_presentations']['TRAIN_target_window_presentations']+low['TRAIN_target_window_presentations'],target_presentations_upper=60588960+partial['C_TRAIN_presentations']['TRAIN_target_window_presentations']+high['TRAIN_target_window_presentations'],interrupted_fit_CPU_seconds=None,H_exact_terminal_updates=None)
    formal_component_bounds=[]
    for key,count in completed.items():
        if key.startswith('current_'):hkey='current_prefix_projection_head_invoked_bound_each'
        elif key=='frozen_encoder_prefix_windows':hkey='E0_prefix_invoked_bound'
        elif key=='attention_query_windows':hkey='attention_query_invoked_bound'
        else:hkey=None
        lo=hb[0][hkey] if hkey else 0;hi=hb[1][hkey] if hkey else 0
        formal_component_bounds.append(dict(component=key,complete12=count,interrupted_C=c['invoked_component_counts'][key],interrupted_H_lower=lo,interrupted_H_upper=hi,formal_all_attempts_lower=count+c['invoked_component_counts'][key]+lo,formal_all_attempts_upper=count+c['invoked_component_counts'][key]+hi,note='Invoked windows per component, not additive unique data or FLOPs; H includes possible incomplete invocation'))
    facts=dict(three_seed_means=s['three_seed_means'],paired_comparisons=s['paired_comparisons'],mean_H_minus_C=s['mean_H_minus_C'],fixed_checkpoint_three_seed_means=groupmeans,selected_TRAIN_gaps=s['selected_TRAIN_gaps'],person_direction_descriptive=person_directions,first_batch_gradient_activity=activity,formal_all_attempt_cost_bounds=cumulative,formal_component_bounds=formal_component_bounds,scientific_gate=s['scientific_all_seed_history_gate'],sources=INPUTS,new_NN=0,new_backward=0,new_fits=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),CPU_seconds=time.process_time()-CPU,wall_seconds=time.time()-START,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,timer_scope='After standard imports through saved-metric calculations; before final JSON/Markdown serialization. Not added to training or presentation nested timers.')
    dump(OUT/'NARRATIVE_NUMERIC_FACTS.json',facts)
    selected_table=table(['seed','臂','selected epoch','selected VAL r','last VAL r','对应固定TRAIN'],[[x['seed'],LABELS[x['arm']],x['selected_epoch'],f"{x['selected_VAL_r']:.9f}",f"{x['last_VAL_r']:.9f}",'有' if x['selected_TRAIN_available'] else '缺失，未补跑'] for x in s['rows']])
    mean_table=table(['臂','selected VAL r均值','相对旧锚点','last VAL r均值','last TRAIN r均值'],[[LABELS[x['arm']],f"{x['selected_VAL_r']:.9f}",f"{x['selected_delta_vs_anchor']:+.9f}",f"{x['last_VAL_r']:.9f}",f"{x['last_TRAIN_r']:.9f}"] for x in s['three_seed_means']])
    paired_table=table(['seed','H−C','H−T','H−P','H−anchor','H near−older'],[[x['seed']]+[f'{x[k]:+.9f}' for k in ('H_minus_C_current','H_minus_T_continue','H_minus_P_prediction_history','H_minus_anchor','H_near_minus_older')] for x in s['paired_comparisons']])
    fixed_table=table(['臂','epoch','TRAIN r均值','同epoch VAL r均值','TRAIN负cosine','TRAIN物理MSE','TRAIN窗口MSE'],[[LABELS[x['arm']],x['epoch'],f"{x['TRAIN_r']:.9f}",f"{x['VAL_r']:.9f}",f"{x['TRAIN_negative_cosine']:.9f}",f"{x['TRAIN_MSE']:.9f}",f"{x['TRAIN_window_MSE']:.9f}"] for x in groupmeans if x['epoch'] in (-1,19)])
    with (OUT/'NUMERIC_TABLES.md').open('x') as f:f.write('# D18完整已存指标的正文数值表\n\n本文件为2203标准库汇总，不是新的独立验证、显著性检验或神经评估。最终解释仍需逐图目视与协调科学审阅。\n\n'+mean_table+'\n\n'+selected_table+'\n\n'+paired_table+'\n\n'+fixed_table+'\n')
    print(json.dumps(dict(passed=True,complete_fits=12,new_NN=0,output=str(OUT/'NARRATIVE_NUMERIC_FACTS.json'))))
if __name__=='__main__':main()
