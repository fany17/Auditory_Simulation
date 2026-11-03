"""D18 saved-JSON tables/static figures only, server2203, after all12fits.

Reuses D16 presentation conventions, adapted to D18 gzip/full metric schema.
No torch/runtime/model imports, dataset reads, prediction arrays or new fitting.
"""
import argparse,csv,gzip,hashlib,json,os,resource,shutil,time
from pathlib import Path
START=time.time();CPU=time.process_time()
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_preprojection_memory_20261103')
SEEDS=(1024,2026,31415)
ARMS=('H_history','C_current','T_continue','P_prediction_history')
EPOCHS=(-1,0,4,9,19)
MANIFEST={}

def read(path):
    path=Path(path);assert path.is_relative_to(ROOT)
    assert path.name.endswith(('.json','.json.gz','.jsonl')),path
    raw=path.read_bytes();entry=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    assert str(path) not in MANIFEST or MANIFEST[str(path)]==entry
    MANIFEST[str(path)]=entry
    text=gzip.decompress(raw).decode() if path.suffix=='.gz' else raw.decode()
    return [json.loads(x) for x in text.splitlines()] if path.suffix=='.jsonl' else json.loads(text)

def write_json(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def csvwrite(path,rows):
    assert rows,path
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)

def readiness():
    missing=[f'{arm}_seed{seed}' for seed in SEEDS for arm in ARMS if not (ROOT/f'fits/{arm}_seed{seed}/COMPLETED.json').exists()]
    ready=not missing and (ROOT/'QUEUE_COMPLETED.json').exists()
    q=read(ROOT/'QUEUE_LAUNCH.json')
    return dict(ready=ready,completed_fits=12-len(missing),missing=missing,wait_handle=str(ROOT/'QUEUE_COMPLETED.json'),queue_pid=q['pid'],protocol_sha256=q['protocol_sha256'],new_NN=0,new_backward=0,new_fits=0)

def main():
    ready=readiness();assert ready['ready'],ready
    # The output directory is never created for an incomplete matrix.
    q=read(ROOT/'QUEUE_COMPLETED.json');protocol=read(ROOT/'PROTOCOL.json')
    assert q['formal_fits']==12 and q['total_optimizer_updates']==q['total_update_positions']==946800
    assert q['target_window_presentations']==60588960
    root_before=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    assert root_before+64*1024**2<768*1024**2 and shutil.disk_usage(ROOT).free>=20*1024**3
    import numpy as np
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'reporting/matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'report01';out.mkdir(exist_ok=False);figs=out/'figures';figs.mkdir()
    selected=[];fixed=[];curves=[];people=[];sources=[];blocks=[];boundaries=[];grads=[];activations=[];costs=[];components=[];gaps=[]
    metrics={};incumbents={};old_history={}
    scope=read(ROOT/'COORDINATOR_SPLIT_SCOPE.json')
    seen={x['source_family']:x['seen_in_frozen_train'] for x in scope['validation_source_families']}
    def export_metric(seed,arm,kind,epoch,value):
        meta=dict(seed=seed,arm=arm,kind=kind,checkpoint_epoch=epoch)
        for row in value['people']:people.append(dict(**meta,**row))
        for row in value['source_family_descriptive']:sources.append(dict(**meta,**row,source_seen_in_frozen_TRAIN=seen[row['source_family']]))
        for row in value['segments']:blocks.append(dict(**meta,**row))
        for row in value['records']:
            b=row['boundary_tail'];flat={}
            for name in ('prefix','first_learning_overlap','tail','last_single_window'):
                flat.update({name+'_'+k:v for k,v in b[name].items()})
            boundaries.append(dict(**meta,record=row['record'],subject=row['subject'],source_family=row['source_family'],**flat,maximum_window_abs=b['maximum_window_abs'],maximum_fused_abs=b['maximum_fused_abs']))
    for seed in SEEDS:
        for arm in ARMS:
            folder=ROOT/f'fits/{arm}_seed{seed}';done=read(folder/'COMPLETED.json');hist=read(folder/'HISTORY.jsonl');old_history[(seed,arm)]=hist
            assert len(hist)==20 and [x['epoch'] for x in hist]==list(range(20))
            assert done['epochs']==20 and done['optimizer_updates']==done['update_positions']==78900 and done['empty_positions']==0
            costs.append(done)
            for row in read(folder/'COMPONENT_COUNTS.json')['stages']:components.append(dict(seed=seed,arm=arm,**row))
            for row in hist:curves.append(dict(seed=seed,arm=arm,epoch=row['epoch'],online_trajectory_negative_cosine=row['online_trajectory_negative_cosine'],online_trajectory_window_MSE=row['online_trajectory_window_MSE'],VAL_r=row['validation']['r'],VAL_MSE=row['validation']['MSE'],VAL_window_MSE=row['validation']['window_MSE'],VAL_overlap_MSE=row['validation']['overlap_MSE'],epoch_wall_seconds=row['epoch_wall_seconds'],online_is_fixed_model=False))
            inc=read(folder/'INCUMBENT.json.gz');sm=read(folder/'SELECTED_METRICS.json.gz');metrics[(seed,arm)]=sm
            assert abs(sm['phases']['val']['r']-done['selected_validation_r'])<=1e-10
            if seed not in incumbents:
                incumbents[seed]=inc;export_metric(seed,'Frozen_Ccap','incumbent',-1,inc)
            else:assert inc['phases']==incumbents[seed]['phases']
            export_metric(seed,arm,'selected',done['selected_epoch'],sm)
            older=read(folder/'OLDER_METRICS.json.gz') if arm=='H_history' else None
            if older is not None:
                metrics[(seed,'H_older')]=older;export_metric(seed,arm,'older_same_current_query',done['selected_epoch'],older)
            checkpoints={}
            for epoch in EPOCHS:
                s=read(folder/f'OFFLINE_TRAIN_epoch{epoch}.json.gz');tp=s['phases']['train'];vp=inc['phases']['val'] if epoch==-1 else hist[epoch]['validation']
                row=dict(seed=seed,arm=arm,epoch=epoch,TRAIN_r=tp['r'],TRAIN_MSE=tp['MSE'],TRAIN_window_MSE=tp['window_MSE'],TRAIN_overlap_MSE=tp['overlap_MSE'],TRAIN_overlap_NRMSE=tp['overlap_NRMSE'],TRAIN_negative_cosine=s['negative_cosine'],TRAIN_prediction_RMS=s['window_prediction_RMS'],TRAIN_prediction_mean=s['window_prediction_mean'],TRAIN_maximum_window_abs=s['maximum_window_abs'],TRAIN_maximum_fused_abs=s['maximum_fused_abs'],VAL_r=vp['r'],VAL_MSE=vp['MSE'],VAL_window_MSE=vp['window_MSE'],VAL_overlap_MSE=vp['overlap_MSE'],TRAIN_people=tp['people'],TRAIN_blocks=tp['blocks'],TRAIN_windows=tp['windows'],VAL_people=vp['people'],VAL_blocks=vp['blocks'],VAL_windows=vp['windows'],same_epoch_TRAIN_VAL=True)
                assert (tp['people'],tp['blocks'],tp['windows'])==(70,6104,252454)
                fixed.append(row);checkpoints[epoch]=row
            epoch=done['selected_epoch'];available=epoch in checkpoints
            selected.append(dict(seed=seed,arm=arm,selected_epoch=epoch,selected_VAL_r=sm['phases']['val']['r'],anchor_VAL_r=inc['phases']['val']['r'],selected_delta_vs_anchor=sm['phases']['val']['r']-inc['phases']['val']['r'],last_VAL_r=hist[-1]['validation']['r'],best_trained_epoch=int(np.argmax([h['validation']['r'] for h in hist])),best_trained_VAL_r=max(h['validation']['r'] for h in hist),selected_VAL_MSE=sm['phases']['val']['MSE'],selected_VAL_window_MSE=sm['phases']['val']['window_MSE'],selected_VAL_overlap_MSE=sm['phases']['val']['overlap_MSE'],selected_VAL_prediction_RMS=sm['window_prediction_RMS'],selected_VAL_maximum_window_abs=sm['maximum_window_abs'],selected_VAL_maximum_fused_abs=sm['maximum_fused_abs'],last_TRAIN_r=checkpoints[19]['TRAIN_r'],anchor_TRAIN_r=checkpoints[-1]['TRAIN_r'],last_TRAIN_MSE=checkpoints[19]['TRAIN_MSE'],last_TRAIN_negative_cosine=checkpoints[19]['TRAIN_negative_cosine'],anchor_TRAIN_negative_cosine=checkpoints[-1]['TRAIN_negative_cosine'],selected_TRAIN_available=available,selected_TRAIN_r=checkpoints[epoch]['TRAIN_r'] if available else None,selected_TRAIN_MSE=checkpoints[epoch]['TRAIN_MSE'] if available else None,H_older_VAL_r=older['phases']['val']['r'] if older else None,H_near_minus_older=sm['phases']['val']['r']-older['phases']['val']['r'] if older else None,registered_trainable_parameters=done['registered_trainable_parameters']))
            gaps.append(dict(seed=seed,arm=arm,selected_epoch=epoch,fixed_TRAIN_available=available,reason='Saved fixed checkpoint exists' if available else 'No saved fixed TRAIN evaluation at selected epoch; not evaluated or interpolated for presentation'))
            for log in read(folder/'FIRST_BATCH_GRADIENTS.jsonl'):
                for module,value in log['modules'].items():
                    if isinstance(value,dict):grads.append(dict(seed=seed,arm=arm,epoch=log['epoch'],module=module,**value,extra_NN=log['extra_NN'],scope='Existing shuffled first batch only; varying inputs/weights across epochs'))
                if log['activation_diagnostics'] is not None:activations.append(dict(seed=seed,arm=arm,epoch=log['epoch'],**log['activation_diagnostics'],extra_NN=0))
    by={(x['seed'],x['arm']):x for x in selected};paired=[];person_paired=[];source_paired=[]
    for seed in SEEDS:
        h=by[(seed,'H_history')]
        row=dict(seed=seed,H_selected_epoch=h['selected_epoch'],**{'H_minus_'+a:h['selected_VAL_r']-by[(seed,a)]['selected_VAL_r'] for a in ARMS if a!='H_history'},H_minus_anchor=h['selected_delta_vs_anchor'],H_near_minus_older=h['H_near_minus_older'])
        paired.append(row)
        pm={a:{p['subject']:p for p in metrics[(seed,a)]['people']} for a in (*ARMS,'H_older')};anchor={p['subject']:p for p in incumbents[seed]['people']}
        assert all(set(pm[a])==set(anchor) for a in pm)
        for subject in sorted(anchor):person_paired.append(dict(seed=seed,subject=subject,H_r=pm['H_history'][subject]['r'],**{'H_minus_'+a:pm['H_history'][subject]['r']-pm[a][subject]['r'] for a in ARMS if a!='H_history'},H_minus_anchor=pm['H_history'][subject]['r']-anchor[subject]['r'],H_near_minus_older=pm['H_history'][subject]['r']-pm['H_older'][subject]['r']))
        fm={a:{p['source_family']:p for p in metrics[(seed,a)]['source_family_descriptive']} for a in (*ARMS,'H_older')}
        for family in sorted(seen):source_paired.append(dict(seed=seed,source_family=family,source_seen_in_frozen_TRAIN=seen[family],**{'H_minus_'+a:fm['H_history'][family]['r']-fm[a][family]['r'] for a in ARMS if a!='H_history'},H_near_minus_older=fm['H_history'][family]['r']-fm['H_older'][family]['r']))
        for key in ('H_minus_C_current','H_minus_T_continue','H_minus_P_prediction_history','H_minus_anchor','H_near_minus_older'):
            assert abs(np.mean([x[key] for x in person_paired if x['seed']==seed])-row[key])<1e-10
    means=[]
    for arm in ARMS:
        rr=[x for x in selected if x['arm']==arm];keys=('selected_VAL_r','anchor_VAL_r','selected_delta_vs_anchor','last_VAL_r','selected_VAL_MSE','last_TRAIN_r','last_TRAIN_MSE','last_TRAIN_negative_cosine')
        means.append(dict(arm=arm,**{k:float(np.mean([x[k] for x in rr])) for k in keys}))
    files={'SELECTED_SUMMARY.csv':selected,'THREE_SEED_MEANS.csv':means,'PAIRED_COMPARISONS.csv':paired,'FIXED_TRAIN_MATCHED_VAL.csv':fixed,'ONLINE_TRAJECTORY_AND_VAL.csv':curves,'SELECTED_TRAIN_GAPS.csv':gaps,'PEOPLE_SAVED_METRICS.csv':people,'PERSON_PAIRED.csv':person_paired,'SOURCE_SAVED_METRICS.csv':sources,'SOURCE_PAIRED.csv':source_paired,'BLOCK_SAVED_METRICS.csv':blocks,'BOUNDARY_SELECTED.csv':boundaries,'FIRST_BATCH_GRADIENTS.csv':grads,'FIRST_BATCH_ACTIVATIONS.csv':activations,'COMPONENT_COUNTS_BY_STAGE.csv':components,'FIT_RESOURCE_COSTS.csv':costs}
    for name,rows in files.items():csvwrite(out/name,rows)
    totals={key:sum(x['component_forward_counts'][key] for x in costs) for key in costs[0]['component_forward_counts']}
    assert totals==q['component_forward_counts']
    eng=read(ROOT/'ENGINEERING_CHECK.json');partial=read(ROOT/'INTERRUPTED_FORMAL_COSTS.json');ref=read(ROOT/'COORDINATOR_REFERENCE_SELFCHECK.json')
    eng_total={k:v+eng['preserved_prior_attempt_costs']['production'].get(k,0) for k,v in eng['production_component_counts'].items()}
    eng_extra={k:v+eng['preserved_prior_attempt_costs']['extra'].get(k,0) for k,v in eng['extra_component_counts'].items()}
    assert eng_extra['temporary_optimizer_updates']==28
    review_costs=[]
    for path in sorted(ROOT.glob('COORDINATOR_*.json')):
        value=read(path);timers={k:v for k,v in value.items() if isinstance(v,(int,float)) and ('cpu' in k.lower() or 'wall' in k.lower()) and ('second' in k.lower() or k.lower().endswith('_s'))}
        review_costs.append(dict(file=path.name,reported_top_level_timers=json.dumps(timers,sort_keys=True),scope=value.get('scope',value.get('timer_scope','See full receipt')),not_summed_as_disjoint_CPU=True,note='Parent/child timers and stored evidence may overlap. Preserve individual receipts; do not add all nested timer values.'))
    csvwrite(out/'REVIEW_COST_INVENTORY.csv',review_costs)
    checkpoint_sizes=[dict(path=str(p),bytes=p.stat().st_size) for p in sorted((ROOT/'fits').glob('*/*.pt'))]
    assert len(checkpoint_sizes)==24;csvwrite(out/'CHECKPOINT_FILE_SIZES.csv',checkpoint_sizes)
    storage=read(ROOT/'COORDINATOR_CHECKPOINT_AUDIT_seed1024.json')['storage_estimate_reconciliation']
    ledger=dict(completed_matrix=dict(fits=12,optimizer_updates=sum(x['optimizer_updates'] for x in costs),target_presentations=sum(x['target_window_presentations'] for x in costs),components=totals,worker_CPU_seconds=sum(x['CPU_seconds'] for x in costs),queue_global_wall_seconds=q['wall_seconds'],timing_scope='Queue wall already includes interrupted first attempt/recovery downtime/foreign-GPU waits; never add them again. Completed fit CPU includes setup/train/eval; epoch CPU/wall is nested, not additive.'),interrupted_formal_attempt01=partial,discarded_engineering=dict(production_components=eng_total,extra_operations=eng_extra,final_attempt_CPU_seconds=eng['CPU_seconds'],final_attempt_wall_seconds=eng['wall_seconds'],prior_attempt_timing=eng['preserved_prior_attempt_costs']['attempts'],note='28 discarded optimizer updates total, no final initialization from engineering.'),reference_synthetic_selfcheck=ref,independent_reviews='See REVIEW_COST_INVENTORY and original receipts. No new neural replay in presentation; no invented additive total for nested timers.',reused_upstream='Original FT encoder/head, D8 latent/P caches and frozen Ccap assets are shared historical work. P reads frozen predicted history; its upstream cost is not zero or fully allocated by D18 runtime counters.',checkpoint_bytes_actual_total=sum(x['bytes'] for x in checkpoint_sizes),storage_estimate_correction=storage,components_are_not_unique_examples_or_FLOPs=True,new_NN_in_presentation=0,new_backward_in_presentation=0)
    write_json(out/'COST_LEDGER.json',ledger)
    # Static figure conventions reused from D16. English labels avoid missing
    # CJK fonts on2203; complete Chinese Results interpretations follow viewing.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    seedcolors=['#306998','#c86b31','#3f886e'];armcolors=['#235789','#c45824','#757575','#38856c']
    short=['H history','C current','T continue','P prior']
    def save(fig,name):fig.savefig(figs/name,dpi=170,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,seed in zip(axes,SEEDS):
        for arm,color,label in zip(ARMS,armcolors,short):
            hh=old_history[(seed,arm)];row=by[(seed,arm)];ax.plot([x['epoch'] for x in hh],[x['validation']['r'] for x in hh],color=color,label=label);ax.scatter(row['selected_epoch'],row['selected_VAL_r'],s=70,marker='*',color=color)
        ax.axhline(incumbents[seed]['phases']['val']['r'],color='black',ls='--',label='Frozen Ccap');ax.set(title=f'Seed {seed}',xlabel='Epoch (0-based; anchor -1)',ylabel='VAL 60s block -> person macro r');ax.legend(fontsize=8)
    fig.suptitle('Repeated-development VAL; strict earliest-best selection includes frozen incumbent')
    save(fig,'01_VAL_TRAJECTORIES.png')
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for seed,color in zip(SEEDS,seedcolors):
        axes[0].plot(range(4),[by[(seed,a)]['selected_VAL_r'] for a in ARMS],marker='o',color=color,label=str(seed))
        row=next(x for x in paired if x['seed']==seed);keys=('H_minus_C_current','H_minus_T_continue','H_minus_P_prediction_history','H_minus_anchor')
        axes[1].plot(range(4),[row[k] for k in keys],marker='o',color=color,label=str(seed));axes[2].scatter(SEEDS.index(seed),row['H_near_minus_older'],color=color,s=55)
    axes[0].set(xticks=range(4),xticklabels=short,ylabel='Selected VAL r',title='All selected controls');axes[0].legend()
    axes[1].axhline(0,color='black',lw=.8);axes[1].set(xticks=range(4),xticklabels=['H-C','H-T','H-P','H-anchor'],ylabel='Paired selected VAL delta r',title='All-direction gate uses every seed');axes[1].legend()
    axes[2].axhline(0,color='black',lw=.8);axes[2].set(xticks=range(3),xticklabels=SEEDS,ylabel='Near minus older VAL r',title='H: same checkpoint / current query')
    fig.suptitle('Seeds are repeated training runs, not independent participants; no significance claim')
    save(fig,'02_SELECTED_AND_CONTROLS.png')
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for j,arm in enumerate(ARMS):
        for seed,color in zip(SEEDS,seedcolors):
            rr=[x for x in fixed if x['seed']==seed and x['arm']==arm]
            for i,key in enumerate(('TRAIN_r','VAL_r')):axes[i,j].plot(EPOCHS,[x[key] for x in rr],marker='o',color=color,label=str(seed))
        for i in range(2):axes[i,j].set(title=short[j],xlabel='Fixed epoch (-1 = anchor)',ylabel=('TRAIN' if i==0 else 'Same-epoch VAL')+' physical60s/person r');axes[i,j].legend(fontsize=8)
    fig.suptitle('Only saved fixed TRAIN epochs -1/0/4/9/19; no selected-epoch TRAIN interpolation')
    save(fig,'03_FIXED_TRAIN_AND_MATCHED_VAL.png')
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for j,arm in enumerate(ARMS):
        for seed,color in zip(SEEDS,seedcolors):
            rr=[x for x in fixed if x['seed']==seed and x['arm']==arm]
            for i,key in enumerate(('TRAIN_negative_cosine','TRAIN_window_MSE')):axes[i,j].plot(EPOCHS,[x[key] for x in rr],marker='o',color=color,label=str(seed))
        for i,label in enumerate(('Fixed TRAIN centered negative cosine','Fixed TRAIN window MSE')):axes[i,j].set(title=short[j],xlabel='Fixed checkpoint epoch',ylabel=label);axes[i,j].legend(fontsize=8)
    fig.suptitle('Fixed models; online optimization trajectories are exported separately')
    save(fig,'04_FIXED_OBJECTIVE_AND_WINDOW_ERROR.png')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    pairs=[('TRAIN_prediction_RMS','selected_VAL_prediction_RMS','Prediction RMS'),('TRAIN_MSE','selected_VAL_MSE','Physical60s MSE'),('TRAIN_overlap_MSE','selected_VAL_overlap_MSE','Overlap MSE')]
    for j,(tk,vk,label) in enumerate(pairs):
        for arm,color,name in zip(ARMS,armcolors,short):
            for seed in SEEDS:
                rr=[x for x in fixed if x['seed']==seed and x['arm']==arm];axes[0,j].plot(EPOCHS,[x[tk] for x in rr],color=color,alpha=.7,label=name if seed==1024 else None)
        for seed,color in zip(SEEDS,seedcolors):axes[1,j].plot(range(4),[by[(seed,a)][vk] for a in ARMS],marker='o',color=color,label=str(seed))
        axes[0,j].set(title='Fixed TRAIN',xlabel='Checkpoint epoch',ylabel=label);axes[0,j].legend(fontsize=8)
        axes[1,j].set(title='Selected VAL',xticks=range(4),xticklabels=short,ylabel=label);axes[1,j].legend(fontsize=8)
    save(fig,'05_AMPLITUDE_PHYSICAL_ERROR_AND_OVERLAP.png')
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    subjects=sorted({x['subject'] for x in person_paired});families=sorted(seen)
    for seed,color in zip(SEEDS,seedcolors):
        rr={x['subject']:x for x in person_paired if x['seed']==seed};ss={x['source_family']:x for x in source_paired if x['seed']==seed}
        axes[0].scatter(range(49),[rr[s]['H_minus_C_current'] for s in subjects],color=color,s=15,alpha=.75,label=str(seed));axes[2].scatter(range(49),[rr[s]['H_near_minus_older'] for s in subjects],color=color,s=15,alpha=.75,label=str(seed))
        axes[1].plot(range(5),[ss[f]['H_minus_C_current'] for f in families],color=color,marker='o',label=str(seed))
    for ax in axes:ax.axhline(0,color='black',lw=.8);ax.legend(fontsize=8)
    axes[0].set(title='Same49 seen participants',xlabel='Subject index (sorted IDs in CSV)',ylabel='Selected H-C person delta r')
    axes[1].set(title='Descriptive source groups',xticks=range(5),xticklabels=[f+('\nseen source' if seen[f] else '\nrelative unseen') for f in families],ylabel='H-C source-specific macro delta r');axes[1].tick_params(axis='x',labelsize=8)
    axes[2].set(title='H paired history replacement',xlabel='Subject index (same ordering)',ylabel='Near-older person delta r')
    fig.suptitle('Mixed-source, repeated-development VAL; subgroup displays are not new selection/tests')
    save(fig,'06_PERSON_AND_SOURCE_DESCRIPTIVE.png')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for j,arm in enumerate(('H_history','C_current','P_prediction_history')):
        for seed,color in zip(SEEDS,seedcolors):
            for module,style in zip(('memory_Q_weight','memory_K_weight','memory_V_weight'),('-','--',':')):
                rr=[x for x in grads if x['seed']==seed and x['arm']==arm and x['module']==module];axes[0,j].plot([x['epoch'] for x in rr],[x['L2_norm'] for x in rr],color=color,ls=style,label=f'{seed} {module[7]}')
            rr=[x for x in activations if x['seed']==seed and x['arm']==arm];axes[1,j].plot([x['epoch'] for x in rr],[x['delta_RMS'] for x in rr],color=color,label=str(seed))
        axes[0,j].set_yscale('symlog',linthresh=1e-12);axes[0,j].set(title=arm,ylabel='Q/K/V weight-gradient L2 (symlog)',xlabel='Existing first batch epoch');axes[0,j].legend(fontsize=6,ncol=3)
        axes[1,j].set(title=arm,ylabel='Memory output delta RMS',xlabel='Existing first batch epoch');axes[1,j].legend(fontsize=8)
    fig.suptitle('Activity samples only; fresh zero output gives zero upstream gradient, not useful-history evidence')
    save(fig,'07_MEMORY_ACTIVITY_SAMPLES.png')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    panels=[('current_encoder_prefix_windows','Current prefix invoked windows'),('frozen_encoder_prefix_windows','Frozen E0 prefix invoked windows'),('attention_query_windows','Attention query invoked windows'),('prior_projection_windows','P Conv projection invoked windows'),('wall_seconds','Completed fit wall seconds'),('registered_trainable_parameters','Trainable parameters')]
    for ax,(key,label) in zip(axes.flat,panels):
        for seed,color in zip(SEEDS,seedcolors):
            vv=[next(x for x in costs if x['seed']==seed and x['arm']==arm) for arm in ARMS];y=[x['component_forward_counts'][key] if key in totals else x[key] for x in vv]
            ax.plot(range(4),y,marker='o',color=color,label=str(seed))
        ax.set(xticks=range(4),xticklabels=short,ylabel=label);ax.legend(fontsize=8)
    fig.suptitle('Components are not unique examples/FLOPs; H/C same parameters do not imply equal compute')
    save(fig,'08_COMPONENT_AND_RESOURCE_COSTS.png')
    gate=all(all(x[k]>0 for k in ('H_minus_C_current','H_minus_T_continue','H_minus_P_prediction_history','H_minus_anchor','H_near_minus_older')) for x in paired)
    summary=dict(status='COMPLETE_MATRIX_SAVED_METRICS_PRESENTED_PENDING_SCIENTIFIC_REVIEW',rows=selected,three_seed_means=means,paired_comparisons=paired,mean_H_minus_C=float(np.mean([x['H_minus_C_current'] for x in paired])),scientific_all_seed_history_gate=gate,selected_TRAIN_gaps=[x for x in gaps if not x['fixed_TRAIN_available']],figure_count=8,formal_complete_fits=12,retained_interrupted_attempts=2,actual_attempts_total=14,completed_matrix_components=totals,completed_matrix_optimizer_updates=946800,completed_matrix_target_presentations=60588960,seen_participants_all49=True,mixed_source_scope_receipt='COORDINATOR_SPLIT_SCOPE.json',no_fresh_test=True,no_waveform_independence_check=True,upstream_costs_fully_allocated=False,new_NN=0,new_backward=0,new_training=0,goal_complete=False,independent_review_required=True)
    write_json(out/'SUMMARY.json',summary);csvwrite(out/'SOURCE_MANIFEST.csv',list(MANIFEST.values()))
    root_after=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    assert root_after+64*1024**2<768*1024**2 and shutil.disk_usage(ROOT).free>=20*1024**3
    write_json(out/'PRESENTATION_COST.json',dict(NN_windows=0,backwards=0,optimizer_updates=0,new_fits=0,scientific_arrays_read=False,model_checkpoints_loaded=0,CPU_seconds=time.process_time()-CPU,wall_seconds=time.time()-START,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='After standard imports, includes numpy/matplotlib, gzip JSON parsing, tables and8plots. No torch/model/runtime/dataset loading.',all_eight_figures_produced=True,root_bytes_before=root_before,root_bytes_after_outputs=root_after,physical_output_path=str(ROOT.resolve()),presentation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(passed=True,formal_fits=12,figures=8,scientific_all_seed_history_gate=gate,output=str(out),new_NN=0)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--readiness',action='store_true');args=parser.parse_args()
    print(json.dumps(readiness())) if args.readiness else main()
