"""D15 JSON-only result tables and four static figures; zero neural forwards."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01');ROOT=BASE/'s03_full_ft_objective_control_20261102';OLD=BASE/'s03_full_ft_order_control_20261102'
SEEDS=(1024,2026,31415);ARMS=('H_full','C_match','P_prior')
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
assert read(ROOT/'QUEUE_COMPLETED.json')['formal_residual_NN_windows']==59473248
out=ROOT/'report01';out.mkdir(exist_ok=False);figs=out/'figures';figs.mkdir()
rows=[];offline=[];curves=[];gradients=[];costs=[];val_scales=[]
for seed in SEEDS:
    for arm in ARMS:
        folder=ROOT/f'fits/{arm}_seed{seed}';prior=OLD/f'fits/{arm}_seed{seed}'
        cost=read(folder/'COMPLETED.json');costs.append(cost);hist=lines(folder/'HISTORY.jsonl');oldhist=lines(prior/'HISTORY.jsonl')
        assert [h['epoch'] for h in hist]==list(range(20))
        for h in hist:curves.append(dict(seed=seed,arm=arm,epoch=h['epoch'],online_negative_cosine=h['online_trajectory_negative_cosine'],online_window_MSE=h['online_trajectory_window_MSE'],val_r=h['validation']['r'],val_MSE=h['validation']['MSE']))
        stages={e:read(folder/f'OFFLINE_TRAIN_epoch{e}.json') for e in [-1,0,4,9,19]};ref=read(prior/'OFFLINE_TRAIN_epoch19.json')
        for e,s in stages.items():offline.append(dict(seed=seed,arm=arm,epoch=e,negative_cosine=s['negative_cosine'],window_MSE=s['window_MSE'],physical60s_r=s['physical60s_r'],physical60s_MSE=s['physical60s_MSE'],maximum_window_abs=s['maximum_window_abs'],maximum_fused_abs=s['maximum_fused_abs'],prediction_mean=s['window_prediction_mean'],prediction_RMS=s['window_prediction_RMS'],overlap_MSE=s['overlap_MSE']))
        inc=read(folder/'INCUMBENT.json')['phases']['val'];sm=read(folder/'SELECTED_METRICS.json');selected=sm['phases']['val'];oldsm=read(prior/'SELECTED_METRICS.json');oldselected=oldsm['phases']['val']
        older=read(folder/'OLDER_METRICS.json')['phases']['val'] if arm=='H_full' else None
        last=stages[19];anchor=stages[-1]
        rows.append(dict(seed=seed,arm=arm,selected_epoch=cost['selected_epoch'],anchor_TRAIN_negative_cosine=anchor['negative_cosine'],new_last_TRAIN_negative_cosine=last['negative_cosine'],anchor_TRAIN_window_MSE=anchor['window_MSE'],old_MSE_last_TRAIN_window_MSE=ref['window_MSE'],new_PCC_last_TRAIN_window_MSE=last['window_MSE'],anchor_TRAIN_r=anchor['physical60s_r'],old_MSE_last_TRAIN_r=ref['physical60s_r'],new_PCC_last_TRAIN_r=last['physical60s_r'],old_MSE_last_TRAIN_MSE=ref['physical60s_MSE'],new_PCC_last_TRAIN_MSE=last['physical60s_MSE'],anchor_VAL_r=inc['r'],old_MSE_selected_VAL_r=oldselected['r'],new_PCC_selected_VAL_r=selected['r'],selected_delta_VAL_vs_anchor=selected['r']-inc['r'],selected_delta_VAL_vs_MSE=selected['r']-oldselected['r'],old_MSE_last_VAL_r=oldhist[-1]['validation']['r'],new_PCC_last_VAL_r=hist[-1]['validation']['r'],new_PCC_selected_VAL_MSE=selected['MSE'],best_trained_VAL_epoch=int(np.argmax([h['validation']['r'] for h in hist])),best_trained_VAL_r=max(h['validation']['r'] for h in hist),H_older_VAL_r=older['r'] if older else None,H_near_minus_older=selected['r']-older['r'] if older else None,last_TRAIN_maximum_window_abs=last['maximum_window_abs'],last_TRAIN_maximum_fused_abs=last['maximum_fused_abs'],last_TRAIN_prediction_mean=last['window_prediction_mean'],last_TRAIN_prediction_RMS=last['window_prediction_RMS'],last_TRAIN_overlap_MSE=last['overlap_MSE']))
        for kind,m in [('PCC',sm),('MSE',oldsm)]:
            val_scales.append(dict(seed=seed,arm=arm,objective=kind,maximum_window_abs=max(x['maximum_window_abs'] for x in m['boundary_tail_records']),maximum_fused_abs=max(x['maximum_fused_abs'] for x in m['boundary_tail_records']),overlap_MSE=m['phases']['val']['overlap_MSE'],window_MSE=m['phases']['val']['window_MSE']))
        logs=lines(folder/'FIRST_BATCH_GRADIENTS.jsonl');assert len(logs)==20
        for x in logs:
            for module,g in x['modules'].items():gradients.append(dict(seed=seed,arm=arm,epoch=x['epoch'],module=module,nonzero_scalars=g['nonzero_scalars'],L2_norm=g['L2_norm'],effective_windows=x['effective_windows'],fallback_windows=x['fallback_windows'],expected_fresh_zero=x['expected_upstream_zero_at_fresh_first_batch']))
def csvwrite(name,rr):
    with (out/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
for name,rr in [('MATCHED_SUMMARY.csv',rows),('OFFLINE_TRAIN.csv',offline),('ONLINE_VAL_CURVES.csv',curves),('FIRST_BATCH_GRADIENTS.csv',gradients),('VAL_SELECTED_AMPLITUDE.csv',val_scales)]:csvwrite(name,rr)
means=[]
for arm in ARMS:
    rr=[r for r in rows if r['arm']==arm];means.append(dict(arm=arm,**{k:float(np.mean([r[k] for r in rr])) if all(r[k] is not None for r in rr) else None for k in rows[0] if k not in ('arm','seed','selected_epoch','best_trained_VAL_epoch')}))
csvwrite('THREE_SEED_MEANS.csv',means)
paired=[]
for seed in SEEDS:
    h,c,p=[next(r for r in rows if r['seed']==seed and r['arm']==a) for a in ARMS]
    paired.append(dict(seed=seed,H_selected_epoch=h['selected_epoch'],H_selected_is_zero=h['selected_epoch']==-1,H_minus_C_selected=h['new_PCC_selected_VAL_r']-c['new_PCC_selected_VAL_r'],H_minus_P_selected=h['new_PCC_selected_VAL_r']-p['new_PCC_selected_VAL_r'],H_near_minus_older=h['H_near_minus_older']))
csvwrite('PAIRED_COMPARISONS.csv',paired)
activity=[]
for arm in ARMS:
    for seed in SEEDS:
        for module in ['token','gru','film','current','output']:
            rr=[x for x in gradients if x['arm']==arm and x['seed']==seed and x['module']==module]
            activity.append(dict(seed=seed,arm=arm,module=module,epoch0_nonzero=rr[0]['nonzero_scalars'],zero_first_batches_epochs1_19=sum(x['nonzero_scalars']==0 for x in rr[1:]),minimum_positive_norm=min((x['L2_norm'] for x in rr[1:] if x['L2_norm']>0),default=None),last_first_batch_nonzero=rr[-1]['nonzero_scalars'],last_first_batch_norm=rr[-1]['L2_norm']))
csvwrite('GRADIENT_ACTIVITY.csv',activity)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});colors=['#306998','#c86b31','#3f886e']
def save(fig,name):fig.savefig(figs/name,dpi=160,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for ax,arm in zip(axes,ARMS):
    for color,seed in zip(colors,SEEDS):
        ss=[x for x in offline if x['seed']==seed and x['arm']==arm];ax.plot([x['epoch'] for x in ss],[x['negative_cosine'] for x in ss],marker='o',color=color,label=str(seed))
    ax.set(title=arm,xlabel='Fixed checkpoint epoch (-1 = anchor)',ylabel='Offline TRAIN centered negative cosine');ax.legend(fontsize=8)
fig.suptitle('Float64 native formula eps1e-6; all windows retained; lower is better')
save(fig,'01_OFFLINE_TRAIN_OBJECTIVE.png')
fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
for j,arm in enumerate(ARMS):
    for color,seed in zip(colors,SEEDS):
        ss=[x for x in offline if x['seed']==seed and x['arm']==arm];r=next(x for x in rows if x['seed']==seed and x['arm']==arm)
        for i,key,oldkey in [(0,'window_MSE','old_MSE_last_TRAIN_window_MSE'),(1,'physical60s_r','old_MSE_last_TRAIN_r')]:
            axes[i,j].plot([x['epoch'] for x in ss],[x[key] for x in ss],marker='o',color=color,label=str(seed));axes[i,j].scatter([19],[r[oldkey]],marker='X',color=color,s=75)
    for i in range(2):axes[i,j].set(title=arm,xlabel='Fixed checkpoint epoch',ylabel='TRAIN window MSE' if i==0 else 'TRAIN physical60s/person r');axes[i,j].legend(fontsize=8)
fig.suptitle('X = D14 MSE-trained last19; PCC loss does not constrain scale/bias')
save(fig,'02_TRAIN_MSE_AND_PHYSICAL_R.png')
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for ax,seed in zip(axes,SEEDS):
    for color,arm in zip(colors,ARMS):
        ss=[r for r in curves if r['seed']==seed and r['arm']==arm];r=next(x for x in rows if x['seed']==seed and x['arm']==arm)
        ax.plot([x['epoch'] for x in ss],[x['val_r'] for x in ss],color=color,label=arm);ax.scatter([r['selected_epoch']],[r['new_PCC_selected_VAL_r']],color=color,marker='*',s=70)
    ax.axhline(r['anchor_VAL_r'],color='black',ls='--',label='Frozen Ccap');ax.set(title=f'Seed{seed}',xlabel='Epoch (incumbent -1 retained)',ylabel='VAL physical60s/person r');ax.legend(fontsize=8)
save(fig,'03_VAL_FIXED_SELECTION_CURVES.png')
fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained');keys=['maximum_window_abs','maximum_fused_abs','overlap_MSE']
for j,key in enumerate(keys):
    for a,arm in enumerate(ARMS):
        ss=[r for r in offline if r['arm']==arm]
        for seed in SEEDS:
            z=[x for x in ss if x['seed']==seed];axes[0,j].plot([x['epoch'] for x in z],[x[key] for x in z],color=colors[a],alpha=.65,label=arm if seed==1024 else None)
    axes[0,j].set(xlabel='Fixed TRAIN checkpoint epoch',ylabel=key,title='TRAIN fixed parameters');axes[0,j].set_yscale('log');axes[0,j].legend(fontsize=8)
    for s,seed in enumerate(SEEDS):
        for kind,marker in [('MSE','x'),('PCC','o')]:
            yy=[next(v[key] for v in val_scales if v['seed']==seed and v['arm']==a and v['objective']==kind) for a in ARMS]
            axes[1,j].scatter(np.arange(3)+(s-1)*.10,yy,marker=marker,color=colors[s],label=f'{seed} {kind}')
    axes[1,j].set(xticks=np.arange(3),xticklabels=ARMS,ylabel=key,title='VAL selected: D14 MSE(x) / D15 PCC(o)');axes[1,j].set_yscale('log');axes[1,j].legend(fontsize=7)
save(fig,'04_AMPLITUDE_AND_OVERLAP.png')
summary=dict(rows=rows,three_seed_means=means,paired_comparisons=paired,gradient_activity=activity,formal_fits=9,optimizer_updates=sum(x['updates'] for x in costs),formal_residual_NN_windows=sum(x['new_residual_NN_window_forwards'] for x in costs),new_encoder_head=0,CPU_worker_seconds=sum(x['CPU_seconds'] for x in costs),queue_wall_seconds=read(ROOT/'QUEUE_COMPLETED.json')['wall_seconds'],max_RSS_KiB=max(x['peak_RSS_KiB'] for x in costs),max_GPU_allocated_bytes=max(x['peak_GPU_allocated_bytes'] for x in costs),new_NN_in_presentation=0,goal_complete=False,independent_review_required=True)
assert summary['formal_residual_NN_windows']==59473248 and summary['optimizer_updates']==710100
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary,allow_nan=False))
