"""D14 fixed-result tables/static figures; JSON-only, zero NN/data-array access."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01');ROOT=BASE/'s03_full_ft_order_control_20261102';OLD=BASE/'s03_full_ft_history_20261102'
SEEDS=(1024,2026,31415);ARMS=('H_full','C_match','P_prior')
def read(p):return json.loads(p.read_text())
assert read(ROOT/'QUEUE_COMPLETED.json')['formal_residual_NN_windows']==61710828
out=ROOT/'report01';out.mkdir(exist_ok=False);figs=out/'figures';figs.mkdir()
rows=[];curves=[];offline=[];costs=[]
for seed in SEEDS:
    for arm in ARMS:
        folder=ROOT/f'fits/{arm}_seed{seed}';prior=OLD/f'fits/{arm}_seed{seed}'
        cost=read(folder/'COMPLETED.json');costs.append(cost)
        history=[json.loads(x) for x in (folder/'HISTORY.jsonl').read_text().splitlines()]
        oldhistory=[json.loads(x) for x in (prior/'HISTORY.jsonl').read_text().splitlines()]
        assert [x['epoch'] for x in history]==list(range(20))
        orders=[json.loads(x) for x in (folder/'PERMUTATIONS.jsonl').read_text().splitlines()];assert len(orders)==20
        for item in history:curves.append(dict(seed=seed,arm=arm,epoch=item['epoch'],online_trajectory_window_MSE=item['online_trajectory_window_MSE'],val_r=item['validation']['r'],val_MSE=item['validation']['MSE']))
        stages={e:read(folder/f'OFFLINE_TRAIN_epoch{e}.json') for e in [-1,0,4,9,19]}
        for epoch,s in stages.items():offline.append(dict(seed=seed,arm=arm,epoch=epoch,window_MSE=s['window_MSE'],physical60s_r=s['physical60s_r'],physical60s_MSE=s['physical60s_MSE']))
        reference=read(folder/'OLD_LAST19_TRAIN_REFERENCE.json');inc=read(folder/'INCUMBENT.json')['phases']['val'];selected=read(folder/'SELECTED_METRICS.json')['phases']['val'];last=history[-1]['validation']
        older=read(folder/'OLDER_METRICS.json')['phases']['val'] if arm=='H_full' else None
        rows.append(dict(seed=seed,arm=arm,selected_epoch=cost['selected_epoch'],anchor_TRAIN_window_MSE=stages[-1]['window_MSE'],old_last_TRAIN_window_MSE=reference['window_MSE'],new_last_TRAIN_window_MSE=stages[19]['window_MSE'],delta_TRAIN_window_MSE_vs_old=stages[19]['window_MSE']-reference['window_MSE'],anchor_TRAIN_r=stages[-1]['physical60s_r'],old_last_TRAIN_r=reference['physical60s_r'],new_last_TRAIN_r=stages[19]['physical60s_r'],delta_TRAIN_r_vs_old=stages[19]['physical60s_r']-reference['physical60s_r'],anchor_TRAIN_MSE=stages[-1]['physical60s_MSE'],old_last_TRAIN_MSE=reference['physical60s_MSE'],new_last_TRAIN_MSE=stages[19]['physical60s_MSE'],anchor_VAL_r=inc['r'],old_last_VAL_r=oldhistory[-1]['validation']['r'],new_last_VAL_r=last['r'],new_selected_VAL_r=selected['r'],selected_delta_VAL_r_vs_anchor=selected['r']-inc['r'],last_delta_VAL_r_vs_anchor=last['r']-inc['r'],last_delta_VAL_r_vs_old=last['r']-oldhistory[-1]['validation']['r'],new_selected_VAL_MSE=selected['MSE'],H_older_VAL_r=older['r'] if older else None,H_near_minus_older=selected['r']-older['r'] if older else None))
        rows[-1].update(best_trained_VAL_epoch=int(np.argmax([h['validation']['r'] for h in history])),best_trained_VAL_r=max(h['validation']['r'] for h in history),old_best_trained_VAL_r=max(h['validation']['r'] for h in oldhistory),old_online_last_window_MSE=oldhistory[-1]['train_window_MSE'],new_online_last_window_MSE=history[-1]['online_trajectory_window_MSE'])
def csvwrite(name,rr):
    with (out/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
csvwrite('MATCHED_SUMMARY.csv',rows);csvwrite('OFFLINE_TRAIN.csv',offline);csvwrite('ONLINE_VAL_CURVES.csv',curves)
means=[]
for arm in ARMS:
    rr=[r for r in rows if r['arm']==arm];means.append(dict(arm=arm,**{k:float(np.mean([r[k] for r in rr])) if all(r[k] is not None for r in rr) else None for k in rows[0] if k not in ('arm','seed','selected_epoch','best_trained_VAL_epoch')}))
csvwrite('THREE_SEED_MEANS.csv',means)
paired=[]
for seed in SEEDS:
    hh,cc,pp=[next(r for r in rows if r['seed']==seed and r['arm']==a) for a in ARMS]
    paired.append(dict(seed=seed,H_selected_epoch=hh['selected_epoch'],H_selected_is_zero_incumbent=hh['selected_epoch']==-1,H_minus_C_selected_r=hh['new_selected_VAL_r']-cc['new_selected_VAL_r'],H_minus_P_selected_r=hh['new_selected_VAL_r']-pp['new_selected_VAL_r'],H_minus_C_last19_r=hh['new_last_VAL_r']-cc['new_last_VAL_r'],H_minus_P_last19_r=hh['new_last_VAL_r']-pp['new_last_VAL_r'],H_near_minus_older=hh['H_near_minus_older']))
csvwrite('PAIRED_COMPARISONS.csv',paired)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=['#306998','#c86b31','#3f886e']
def save(fig,name):fig.savefig(figs/name,dpi=160,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for ax,arm in zip(axes,ARMS):
    for color,seed in zip(colors,SEEDS):
        rr=next(r for r in rows if r['seed']==seed and r['arm']==arm);ss=[x for x in offline if x['seed']==seed and x['arm']==arm]
        ax.plot([x['epoch'] for x in ss],[x['window_MSE'] for x in ss],marker='o',color=color,label=str(seed))
        ax.scatter([19],[rr['old_last_TRAIN_window_MSE']],color=color,marker='X',s=85)
    ax.set(title=arm,xlabel='Fixed checkpoint epoch (-1 = anchor)',ylabel='Offline TRAIN window MSE');ax.legend(fontsize=8)
fig.suptitle('Same fixed parameters across entire TRAIN; X = old ordered last19')
save(fig,'01_OFFLINE_TRAIN_WINDOW_MSE.png')
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for ax,arm in zip(axes,ARMS):
    for color,seed in zip(colors,SEEDS):
        rr=next(r for r in rows if r['seed']==seed and r['arm']==arm);ss=[x for x in offline if x['seed']==seed and x['arm']==arm]
        ax.plot([x['epoch'] for x in ss],[x['physical60s_r'] for x in ss],marker='o',color=color,label=str(seed));ax.scatter([19],[rr['old_last_TRAIN_r']],color=color,marker='X',s=85)
    ax.set(title=arm,xlabel='Fixed checkpoint epoch',ylabel='Offline TRAIN physical60s/person r');ax.legend(fontsize=8)
fig.suptitle('Original6104 blocks/70people; X = old ordered last19; no online-loss substitution')
save(fig,'02_OFFLINE_TRAIN_PHYSICAL_R.png')
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for ax,seed in zip(axes,SEEDS):
    for color,arm in zip(colors,ARMS):
        ss=[r for r in curves if r['seed']==seed and r['arm']==arm];rr=next(r for r in rows if r['seed']==seed and r['arm']==arm)
        ax.plot([r['epoch'] for r in ss],[r['val_r'] for r in ss],color=color,label=arm)
        ax.scatter([rr['selected_epoch']],[rr['new_selected_VAL_r']],marker='*',color=color,s=65)
    ax.axhline(rr['anchor_VAL_r'],color='black',ls='--',lw=1,label='Frozen Ccap');ax.set(title=f'Seed{seed}',xlabel='Epoch (-1 candidate retained)',ylabel='VAL physical60s/person r');ax.legend(fontsize=8)
save(fig,'03_VAL_FIXED_SELECTION_CURVES.png')
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');x=np.arange(3)
keys=['delta_TRAIN_window_MSE_vs_old','delta_TRAIN_r_vs_old','last_delta_VAL_r_vs_old'];labels=['TRAIN window MSE (negative better)','TRAIN physical r (positive better)','VAL physical r (positive better)']
for ax,key,label in zip(axes,keys,labels):
    for i,seed in enumerate(SEEDS):ax.scatter(x+(i-1)*.08,[next(r[key] for r in rows if r['seed']==seed and r['arm']==a) for a in ARMS],label=str(seed),color=colors[i])
    ax.axhline(0,color='black',lw=.8);ax.set(xticks=x,xticklabels=ARMS,ylabel='Shuffled last19 minus old ordered last19',title=label);ax.legend(fontsize=8)
save(fig,'04_MATCHED_LAST19_DIFFERENCES.png')
summary=dict(rows=rows,three_seed_means=means,paired_comparisons=paired,formal_fits=9,optimizer_updates=sum(x['updates'] for x in costs),formal_residual_NN_windows=sum(x['new_residual_NN_window_forwards'] for x in costs),engineering_success_NN=read(ROOT/'ENGINEERING_CHECK.json')['NN_windows'],engineering_failure_path_reconstruction_NN=362,new_encoder_head=0,CPU_worker_seconds=sum(x['CPU_seconds'] for x in costs),queue_wall_seconds=read(ROOT/'QUEUE_COMPLETED.json')['wall_seconds'],max_RSS_KiB=max(x['peak_RSS_KiB'] for x in costs),max_GPU_allocated_bytes=max(x['peak_GPU_allocated_bytes'] for x in costs),goal_complete=False,independent_review_required=True,new_NN_in_presentation=0)
assert summary['formal_residual_NN_windows']==61710828 and summary['optimizer_updates']==710100
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary,allow_nan=False))
