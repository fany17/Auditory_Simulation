"""D5 completed light metrics -> static figures/tables on2203; no new fits."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_history_interaction_probe_20261101')
OUT=ROOT/'report01';SEEDS=(1024,2026,31415);ARMS=('Add','CC','Near','Older');LAMBDAS=(.0001,.01,1.)
PAIRS=(('Near','Add'),('Near','CC'),('Older','Add'),('Older','CC'),('Near','Older'))
def read(p):return json.loads(p.read_text())
def dump(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def csvfile(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert (ROOT/'COMPLETED.json').exists();OUT.mkdir(exist_ok=False)
s=read(ROOT/'SUMMARY.json');rr=s['all_solutions'];assert len(rr)==36
lookup={(r['arm'],r['seed'],r['lam']):r for r in rr};base={int(k):v for k,v in s['baseline'].items()};sels={r['arm']:r for r in s['selection']}
selected=[];people=[];blocks=[];records=[];models={}
for seed in SEEDS:
    for arm in ARMS:
        lam=sels[arm]['selected_lambda'];m=base[seed] if lam is None else read(ROOT/f'seed{seed}/{arm}_lambda{lam:g}_METRICS.json');models[(arm,seed)]=m
        selected.append(dict(arm=arm,seed=seed,selected_lambda=lam,anchor_selected=lam is None,r3s=m['r3s'],r60s=m['r60s'],delta60s_over_anchor=m['r60s']-base[seed]['r60s']))
    for arm,lam in [('Anchor',None)]+[(a,l) for a in ARMS for l in LAMBDAS]:
        m=base[seed] if arm=='Anchor' else read(ROOT/f'seed{seed}/{arm}_lambda{lam:g}_METRICS.json')
        for key,out in [('people',people),('blocks',blocks),('records',records)]:
            for row in m[key]:out.append(dict(seed=seed,arm=arm,lam=lam,**row))
csvfile(OUT/'SELECTED.csv',selected);csvfile(OUT/'ALL_PERSON.csv',people);csvfile(OUT/'ALL_BLOCK.csv',blocks);csvfile(OUT/'ALL_RECORD_3S.csv',records)
contrasts=[];summaries=[]
for a,b in PAIRS:
    for choice in list(LAMBDAS)+['shared_selected']:
        vals=[]
        for seed in SEEDS:
            x=models[(a,seed)] if isinstance(choice,str) else lookup[(a,seed,choice)];y=models[(b,seed)] if isinstance(choice,str) else lookup[(b,seed,choice)]
            delta=x['r60s']-y['r60s'];vals.append(delta);contrasts.append(dict(comparison=f'{a}-{b}',choice=choice,seed=seed,delta60s=delta,delta3s=x['r3s']-y['r3s']))
        summaries.append(dict(comparison=f'{a}-{b}',choice=choice,deltas=vals,mean=float(np.mean(vals)),positive_seeds=sum(v>0 for v in vals)))
csvfile(OUT/'CONTRASTS.csv',contrasts)
means=[]
for arm in ARMS:
    rows=[r for r in selected if r['arm']==arm];lam=sels[arm]['selected_lambda']
    means.append(dict(arm=arm,selected_lambda=lam,anchor_selected=lam is None,mean3s=float(np.mean([r['r3s'] for r in rows])),mean60s=float(np.mean([r['r60s'] for r in rows])),paired_deltas_over_anchor=[r['delta60s_over_anchor'] for r in rows],selected_effective_df=[lookup[(arm,seed,lam)]['effective_df'] for seed in SEEDS] if lam is not None else None))
cost=read(ROOT/'COST.json');seedcost=[{k:v for k,v in read(ROOT/f'seed{seed}/COMPLETED.json').items() if k!='results'} for seed in SEEDS]
checks=[dict(seed=seed,max_metric_abs=read(ROOT/f'seed{seed}/ANCHOR_CHECK.json')['max_metric_abs']) for seed in SEEDS]
summary=dict(selected_means=means,contrasts=summaries,anchor_means={k:float(np.mean([v[k] for v in base.values()])) for k in ('r3s','r60s')},anchor_checks=checks,seed_costs=seedcost,cost=read(ROOT/'COMPLETED.json'),stage_costs=cost['stages'],
    scientific_limit='CC includes shared near/older additive effects; pure-current is frozen Ccap anchor. Repeated VAL development, finite bilinear features only.')
dump(OUT/'NUMERICAL_SUMMARY.json',summary)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
colors=dict(zip(ARMS,plt.cm.tab10.colors[:4]));sc=plt.cm.Set1.colors[:3]
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for ax,key,label in zip(axes,('r3s','r60s'),('3-s mean Pearson r','60-s / person macro Pearson r')):
    for arm in ARMS:
        for seed in SEEDS:ax.plot(LAMBDAS,[lookup[(arm,seed,l)][key] for l in LAMBDAS],color=colors[arm],alpha=.17,lw=.8)
        ax.plot(LAMBDAS,[np.mean([lookup[(arm,seed,l)][key] for seed in SEEDS]) for l in LAMBDAS],'o-',color=colors[arm],label=arm)
    ax.axhline(summary['anchor_means'][key],color='black',ls='--',label='Frozen current Ccap anchor (mean)')
    ax.set_xscale('log');ax.set_xticks(LAMBDAS,labels=['1e-4','1e-2','1']);ax.set_xlabel('Fixed lambda');ax.set_ylabel(label);ax.grid(alpha=.2)
axes[0].legend(fontsize=8);fig.suptitle('All36 fits: common current + near + older additive terms; CC is NOT history-free')
fig.savefig(OUT/'01_COMPLETE_LAMBDA_MATRIX.png');plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for ax,(a,b) in zip(axes.flat,PAIRS):
    for i,seed in enumerate(SEEDS):
        rows=[r for r in contrasts if r['comparison']==f'{a}-{b}' and r['seed']==seed]
        ax.plot(range(3),[r['delta60s'] for r in rows[:3]],'o-',color=sc[i],label=str(seed));ax.scatter(3+(i-1)*.045,rows[3]['delta60s'],marker='D',color=sc[i],s=30)
    ax.axhline(0,color='black',lw=.8);ax.axvline(2.5,color='gray',ls=':',lw=.8);ax.set_title(f'{a} minus {b}');ax.set_ylabel('Paired VAL60s delta r');ax.set_xticks(range(4),['1e-4','1e-2','1','Shared-selected\n(each arm / anchor)']);ax.grid(alpha=.2)
axes.flat[0].legend(title='seed',fontsize=8);axes.flat[-1].axis('off');axes.flat[-1].text(.03,.8,'Near: current × previous3–0s\nOlder: current × previous6–3s\nCC: current × current\nAll learned arms include identical\ncurrent/near/older additive main effects.\nNo confidence interval.',va='top',linespacing=1.7)
fig.suptitle('Prespecified interaction / capacity / distance contrasts (all fixed lambdas)')
fig.savefig(OUT/'02_FIXED_CONTRASTS.png');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
for arm in ARMS:
    rows=[r for r in rr if r['arm']==arm];axes[0].scatter([r['parameters'] for r in rows],[r['effective_df'] for r in rows],label=arm,color=colors[arm],s=28,alpha=.7)
axes[0].set_xlabel('Nominal coefficients incl.192 intercepts');axes[0].set_ylabel('Effective df per output (intercept excluded)');axes[0].legend(fontsize=8)
stages=list(dict.fromkeys(r['stage'] for r in cost['stages']));palette=plt.cm.tab20.colors
for ax,key,label in zip(axes[1:],('wall_seconds','CPU_seconds'),('Measured stage wall seconds','Measured stage CPU seconds')):
    bottom=np.zeros(3)
    for i,stage in enumerate(stages):
        vals=np.array([sum(r[key] for r in cost['stages'] if r['seed']==seed and r['stage']==stage) for seed in SEEDS]);ax.bar(range(3),vals,bottom=bottom,label=stage,color=palette[i]);bottom+=vals
    ax.set_xticks(range(3),map(str,SEEDS));ax.set_ylabel(label);ax.set_xlabel('Seed; stage timers exclude some bookkeeping')
axes[2].legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=6.5)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.suptitle('All36 solutions: capacity and measured shared-CPU cost; GPU/head/encoder forward=0')
fig.savefig(OUT/'03_CAPACITY_COST.png');plt.close(fig)
dump(OUT/'COMPLETED.json',dict(plots=3,solutions=36,person_rows=len(people),block_rows=len(blocks),record_rows=len(records),new_fits_or_signal_reads=False))
print(json.dumps({k:v for k,v in summary.items() if k!='stage_costs'},indent=2))
