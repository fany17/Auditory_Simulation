"""D4 light numerical tables/static figures from completed2203 summaries only."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_conditional_history_probe_20261101')
OUT=ROOT/'report01'
SEEDS=(1024,2026,31415);ARMS=('C','Ccap','CS','CF','CO','CF48');LAMBDAS=(.0001,.01,1.)
def read(p):return json.loads(p.read_text())
def dump(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def csvfile(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert (ROOT/'COMPLETED.json').exists();OUT.mkdir(exist_ok=False)
s=read(ROOT/'SUMMARY.json');rows=s['all_solutions'];assert len(rows)==54
lookup={(r['arm'],r['seed'],r['lam']):r for r in rows}
base={int(k):v for k,v in s['baseline'].items()};selections={r['arm']:r for r in s['selection']}
selected=[];people=[];blocks=[];records=[];models={}
for seed in SEEDS:
    for arm in ARMS:
        q=selections[arm];lam=q['selected_lambda']
        metric=base[seed] if lam is None else read(ROOT/f'seed{seed}/{arm}_lambda{lam:g}_METRICS.json')
        models[(arm,seed)]=metric
        row=dict(arm=arm,seed=seed,selected_lambda=lam,incumbent_P_selected=lam is None,r3s=metric['r3s'],r60s=metric['r60s'],gain_over_P60s=metric['r60s']-base[seed]['r60s'])
        selected.append(row)
    for arm,lam in [('P',None)]+[(a,l) for a in ARMS for l in LAMBDAS]:
        m=base[seed] if arm=='P' else read(ROOT/f'seed{seed}/{arm}_lambda{lam:g}_METRICS.json')
        for kind,out in [('people',people),('blocks',blocks),('records',records)]:
            for v in m[kind]:out.append(dict(seed=seed,arm=arm,lam=lam,**v))
csvfile(OUT/'SELECTED.csv',selected);csvfile(OUT/'ALL_PERSON.csv',people);csvfile(OUT/'ALL_BLOCK.csv',blocks);csvfile(OUT/'ALL_RECORD_3S.csv',records)
pairs=(('CF','CS'),('CF','Ccap'),('CF','CO'),('CF48','CS'))
contrasts=[]
for a,b in pairs:
    for lam in LAMBDAS:
        for seed in SEEDS:contrasts.append(dict(a=a,b=b,comparison=f'{a}-{b}',selection='common_lambda',lam=lam,seed=seed,delta60s=lookup[(a,seed,lam)]['r60s']-lookup[(b,seed,lam)]['r60s'],delta3s=lookup[(a,seed,lam)]['r3s']-lookup[(b,seed,lam)]['r3s']))
    for seed in SEEDS:contrasts.append(dict(a=a,b=b,comparison=f'{a}-{b}',selection='own_shared_lambda_or_P',lam=None,seed=seed,delta60s=models[(a,seed)]['r60s']-models[(b,seed)]['r60s'],delta3s=models[(a,seed)]['r3s']-models[(b,seed)]['r3s']))
csvfile(OUT/'CONTRASTS.csv',contrasts)
means=[]
for arm in ARMS:
    rr=[r for r in selected if r['arm']==arm]
    means.append(dict(arm=arm,selected_lambda=selections[arm]['selected_lambda'],P_selected=selections[arm]['incumbent_P_selected'],mean3s=float(np.mean([r['r3s'] for r in rr])),mean60s=float(np.mean([r['r60s'] for r in rr])),paired_deltas_over_P=[r['gain_over_P60s'] for r in rr]))
contrast_summary=[]
for a,b in pairs:
    for choice in ['own_shared_lambda_or_P']+list(LAMBDAS):
        rr=[r for r in contrasts if r['a']==a and r['b']==b and (r['selection']==choice if isinstance(choice,str) else r['selection']=='common_lambda' and r['lam']==choice)]
        vals=[r['delta60s'] for r in rr];assert len(vals)==3
        contrast_summary.append(dict(comparison=f'{a}-{b}',choice=choice,deltas=vals,mean=float(np.mean(vals)),positive_seeds=sum(v>0 for v in vals)))
costs=[read(ROOT/f'seed{seed}/COMPLETED.json') for seed in SEEDS]
cal=[dict(seed=seed,**{k:v for k,v in read(ROOT/f'seed{seed}/P_CHECK.json').items() if k!='baseline'}) for seed in SEEDS]
summary=dict(selected_means=means,contrasts=contrast_summary,baseline_means={k:float(np.mean([v[k] for v in base.values()])) for k in ('r3s','r60s')},calibrations=cal,cost=read(ROOT/'COMPLETED.json'),seed_costs=[{k:v for k,v in c.items() if k!='results'} for c in costs],
    fit_windows_per_seed=250750,history_eligible_VAL_windows=30304,all_VAL_windows=30500,original_train_windows=252454,
    limits='Repeated source VAL, additive linear residual only; shared lambda selection uses all3seed VAL means; not confirmation or information-theoretic causality.')
dump(OUT/'NUMERICAL_SUMMARY.json',summary)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
colors=dict(zip(ARMS,plt.cm.tab10.colors[:6]));seedcolors=plt.cm.Set1.colors[:3]
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for ax,key,label in zip(axes,('r3s','r60s'),('3-s window mean Pearson r','60-s block to person macro Pearson r')):
    for arm in ARMS:
        vals=[np.mean([lookup[(arm,seed,lam)][key] for seed in SEEDS]) for lam in LAMBDAS]
        ax.plot(LAMBDAS,vals,'o-',label=arm,color=colors[arm])
        for seed in SEEDS:ax.plot(LAMBDAS,[lookup[(arm,seed,lam)][key] for lam in LAMBDAS],color=colors[arm],alpha=.18,lw=.8)
    ax.axhline(summary['baseline_means'][key],color='black',ls='--',label='Original P (3-seed mean)')
    ax.set_xscale('log');ax.set_xticks(LAMBDAS,labels=['1e-4','1e-2','1']);ax.set_xlabel('Fixed ridge lambda');ax.set_ylabel(label);ax.grid(alpha=.2)
axes[0].legend(fontsize=8,ncol=2);fig.suptitle('SOURCE VAL development: complete fixed lambda matrix (thin: seeds, thick: mean)')
fig.savefig(OUT/'01_COMPLETE_LAMBDA_MATRIX.png');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
for ax,(a,b) in zip(axes.flat,pairs):
    for i,seed in enumerate(SEEDS):
        rr=[r for r in contrasts if r['a']==a and r['b']==b and r['seed']==seed and r['selection']=='common_lambda']
        ax.plot(range(3),[r['delta60s'] for r in rr],'o-',color=seedcolors[i],label=f'seed {seed}')
        v=[r['delta60s'] for r in contrasts if r['a']==a and r['b']==b and r['seed']==seed and r['selection']=='own_shared_lambda_or_P'][0]
        ax.scatter(3+(i-1)*.05,v,color=seedcolors[i],marker='D',s=30)
    ax.axhline(0,color='black',lw=.8);ax.axvline(2.5,color='gray',ls=':',lw=.8)
    ax.set_xticks(range(4),['1e-4','1e-2','1','Shared-selected\n(each arm / P)']);ax.set_title(f'{a} minus {b}');ax.set_ylabel('Paired VAL60s delta r');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('History / representation / capacity contrasts; three seeds, no confidence interval')
fig.savefig(OUT/'02_FIXED_CONTRASTS.png');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
for arm in ARMS:
    rr=[r for r in rows if r['arm']==arm]
    axes[0].scatter([r['parameters'] for r in rr],[r['effective_df'] for r in rr],label=arm,color=colors[arm],s=22,alpha=.75)
axes[0].set_xlabel('Nominal fitted coefficients (incl. intercept)');axes[0].set_ylabel('Ridge effective df (per output; no intercept)');axes[0].legend(fontsize=8,ncol=2);axes[0].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
axes[1].bar(range(3),[c['seconds']/60 for c in costs],color=seedcolors);axes[1].set_xticks(range(3),map(str,SEEDS));axes[1].set_ylabel('Whole seed elapsed minutes');axes[1].set_xlabel('18 solutions + statistics / validation per seed')
axes[2].bar(range(3),[sum(r['solve_score_seconds'] for r in rows if r['seed']==seed)/60 for seed in SEEDS],color=seedcolors);axes[2].set_xticks(range(3),map(str,SEEDS));axes[2].set_ylabel('Sum recorded per-lambda minutes');axes[2].set_xlabel('Coefficient formation + prediction / scoring\nShared eigen solve excluded')
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.suptitle('All54 solutions: effective capacity and actual bounded cost (shared resources)')
fig.savefig(OUT/'03_CAPACITY_COST.png');plt.close(fig)
dump(OUT/'COMPLETED.json',dict(plots=3,solutions=54,all_person_rows=len(people),all_block_rows=len(blocks),all_record_rows=len(records),no_new_fits_or_signal_reads=True))
print(json.dumps(summary,indent=2))
