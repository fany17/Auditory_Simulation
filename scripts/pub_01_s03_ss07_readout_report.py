"""D6 completed summaries -> all-grid tables and static figures on2203."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_readout_attribution_20261101');OUT=ROOT/'report01'
SEEDS=(1024,2026,31415);ARMS=('Ponly','S32','U32','C','Ccap');LAMBDAS=(.0001,.01,1.)
PAIRS=(('U32','S32'),('C','U32'),('Ccap','C'),('S32','Ponly'),('U32','Ponly'),('C','Ponly'),('Ccap','Ponly'))
def read(p):return json.loads(p.read_text())
def dump(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def csvfile(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert (ROOT/'COMPLETED.json').exists();OUT.mkdir(exist_ok=False)
s=read(ROOT/'SUMMARY.json');rr=s['all_solutions'];assert len(rr)==45
lookup={(r['arm'],r['seed'],r['lam']):r for r in rr};sels={r['arm']:r for r in s['selection']}
base={seed:read(ROOT/f'seed{seed}/P_BASELINE.json') for seed in SEEDS};models={};selected=[];people=[];blocks=[];records=[];overlaprecords=[]
for seed in SEEDS:
    for arm in ARMS:
        lam=sels[arm]['selected_lambda'];m=base[seed] if lam is None else read(ROOT/f'seed{seed}/{arm}_lambda{lam:g}_METRICS.json');models[(arm,seed)]=m
        aux=m['secondary'];selected.append(dict(arm=arm,seed=seed,selected_lambda=lam,P_selected=lam is None,r3s=m['r3s'],r60s=m['r60s'],delta60s_over_P=m['r60s']-base[seed]['r60s'],
            overlap_MSE=aux['overlap_MSE'],overlap_NRMSE=aux['overlap_NRMSE'],window_target_MSE=aux['window_target_MSE'],**aux['macro_MSE']))
    for arm,lam in [('P',None)]+[(a,l) for a in ARMS for l in LAMBDAS]:
        m=base[seed] if arm=='P' else read(ROOT/f'seed{seed}/{arm}_lambda{lam:g}_METRICS.json');aux=m['secondary']
        sm={p['subject']:p for p in aux['people']};sb={(b['record'],b['start']):b for b in aux['blocks']}
        for row in m['people']:people.append(dict(seed=seed,arm=arm,lam=lam,**row,**{k:v for k,v in sm[row['subject']].items() if k!='subject'}))
        for row in m['blocks']:blocks.append(dict(seed=seed,arm=arm,lam=lam,**row,**{k:v for k,v in sb[(row['record'],row['start'])].items() if k not in ('record','subject','start')}))
        for row in m['records']:records.append(dict(seed=seed,arm=arm,lam=lam,**row))
        for row in aux['records']:overlaprecords.append(dict(seed=seed,arm=arm,lam=lam,**row))
csvfile(OUT/'SELECTED.csv',selected);csvfile(OUT/'ALL_PERSON.csv',people);csvfile(OUT/'ALL_BLOCK.csv',blocks);csvfile(OUT/'ALL_RECORD_3S.csv',records);csvfile(OUT/'ALL_RECORD_CONTINUITY.csv',overlaprecords)
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
    means.append(dict(arm=arm,selected_lambda=lam,**{k:float(np.mean([r[k] for r in rows])) for k in ('r3s','r60s','delta60s_over_P','overlap_MSE','overlap_NRMSE','window_target_MSE','weighted_single_MSE','fused_MSE','disagreement')},paired_deltas_over_P=[r['delta60s_over_P'] for r in rows],selected_effective_df_total=[lookup[(arm,seed,lam)]['effective_df_total'] for seed in SEEDS] if lam else None))
base_means={k:float(np.mean([v[k] for v in base.values()])) for k in ('r3s','r60s')}
base_means.update({k:float(np.mean([v['secondary'][k] for v in base.values()])) for k in ('overlap_MSE','overlap_NRMSE','window_target_MSE')})
base_means.update({k:float(np.mean([v['secondary']['macro_MSE'][k] for v in base.values()])) for k in ('weighted_single_MSE','fused_MSE','disagreement')})
summary=dict(selected_means=means,contrasts=summaries,baseline_means=base_means,cost=read(ROOT/'COMPLETED.json'),seed_costs=[{k:v for k,v in read(ROOT/f'seed{seed}/COMPLETED.json').items() if k!='results'} for seed in SEEDS],
    maximum_reused_primary_abs=max(r['reused_primary_max_abs'] or 0 for r in rr),maximum_normal_equation_abs=max(r['normal_equation_max_abs'] for r in rr),maximum_Hann_identity_abs=max(r['Hann_identity_max_abs'] for r in rr),
    scope='Ponly common output reread; S32/U32 same local Z support, denseP common; sharing after D4 coordinate normalization. Repeated known-subject source VAL, no independent loss causality.')
dump(OUT/'NUMERICAL_SUMMARY.json',summary)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
colors=dict(zip(ARMS,plt.cm.tab10.colors[:5]));sc=plt.cm.Set1.colors[:3]
fig,axes=plt.subplots(1,3,figsize=(16,4.7),layout='constrained')
for ax,key,label in zip(axes,('r3s','r60s','fused_macro_MSE'),('3-s mean Pearson r','60-s / person macro Pearson r','60-s / person fused target MSE')):
    for arm in ARMS:
        for seed in SEEDS:ax.plot(LAMBDAS,[lookup[(arm,seed,l)][key] for l in LAMBDAS],color=colors[arm],alpha=.16,lw=.7)
        ax.plot(LAMBDAS,[np.mean([lookup[(arm,seed,l)][key] for seed in SEEDS]) for l in LAMBDAS],'o--' if arm in ('C','Ccap') else 'o-',color=colors[arm],label=arm+(' (reused)' if arm in ('C','Ccap') else ''))
    bm=base_means['fused_MSE'] if key=='fused_macro_MSE' else base_means[key];ax.axhline(bm,color='black',ls=':',label='P* mean')
    ax.set_xscale('log');ax.set_xticks(LAMBDAS,labels=['1e-4','1e-2','1']);ax.set_xlabel('Fixed lambda');ax.set_ylabel(label);ax.grid(alpha=.2)
axes[0].legend(fontsize=7.5);fig.suptitle('45 candidates:27 new /18 immutable D4 reuse; same MSE residual start, no history inputs')
fig.savefig(OUT/'01_COMPLETE_MATRIX.png');plt.close(fig)
fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
for ax,(a,b) in zip(axes.flat,PAIRS):
    for i,seed in enumerate(SEEDS):
        rows=[r for r in contrasts if r['comparison']==f'{a}-{b}' and r['seed']==seed];ax.plot(range(3),[r['delta60s'] for r in rows[:3]],'o-',color=sc[i],label=str(seed));ax.scatter(3+(i-1)*.04,rows[3]['delta60s'],marker='D',color=sc[i],s=25)
    ax.axhline(0,color='black',lw=.8);ax.axvline(2.5,color='gray',ls=':',lw=.8);ax.set_title(f'{a} minus {b}');ax.set_xticks(range(4),['1e-4','1e-2','1','Shared\nselected']);ax.set_ylabel('Paired VAL60s delta r');ax.grid(alpha=.2)
axes.flat[0].legend(title='seed',fontsize=7)
for ax in list(axes.flat)[7:]:ax.axis('off')
axes.flat[7].text(.02,.9,'Top row: prespecified primary contrasts\nBottom: prespecified P-only attribution\nAll methods retain same dense P path.\nS32 / U32 have identical local Z support.\nNo confidence interval.',va='top',linespacing=1.8)
fig.suptitle('Weight sharing, latent support and feature expansion; all fixed lambda contrasts')
fig.savefig(OUT/'02_ATTRIBUTION_CONTRASTS.png');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained');names=('P',)+ARMS
allmeans=[dict(arm='P',**base_means)]+means
axes[0].bar(range(6),[r['overlap_MSE'] for r in allmeans],color=['gray']+[colors[a] for a in ARMS]);axes[0].set_xticks(range(6),names);axes[0].set_ylabel('Uniform adjacent96 overlap MSE (before Hann)');axes[0].set_title('Continuity alone is not accuracy')
fused=np.array([r['fused_MSE'] for r in allmeans]);disc=np.array([r['disagreement'] for r in allmeans])
axes[1].bar(range(6),fused,label='Fused target MSE',color='steelblue');axes[1].bar(range(6),disc,bottom=fused,label='Hann disagreement',color='orange');axes[1].set_xticks(range(6),names);axes[1].set_ylabel('60-s / person macro MSE');axes[1].set_title('Sum = weighted single-window target MSE');axes[1].legend(fontsize=8)
for row in allmeans:
    arm=row['arm'];axes[2].scatter(row['overlap_MSE'],row['r60s'],s=60,color='gray' if arm=='P' else colors[arm]);axes[2].annotate(arm,(row['overlap_MSE'],row['r60s']),xytext=(4,5),textcoords='offset points',fontsize=8)
axes[2].set_xlabel('Uniform overlap MSE');axes[2].set_ylabel('VAL60s/person mean r');axes[2].set_title('Accuracy and disagreement jointly reported')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Shared-selected models only: fixed Hann decomposition; no selection by continuity')
fig.savefig(OUT/'03_CONTINUITY_ACCURACY.png');plt.close(fig)
dump(OUT/'COMPLETED.json',dict(plots=3,candidates=45,new_models=27,reused_models=18,person_rows=len(people),block_rows=len(blocks),record_rows=len(records),new_fits_or_signal_reads=False))
print(json.dumps({k:v for k,v in summary.items() if k!='seed_costs'},indent=2))
