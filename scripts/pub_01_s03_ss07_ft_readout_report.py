"""D8 finished TRAIN/VAL summaries to complete tables/static plots on2203."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_finetuned_readout_transfer_20261101');OUT=ROOT/'report01'
SEEDS=(1024,2026,31415);ARMS=('Ponly','C','Ccap');LAMBDAS=(.0001,.01,1.)
PAIRS=(('Ccap','P_FT'),('Ccap','Ponly'),('C','Ponly'),('Ccap','C'))
def read(p):return json.loads(p.read_text())
def dump(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def csvfile(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert (ROOT/'COMPLETED.json').exists();OUT.mkdir(exist_ok=False)
s=read(ROOT/'SUMMARY.json');allrows=s['all_solutions'];assert len(allrows)==27
sels={r['arm']:r for r in s['selection']};models={};selected=[];people=[];blocks=[];records=[];grid={}
for seed in SEEDS:
    base=read(ROOT/f'seed{seed}/P_FT_BASELINE.json');models[(seed,'P_FT')]=base
    for arm,lam in [('P_FT',None)]+[(a,l) for a in ARMS for l in LAMBDAS]:
        m=base if arm=='P_FT' else read(ROOT/f'seed{seed}/{arm}_lambda{lam:g}_METRICS.json')['metrics'];grid[(seed,arm,lam)]=m
        for key,out in [('people',people),('segments',blocks)]:
            for row in m[key]:out.append(dict(seed=seed,arm=arm,lam=lam,**row))
        for row in m['records']:records.append(dict(seed=seed,arm=arm,lam=lam,**{k:v for k,v in row.items() if k!='segments'}))
    for arm in ('P_FT',)+ARMS:
        lam=None if arm=='P_FT' else sels[arm]['selected_lambda'];m=base if lam is None else grid[(seed,arm,lam)];models[(seed,arm)]=m
        v=m['phases']['val'];selected.append(dict(seed=seed,arm=arm,selected_lambda=lam,baseline_selected=arm!='P_FT' and lam is None,**v))
csvfile(OUT/'SELECTED.csv',selected);csvfile(OUT/'ALL_PERSON.csv',people);csvfile(OUT/'ALL_BLOCK.csv',blocks);csvfile(OUT/'ALL_RECORD.csv',records)
contrasts=[];paired=[]
for index,(a,b) in enumerate(PAIRS):
    for choice in list(LAMBDAS)+['shared_selected']:
        vals=[]
        for seed in SEEDS:
            ma=models[(seed,a)] if isinstance(choice,str) else grid[(seed,a,choice)]
            mb=models[(seed,b)] if isinstance(choice,str) or b=='P_FT' else grid[(seed,b,choice)]
            x,y=ma['phases']['val'],mb['phases']['val'];delta=x['r']-y['r'];vals.append(delta)
            contrasts.append(dict(comparison=f'{a}-{b}',status='primary' if index<2 else 'secondary',choice=choice,seed=seed,delta_r=delta,delta_MSE=x['MSE']-y['MSE'],delta_disagreement=x['disagreement']-y['disagreement']))
        paired.append(dict(comparison=f'{a}-{b}',status='primary' if index<2 else 'secondary',choice=choice,deltas=vals,mean=float(np.mean(vals)),positive_seeds=sum(v>0 for v in vals)))
csvfile(OUT/'PAIRED_CONTRASTS.csv',contrasts)
keys=('r','r3s','MSE','weighted_single_MSE','disagreement','overlap_MSE','window_MSE')
means=[]
for arm in ('P_FT',)+ARMS:
    rr=[r for r in selected if r['arm']==arm]
    means.append(dict(arm=arm,selected_lambda=None if arm=='P_FT' else sels[arm]['selected_lambda'],**{k:float(np.mean([r[k] for r in rr])) if all(r[k] is not None for r in rr) else None for k in keys},seed_r=[r['r'] for r in rr],undefined_3s=[r['window_r_undefined'] for r in rr]))
summary=dict(scientific_status='REPEATED_SOURCE_TRAIN_VAL_DEVELOPMENT_ONLY',selected_means=means,paired=paired,all_lambda_selection=s['selection'],cost=read(ROOT/'COMPLETED.json'),bridges=[read(ROOT/f'seed{seed}/FT_VAL_BRIDGE.json') for seed in SEEDS],calibration=[dict(seed=seed,**read(ROOT/f'seed{seed}/P_FT_BASELINE.json')['calibration']) for seed in SEEDS],maximum_normal_equation_abs=max(r['normal_equation_max_abs'] for r in allrows),maximum_Hann_identity_abs=max(r['Hann_identity_max_abs'] for r in allrows),total_undefined_3s=sum(r['undefined_3s'] for r in allrows),goal_complete=False)
dump(OUT/'NUMERICAL_SUMMARY.json',summary)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
colors=dict(zip(ARMS,plt.cm.tab10.colors[:3]));sc=plt.cm.Set1.colors[:3]
fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
for ax,key,label in zip(axes,('r60s','r3s','fused_MSE'),('60-s / person macro Pearson r','3-s mean Pearson r','60-s / person fused target MSE')):
    for arm in ARMS:
        for seed in SEEDS:ax.plot(LAMBDAS,[next(r for r in allrows if r['arm']==arm and r['seed']==seed and r['lam']==l)[key] for l in LAMBDAS],color=colors[arm],alpha=.18,lw=.7)
        ax.plot(LAMBDAS,[np.mean([r[key] for r in allrows if r['arm']==arm and r['lam']==l]) if all(r[key] is not None for r in allrows if r['arm']==arm and r['lam']==l) else np.nan for l in LAMBDAS],'o-',color=colors[arm],label=arm)
    base=means[0]['MSE' if key=='fused_MSE' else ('r' if key=='r60s' else key)]
    if base is not None:ax.axhline(base,color='black',ls=':',label='P_FT* mean')
    ax.set_xscale('log');ax.set_xticks(LAMBDAS,labels=['1e-4','1e-2','1']);ax.set_xlabel('Fixed lambda');ax.set_ylabel(label);ax.grid(alpha=.2)
axes[0].legend(fontsize=8);fig.suptitle('D8: 27 TRAIN ridge fits on frozen full-FT encoder; repeated source VAL only')
fig.savefig(OUT/'01_COMPLETE_MATRIX.png');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for index,(ax,(a,b)) in enumerate(zip(axes.flat,PAIRS)):
    for i,seed in enumerate(SEEDS):
        rr=[r for r in contrasts if r['comparison']==f'{a}-{b}' and r['seed']==seed];ax.plot(range(3),[r['delta_r'] for r in rr[:3]],'o-',color=sc[i],label=str(seed));ax.scatter(3+(i-1)*.04,rr[3]['delta_r'],marker='D',color=sc[i],s=28)
    ax.axhline(0,color='black',lw=.8);ax.axvline(2.5,color='gray',ls=':',lw=.8);ax.set_title(('PRIMARY: ' if index<2 else 'Secondary: ')+f'{a} minus {b}');ax.set_xticks(range(4),['1e-4','1e-2','1','Shared\nselected']);ax.set_ylabel('Paired VAL60s delta r');ax.grid(alpha=.2)
axes.flat[0].legend(title='seed',fontsize=8);fig.suptitle('All fixed lambda contrasts; zero-residual FT baseline eligible; no confidence intervals')
fig.savefig(OUT/'02_FIXED_CONTRASTS.png');plt.close(fig)
names=('P_FT',)+ARMS;fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
axes[0].bar(range(4),[r['overlap_MSE'] for r in means],color=['gray']+[colors[a] for a in ARMS]);axes[0].set_xticks(range(4),names);axes[0].set_ylabel('Uniform96 overlap MSE before Hann');axes[0].set_title('Continuity is not selection endpoint')
fused=np.array([r['MSE'] for r in means]);weighted=np.array([r['weighted_single_MSE'] for r in means]);axes[1].vlines(range(4),fused,weighted,color='gray',lw=2,label='Gap = Hann disagreement');axes[1].scatter(range(4),fused,color='steelblue',label='Fused target MSE');axes[1].scatter(range(4),weighted,color='darkorange',label='Weighted single-window MSE');axes[1].set_xticks(range(4),names);axes[1].set_ylabel('60-s / person macro MSE');axes[1].set_title('Fixed target-error decomposition');axes[1].ticklabel_format(axis='y',style='plain',useOffset=False);axes[1].legend(fontsize=7)
for r in means:
    color='gray' if r['arm']=='P_FT' else colors[r['arm']];axes[2].scatter(r['overlap_MSE'],r['r'],color=color,s=55);axes[2].annotate(r['arm'],(r['overlap_MSE'],r['r']),xytext=(5,5),textcoords='offset points',fontsize=8)
axes[2].set_xlabel('Uniform overlap MSE');axes[2].set_ylabel('VAL60s / person macro r');axes[2].set_title('Accuracy and disagreement jointly shown')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Shared-selected readouts: descriptive metrics only; no continuity objective or new NN updates')
fig.savefig(OUT/'03_CONTINUITY_MSE.png');plt.close(fig)
dump(OUT/'COMPLETED.json',dict(plots=3,new_ridge_fits=27,baseline_scores=3,person_rows=len(people),block_rows=len(blocks),record_rows=len(records),new_signal_or_model_reads=False))
print(json.dumps(dict(selected_means=means,selected_pairs=[p for p in paired if p['choice']=='shared_selected'],cost=summary['cost']),indent=2))
