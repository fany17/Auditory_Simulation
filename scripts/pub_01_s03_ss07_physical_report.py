"""D9 completed six solves plus immutable D8 references -> tables/3 plots."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01');ROOT=BASE/'s03_physical_objective_20261101';D8=BASE/'s03_finetuned_readout_transfer_20261101';OUT=ROOT/'report01'
SEEDS=(1024,2026,31415);ROLES=('P_FT','Ponly','C','Ccap','W','F')
def read(p):return json.loads(p.read_text())
def dump(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def table(p,rows):
    with p.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
assert (ROOT/'COMPLETED.json').exists();OUT.mkdir(exist_ok=False)
lookup={};absolute=[];people=[];blocks=[];records=[];solver=[];train=[]
for seed in SEEDS:
    for role in ROLES:
        if role=='P_FT':m=read(D8/f'seed{seed}/P_FT_BASELINE.json')
        elif role in ('Ponly','C','Ccap'):
            lam=.0001 if role=='C' else .01;m=read(D8/f'seed{seed}/{role}_lambda{lam:g}_METRICS.json')['metrics']
        else:m=read(ROOT/f'seed{seed}/{role}_METRICS.json')['metrics']
        lookup[(seed,role)]=m;absolute.append(dict(seed=seed,role=role,source='D9_NEW' if role in ('W','F') else 'D8_FROZEN',**m['phases']['val']))
        for key,out in [('people',people),('segments',blocks)]:
            for row in m[key]:out.append(dict(seed=seed,role=role,**row))
        for row in m['records']:records.append(dict(seed=seed,role=role,**{k:v for k,v in row.items() if k!='segments'}))
    train.append(dict(seed=seed,role='Ccap',**read(ROOT/f'seed{seed}/INITIAL_TRAIN_OBJECTIVES.json')))
    for role in ('W','F'):
        s=read(ROOT/f'seed{seed}/{role}_SOLVER.json');solver.append(s);train.append(dict(seed=seed,role=role,**s['train_objectives']))
table(OUT/'ABSOLUTE.csv',absolute);table(OUT/'ALL_PERSON.csv',people);table(OUT/'ALL_BLOCK.csv',blocks);table(OUT/'ALL_RECORD.csv',records);table(OUT/'TRAIN_OBJECTIVES.csv',train)
means=[]
for role in ROLES:
    rr=[r for r in absolute if r['role']==role]
    means.append(dict(role=role,seed_r=[r['r'] for r in rr],**{k:float(np.mean([r[k] for r in rr])) if all(r[k] is not None for r in rr) else None for k in ('r','r3s','MSE','weighted_single_MSE','disagreement','window_MSE','overlap_MSE')},undefined_3s=[r['window_r_undefined'] for r in rr]))
pairs=[];paired=[]
for a,b in (('F','W'),('W','Ccap'),('F','Ccap')):
    for seed in SEEDS:
        x=lookup[(seed,a)]['phases']['val'];y=lookup[(seed,b)]['phases']['val'];pairs.append(dict(comparison=f'{a}-{b}',status='primary' if b=='W' else 'secondary',seed=seed,**{f'delta_{k}':x[k]-y[k] for k in ('r','MSE','weighted_single_MSE','disagreement','overlap_MSE')}))
    rr=[r for r in pairs if r['comparison']==f'{a}-{b}'];paired.append(dict(comparison=f'{a}-{b}',status=rr[0]['status'],seed_delta_r=[r['delta_r'] for r in rr],positive_r_seeds=sum(r['delta_r']>0 for r in rr),**{k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k.startswith('delta_')}))
table(OUT/'PAIRED_CONTRASTS.csv',pairs)
train_means=[dict(role=role,**{k:float(np.mean([r[k] for r in train if r['role']==role])) for k in ('W_data','F_data','disagreement','regularizer','W_total','F_total','identity_error')}) for role in ('Ccap','W','F')]
summary=dict(scope='REPEATED_SOURCE_TRAIN_VAL_DEVELOPMENT_ONLY; no VAL selection, D8 references immutable',absolute_means=means,paired=paired,train_means=train_means,solvers=solver,cost=read(ROOT/'COMPLETED.json'),Hann_identity_max_abs=max(m['Hann_identity_max_abs'] for m in lookup.values()),goal_complete=False)
dump(OUT/'NUMERICAL_SUMMARY.json',summary)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180});sc=plt.cm.Set1.colors[:3]
fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
for i,seed in enumerate(SEEDS):axes[0].plot(range(6),[lookup[(seed,role)]['phases']['val']['r'] for role in ROLES],'o-',color=sc[i],alpha=.65,label=str(seed))
axes[0].plot(range(6),[r['r'] for r in means],'s--',color='black',label='3-seed mean');axes[0].axvline(3.5,color='gray',ls=':');axes[0].set_xticks(range(6),ROLES);axes[0].set_ylabel('VAL60s / person macro Pearson r');axes[0].set_title('Four frozen strong references + two new solves');axes[0].legend(fontsize=7)
for j,comparison in enumerate(('F-W','W-Ccap','F-Ccap')):
    for i,seed in enumerate(SEEDS):axes[1].scatter(j+(i-1)*.05,next(r['delta_r'] for r in pairs if r['comparison']==comparison and r['seed']==seed),color=sc[i])
    axes[1].scatter(j,next(r['delta_r'] for r in paired if r['comparison']==comparison),color='black',marker='D',s=30)
axes[1].axhline(0,color='black',lw=.8);axes[1].set_xticks(range(3),['PRIMARY\nF-W','W-Ccap','F-Ccap']);axes[1].set_ylabel('Paired VAL60s delta r');axes[1].set_title('No validation selection or confidence interval')
for i,seed in enumerate(SEEDS):axes[2].plot(range(3),[lookup[(seed,role)]['phases']['val']['MSE'] for role in ('Ccap','W','F')],'o-',color=sc[i],alpha=.65)
axes[2].set_xticks(range(3),['Ccap','W','F']);axes[2].set_ylabel('VAL60s / person fused target MSE');axes[2].set_title('Target MSE reported beside correlation');axes[2].ticklabel_format(axis='y',style='plain',useOffset=False)
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('D9: matched full-Hann targets, frozen FT+Ccap features; repeated source VAL only');fig.savefig(OUT/'01_MATCHED_RESULTS.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for ax,role in zip(axes,('W','F')):
    for i,seed in enumerate(SEEDS):
        s=next(r for r in solver if r['seed']==seed and r['role']==role);h=s['history'];ax.plot([r['iteration'] for r in h],[max(r['true_preconditioned_relative'],1e-30) for r in h],'o-',color=sc[i],label=f"{seed}: {s['iterations']} updates / {s['hessian_actions']} actions")
    ax.axhline(1e-8,color='black',ls=':',label='Frozen true-residual tolerance');ax.set_yscale('log');ax.set_xlabel('CG updates (cap 200)');ax.set_ylabel('True preconditioned residual / RHS norm');ax.set_title(role+' objective (Hessian action cap214)');ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.suptitle('Actual TRAIN stopping evidence; convergence checked on true residual, never VAL');fig.savefig(OUT/'02_TRUE_RESIDUALS.png');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained');roles=('Ccap','W','F')
tr=[next(r for r in train_means if r['role']==role) for role in roles]
axes[0].vlines(range(3),[r['F_data'] for r in tr],[r['W_data'] for r in tr],color='gray',lw=2,label='Gap = disagreement');axes[0].scatter(range(3),[r['F_data'] for r in tr],label='TRAIN fused physical MSE',color='steelblue');axes[0].scatter(range(3),[r['W_data'] for r in tr],label='TRAIN weighted-window MSE',color='darkorange');axes[0].set_ylabel('TRAIN unique physical-point mean MSE');axes[0].set_title('Same24,112,044 target points');axes[0].legend(fontsize=7)
val=[next(r for r in means if r['role']==role) for role in roles]
axes[1].vlines(range(3),[r['MSE'] for r in val],[r['weighted_single_MSE'] for r in val],color='gray',lw=2);axes[1].scatter(range(3),[r['MSE'] for r in val],label='VAL fused target MSE',color='steelblue');axes[1].scatter(range(3),[r['weighted_single_MSE'] for r in val],label='VAL weighted-window MSE',color='darkorange');axes[1].set_ylabel('VAL60s / person macro MSE');axes[1].set_title('VAL physical/subject weighting differs from TRAIN');axes[1].legend(fontsize=7)
axes[2].bar(range(3),[r['overlap_MSE'] for r in val],color=['gray','tab:blue','tab:orange']);axes[2].set_ylabel('VAL uniform96 overlap MSE');axes[2].set_title('Disagreement is not accuracy or a selection rule')
for ax in axes:ax.set_xticks(range(3),roles);ax.ticklabel_format(axis='y',style='plain',useOffset=False);ax.grid(alpha=.2)
fig.suptitle('Supervised physical-target comparison; no history input, no consistency penalty');fig.savefig(OUT/'03_PHYSICAL_DECOMPOSITION.png');plt.close(fig)
dump(OUT/'COMPLETED.json',dict(plots=3,new_solves=6,immutable_reference_roles=4,person_rows=len(people),block_rows=len(blocks),record_rows=len(records),new_signal_or_model_reads=False))
print(json.dumps(dict(absolute_means=means,paired=paired,train_means=train_means,convergence=[{k:s[k] for k in ('seed','role','converged','iterations','hessian_actions','true_preconditioned_relative')} for s in solver]),indent=2))
