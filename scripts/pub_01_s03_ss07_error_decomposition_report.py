"""Four static plots and compact summaries of already-computed diagnostic rows."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pub_01_s03_ss07_error_decomposition_core import *
assert read(ROOT/'COMPLETED.json')['effective_residual_NN_windows']==2508111
out=ROOT/'results/report01';out.mkdir(exist_ok=False);figs=out/'figures';figs.mkdir()
phases=[];gradients=[];records=[]
for seed in SEEDS:
    folder=ROOT/f'results/seed{seed}';phases+=read(folder/'SUMMARY.json');gradients.append(read(folder/'GRADIENT_SUMMARY.json'))
    records+=list(csv.DictReader((folder/'GRADIENT_RECORDS.csv').open()))
def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
csvwrite(out/'PHASE_SUMMARY.csv',phases);csvwrite(out/'GRADIENT_SEED.csv',gradients)
def mean(rows,key):return float(np.mean([x[key] for x in rows])) if all(x[key] is not None for x in rows) else None
model=[]
for phase in ['train','val']:
    for arm in ARMS:
        rr=[r for r in phases if r['arm']==arm and r['phase']==phase]
        model.append(dict(phase=phase,arm=arm,**{k:mean(rr,k) for k in NUMERIC},undefined_projection_blocks=sum(r['undefined_projection_blocks'] for r in rr),undefined_Pearson_blocks=sum(r['undefined_Pearson_blocks'] for r in rr)))
csvwrite(out/'MODEL_SUMMARY.csv',model)
dump(out/'NUMERICAL_SUMMARY.json',dict(model_summaries=model,gradients=gradients,new_NN=0,all_statistics_descriptive=True,goal_complete=False))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=['#6c757d','#457b9d','#e9a23b'];x=np.arange(3)
def save(fig,name):fig.savefig(figs/name,dpi=160,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for j,phase in enumerate(['train','val']):
    for z,key in enumerate(['MSE_bias','MSE_parallel','MSE_orthogonal']):axes[j].bar(x+(z-1)*.22,[next(v for v in model if v['phase']==phase and v['arm']==a)[key] for a in ARMS],width=.22,color=colors[z],label=key)
    axes[j].scatter(x,[next(v for v in model if v['phase']==phase and v['arm']==a)['delta_MSE'] for a in ARMS],color='black',marker='D',label='Observed delta MSE',zorder=5)
    axes[j].axhline(0,color='black',lw=.8);axes[j].set(xticks=x,xticklabels=ARMS,ylabel='Person-macro delta MSE',title=phase.upper()+': same physical60s support');axes[j].legend(fontsize=8)
save(fig,'01_MSE_DECOMPOSITION.png')
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for j,phase in enumerate(['train','val']):
    for z,key in enumerate(['r_anchor_normalization_delta','r_orthogonal_delta']):axes[j].bar(x+(z-.5)*.28,[next(v for v in model if v['phase']==phase and v['arm']==a)[key] for a in ARMS],width=.28,color=colors[z+1],label=key)
    axes[j].scatter(x,[next(v for v in model if v['phase']==phase and v['arm']==a)['delta_r'] for a in ARMS],color='black',marker='D',label='Observed delta r',zorder=5)
    axes[j].axhline(0,color='black',lw=.8);axes[j].set(xticks=x,xticklabels=ARMS,ylabel='Person-macro delta Pearson r',title=phase.upper()+': evaluate identities per block, then average');axes[j].legend(fontsize=8)
save(fig,'02_PEARSON_DECOMPOSITION.png')
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');pairs=['cos_M_F','cos_F_R','cos_M_R']
for i,g in enumerate(gradients):
    axes[0].scatter(x+(i-1)*.08,[g[k] for k in pairs],label=str(g['seed']),s=45)
    axes[1].plot([0,1],[g['physical_cos_F_R_before_Hann'],g['cos_F_R']],marker='o',label=str(g['seed']))
for ax in axes:ax.axhline(0,color='black',lw=.8);ax.set_ylim(-1.05,1.05);ax.legend()
axes[0].set(xticks=x,xticklabels=['window MSE / fused MSE','fused MSE / -Pearson','window MSE / -Pearson'],ylabel='Cosine of summed TRAIN output gradients',title='At frozen anchor; all use same learnability mask')
axes[0].tick_params(axis='x',labelrotation=12)
axes[1].set(xticks=[0,1],xticklabels=['Physical before Hann','Window adjoint + i>=9'],ylabel='Cosine gF / gR',title='Coordinate/support effects; no parameter-gradient claim')
save(fig,'03_GRADIENT_COSINES.png')
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for ax,key in zip(axes,pairs):
    for seed in SEEDS:
        rr=[r for r in records if int(r['seed'])==seed];v=[float(r[key]) for r in rr if r[key] not in ('','None')]
        ax.hist(v,bins=np.linspace(-1,1,31),histtype='step',label=f"{seed}, defined {len(v)}/{len(rr)}")
    ax.axvline(0,color='black',lw=.8);ax.set(title=key,xlabel='Record-level cosine (not independent records)',ylabel='Record count');ax.legend(fontsize=8)
save(fig,'04_GRADIENT_RECORD_DISTRIBUTIONS.png')
dump(out/'PRESENTATION_COMPLETED.json',dict(figures=4,new_NN=0,independent_review_pending=True))
print(json.dumps(dict(model=model,gradients=gradients),allow_nan=False),flush=True)
