"""D10 completed frozen FT scores -> static summaries/tables only on2203."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01');ROOT=BASE/'s03_finetuned_internal_audit_20261101';OUT=ROOT/'report01'
SEEDS=(1024,2026,31415);ROLES=('P_FT','Ponly','C','Ccap','W','F');PHASES=('test_seen','test_heldout')
def read(p):return json.loads(p.read_text())
def dump(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def csvfile(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert (ROOT/'COMPLETED.json').exists();OUT.mkdir(exist_ok=False)
s=read(ROOT/'SUMMARY.json');protocol=read(ROOT/'PROTOCOL.json');metrics={}
people=[];blocks=[];records=[];families=[];absolute=[]
for seed in SEEDS:
    for role in ROLES:
        m=read(ROOT/f'seed{seed}/{role}_METRICS.json');metrics[(seed,role)]=m
        for key,out in [('people',people),('segments',blocks)]:
            for row in m[key]:out.append(dict(seed=seed,role=role,**row))
        for row in m['records']:records.append(dict(seed=seed,role=role,**{k:v for k,v in row.items() if k!='segments'}))
        for phase in PHASES:
            for family in sorted({b['source_family'] for b in m['segments'] if b['phase']==phase}):
                bb=[b for b in m['segments'] if b['phase']==phase and b['source_family']==family];person_values=[np.mean([b['r'] for b in bb if b['subject']==person]) for person in sorted({b['subject'] for b in bb})]
                families.append(dict(seed=seed,role=role,phase=phase,source_family=family,people=len(person_values),records=len({b['record'] for b in bb}),blocks=len(bb),family_person_macro_r=float(np.mean(person_values))))
            absolute.append(dict(seed=seed,role=role,phase=phase,**m['phases'][phase]))
        absolute.append(dict(seed=seed,role=role,phase='weighted_2to1',**{k:m['weighted'][k] for k in ('r','MSE','weighted_single_MSE','disagreement')}))
csvfile(OUT/'ALL_PERSON.csv',people);csvfile(OUT/'ALL_BLOCK.csv',blocks);csvfile(OUT/'ALL_RECORD.csv',records);csvfile(OUT/'FAMILY_DESCRIPTIVE.csv',families)
csvfile(OUT/'ABSOLUTE.csv',absolute)
pairs=[tuple(p) for p in protocol['primary_pairs']+protocol['secondary_pairs']];contrasts=[];contrasts_summary=[]
for index,(a,b) in enumerate(pairs):
    for phase in PHASES+('weighted_2to1',):
        vals=[]
        for seed in SEEDS:
            ma=metrics[(seed,a)]['weighted'] if phase=='weighted_2to1' else metrics[(seed,a)]['phases'][phase]
            mb=metrics[(seed,b)]['weighted'] if phase=='weighted_2to1' else metrics[(seed,b)]['phases'][phase]
            delta=ma['r']-mb['r'];vals.append(delta);contrasts.append(dict(comparison=f'{a}-{b}',status='primary' if index<3 else 'secondary',phase=phase,seed=seed,delta_r=delta,delta_MSE=ma['MSE']-mb['MSE']))
        contrasts_summary.append(dict(comparison=f'{a}-{b}',status='primary' if index<3 else 'secondary',phase=phase,deltas=vals,mean=float(np.mean(vals)),positive_seeds=sum(v>0 for v in vals)))
csvfile(OUT/'PAIRED_CONTRASTS.csv',contrasts)
means=[]
for role in ROLES:
    for phase in PHASES+('weighted_2to1',):
        rows=[r for r in absolute if r['role']==role and r['phase']==phase]
        item=dict(role=role,phase=phase,mean_r=float(np.mean([r['r'] for r in rows])),seed_r=[r['r'] for r in rows],mean_MSE=float(np.mean([r['MSE'] for r in rows])),mean_weighted_single_MSE=float(np.mean([r['weighted_single_MSE'] for r in rows])),mean_disagreement=float(np.mean([r['disagreement'] for r in rows])))
        if phase!='weighted_2to1':
            item.update(mean_r3s=float(np.mean([r['r3s'] for r in rows])) if all(r['r3s'] is not None for r in rows) else None,
                window_r_undefined=[r['window_r_undefined'] for r in rows],mean_overlap_MSE=float(np.mean([r['overlap_MSE'] for r in rows])),mean_overlap_NRMSE=float(np.mean([r['overlap_NRMSE'] for r in rows])) if all(r['overlap_NRMSE'] is not None for r in rows) else None)
        means.append(item)
bridges=[dict(seed=seed,**{k:v for k,v in read(ROOT/f'seed{seed}/RAW_FT_ALL144_PASSED.json').items() if k!='records'}) for seed in SEEDS]
summary=dict(scientific_status='POST-HOC INTERNAL EXPLORATORY',absolute_means=means,paired=contrasts_summary,bridges=bridges,cost=read(ROOT/'COMPLETED.json'),
    seed_costs=[read(ROOT/f'seed{seed}/COMPLETED.json') for seed in SEEDS],maximum_Hann_identity_abs=max(m['Hann_identity_max_abs'] for m in metrics.values()),new_fits=0,FT_raw_encoder_cache=True,
    limitations='Old tests exposed; only original3seeds and fixed lambdas; do not retrain/select from this audit; no independent confirmation or history contribution claim.')
dump(OUT/'NUMERICAL_SUMMARY.json',summary)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
colors=dict(zip(ROLES,('gray',)+plt.cm.tab10.colors[:5]));sc=plt.cm.Set1.colors[:3]
fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
for ax,phase,label in zip(axes,PHASES+('weighted_2to1',),('Seen subjects','Held-out subjects','Original 2:1 weighting')):
    for i,seed in enumerate(SEEDS):
        vals=[(metrics[(seed,role)]['weighted'] if phase=='weighted_2to1' else metrics[(seed,role)]['phases'][phase])['r'] for role in ROLES]
        ax.plot(range(6),vals,'o-',color=sc[i],alpha=.65,label=f'seed {seed}')
    ax.plot(range(6),[next(m for m in means if m['role']==role and m['phase']==phase)['mean_r'] for role in ROLES],'s--',color='black',label='3-seed mean',lw=1.7)
    ax.set_xticks(range(6),['P_FT*','P-only','C','Ccap','W','F']);ax.set_ylabel('60-s block to person macro Pearson r');ax.set_title(label);ax.grid(alpha=.2)
axes[0].legend(fontsize=8);fig.suptitle('D10: frozen FT readouts; POST-HOC INTERNAL EXPLORATORY, not independent confirmation')
fig.savefig(OUT/'01_ABSOLUTE_PAIRED.png');plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
for index,(ax,(a,b)) in enumerate(zip(axes.flat,pairs)):
    for i,seed in enumerate(SEEDS):
        rr=[r for r in contrasts if r['comparison']==f'{a}-{b}' and r['seed']==seed]
        ax.plot(range(3),[r['delta_r'] for r in rr],'o-',color=sc[i],label=f'{seed}')
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),['Seen','Held-out','2:1']);ax.set_title(('PRIMARY: ' if index<3 else 'Secondary: ')+f'{a} minus {b}');ax.set_ylabel('Paired delta r');ax.grid(alpha=.2)
axes.flat[0].legend(title='seed',fontsize=8);fig.suptitle('All prespecified paired contrasts; no model selection or confidence intervals')
fig.savefig(OUT/'02_FROZEN_CONTRASTS.png');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
for ax,phase,label in zip(axes[:2],PHASES,('Seen','Held-out')):
    vals=[next(m for m in means if m['role']==role and m['phase']==phase)['mean_overlap_MSE'] for role in ROLES]
    ax.bar(range(6),vals,color=[colors[r] for r in ROLES]);ax.set_xticks(range(6),['P_FT*','P-only','C','Ccap','W','F']);ax.set_yscale('log');ax.set_ylabel('Uniform96 overlap MSE before Hann (log)');ax.set_title(label+' continuity (not a selection endpoint)');ax.grid(alpha=.2)
selected=[next(m for m in means if m['role']==role and m['phase']=='weighted_2to1') for role in ROLES]
fused=np.array([r['mean_MSE'] for r in selected]);weighted=np.array([r['mean_weighted_single_MSE'] for r in selected])
axes[2].vlines(range(6),fused,weighted,color='gray',lw=2,label='Gap = Hann disagreement');axes[2].scatter(range(6),fused,color='steelblue',label='Fused target MSE');axes[2].scatter(range(6),weighted,color='darkorange',label='Weighted single-window MSE')
axes[2].set_xticks(range(6),['P_FT*','P-only','C','Ccap','W','F']);axes[2].set_ylabel('Original 2:1 weighted person macro MSE');axes[2].set_title('Accuracy / disagreement decomposition');axes[2].legend(fontsize=7);axes[2].grid(alpha=.2)
fig.suptitle('Descriptive secondary metrics only: unchanged Hann weights, no fitting or consistency optimization')
fig.savefig(OUT/'03_CONTINUITY_MSE.png');plt.close(fig)
dump(OUT/'COMPLETED.json',dict(plots=3,role_split_scores=36,people_rows=len(people),block_rows=len(blocks),record_rows=len(records),family_rows=len(families),new_model_or_input_signal_reads=False))
print(json.dumps({k:v for k,v in summary.items() if k not in ('seed_costs',)},indent=2))
