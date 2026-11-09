"""D11 completed JSON summaries to three static figures; server2203 only."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_observation_state_probe_20261101')
OUT=ROOT/'report01'
SEEDS=(1024,2026,31415)
ROLES=('Ccap','Add','CurrentCov','RecordPrefix','PersonTRAIN','PastCov','PastOlder','PastNeutral')
PAIRS=(('PastCov','Ccap'),('PastCov','CurrentCov'),('PastCov','RecordPrefix'),
       ('PastCov','PersonTRAIN'),('PastCov','PastOlder'),('PastCov','PastNeutral'))


def read(path):
    return json.loads(path.read_text())


def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def table(path,rows):
    assert rows
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


assert (ROOT/'COMPLETED.json').exists()
OUT.mkdir(exist_ok=False)
summary=read(ROOT/'SUMMARY.json')
models={};absolute=[];people=[];blocks=[];records=[]
for seed in SEEDS:
    for role in ROLES:
        data=read(ROOT/f'seed{seed}/{role}_METRICS.json')
        models[(seed,role)]=data
        full=data['full'];v=full['phases']['val']
        absolute.append(dict(seed=seed,role=role,**v,
             warm_r=data['subsets']['warm']['phases']['val']['r'],
             cold_r=data['subsets']['cold']['phases']['val']['r']))
        for subset,m in [('full',full),*data['subsets'].items()]:
            people.extend(dict(seed=seed,role=role,subset=subset,**p) for p in m['people'])
            blocks.extend(dict(seed=seed,role=role,subset=subset,**p) for p in m['segments'])
        records.extend(dict(seed=seed,role=role,**{k:v for k,v in r.items() if k!='segments'})
                       for r in full['records'])
table(OUT/'ABSOLUTE.csv',absolute)
table(OUT/'ALL_PERSON.csv',people)
table(OUT/'ALL_BLOCK.csv',blocks)
table(OUT/'ALL_RECORD.csv',records)
contrasts=[];person_deltas=[]
for a,b in PAIRS+(('PastCov','Add'),):
    for seed in SEEDS:
        ma,mb=models[(seed,a)],models[(seed,b)]
        for subset in ('full','warm','cold'):
            aa=ma['full'] if subset=='full' else ma['subsets'][subset]
            bb=mb['full'] if subset=='full' else mb['subsets'][subset]
            contrasts.append(dict(comparison=f'{a}-{b}',status='secondary' if b=='Add' else 'primary',seed=seed,subset=subset,
                                  delta_r=aa['phases']['val']['r']-bb['phases']['val']['r']))
            kb={(p['phase'],p['subject']):p for p in bb['people']}
            for p in aa['people']:
                other=kb[(p['phase'],p['subject'])]
                person_deltas.append(dict(comparison=f'{a}-{b}',status='secondary' if b=='Add' else 'primary',seed=seed,subset=subset,
                    subject=p['subject'],delta_r=p['r']-other['r']))
table(OUT/'PAIRED_CONTRASTS.csv',contrasts)
table(OUT/'PERSON_DIFFERENCES.csv',person_deltas)
table(OUT/'SECONDARY_PAST_MINUS_ADD.csv',[r for r in contrasts if r['status']=='secondary'])
table(OUT/'SECONDARY_PAST_MINUS_ADD_PERSON.csv',[r for r in person_deltas if r['status']=='secondary'])
references=[];reference_pins={}
for seed in SEEDS:
    source=ROOT.parent/f's03_physical_objective_20261101/seed{seed}/F_METRICS.json'
    reference_pins[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    old=read(source)['metrics']['phases']['val']
    past=next(r for r in absolute if r['seed']==seed and r['role']=='PastCov')
    references.append(dict(seed=seed,role='D9_F_existing_full_VAL_only',source_file=str(source),
        source_sha256=reference_pins[str(source)],r=old['r'],MSE=old['MSE'],r3s=old['r3s'],
        window_MSE=old['window_MSE'],disagreement=old['disagreement'],
        PastCov_r=past['r'],PastCov_minus_F=past['r']-old['r'],new_prediction_or_scoring=False))
table(OUT/'F_FROZEN_FULL_REFERENCE.csv',references)
dump(OUT/'REFERENCE_PROVENANCE.json',dict(source_pins=reference_pins,
    scope='Existing frozen D9 full-VAL JSON only; no new F prediction, warm or cold score'))
fit_diagnostics=[]
for seed in SEEDS:
    for role in ROLES[1:6]:
        normal=read(ROOT/f'seed{seed}/{role}_NORMAL_CHECK.json')
        dimension=32 if role=='Add' else 296
        fit_diagnostics.append(dict(seed=seed,role=role,dimension=dimension,
             parameters=(dimension+1)*192,lambda_value=.01,
             effective_df_excludes_intercept=normal['effective_df'],
             normal_equation_max_abs=normal['normal_equation_max_abs']))
table(OUT/'FIT_DIAGNOSTICS.csv',fit_diagnostics)
family=[]
for seed in SEEDS:
    for role in ROLES:
        rr=models[(seed,role)]['full']['segments']
        for name in sorted({r['source_family'] for r in rr}):
            group=[r for r in rr if r['source_family']==name]
            family.append(dict(seed=seed,role=role,source_family=name,
                               blocks=len(group),r=float(np.mean([r['r'] for r in group])),
                               definition='existing story fallback label, not verified speaker/book family'))
table(OUT/'FAMILY_DESCRIPTIVE.csv',family)
means=[]
for role in ROLES:
    rows=[r for r in absolute if r['role']==role]
    keys=('r','r3s','MSE','window_MSE','weighted_single_MSE','disagreement','overlap_MSE','warm_r','cold_r')
    means.append(dict(role=role,seed_r=[r['r'] for r in rows],
                 **{k:float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None for k in keys}))
details=dict(means=means,primary_contrasts=summary['primary_contrasts'],
             fit_diagnostics=fit_diagnostics,
             secondary_Past_minus_Add=[r for r in contrasts if r['status']=='secondary'],
             frozen_F_full_references=references,
             condition_met=summary['condition_met'],cost=read(ROOT/'COMPLETED.json'),
             state=read(ROOT/'STATE_NUMERICS.json'),
             maximum_Hann_identity=max(models[key]['full']['Hann_identity_max_abs'] for key in models),
             actual_scoring_roles_per_seed=8,new_VAL_score_sets=24,
             first_warm_grid_start=7680,cold_block_start=0,
             goal_complete=False,scientific_status=summary['scientific_status'])
dump(OUT/'NUMERICAL_SUMMARY.json',details)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
colors=['gray','slategray','steelblue','darkorange','purple','seagreen','indianred','saddlebrown']
fig,axes=plt.subplots(1,2,figsize=(15,5.5),layout='constrained')
for ax,key,label in zip(axes,('r','warm_r'),('Full original VAL60s/person r','Warm original blocks (start >=7680) r')):
    for i,row in enumerate(means):
        ax.bar(i,row[key],color=colors[i],alpha=.7)
        for j,seed in enumerate(SEEDS):
            raw=next(r for r in absolute if r['role']==row['role'] and r['seed']==seed)
            ax.scatter(i+(j-1)*.13,raw[key],s=20,color='black',alpha=.65)
    ax.set_xticks(range(8),ROLES,rotation=35,ha='right');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
axes[0].axhline(np.mean([r['r'] for r in references]),color='black',ls='--',lw=1,
    label='Existing D9 F full VAL mean')
axes[0].legend(fontsize=8)
fig.suptitle('D11: 15 fixed TRAIN fits + 6 nofit interventions; repeated known-person VAL')
fig.savefig(OUT/'01_COMPLETE_MATRIX.png');plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for ax,(a,b) in zip(axes.flat,PAIRS):
    for j,seed in enumerate(SEEDS):
        rows=[r for r in contrasts if r['comparison']==f'{a}-{b}' and r['seed']==seed]
        ax.plot((0,1),[next(r['delta_r'] for r in rows if r['subset']==s) for s in ('full','warm')],
                'o-',alpha=.75,label=str(seed))
    ax.axhline(0,color='black',lw=.8)
    ax.set_xticks((0,1),('Full','Fixed warm'));ax.set_title(f'{a} minus {b}')
    ax.set_ylabel('Paired delta r');ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=8)
fig.suptitle('Six fixed descriptive conditions; no significance test or new interval')
fig.savefig(OUT/'02_FIXED_CONTRASTS.png');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout='constrained')
for i,row in enumerate(means):
    axes[0].scatter(row['MSE'],row['r'],color=colors[i])
    axes[0].annotate(row['role'],(row['MSE'],row['r']),fontsize=7,xytext=(3,3),textcoords='offset points')
    axes[1].plot([i,i],[row['MSE'],row['weighted_single_MSE']],color=colors[i],lw=2)
    axes[1].scatter(i,row['MSE'],color='steelblue',s=16)
    axes[1].scatter(i,row['weighted_single_MSE'],color='darkorange',s=16)
axes[0].set(xlabel='Full fused target MSE',ylabel='Full VAL r',title='Accuracy and error jointly reported')
axes[1].set_xticks(range(8),ROLES,rotation=45,ha='right');axes[1].set_ylabel('Target MSE')
axes[1].set_title('Blue fused / orange weighted single; gap D')
axes[2].bar(range(1,9),details['state']['PCA_eigenvalues'],color='seagreen')
axes[2].set(xlabel='Fixed pooled TRAIN PCA component',ylabel='Eigenvalue',title='State spectrum is not evidence of task gain')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('D11 numerical checks and limits: no covariance inverse, no NN update, no causal claim')
fig.savefig(OUT/'03_ERROR_AND_STATE.png');plt.close(fig)
dump(OUT/'COMPLETED.json',dict(plots=3,VAL_score_sets=24,new_signal_or_model_reads=False,
                              person_rows=len(people),block_rows=len(blocks),record_rows=len(records)))
print(json.dumps(dict(means=means,condition_met=details['condition_met'],cost=details['cost']),indent=2))
