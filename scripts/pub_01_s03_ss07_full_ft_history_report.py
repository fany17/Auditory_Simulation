"""All-nine-fit results presentation. JSON only, zero neural forwards."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pub_01_s03_ss07_full_ft_history_core import ROOT,SEEDS,ARMS,read,dump

def write_csv(p,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

assert read(ROOT/'QUEUE_COMPLETED.json')['new_fits']==9
output=ROOT/'report01';output.mkdir(exist_ok=False);figures=output/'figures';figures.mkdir()
fits=[];epochs=[];people=[];blocks=[];boundary=[];family=[];scores={};costs=[]
for seed in SEEDS:
    anchor=read(ROOT/f'fits/H_full_seed{seed}/INCUMBENT.json');scores['Anchor',seed]=anchor
    for arm in ARMS:
        folder=ROOT/f'fits/{arm}_seed{seed}';cp=read(folder/'COMPLETED.json');assert cp['epochs']==20 and cp['updates']==78900
        history=[json.loads(x) for x in (folder/'HISTORY.jsonl').read_text().splitlines()];assert len(history)==20 and [x['epoch'] for x in history]==list(range(20))
        assert np.argmax([anchor['phases']['val']['r']]+[x['validation']['r'] for x in history])-1==cp['selected_epoch']
        metrics=read(folder/'SELECTED_METRICS.json');scores[arm,seed]=metrics;v=metrics['phases']['val']
        fits.append(dict(seed=seed,arm=arm,selected_epoch=cp['selected_epoch'],r=v['r'],MSE=v['MSE'],r3s=v['r3s'],window_MSE=v['window_MSE'],
            overlap_MSE=v['overlap_MSE'],warm_r=metrics['fixed_warm_auxiliary']['r'],parameters=cp['new_trainable_parameters'],
            wall_seconds=cp['wall_seconds'],CPU_seconds=cp['CPU_seconds'],peak_RSS_KiB=cp['peak_RSS_KiB'],peak_GPU_allocated_bytes=cp['peak_GPU_allocated_bytes'],updates=cp['updates']))
        costs.append(cp)
        for h in history:epochs.append(dict(seed=seed,arm=arm,epoch=h['epoch'],train_window_MSE=h['train_window_MSE'],validation_r=h['validation']['r'],validation_MSE=h['validation']['MSE'],updates=h['updates'],epoch_wall_seconds=h['epoch_wall_seconds']))
    scores['H_older',seed]=read(ROOT/f'fits/H_full_seed{seed}/OLDER_METRICS.json')
for (role,seed),m in scores.items():
    for row in m['people']:people.append(dict(role=role,seed=seed,**row))
    for row in m['segments']:blocks.append(dict(role=role,seed=seed,**row))
    for row in m['boundary_tail_records']:boundary.append(dict(role=role,seed=seed,**row))
    for fam in sorted({b['source_family'] for b in m['segments']}):
        bb=[b for b in m['segments'] if b['source_family']==fam];ss=sorted({b['subject'] for b in bb})
        family.append(dict(role=role,seed=seed,source_family=fam,people=len(ss),blocks=len(bb),
            r=float(np.mean([np.mean([b['r'] for b in bb if b['subject']==s]) for s in ss])),descriptive_only=True))
contrasts=[]
for b in ['Anchor','C_match','P_prior','H_older']:
    diff=[scores['H_full',s]['phases']['val']['r']-scores[b,s]['phases']['val']['r'] for s in SEEDS]
    contrasts.append(dict(comparison='H_full-'+b,mean_delta=float(np.mean(diff)),positive_seeds=sum(x>0 for x in diff),seed1024=diff[0],seed2026=diff[1],seed31415=diff[2]))
summary=dict(fits=9,epochs=180,total_updates=sum(c['updates'] for c in costs),all_prespecified_seed_direction_conditions_met=all(c['positive_seeds']==3 for c in contrasts),
    contrasts=contrasts,selected_incumbents=sum(f['selected_epoch']==-1 for f in fits),new_encoder_head_forwards=0,
    formal_new_residual_NN_window_forwards=sum(c['new_residual_NN_window_forwards'] for c in costs),
    summed_CPU_seconds=sum(c['CPU_seconds'] for c in costs),parallel_fit_wall_times_not_total_wall=True,descriptive_development_only=True,new_report_NN=0,goal_complete=False)
for name,rows in [('FIT_RESULTS.csv',fits),('ALL_EPOCHS.csv',epochs),('ALL_PERSON.csv',people),('ALL_BLOCKS.csv',blocks),('FAMILY_DESCRIPTIVE.csv',family),('CONTRASTS.csv',contrasts)]:write_csv(output/name,rows)
dump(output/'NUMERICAL_SUMMARY.json',summary);dump(output/'BOUNDARY_TAIL.json',boundary);dump(output/'FORMAL_COSTS.json',costs)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=['#6c757d','#457b9d','#e9a23b','#b23a48','#8b6bb1'];roles=['Anchor','H_full','C_match','P_prior','H_older']
def save(fig,name):fig.savefig(figures/name,dpi=160,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for i,role in enumerate(roles):
    rr=[scores[role,s]['phases']['val']['r'] for s in SEEDS];axes[0].scatter(i+np.array([-.08,0,.08]),rr,color=colors[i],s=35)
    axes[0].plot([i-.15,i+.15],[np.mean(rr)]*2,color=colors[i],lw=3)
for i,c in enumerate(contrasts):
    vals=[c[f'seed{s}'] for s in SEEDS];axes[1].scatter(vals,[i-.08,i,i+.08],color=['#457b9d','#e9a23b','#b23a48']);axes[1].plot([c['mean_delta']],[i],marker='D',color='black')
axes[0].set(xticks=range(5),xticklabels=roles,ylabel='Original VAL60s/person Pearson r',title='All three seeds; earliest best includes frozen anchor')
axes[1].axvline(0,color='black',lw=.8);axes[1].set(yticks=range(4),yticklabels=[c['comparison'] for c in contrasts],xlabel='Paired seed difference in r',title='Fixed development contrasts; no inferential CIs')
save(fig,'01_ABSOLUTE_AND_CONTRASTS.png')
fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
for i,arm in enumerate(ARMS):
    for si,seed in enumerate(SEEDS):
        rows=[x for x in epochs if x['arm']==arm and x['seed']==seed];f=next(x for x in fits if x['arm']==arm and x['seed']==seed);c=['#457b9d','#e9a23b','#b23a48'][si]
        axes[i,0].plot(range(1,21),[x['train_window_MSE'] for x in rows],color=c,label=str(seed))
        vr=[scores['Anchor',seed]['phases']['val']['r']]+[x['validation_r'] for x in rows];axes[i,1].plot(range(21),vr,color=c,label=str(seed));axes[i,1].scatter([f['selected_epoch']+1],[f['r']],marker='s',s=45,color=c)
    axes[i,0].set(title=arm+' online TRAIN window MSE',xlabel='Added epoch',ylabel='MSE');axes[i,1].set(title=arm+' full VAL60s/person r',xlabel='Added epoch (0 = incumbent)',ylabel='r');axes[i,1].legend()
save(fig,'02_ALL_TRAIN_VAL_CURVES.png')
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for i,role in enumerate(roles):
    for j,key in enumerate(['MSE','window_MSE','overlap_MSE']):
        vals=[scores[role,s]['phases']['val'][key] for s in SEEDS];axes[j].scatter(i+np.array([-.08,0,.08]),vals,color=colors[i]);axes[j].set(xticks=range(5),xticklabels=roles,title=key,ylabel='Fixed envelope numeric units squared')
save(fig,'03_WINDOW_AND_FUSED_ERRORS.png')
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');labels=[f"{f['arm']}\n{f['seed']}" for f in fits]
axes[0].bar(range(9),[f['wall_seconds']/60 for f in fits],color=['#457b9d','#e9a23b','#b23a48']*3);axes[0].set(xticks=range(9),xticklabels=labels,ylabel='Fit elapsed minutes',title='Each fit: 20 epochs, 78,900 updates; parallel elapsed not summed')
axes[1].bar(range(9),[f['selected_epoch']+1 for f in fits],color=['#457b9d','#e9a23b','#b23a48']*3);axes[1].set(xticks=range(9),xticklabels=labels,ylim=(0,21),ylabel='Selected added epoch (0 = incumbent)',title='Actual earliest checkpoint, including zero-residual incumbent')
save(fig,'04_COSTS_AND_SELECTED_EPOCHS.png')
dump(output/'PRESENTATION_COMPLETED.json',dict(figures=4,NN_forwards=0,all_nine_fits=True,independent_review_still_required=True))
print(json.dumps(summary),flush=True)
