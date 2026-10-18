"""Static summary from actual completed metrics; execute on server2203 only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261015/transformer_available01')
contrast=root/'training_seed1024_attempt04';envelope=root/'envelope_readout_seed1024'
c=json.loads((contrast/'result.json').read_text());e=json.loads((envelope/'result.json').read_text())
ch=json.loads((contrast/'history.json').read_text());eh=json.loads((envelope/'history.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for ax,h,key,title,best in [(axes[0,0],ch,'validation_loss','A  Contrastive validation loss',c['best_epoch']),
    (axes[0,1],eh,'validation_negative_Pearson','B  Frozen encoder readout validation',e['best_epoch'])]:
    epochs=np.array([r['epoch']+1 for r in h]);ax.plot(epochs,[r[key] for r in h],color='#28618f',lw=1.8)
    ax.axvline(best+1,color='#888888',ls='--',lw=1);ax.set(xlabel='Epoch',ylabel=key.replace('_',' '),title=title)
    ax.text(.98,.98,f'Selected epoch {best+1}',ha='right',va='top',transform=ax.transAxes)
groups=['test_seen','test_heldout'];labels=['Seen subjects','Held-out subjects'];x=np.arange(2);w=.32
for ax,result,key,nullkey,title in [(axes[1,0],c,'accuracy_2afc','EEG_shuffle_accuracy','C  Derived 2AFC: same story, 4 s offset'),
    (axes[1,1],e,'pearson_r','EEG_shuffle_r','D  Direct envelope reconstruction')]:
    vals=np.array([result['test'][g][key] for g in groups]);null=np.array([result['test'][g][nullkey] for g in groups])
    ci=np.array([result['test'][g]['subject_bootstrap95'] for g in groups]);error=np.vstack([vals-ci[:,0],ci[:,1]-vals])
    if key=='accuracy_2afc':vals*=100;null*=100;error*=100
    ax.bar(x-w/2,vals,w,color='#28618f',label='EEG',yerr=error,capsize=3)
    ax.bar(x+w/2,null,w,color='#a9a9a9',label='EEG shuffled')
    for i,v in enumerate(vals):ax.annotate(f'{v:.2f}' if key=='accuracy_2afc' else f'{v:.3f}',(i-w/2,v+error[1,i]),xytext=(0,5),textcoords='offset points',ha='center')
    ax.set_xticks(x,labels);ax.set_title(title);ax.set_ylabel('Subject-macro accuracy (%)' if key=='accuracy_2afc' else 'Subject-macro Pearson r')
    ax.legend(frameon=True,facecolor='white',framealpha=.95,loc='lower right' if key=='accuracy_2afc' else 'upper right');ax.margins(y=.3)
    if key=='accuracy_2afc':ax.axhline(50,color='#555555',ls=':',lw=1);ax.set_ylim(0,100)
fig.suptitle('Transformer-CLIP functional reproduction | available public folds | one seed, one model',fontsize=13)
fig.savefig(root/'test_summary.png');plt.close(fig)
print(root/'test_summary.png')
