"""Static scientific figures from reviewed SS03 metrics; server2203 only."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_architecture_20261019');out=root/'review01'
with (out/'RESULTS.csv').open() as f:rows=list(csv.DictReader(f))
with (out/'paired_comparisons.csv').open() as f:paired=list(csv.DictReader(f))
protocol=json.loads((root/'formal_protocol.json').read_text());variants=protocol['variants'];colors=['#888888','#d8912d','#276d9e','#a84953']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
for variant,color in zip(variants,colors):
    histories=[json.loads((root/'formal01'/f'{variant}_seed{seed}'/'history.json').read_text()) for seed in protocol['seeds']]
    steps=np.array([r['step'] for r in histories[0]])/1000;values=np.array([[r['validation_subject_macro_60s_r'] for r in h] for h in histories])
    axes[0].plot(steps,values.mean(0),color=color,label=variant,lw=1.8);axes[0].fill_between(steps,values.min(0),values.max(0),color=color,alpha=.12)
axes[0].set(xlabel='Optimizer steps (thousands)',ylabel='Validation subject-macro r',title='A  Same budget, validation-only selection');axes[0].legend(frameon=False,fontsize=8)
for ax,group,title in zip(axes[1:],['test_seen','test_heldout'],['B  Seen subjects, held-out story IDs','C  Unseen subjects']):
    for i,(variant,color) in enumerate(zip(variants,colors)):
        selected=[r for r in rows if r['variant']==variant and r['group']==group];values=np.array([float(r['pearson_r']) for r in selected]);wrong=np.mean([float(r['wrong_EEG_r']) for r in selected])
        ax.bar(i,values.mean(),width=.65,color=color,alpha=.8,yerr=values.std(ddof=1),capsize=3)
        ax.scatter(i+np.linspace(-.12,.12,3),values,s=23,c='white',edgecolors='black',lw=.7,zorder=3)
        ax.scatter(i,wrong,marker='x',c='black',s=35,zorder=4)
        ax.text(i,values.max()+.008,f'{values.mean():.3f}',ha='center',fontsize=9)
    ax.axhline(0,color='#888888',lw=1);ax.set_xticks(range(4),variants,rotation=20);ax.set(ylabel='Test subject-macro Pearson r',title=title)
    ax.margins(y=.25);ax.text(.02,.98,'Dots: seeds; bars: mean ± seed SD\n×: wrong-EEG control',transform=ax.transAxes,va='top',fontsize=8)
fig.suptitle('SS03 | raw 64-Hz EEG → envelope | 6001 parameters each | 25,000 optimizer steps',fontsize=13)
fig.savefig(out/'learning_and_results.png');plt.close(fig)

fig,ax=plt.subplots(figsize=(9,4),layout='constrained');labels=[]
for i,r in enumerate(paired):
    value=float(r['mean_delta_r']);lo=float(r['Bonferroni_family95_low']);hi=float(r['Bonferroni_family95_high'])
    color='#276d9e' if r['group']=='test_seen' else '#a84953'
    ax.errorbar(value,i,xerr=np.array([[value-lo],[hi-value]]),fmt='o',color=color,capsize=4)
    labels.append(('Seen' if r['group']=='test_seen' else 'Unseen')+' / '+r['contrast'].replace('_',' '))
ax.axvline(0,color='#555555',ls='--',lw=1);ax.set_yticks(range(4),labels);ax.invert_yaxis()
ax.set(xlabel='Paired Δ Pearson r (multiscale − single scale)',title='Planned contrasts: subject bootstrap, family 95% intervals')
ax.text(.02,-.24,'Each person: average the 3 seed scores first. Four contrasts; Bonferroni intervals (98.75% each).',transform=ax.transAxes,fontsize=9)
fig.savefig(out/'paired_differences.png',bbox_inches='tight');plt.close(fig)
print('STATIC_FIGURES_SAVED')
