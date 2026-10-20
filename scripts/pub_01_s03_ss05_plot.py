"""Static scientific figures, only after real heldout results and review exist."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pub_01_s03_ss05_data import ROOT


def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def main():
    out=ROOT/'review01';review=json.loads((out/'review.json').read_text());assert review['technical']=='PASS'
    rows=read(out/'RESULTS.csv');summary=read(out/'SUMMARY.csv');primary=read(out/'primary_contrasts.csv')
    cfg=json.loads((ROOT/'FORMAL_PROTOCOL.json').read_text());variants=cfg['variants'];colors=['#888888','#b69246','#2779a1','#a84953','#469081']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for variant,color in zip(variants,colors):
        histories=[json.loads((ROOT/'formal01'/f'{variant}_seed{seed}'/'history.json').read_text()) for seed in cfg['seeds']]
        steps=np.array([r['step'] for r in histories[0]]);values=np.array([[r['validation_record_then_subject_macro_60s_r'] for r in h] for h in histories])
        axes[0].plot(steps,values.mean(0),color=color,label=variant);axes[0].fill_between(steps,values.min(0),values.max(0),color=color,alpha=.1)
    axes[0].set(xlabel='Optimizer steps',ylabel='Validation subject-macro60s r',title='A  Frozen budget and validation-only selection');axes[0].legend(frameon=False,fontsize=8)
    conditions=['true','reset','wrong_record'];x=np.arange(5);width=.24
    for j,(condition,color) in enumerate(zip(conditions,['#2779a1','#a9a9a9','#d8912d'])):
        means=[];sds=[]
        for variant in variants:
            values=np.array([float(r['pearson_r']) for r in rows if r['variant']==variant and r['condition']==condition]);means.append(values.mean());sds.append(values.std(ddof=1))
        axes[1].bar(x+(j-1)*width,means,width,yerr=sds,capsize=2,color=color,label=condition)
    axes[1].set_xticks(x,variants,rotation=22);axes[1].set(ylabel='New internal holdout subject-macro r',title='B  Same current EEG, different past');axes[1].margins(y=.3);axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('SS05 | 5 variants ×3 seeds | newtrain-only normalization | prospective INTERNAL family holdout',fontsize=12)
    fig.savefig(out/'training_and_history_scores.png');plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');labels=[]
    for i,r in enumerate(primary):
        v=float(r['mean_delta_r']);lo=float(r['subject_CI95_low']);hi=float(r['subject_CI95_high']);cl=float(r['crossed_family95_low']);ch=float(r['crossed_family95_high'])
        ax.errorbar(v,i-.07,xerr=np.array([[v-lo],[hi-v]]),fmt='o',color='#9aaab2',capsize=3,label='Subject95%' if i==0 else None)
        ax.errorbar(v,i+.07,xerr=np.array([[v-cl],[ch-v]]),fmt='o',color='#2779a1',capsize=4,label='Crossed subject/family, family95%' if i==0 else None)
        labels.append(r['contrast'].replace('_',' '))
    ax.axvline(0,color='#555555',ls='--');ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set(xlabel='Paired Δ Pearson r; 3-seed mean within recording',title='Six frozen primary contrasts');ax.legend(frameon=False,fontsize=8,loc='best')
    fig.savefig(out/'primary_paired_differences.png',bbox_inches='tight');plt.close(fig)

    modes=['reset','wrong_record','wrong_time','shuffled','replace_high_similarity','replace_low_similarity'];matrix=[]
    for variant in variants:
        lookup={r['condition']:float(r['mean_r']) for r in summary if r['variant']==variant};matrix.append([lookup['true']-lookup[m] for m in modes])
    matrix=np.array(matrix);limit=max(float(np.abs(matrix).max()),1e-8)
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained');im=ax.imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_yticks(range(5),variants);ax.set_xticks(range(6),['Reset','Other record','Earlier time','Shuffle','Replace high sim','Replace low sim'],rotation=18,ha='right')
    for i in range(5):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:+.5f}',ha='center',va='center',color='white' if abs(matrix[i,j])>limit*.6 else 'black',fontsize=9)
    ax.set_title('True history − altered history; current EEG and target fixed\nSimilarity is an EEG-token proxy, not labeled event relevance')
    fig.colorbar(im,ax=ax,label='Δ subject-macro Pearson r');fig.savefig(out/'history_interventions.png');plt.close(fig)
    print('THREE_STATIC_SCIENTIFIC_FIGURES_SAVED')


if __name__=='__main__':main()
