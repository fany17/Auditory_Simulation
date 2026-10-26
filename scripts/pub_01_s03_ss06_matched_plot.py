"""Aggregate-only static scientific plots on2203; no signal arrays exported."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pub_01_s03_ss06_matched import ROOT,OLD,SEEDS,read


def table(p):
    with p.open() as f:return list(csv.DictReader(f))


def main():
    assert read(ROOT/'review01/REVIEW.json')['technical']=='PASS'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.dpi':160,'figure.facecolor':'white'})
    out=ROOT/'figures';out.mkdir(exist_ok=False)
    new=table(ROOT/'audit01/RESULTS.csv');old=table(OLD/'review01/RESULTS.csv')
    variants=['m1_multi','m2','gru','m3_finetune','m4_adaptive','m5_full','C','O','P']
    labels=['M1 multi','M2','GRU','M3 + EEG SSL','M4 adaptive','M5 full','C: causal strong','O: offline direct','P: two-stage']
    fig,ax=plt.subplots(figsize=(10,4.7));colors=['#9aabb4']*6+['#197f86','#b16d76','#b16d76']
    for i,v in enumerate(variants):
        values=[float(r['r']) for r in new if r['condition']==v] if v in ('C','O','P') else [float(r['pearson_r']) for r in old if r['variant']==v and r['condition']=='true']
        assert len(values)==3
        ax.bar(i,np.mean(values),color=colors[i],width=.65,zorder=2)
        ax.scatter(i+np.linspace(-.13,.13,3),values,s=24,color='#24323b',zorder=3)
        ax.text(i,max(values)+.004,f'{np.mean(values):.3f}',ha='center',fontsize=9)
    ax.axvline(5.5,color='#707070',lw=.8);ax.axvline(6.5,color='#707070',lw=.8,ls='--')
    ax.set_xticks(range(9),labels,rotation=25,ha='right');ax.set_ylabel('Record-to-person macro Pearson r')
    ax.set_title('Same 41 records / 538 blocks; three-seed dots, mean bars')
    fig.text(.11,.015,'POST-HOC INTERNAL EXPLORATORY. O/P: offline future EEG; P/M3: extra pretraining. No total-budget or pure-latency claim.',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(out/'performance.png');plt.close(fig)
    contrasts=table(ROOT/'review01/primary_contrasts.csv');fig,ax=plt.subplots(figsize=(8.2,4.3))
    for i,r in enumerate(contrasts):
        d=float(r['delta_r']);lo=float(r['crossed_Bonferroni_CI_low']);hi=float(r['crossed_Bonferroni_CI_high'])
        color='#197f86' if lo>0 else '#b16d76' if hi<0 else '#536875'
        ax.plot([lo,hi],[i,i],color=color,lw=2);ax.scatter([d],[i],color=color,s=35,zorder=3)
    ax.axvline(0,color='#555555',lw=1,ls='--');ax.set_yticks(range(6),[r['contrast'].replace('C_minus_','C - ') for r in contrasts]);ax.invert_yaxis()
    ax.set_xlabel('Delta r (C minus run01 comparator)');ax.set_title('Crossed person/family bootstrap; 99.167% Bonferroni intervals')
    fig.text(.14,.025,'10,000 draws; five source families; seed means first. Exploratory intervals do not establish external confirmation.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(out/'paired_contrasts.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9.5,6.4));colors=['#197f86','#b16d76','#a18231']
    for i,name in enumerate(('C','O','P_pretrain','P_head')):
        ax=axes.flat[i]
        for seed,color in zip(SEEDS,colors):
            h=read(ROOT/f'formal01/{name}_seed{seed}/history.json')
            xs=[r['step'] for r in h]
            ys=[r['validation']['r'] if name in ('C','O') else r['validation_InfoNCE'] if name=='P_pretrain' else -r['validation_3s_negative_r'] for r in h]
            ax.plot(xs,ys,'o-',color=color,label=str(seed),ms=4)
            selected=read(ROOT/f'formal01/{name}_seed{seed}/completed.json')['selected_step'];j=xs.index(selected)
            ax.scatter([xs[j]],[ys[j]],s=70,facecolors='none',edgecolors=color)
        ax.set_title(name);ax.set_xlabel('Optimizer steps')
        ax.set_ylabel('60s val macro r' if name in ('C','O') else 'Val symmetric InfoNCE' if name=='P_pretrain' else 'Val mean 3s r')
    axes.flat[0].legend(title='Seed',fontsize=8);fig.suptitle('Frozen validation checks; hollow ring = selected checkpoint')
    fig.tight_layout();fig.savefig(out/'validation_curves.png');plt.close(fig)
    print(json.dumps({'figures':[str(p) for p in out.glob('*.png')],'source':'aggregate metrics only'}),flush=True)

if __name__=='__main__':main()
