"""Static SS06 figures from reviewed summary tables, on server2203 only."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pub_01_s03_ss06_data import ROOT,read


def rows(name):
    with (ROOT/'review01'/name).open() as f:return list(csv.DictReader(f))


LABELS={'local':'Local','m1_short':'M1 short (4 branches)','m1_long':'M1 long (4 branches)',
        'm1_multi':'M1 multiscale','m2':'M2 persistent','gru':'GRU / M3 no pretrain',
        'm3_frozen':'M3 pretrained frozen','m3_finetune':'M3 pretrained finetune',
        'm4_fixed':'M4 fixed lag mixture','m4_adaptive':'M4 adaptive lag mixture',
        'm5_eeg':'M5 EEG only','m5_prior':'M5 autonomous prior','m5_full':'M5 full innovation'}


def main():
    out=ROOT/'review01';p=read(ROOT/'FORMAL_PROTOCOL.json');plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    selected=('local','m1_multi','m2','gru','m3_frozen','m3_finetune','m4_adaptive','m5_full')
    fig,ax=plt.subplots(figsize=(9,5))
    for variant in selected:
        histories=[read(ROOT/'formal01'/f'{variant}_seed{seed}'/'history.json') for seed in p['seeds']]
        x=[r['step'] for r in histories[0]];y=np.array([[r['validation']['r'] for r in h] for h in histories])
        ax.plot(x,y.mean(0),marker='o',markersize=3,label=LABELS[variant]);ax.fill_between(x,y.min(0),y.max(0),alpha=.08)
    ax.set(xlabel='Supervised optimizer steps (one full train pass)',ylabel='Validation subject-macro Pearson r',title='Full-record training; 3 seeds; bands show seed range')
    ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=200);plt.close(fig)
    summary=rows('SUMMARY.csv');entries=[(v,'true',LABELS[v]) for v in p['variants']]
    entries += [('local','local_ema','Local + causal EMA'),('local','local_ar','Local + predicted-output AR'),('m2','periodic_reset3s','M2 reset every 3s')]
    fig,ax=plt.subplots(figsize=(9,7));positions=np.arange(len(entries))
    for i,(v,c,label) in enumerate(entries):
        r=next(r for r in summary if r['variant']==v and r['condition']==c)
        if r['mean_r']:
            ax.errorbar(float(r['mean_r']),i,xerr=float(r['seed_SD']) if r['seed_SD'] else 0,fmt='o',color='#22577a',capsize=3)
        else:ax.text(.02,i,'undefined r (near-constant)',fontsize=8,color='#777777',transform=ax.get_yaxis_transform())
    ax.set_yticks(positions,[v[2] for v in entries]);ax.invert_yaxis();ax.set(xlabel='Internal audit subject-macro Pearson r',title='True continuous outputs; points = seed mean, bars = seed SD')
    ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(out/'model_scores.png',dpi=200);plt.close(fig)
    contrasts=rows('primary_contrasts.csv');fig,ax=plt.subplots(figsize=(10,5.5))
    for i,r in enumerate(contrasts):
        mean=float(r['mean_delta_r']);lo=float(r['crossed_family95_low']);hi=float(r['crossed_family95_high'])
        ax.plot([float(r['subject_CI95_low']),float(r['subject_CI95_high'])],[i-.08,i-.08],color='#bbbbbb',lw=3)
        ax.plot([lo,hi],[i,i],color='#22577a',lw=2);ax.scatter([mean],[i],color='#22577a',s=25)
    ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(contrasts)),[r['contrast'].replace('_minus_',' − ').replace('_',' ') for r in contrasts]);ax.invert_yaxis()
    ax.set(xlabel='Paired Δr (positive favors first model)',title='8 prespecified contrasts: crossed subject / source-family adjusted intervals')
    ax.text(.01,-.2,'Blue: Bonferroni-adjusted 99.375% each; gray: subject-only 95%. Only 5 audit families.',transform=ax.transAxes,fontsize=8)
    fig.tight_layout();fig.savefig(out/'primary_paired_differences.png',dpi=200);plt.close(fig)
    h=rows('history_deltas.csv');variants=[v for v in p['variants'] if v!='m5_prior'];conditions=['state_reset','state_wrong','prefix_reset','prefix_wrong']
    values=np.array([[float(next(r for r in h if r['variant']==v and r['contrast']=='true_minus_'+c)['mean_delta_r']) for c in conditions] for v in variants])
    bound=max(float(np.abs(values).max()),1e-6);fig,ax=plt.subplots(figsize=(8,7))
    im=ax.imshow(values,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
    ax.set_xticks(range(4),['State reset','Wrong state','Prefix reset','Wrong prefix']);ax.set_yticks(range(len(variants)),[LABELS[v] for v in variants])
    for i in range(len(variants)):
        for j in range(4):ax.text(j,i,f'{values[i,j]:+.4f}',ha='center',va='center',fontsize=8,color='black' if abs(values[i,j])<.7*bound else 'white')
    ax.set(title='First scored 60s only: true − intervention Δr\nState controls preserve exact current encoder features; prefix controls are separate')
    fig.colorbar(im,ax=ax,label='Δr (color is not significance)');fig.tight_layout();fig.savefig(out/'history_interventions.png',dpi=200);plt.close(fig)
    print(json.dumps({'figures':4,'source':'reviewed CSV only','rendered_on':'server2203'}),flush=True)


if __name__=='__main__':main()
