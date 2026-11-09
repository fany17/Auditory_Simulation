"""Completed native-baseline train/val curves, no signal/test reads."""
import csv
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pub_01_s03_ss07_baseline import ROOT,read,dump,sha


def main():
    folder=ROOT/'baseline_curve_snapshot01';folder.mkdir(exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.4),layout='constrained');rows=[];sources={}
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
    for seed,color in zip([1024,2026,31415,42,12345],colors):
        pre=ROOT/f'baseline/pretrain_seed{seed}';head=ROOT/f'baseline/head_seed{seed}'
        pc=read(pre/'completed.json');hc=read(head/'completed.json')
        ph=read(pre/'history.json');hh=read(head/'history.json')
        assert len(ph)==pc['epochs'] and len(hh)==hc['epochs']
        axes[0].plot([r['epoch']+1 for r in ph],[r['validation_loss'] for r in ph],color=color,lw=1.5,label=str(seed))
        selected=pc['selected_epoch'];axes[0].scatter(selected+1,ph[selected]['validation_loss'],s=26,color=color,zorder=4)
        axes[1].plot([r['epoch']+1 for r in hh],[-r['validation_negative_Pearson'] for r in hh],color=color,lw=1.5)
        selected=hc['selected_epoch'];axes[1].scatter(selected+1,-hh[selected]['validation_negative_Pearson'],s=26,color=color,zorder=4)
        rows.append({'seed':seed,'pretrain_epochs':pc['epochs'],'pretrain_updates':pc['updates'],
            'pretrain_selected_epoch_0based':pc['selected_epoch'],'pretrain_stopping':pc['stopping'],
            'head_epochs':hc['epochs'],'head_updates':hc['updates'],'head_selected_epoch_0based':hc['selected_epoch'],
            'head_stopping':hc['stopping'],'head_best_validation_3s_r':-hc['best_validation']})
        for p in [pre/'history.json',pre/'completed.json',head/'history.json',head/'completed.json']:sources[str(p)]=sha(p)
    axes[0].set(title='Full native pretraining',xlabel='Epoch (1-based)',ylabel='Validation symmetric InfoNCE')
    axes[1].set(title='Full native readout',xlabel='Epoch (1-based)',ylabel='Mean 3 s validation correlation')
    axes[0].legend(title='Paired seed',frameon=False,ncol=2,fontsize=8,title_fontsize=8)
    for label,ax in zip(['a','b'],axes):
        ax.text(-.13,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=12)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18,lw=.5)
        ax.tick_params(labelsize=8);ax.set_axisbelow(True)
    fig.savefig(folder/'BASELINE_CURVES.png',dpi=220,facecolor='white')
    plt.close(fig)
    with (folder/'BASELINE_BUDGET.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(folder/'SNAPSHOT.json',{'time':time.time(),'all5_native_two_stage_runs_completed':True,
        'train_val_only_no_test_data':True,'sources':sources,'rows':rows,
        'interpretation':'markers selected checkpoints; patience stop vs budget ceiling distinct; no claim of convergence or superiority',
        'metric_warning':'readout panel is original3s checkpoint criterion, not60s person-level primary test endpoint'})
    print(json.dumps({'snapshot':'completed_native5','rows':rows}))


if __name__=='__main__':main()
