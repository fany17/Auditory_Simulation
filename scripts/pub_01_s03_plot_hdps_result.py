"""Plot already frozen metrics only; no inference or parameter selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(run):
    result=json.loads((run/'result.json').read_text());history=json.loads((run/'history.json').read_text());test=result['test']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    figure,axes=plt.subplots(1,2,figsize=(11.4,3.9),constrained_layout=True)
    epochs=np.array([row['epoch']+1 for row in history])
    loss_line,=axes[0].plot(epochs,[r['training_loss'] for r in history],color='#2878B5',marker='o',label='Training InfoNCE')
    axes[0].set(xlabel='Epoch',ylabel='Training loss',title='A  Validation selects the checkpoint')
    second=axes[0].twinx();validation_line,=second.plot(epochs,[r['validation']['Top10']*100 for r in history],color='#C65D3A',marker='s',label='Validation Top-10')
    second.set_ylabel('Validation Top-10 (%)',color='#C65D3A');second.spines['top'].set_visible(False)
    chosen_line=axes[0].axvline(result['best_epoch']+1,color='#666666',linestyle='--',label='Selected epoch')
    axes[0].legend([loss_line,validation_line,chosen_line],['Training InfoNCE','Validation Top-10','Selected epoch'],frameon=False,loc='lower left',fontsize=8)
    labels=['Top1','Top5','Top10'];x=np.arange(3);width=.26
    actual=np.array([test[k]*100 for k in labels]);null=np.array([test['EEG_shuffle_'+k]*100 for k in labels]);chance=np.array([test['chance_'+k]*100 for k in labels])
    intervals=np.array([test[k+'_subject_bootstrap95'] for k in labels])*100
    axes[1].bar(x-width,actual,width,color='#2878B5',label='EEG model',yerr=np.stack([actual-intervals[:,0],intervals[:,1]-actual]),capsize=3)
    axes[1].bar(x,null,width,color='#A9A9A9',label='EEG shuffled')
    axes[1].bar(x+width,chance,width,facecolor='white',edgecolor='#555555',label='Analytic chance')
    axes[1].set(xticks=x,xticklabels=['Top-1','Top-5','Top-10'],ylabel='Test accuracy (%)',ylim=(0,52),title='B  One frozen test evaluation')
    axes[1].legend(frameon=False,loc='upper left',fontsize=8)
    for i,value in enumerate(actual):axes[1].text(i-width,intervals[i,1]+1.0,f'{value:.2f}',ha='center',fontsize=9)
    figure.suptitle('HDPS / Mel-only / 100 Hz / 76 participants / 100–235 candidates',fontsize=12)
    figure.savefig(run/'test_summary.png');plt.close(figure)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);main(parser.parse_args().run)
