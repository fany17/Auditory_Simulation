"""D6 presentation-only: clarify MSE decomposition gap / crowded numeric ticks."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_readout_attribution_20261101')
OUT=ROOT/'presentation02';OUT.mkdir(exist_ok=False)
s=json.loads((ROOT/'report01/NUMERICAL_SUMMARY.json').read_text());arms=('Ponly','S32','U32','C','Ccap');colors=dict(zip(arms,plt.cm.tab10.colors[:5]))
rows=[dict(arm='P',**s['baseline_means'])]+s['selected_means'];names=[r['arm'] for r in rows]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,ax=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
ax[0].bar(range(6),[r['overlap_MSE'] for r in rows],color=['gray']+[colors[a] for a in arms]);ax[0].set_xticks(range(6),names);ax[0].set_ylabel('Uniform adjacent96 overlap MSE');ax[0].set_title('Continuity alone is not accuracy')
fused=np.array([r['fused_MSE'] for r in rows]);weighted=np.array([r['weighted_single_MSE'] for r in rows])
ax[1].vlines(range(6),fused,weighted,color='gray',lw=2,label='Gap = Hann disagreement')
ax[1].scatter(range(6),weighted,color='darkorange',s=35,label='Hann-weighted single-window MSE')
ax[1].scatter(range(6),fused,color='steelblue',s=35,label='Fused target MSE')
ax[1].set_xticks(range(6),names);ax[1].set_ylabel('60-s / person macro target MSE');ax[1].set_title('Same fixed Hann; vertical gap is exact');ax[1].legend(fontsize=7.5)
for r in rows:
    arm=r['arm'];ax[2].scatter(r['overlap_MSE'],r['r60s'],s=50,color='gray' if arm=='P' else colors[arm]);ax[2].annotate(arm,(r['overlap_MSE'],r['r60s']),xytext=(4,6),textcoords='offset points',fontsize=8)
ax[2].set_xlabel('Uniform overlap MSE');ax[2].set_ylabel('VAL60s/person mean r');ax[2].set_title('Accuracy and disagreement jointly reported')
ax[2].xaxis.set_major_locator(MaxNLocator(4));ax[2].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
for a in ax:a.grid(alpha=.2)
fig.suptitle('Shared-selected models: descriptive continuity / accuracy, never used for model selection')
fig.savefig(OUT/'03_CONTINUITY_ACCURACY.png');plt.close(fig)
with (OUT/'PRESENTATION_NOTE.json').open('x') as f:json.dump(dict(source='report01/NUMERICAL_SUMMARY.json',changes='Show fused and weighted-single MSE as dots with disagreement gap; reduce numeric ticks. Original PNG retained. No new fits, data reads, statistics, selection or metrics.'),f,indent=2)
