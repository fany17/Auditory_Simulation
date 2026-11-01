"""Presentation-only view of already reported D9 window/bias pathology."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_physical_objective_20261101');OUT=ROOT/'presentation02'
s=json.loads((ROOT/'report01/NUMERICAL_SUMMARY.json').read_text());d=json.loads((ROOT/'COEFFICIENT_RANGE_DIAGNOSTIC.json').read_text());OUT.mkdir(exist_ok=False)
roles=('Ccap','W','F');rows=[next(r for r in s['absolute_means'] if r['role']==role) for role in roles]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
for ax,key,label,title in [(axes[0],'window_MSE','VAL uniform window target MSE (log)','Fused accuracy conceals window errors'),(axes[1],'overlap_MSE','VAL uniform96 overlap MSE (log)','Same stored predictions, no new evaluation')]:
    values=[r[key] for r in rows];ax.bar(range(3),values,color=['gray','tab:blue','tab:orange']);ax.set_yscale('log');ax.set_xticks(range(3),roles);ax.set_ylabel(label);ax.set_title(title)
    for i,v in enumerate(values):ax.annotate(f'{v:.6g}',(i,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
colors=plt.cm.Set1.colors[:3]
for i,seed in enumerate((1024,2026,31415)):
    rr=[next(r for r in d['results'] if r['seed']==seed and r['role']==role) for role in ('W','F')]
    axes[2].plot(range(2),[r['bias_max_abs'] for r in rr],'o-',color=colors[i],label=str(seed))
    axes[2].scatter(1,rr[1]['interior_Hann_bias_pair_max_abs'],marker='x',color=colors[i])
axes[2].set_yscale('log');axes[2].set_xticks(range(2),['W','F']);axes[2].set_ylabel('Saved bias magnitude (log)');axes[2].set_title('F bias max247-253; fused pair max<.006');axes[2].legend(title='seed; x = F fused bias pair max',fontsize=7)
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Post-result coefficient validation: large window offsets nearly cancel under fixed Hann; no refit')
fig.savefig(OUT/'04_WINDOW_DEGENERACY.png');plt.close(fig)
(OUT/'PRESENTATION_NOTE.json').write_text(json.dumps(dict(scope='PRESENTATION_ONLY_OF_ALREADY_FROZEN_SUMMARIES_AND_COEFFICIENT_DIAGNOSTIC',new_solve_score_signal_read=False,original_three_figures_preserved=True,reason='Show log-scale window errors and saved bias cancellation alongside the small fused-r gain; no scientific matrix change.'),indent=2))
print('PRESENTATION_ONLY_DEGENERACY_FIGURE_COMPLETED')
