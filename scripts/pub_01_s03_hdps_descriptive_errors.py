"""Post-test descriptive candidate-size analysis; never changes a candidate."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261015/hdps_m_available01/training_seed1024')
with (run/'test_per_subject.csv').open() as f:rows=list(csv.DictReader(f))
n=np.array([int(r['candidate_n']) for r in rows]);score=np.array([float(r['Top1']) for r in rows]);null=np.array([float(r['EEG_shuffle_Top1']) for r in rows])
groups=[]
for label,mask in [('100–150',(n<=150)),('151–200',(n>150)&(n<=200)),('201–235',(n>200))]:
    if mask.any():groups.append({'candidate_n_bin':label,'subjects':int(mask.sum()),'subject_macro_Top1':float(score[mask].mean()),
        'EEG_shuffle_Top1':float(null[mask].mean()),'analytic_chance':float((1/n[mask]).mean())})
summary={'post_test_descriptive_only':True,'groups':groups,'subject_top1_min_median_max':[float(score.min()),float(np.median(score)),float(score.max())],
    'subjects_above_own_analytic_chance':int((score>1/n).sum()),'subjects_above_shuffled_EEG':int((score>null).sum()),
    'candidate_n_vs_top1_Pearson_descriptive':float(np.corrcoef(n,score)[0,1]),
    'interpretation':'candidate count is confounded with subject/story/duration; association is descriptive, not a candidate-size causal effect or independent significance test; no pooling or model settings changed'}
with (run/'descriptive_errors.json').open('x') as f:json.dump(summary,f,indent=2)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(6,4),layout='constrained');ax.scatter(n,score*100,s=26,color='#28618f',alpha=.65,label='Participant Top-1')
grid=np.arange(100,236);ax.plot(grid,100/grid,color='#777777',ls='--',label='Analytic chance')
ax.set(xlabel='Frozen candidate count',ylabel='Test Top-1 (%)',title='HDPS / 76 participants: descriptive errors');ax.legend(frameon=False)
fig.savefig(run/'candidate_size_summary.png',dpi=180);plt.close(fig);print(json.dumps(summary),flush=True)
