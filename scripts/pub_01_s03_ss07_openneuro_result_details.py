"""Descriptive report-only metadata and legible primary-axis tick labels; no NN."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
from pub_01_s03_ss07_openneuro_transport import ROOT

def read(p):return json.loads(p.read_text())
def describe(values):
    a=np.asarray(values,dtype='f8');assert np.isfinite(a).all()
    return dict(n=len(a),minimum=float(a.min()),median=float(np.median(a)),maximum=float(a.max()),mean=float(a.mean()))

output=ROOT/'report01';summary=read(ROOT/'EXTERNAL_SUMMARY.json');details={}
for role in ['P_FT_star','Ccap','W','F']:
    records=[r for s in [1024,2026,31415] for r in read(ROOT/f'evaluation_seed{s}/{role}_METRICS.json')['records']]
    details[role]={k:describe([r[k] for r in records]) for k in ['uniform_overlap_MSE','first4_window_MSE','max_abs_window_prediction','max_abs_fused_prediction']}
    for key in ['prefix_before_learning','first_learning_overlap','final_single_window_tail','tail']:
        details[role][key+'_MSE']=describe([r[key]['MSE'] for r in records if r[key]['MSE'] is not None])
precision=list(csv.DictReader((output/'PRECISION_PERSON.csv').open()))
details['precision_person']={role:describe([float(p['FP32_minus_FP64_r']) for p in precision if p['role']==role]) for role in ['W','F']}
details['primary_person_directions']=[dict(comparison=p['comparison'],positive=sum(x['difference']>0 for x in p['person_differences']),
    negative=sum(x['difference']<0 for x in p['person_differences']),**describe([x['difference'] for x in p['person_differences']])) for p in summary['primary']]
details['new_NN']=0;details['descriptive_only']=True
with (output/'RESULTS_DETAILS.json').open('x') as f:json.dump(details,f,indent=2,allow_nan=False)
# Retain first PNG. Change tick density only, with identical data/intervals/limits.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for i,item in enumerate(summary['primary']):
    test=item['inference']
    if test:
        mean=test['mean_difference'];low,high=test['Bonferroni_family95_approximate_CI']
        axes[0].errorbar(mean,i,xerr=np.array([[mean-low],[high-mean]]),fmt='o',capsize=4,color=['#457b9d','#e9a23b'][i])
        axes[0].annotate(f"Holm2 p={test['p_Holm2']:.5g}",(mean,i),xytext=(0,13),textcoords='offset points',ha='center',fontsize=9)
    else:axes[0].text(0,i,'Undefined',ha='center')
axes[0].axvline(0,color='black',lw=.8);axes[0].set(yticks=[0,1],yticklabels=['Ccap - P_FT*','F - W'],xlabel='Difference in person-macro r',ylim=(-.5,1.6),title='Fixed primary contrasts; family95 approximate CIs')
a={r['subject']:r['difference'] for r in summary['primary'][0]['person_differences']};b={r['subject']:r['difference'] for r in summary['primary'][1]['person_differences']}
axes[1].scatter([a[s] for s in sorted(a)],[b[s] for s in sorted(a)],s=35,color='#457b9d')
axes[1].axhline(0,color='black',lw=.8);axes[1].axvline(0,color='black',lw=.8)
axes[1].set(xlabel='Ccap - P_FT*: person difference',ylabel='F - W: person difference',title='19 people; each averaged over three seeds')
axes[1].xaxis.set_major_locator(MaxNLocator(5));axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
path=output/'figures/two_primary_differences_v2.png';assert not path.exists();fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
with (output/'PRESENTATION_TICK_PATCH.json').open('x') as f:json.dump(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    changed='Reduce crowding of primary difference panel x tick labels after actual visual inspection; original PNG retained',new_NN=0,
    scientific_values_unchanged=True,extra_descriptive_summary='RESULTS_DETAILS.json'),f,indent=2)
print('REPORT_ONLY_DETAILS_AND_TICK_LABELS_COMPLETE')
