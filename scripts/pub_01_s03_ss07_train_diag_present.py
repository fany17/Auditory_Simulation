"""Lightweight diagnostic readout and tick repair from saved TRAIN statistics.

No neural arrays, weights, fits or changed scientific diagnostic definitions.
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_train_diagnostics_20261101')
def rows(name):
    with (ROOT/'results01'/name).open() as f:return list(csv.DictReader(f))
def val(row,key):return float(row[key]) if row[key] else None
def describe(values):
    a=np.array([v for v in values if v is not None and np.isfinite(v)])
    return {'n':len(a),'mean':float(a.mean()),'q05':float(np.quantile(a,.05)),'median':float(np.median(a)),'q95':float(np.quantile(a,.95)),'min':float(a.min()),'max':float(a.max())} if len(a) else {'n':0}
def ecdf(ax,values,**kwargs):
    a=np.array([v for v in values if v is not None and np.isfinite(v)]);a.sort()
    if len(a):ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',**kwargs)

def main():
    summary=json.loads((ROOT/'results01/SUMMARY.json').read_text());assert not summary['errors']
    d1=rows('D1_PER_RECORD.csv');d2=rows('D2_PER_RECORD.csv');d3=rows('D3_PER_RECORD.csv')
    records=json.loads((ROOT/'TRAIN_RECORDS.json').read_text());assert len(records)==426
    output={'derivation':'existing light CSV field copies, aggregate arithmetic and quantiles only; no new signal reads, fitting, model or choice',
        'unique_TRAIN_subjects':len({r['subject'] for r in records}),'unique_TRAIN_stories':len({str(r['story']) for r in records}),
        'unique_TRAIN_target_paths':len({r['envelope_path'] for r in records}),'seeds':[],'D3':[]}
    for seed in (1024,2026,31415):
        a=[r for r in d1 if int(r['seed'])==seed];b=[r for r in d2 if int(r['seed'])==seed];assert len(a)==len(b)==426
        assert sum(int(r['windows']) for r in a)==252454 and sum(int(r['overlap_pairs']) for r in a)==252028
        total_mse=sum(int(r['overlap_pairs'])*float(r['raw_overlap_MSE_mean']) for r in a)
        mean_component=sum(int(r['overlap_pairs'])*(float(r['overlap_abs_mean_delta_mean'])**2+float(r['overlap_abs_mean_delta_std'])**2) for r in a)
        item={'seed':seed,'record_window_mean_drift_ratio':describe([val(r,'window_mean_drift_over_median_std') for r in a]),
            'record_window_std_CV':describe([val(r,'window_std_CV') for r in a]),'record_overlap_NRMSE':describe([val(r,'raw_overlap_NRMSE_mean') for r in a]),
            'pooled_overlap_mean_difference_energy_fraction':mean_component/total_mse,
            'record_energy_retained':describe([val(r,'energy_retained_ratio') for r in b]),
            'max_energy_balance_error':max(float(r['energy_balance_relative_max_error']) for r in b)}
        source=next(r for r in summary['seeds'] if r['seed']==seed)
        retained=sum(float(r['E6']) for r in b)/sum(float(r['E0']) for r in b)
        assert abs(retained-source['D2_energy_retained_ratio'])<=1e-12
        for direction in ('forward','reverse'):
            actual=sum(float(r[direction+'_aligned_held_MSE_sum']) for r in a)/sum(float(r[direction+'_paired_raw_MSE_sum']) for r in a)
            assert abs(actual-source[direction+'_held_aligned_raw_ratio'])<=1e-12
            item[direction+'_record_held_ratio']=describe([val(r,direction+'_held_aligned_raw_ratio') for r in a])
        output['seeds'].append(item)
    for seconds in (3,6,12):
        selected=[r for r in d3 if int(r['lag_seconds'])==seconds];assert len(selected)==426
        values=[val(r,'r') for r in selected]
        output['D3'].append({'lag_seconds':seconds,'visited_records':426,'undefined':sum(v is None for v in values),
            'unique_defined_r_values':len({v for v in values if v is not None}),'record_r':describe(values)})
    out=ROOT/'presentation02';out.mkdir(exist_ok=False)
    (out/'REPORT_NUMERICS.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});colors=['#0072B2','#D55E00','#009E73']
    fig,axes=plt.subplots(1,3,figsize=(15,5.1))
    for seed,color in zip((1024,2026,31415),colors):
        selected=[r for r in d1 if int(r['seed'])==seed]
        ecdf(axes[0],[val(r,'window_mean_drift_over_median_std') for r in selected],color=color,label=str(seed))
        ecdf(axes[1],[val(r,'raw_overlap_NRMSE_mean') for r in selected],color=color)
        for direction,ls in [('forward','-'),('reverse','--')]:
            ecdf(axes[2],[val(r,direction+'_held_aligned_raw_ratio') for r in selected],color=color,ls=ls,label=f'{seed} {direction}')
    axes[0].set_xlim(left=0);axes[0].xaxis.set_major_locator(MaxNLocator(5));axes[0].set_xlabel('SD(window mean) / median window SD')
    axes[1].set_xlabel('Mean raw overlap normalized RMSE');axes[2].set_xlabel('Held-half aligned / paired raw MSE')
    axes[2].set_xscale('symlog',linthresh=.1);axes[2].axvline(1,color='grey',lw=1)
    for ax in axes:ax.set_ylabel('Record ECDF (descriptive)')
    axes[0].legend(title='P seed');axes[2].legend(fontsize=8)
    text='Positive-fit coverage (forward / reverse): '+ '; '.join(f"{r['seed']}: {r['forward_positive_fit_fraction']:.2%} / {r['reverse_positive_fit_fraction']:.2%}" for r in summary['seeds'])
    fig.suptitle('D1: SOURCE-TRAIN P overlap consistency; held-half alignment of predictions, no labels')
    fig.text(.5,.015,text,ha='center',fontsize=10);fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(out/'D1_OVERLAP_AFFINE.png',dpi=170);plt.close(fig)
    print(json.dumps(output))

if __name__=='__main__':main()
