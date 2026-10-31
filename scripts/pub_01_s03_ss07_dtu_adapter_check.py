"""Input/geometry only, no model; loaded MAT's target values are unused."""
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from pub_01_s03_ss07_dtu_adapter import native_windows
from pub_01_s03_ss07_history import hann_support_audit

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_continuity_20261101/external_dtu')


def main():
    d=loadmat(ROOT/'extracted/S1_data_preproc.mat',simplify_cells=True)['data']
    eeg=d['eeg'][0];labels=d['dim']['chan']['eeg'][0].tolist()
    names=json.loads((ROOT/'READABILITY_CHANNEL_EVENT_AUDIT.json').read_text())['subjects'][0]['labels'][:64]
    actual,starts,constants=native_windows(eeg,labels,names)
    assert starts.tolist()==list(range(0,2977,96)) and actual.shape==(32,192,64)
    errors=[]
    for factor in [1e-6,1e6]:
        other,_,_=native_windows(eeg*factor,labels,names)
        error=float(np.max(np.abs(actual-other)));assert error<1e-6;errors.append(error)
    changed=eeg.copy();changed[192:]+=12345
    other,_,_=native_windows(changed,labels,names);assert np.array_equal(actual[0],other[0])
    perm=np.random.default_rng(1122).permutation(len(labels))
    reordered,_,_=native_windows(eeg[:,perm],[labels[i] for i in perm],names)
    assert np.array_equal(actual,reordered)
    constant,_,_=native_windows(np.ones((3200,66)),labels,names)
    assert np.count_nonzero(constant)==0 and np.isfinite(constant).all()
    h=np.hanning(192);weight=np.zeros(3200)
    for s in starts:weight[s:s+192]+=h
    covered=np.flatnonzero(weight>1e-8);assert covered.tolist()==list(range(1,3167))
    for i in range(2,len(starts)):assert starts[i-2]+192==starts[i]
    result={'engineering_only':True,'neural_model_or_target_performance_computed':False,
        'proposal_not_yet_external_protocol_approval':True,'window_shape':list(actual.shape),
        'global_positive_unit_scale_invariance_max_abs_float32':errors,
        'future_after_first_current3s_cannot_change_first_window':True,
        'physical_label_permutation_invariance':True,'constant_channels_finite_zero':True,
        'first_trial_zero_variance_channel_counts':constants,
        'Hann':{'covered_samples':len(covered),'first_inclusive':1,'last_inclusive':3166,
            'uncovered_half_open_ranges':[[0,1],[3167,3200]],'last_native_window_start':2976,
            'last_native_window_end_exclusive':3168,'no_appended_irregular_3008_window':True},
        'parity_history_stride_seconds':3,'raw_support_i_minus2_end_equals_current_start':True,
        'cold_first_two_windows_included':True,'older_history_support_audit':hann_support_audit(32,3200)}
    with (ROOT/'ADAPTER_GEOMETRY_ENGINEERING_CHECK.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='older_history_support_audit'}))


if __name__=='__main__':main()
