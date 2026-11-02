"""Pre-prediction amendment: shared, label-free per-record scalar amplitude calibration.

Output is source-amplitude calibrated numeric units, not verified physical microvolts.
Full-record future EEG is used; this is an offline/transductive input adaptation.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy import signal
from pub_01_s03_ss07_openneuro_adapter import (
    SOURCE_NAMES,SCALP_NAMES,geometry,decode_microvolt,envelope64,fixed_norm_windows)

NORM=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261015/transformer_available01/prepared/train_only_normalization.npz')
NORM_SHA='2ed88be6c5d2279a65276e20d20fc3725d2d7e21185a90452886b367759784d8'


def source_normalization():
    assert hashlib.sha256(NORM.read_bytes()).hexdigest()==NORM_SHA
    norm=np.load(NORM,allow_pickle=False)
    mean=np.asarray(norm['mean'],'f8');std=np.asarray(norm['std'],'f8')
    assert mean.shape==std.shape==(64,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all()
    amplitude=float(np.sqrt(np.mean(std**2)))
    return mean,std,amplitude


def eeg64_numeric(payload,header,good_names,matrix):
    """Retain FP64 DSP until record calibration; header scale is a numeric convention."""
    x=decode_microvolt(payload,header,good_names)
    assert np.any(x.max(axis=1)>x.min(axis=1)), 'Mechanical failure: all good channels are temporally constant before filtering'
    sos=signal.butter(1,.5,btype='highpass',fs=512,output='sos')
    padlen=3*(2*len(sos)-min((sos[:,2]==0).sum(),(sos[:,5]==0).sum()))
    x=signal.sosfiltfilt(sos,x,axis=1,padtype='odd',padlen=padlen)
    x=matrix@x;x-=x.mean(0,keepdims=True)
    x=signal.resample_poly(x,1,8,axis=1)
    assert x.shape[0]==64 and x.dtype==np.float64 and np.isfinite(x).all()
    return x


def calibrate_record(x,source_amplitude):
    """Use all L common-duration samples [0,L), including Hann-uncovered boundaries.

    A_record=sqrt(mean_c(mean_t((x_ct-mean_t(x_ct))**2))), ddof=0.
    Temporal centering is solely for estimating gain; the output itself is gain*x.
    """
    x=np.asarray(x,'f8')
    assert x.ndim==2 and x.shape[0]==64 and x.shape[1]>=192 and np.isfinite(x).all()
    assert np.isfinite(source_amplitude) and source_amplitude>0
    centered=x-x.mean(axis=1,keepdims=True)
    amplitude=float(np.sqrt(np.mean(centered**2)))
    assert np.isfinite(amplitude) and amplitude>0,'Mechanical failure: nonpositive/nonfinite record amplitude'
    gain=source_amplitude/amplitude
    calibrated=x*gain
    assert np.isfinite(calibrated).all()
    return calibrated,dict(A_record=amplitude,A_source=source_amplitude,gain=gain,
        time_support=[0,x.shape[1]],variance_ddof=0,channels=64,
        support='All cropped common-duration EEG samples; independent of window/Hann/target values',
        units='source-amplitude calibrated numeric units; physical gain UNVERIFIED',
        uses_full_record_future_EEG=True,label_free=True,clamp=None)


def prepare_record(payload,header,good_names,matrix,audio_samples64,source_amplitude):
    x=eeg64_numeric(payload,header,good_names,matrix)
    length=min(x.shape[1],audio_samples64)
    x,info=calibrate_record(x[:,:length],source_amplitude)
    # Only final stored input is FP32, before the unchanged old TRAIN normalization.
    return np.ascontiguousarray(x,dtype=np.float32),info


def synthetic_check():
    rng=np.random.default_rng(20261102)
    x=rng.normal(size=(64,3000))*np.linspace(.5,2,64)[:,None]+np.arange(64)[:,None]
    _,_,a=source_normalization();reference,info=calibrate_record(x,a)
    errors=[]
    for scale in [1e-9,.01,1.,1e6,1e9]:
        value,_=calibrate_record(x*scale,a)
        error=float(np.max(abs(value-reference)))
        assert error<=1e-11
        errors.append(dict(positive_unit_multiplier=scale,output_max_abs=error))
    # Processing another record cannot mutate previously computed calibration/output.
    held=reference.copy();other,_=calibrate_record(rng.normal(size=(64,1000))*1e4,a)
    assert np.array_equal(held,reference) and other.shape==(64,1000)
    assert abs(np.sqrt(np.mean((reference-reference.mean(1,keepdims=True))**2))-a)<=1e-12
    failed=False
    try:calibrate_record(np.ones((64,192)),a)
    except AssertionError:failed=True
    assert failed
    header=dict(binary_format='IEEE_FLOAT_32',orientation='MULTIPLEXED',channel_count=128,
        sampling_interval_us=1953.125,channels=[dict(name=n,resolution=.1,unit='uV') for n in SOURCE_NAMES])
    dc=np.broadcast_to(np.arange(128,dtype='<f4'),(4096,128)).copy()
    matrix,_=geometry(list(SOURCE_NAMES));failed=False
    try:prepare_record(dc.tobytes(),header,list(SOURCE_NAMES),matrix,512,a)
    except AssertionError:failed=True
    assert failed,'Different per-channel DC must fail before filter roundoff can be amplified'
    return dict(passed=True,A_source=a,positive_scale_invariance=errors,
        record_independence=True,constant_record_rejected=True,raw_constant_DC_before_filter_rejected=True,
        numeric_contract='offline full-record scalar gain; not physical unit correction',NN_forwards=0)
