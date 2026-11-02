"""Single predeclared ds004408->frozen Sparr64 adapter. No models or fitted target stats."""
from pathlib import Path
import hashlib
import inspect
import numpy as np
from scipy import signal
import mne
from mne.channels.interpolation import _make_interpolation_matrix
from brain_pipe.preprocessing.stimulus.audio.envelope import GammatoneEnvelope
from brain_pipe.preprocessing.resample import ResamplePoly
import librosa

SCALP_NAMES=('Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8','AF4','AFz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2')
SOURCE_NAMES=tuple(letter+str(i) for letter in 'ABCD' for i in range(1,33))
UNIT_TO_MICROVOLT={'V':1e6,'mV':1e3,'uV':1.,'µV':1.,'μV':1.}


def geometry(good_names):
    assert mne.__version__=='1.6.1'
    assert len(set(good_names))==len(good_names) and all(x in SOURCE_NAMES for x in good_names)
    assert len(good_names)>=4
    source=mne.channels.make_standard_montage('biosemi128',head_size=.095).get_positions()['ch_pos']
    target=mne.channels.make_standard_montage('biosemi64',head_size=.095).get_positions()['ch_pos']
    assert tuple(source)==SOURCE_NAMES and tuple(target)==SCALP_NAMES
    pos_from=np.array([source[name] for name in good_names],np.float64)
    pos_to=np.array([target[name] for name in SCALP_NAMES],np.float64)
    matrix=_make_interpolation_matrix(pos_from,pos_to,alpha=1e-5)
    assert matrix.shape==(64,len(good_names)) and np.isfinite(matrix).all()
    row_sum_error=float(abs(matrix.sum(1)-1).max())
    assert row_sum_error<=1e-8
    return matrix,dict(good_channels=len(good_names),target_channels=64,
        constant_preservation_max_abs=row_sum_error,
        maximum_abs_coefficient=float(abs(matrix).max()),maximum_row_L1=float(abs(matrix).sum(1).max()),
        source_template='biosemi128',target_template='biosemi64',head_size_m=.095,
        alpha=1e-5,mne_version=mne.__version__,positions_are_assumed_templates=True)


def decode_microvolt(payload,header,good_names):
    assert header['binary_format']=='IEEE_FLOAT_32' and header['orientation']=='MULTIPLEXED'
    assert header['channel_count']==128 and header['sampling_interval_us']==1953.125
    channels=header['channels']
    assert tuple(row['name'] for row in channels)==SOURCE_NAMES
    raw=np.frombuffer(payload,dtype='<f4').reshape(-1,128)
    indices=[SOURCE_NAMES.index(name) for name in good_names]
    scales=np.array([channels[i]['resolution']*UNIT_TO_MICROVOLT[channels[i]['unit'].strip()] for i in indices])
    result=np.asarray(raw[:,indices].T,np.float64)*scales[:,None]
    assert np.isfinite(result).all(),'Nonfinite unmarked-good channel: affected record technical failure'
    return result


def eeg64(payload,header,good_names,matrix):
    x=decode_microvolt(payload,header,good_names)
    # Match the published order1 high-pass and MATLAB-emulation padding rule.
    sos=signal.butter(1,.5,btype='highpass',fs=512,output='sos')
    minimum=min((sos[:,2]==0).sum(),(sos[:,5]==0).sum())
    padlen=3*(2*len(sos)-minimum)
    x=signal.sosfiltfilt(sos,x,axis=1,padtype='odd',padlen=padlen)
    mapped=matrix@x
    mapped-=mapped.mean(0,keepdims=True)
    result=signal.resample_poly(mapped,1,8,axis=1)
    assert result.shape[0]==64 and np.isfinite(result).all()
    return np.ascontiguousarray(result,dtype=np.float32)


def envelope64(path):
    wave,rate=librosa.load(path,sr=None,mono=True,dtype=np.float32)
    assert rate>=10000 and wave.ndim==1 and np.isfinite(wave).all()
    processor=GammatoneEnvelope(power_factor=.6,min_freq=50,max_freq=5000,bands=28)
    value=processor(dict(stimulus_data=wave,stimulus_sr=rate))
    # Existing brain_pipe uses sum(abs(gammatone)^.6), not channel mean.
    value=ResamplePoly(64,data_key='envelope_data',sampling_frequency_key='stimulus_sr')(value)
    result=np.asarray(value['envelope_data'],np.float32).reshape(-1)
    assert result.size>=192 and np.isfinite(result).all()
    return result,dict(native_audio_rate=int(rate),mono='librosa arithmetic average',
        brain_pipe_envelope_source_sha256=hashlib.sha256(Path(inspect.getfile(GammatoneEnvelope)).read_bytes()).hexdigest())


def fixed_norm_windows(eeg,mean,std,starts):
    assert eeg.shape[0]==64 and mean.size==std.size==64 and np.all(std>0)
    x=eeg.T
    return np.stack([(np.asarray(x[s:s+192],np.float32)-mean)/std for s in starts]).astype(np.float32)


def synthetic_check():
    good=list(SOURCE_NAMES)
    matrix,info=geometry(good)
    other,_=geometry([x for x in good if x not in ('B2','C20','C21','C24','C25')])
    assert np.max(abs(matrix@np.ones(128)-1))<=1e-8
    assert np.max(abs(other@np.ones(123)-1))<=1e-8
    rng=np.random.default_rng(20261102)
    raw=rng.normal(size=(4096,128)).astype('<f4')
    header=dict(binary_format='IEEE_FLOAT_32',orientation='MULTIPLEXED',channel_count=128,
                sampling_interval_us=1953.125,
                channels=[dict(name=n,resolution=.1,unit='µV') for n in SOURCE_NAMES])
    decoded=decode_microvolt(raw.tobytes(),header,good)
    assert np.array_equal(decoded,raw.T.astype(np.float64)*.1)
    result=eeg64(raw.tobytes(),header,good,matrix)
    assert result.shape==(64,512) and result.flags.c_contiguous
    return dict(passed=True,template_constant_canary=info,
                physical_unit_pointwise_exact=True,resample512_to64=True,
                CAR64_max_abs=float(abs(result.astype(np.float64).sum(0)).max()),
                synthetic_only=True,no_NN=True)
