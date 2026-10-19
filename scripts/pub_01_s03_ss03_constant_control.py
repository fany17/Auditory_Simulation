"""Fit the predeclared constant control from training targets before test."""
import json
from pathlib import Path
import time
import numpy as np
root=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_architecture_20261019')
if (root/'test_opened.json').exists():raise RuntimeError('constant fit must precede test')
source=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261015/transformer_available01/envelope_readout_seed1024/train_envelope.npy')
target=np.load(source,mmap_mode='r');value=float(np.mean(target,dtype=np.float64))
with (root/'constant_train_mean_control.json').open('x') as f:json.dump({'fit_phase':'train_only','mean':value,
    'prediction_peak_to_peak':0.,'Pearson_r':None,'reason':'UNDEFINED_ZERO_VARIANCE','test_access':False,'utc_epoch':time.time()},f,indent=2)
print('TRAIN_MEAN_CONTROL_FROZEN_PEARSON_UNDEFINED')
