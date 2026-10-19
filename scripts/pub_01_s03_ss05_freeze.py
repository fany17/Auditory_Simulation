"""Actual train/val-dependent protocol freeze BEFORE newholdout signal access."""
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from pub_01_s03_ss05_data import ROOT,dump,split
from pub_01_s03_ss05_train import digest


def main():
    cfg=json.loads((ROOT/'input/pub_01_s03_ss05_eeg_formal_20261019.json').read_text())
    pilot=json.loads((ROOT/'pilot01/pilot.json').read_text());assert pilot['holdout_signal_access'] is False
    assert len(pilot['runs'])==20 and all(r['optimizer_steps']==100 for r in pilot['runs'])
    assert not (ROOT/'cache_holdout').exists() and not (ROOT/'formal01').exists()
    source=split();expected=[r for r in source['records'] if r['phase']=='holdout' and r['common_samples']//192>=60]
    assert len({r['source_family'] for r in expected})>=4 and len({r['subject'] for r in expected})>=10
    names=['pub_01_s03_ss05_predictive_memory.py','pub_01_s03_ss05_window_decoder.py','pub_01_s03_ss05_finite_history.py',
        'pub_01_s03_ss05_data.py','pub_01_s03_ss05_evaluate.py','pub_01_s03_ss05_train.py','pub_01_s03_ss05_test.py',
        'pub_01_s03_ss05_freeze.py','pub_01_s03_ss05_run.py','pub_01_s03_ss05_eeg_formal_20261019.json']
    cache=json.loads((ROOT/'cache_trainval/manifest.json').read_text());norm=json.loads((ROOT/'normalization_contract.json').read_text())
    assert set(norm['fit_record_ids'])=={r['record'] for r in source['records'] if r['phase']=='train'} and not norm['old_normalization_used']
    total=0.;count=0
    for r in cache['records']:
        if r['phase']=='train':
            y=np.load(r['cache_envelope'],mmap_mode='r');total+=float(np.sum(y,dtype=np.float64));count+=len(y)
    constant={'fit_phase':'SS05_train_only','mean':total/count,'Pearson_r':None,'reason':'constant_zero_variance','frozen_before_holdout_payload':True}
    cfg.update(version='SS05-EEG-FORMAL-v1',lag_samples=pilot['selected_lag_samples'],frozen_utc_epoch=time.time(),
        holdout_payload_seen_before_freeze=False,pilot_weights_reused=False,constant_train_mean_control=constant,
        source_sha256={name:digest(ROOT/'input'/name) for name in names},
        input_contract_sha256={'split':digest(ROOT/'split_frozen.json'),'normalizer':digest(ROOT/'newtrain_normalization.npz'),'trainval_cache_manifest':digest(ROOT/'cache_trainval/manifest.json')},
        environment={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'GPU':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
        expected_holdout_from_metadata={'eligible_records':len(expected),'subjects':len({r['subject'] for r in expected}),'source_families':len({r['source_family'] for r in expected})},
        caveat='full-minus-assoc isolates learned history-residual branch and extra reachable predictor weights, not a proven forecasting-error objective; high/low similarity is not EEG semantic relevance groundtruth')
    dump(ROOT/'FORMAL_PROTOCOL.json',cfg)
    dump(ROOT/'freeze_review.json',{'technical':'PASS','newholdout_signal_access':False,'same_current_EEG_history_controls_frozen':True,
        'split_group_disjoint':True,'normalization_newtrain_only':True,'lag_trainval_only':True,'selected_lag':cfg['lag_samples'],
        'formal_runs':15,'optimizer_steps_perrun':2000,'protocol_sha256':digest(ROOT/'FORMAL_PROTOCOL.json')})
    print(json.dumps({'status':'TRAIN_VAL_FROZEN','lag':cfg['lag_samples'],'runs':15,'holdout':cfg['expected_holdout_from_metadata']}),flush=True)


if __name__=='__main__':main()
