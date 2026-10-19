"""Write-once pre-test design after the bounded train/val feasibility pilot."""
import argparse
import json
from pathlib import Path
import time
from pub_01_s03_ss03_data import load_contract,dump,digest
from pub_01_s03_temporal_scales import TemporalScaleDecoder,model_parameters,DILATION_PRESETS


def main(config):
    cfg,_,_,root=load_contract(config)
    assert json.loads((root/'bootstrap_completed.json').read_text())['test_access'] is False
    pilot=json.loads((root/'pilot.json').read_text());assert pilot['optimizer_steps']==100 and not pilot['test_access']
    assert not (root/'formal01').exists() and not (root/'test_opened.json').exists()
    audit=json.loads((root/'cache01/manifest.json').read_text());assert all(audit['checks'].values())
    assert cfg['training']['formal_budget']==25000 and cfg['training']['validation_every_steps']==2500
    names=['pub_01_s03_temporal_scales.py','pub_01_s03_ss03_data.py','pub_01_s03_ss03_bootstrap.py',
        'pub_01_s03_ss03_eval.py','pub_01_s03_ss03_train.py','pub_01_s03_ss03_test.py',
        'pub_01_s03_ss03_freeze.py','pub_01_s03_ss03_run.py',Path(config).name]
    counts={v:model_parameters(TemporalScaleDecoder(variant=v)) for v in DILATION_PRESETS};assert len(set(counts.values()))==1
    protocol={'task':cfg['task'],'version':'SS03-MULTISCALE-FORMAL-v1','frozen_utc_epoch':time.time(),
        'variants':list(DILATION_PRESETS),'dilations':DILATION_PRESETS,'seeds':[1024,2026,31415],
        'parameters':counts,'nominal_conv_MACCs_per_192sample_window':192*(64*48+48*9+48*48+48),
        'optimizer':'AdamW','learning_rate':.001,'weight_decay':.01,'batch_size':64,'optimizer_steps':25000,
        'validation_every_steps':2500,'early_stopping':False,
        'selection':'max complete validation subject-macro60s Pearson r; exact tie picks earliest step',
        'validation_subjects':len(audit['phase_subjects']['val']),
        'batch_schedule':'same numpy.default_rng(seed) uniform64 distinct window IDs each step in every variant',
        'formal_weights_from_scratch':True,'pilot_weights_discarded':True,'test_access_before_freeze':False,
        'test_rule':'all12 training runs complete and reviewed before each selected checkpoint is tested exactly once; no fit after test',
        'primary':'separate seen/heldout subject-macro60s Pearson; paired multiscale-minus-long and minus-middle, seed means within person',
        'decision':'group-specific support requires both paired deltas positive with Bonferroni family95% intervals across4 comparisons; overall H1 only if both groups support; otherwise mixed/negative',
        'intervals':'subject bootstrap10000,seed20261019; report95% descriptive and98.75% Bonferroni for4 planned contrasts',
        'secondary':'2/3seen+1/3heldout; every seed and wrong-EEG control; no ranking against unlike SS02/TRF protocols',
        'constant_control':'constant train mean; Pearson undefined for zero variance; report null with reason, never fabricated0',
        'optional_linear_reference':'NOT_RUN; not needed for the fixed dilation hypothesis',
        'optional_temporal_diagnostics':'NOT_RUN; no validation jitter/modulation inference will be promoted into an unplanned test claim',
        'config':cfg,'source_sha256':{name:digest(root/'input'/name) for name in names},
        'input_contract_sha256':digest(root/'cache01/manifest.json'),'test_target_structural_checks_only_before_freeze':True}
    dump(root/'formal_protocol.json',protocol)
    dump(root/'freeze_review.json',{'technical':'PASS','scope':'common budget/model/seed/selection frozen before any SS03 scientific test',
        'test_scores_seen':False,'parameters_matched':True,'nominal_MACCs_matched':True,'train_val_pilot_steps':100,
        'same_upstream_inputs':True,'no_pretrained_encoder':True,'formal_protocol_sha256':digest(root/'formal_protocol.json')})
    print(json.dumps({'status':'TRAIN_VAL_FROZEN','steps':25000,'validate_every':2500,'runs':12}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);main(p.parse_args().config)
