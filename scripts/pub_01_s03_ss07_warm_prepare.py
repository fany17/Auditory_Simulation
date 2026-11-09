"""Read-only upstream intake or reviewed freeze, never starts a fit."""
import argparse
import json
import shutil
import time
from pub_01_s03_ss07_warm import ROOT,ROUND01,OLD,SEEDS,CONDITIONS,read,dump,sha,disk_guard

def sources():
    paths=list(sorted((ROOT/'input').glob('pub_01_s03_ss07_warm*.py')))
    paths += [ROUND01/'input'/n for n in ['pub_01_s03_ss07_baseline.py','pub_01_s03_ss07_finetune.py',
        'pub_01_s03_ss07_incremental.py','pub_01_s03_ss07_train.py']]
    paths += [OLD/'input/pub_01_s03_transformer_train_v2.py',OLD/'input/pub_01_s03_method_models.py',
        OLD/'training_seed1024_attempt04/config_frozen.json',OLD/'envelope_readout_seed1024/manifest_frozen.json',
        OLD/'prepared/train_only_normalization.npz',ROUND01/'INCREMENTAL_PROTOCOL.json',ROUND01/'block_eligibility/val.json']
    for seed in SEEDS:
        paths += [ROUND01/f'baseline/pretrain_seed{seed}/best_validation.pt',ROUND01/f'baseline/head_seed{seed}/best_validation.pt',
            ROUND01/f'baseline/head_seed{seed}/cache_manifest.json']
    return {str(p):sha(p) for p in paths}

def main(freeze):
    disk_guard();assert not (ROOT/'fits').exists()
    source_map=sources()
    if not freeze:
        payloads=[p for p in ROUND01.rglob('*') if p.is_file() and p.suffix in ('.npy','.npz','.pt')]
        existing=sum(p.stat().st_size for p in payloads)
        allowance=18*2*15*1024**2
        assert existing+allowance<=25*1024**3
        dump(ROOT/'INTAKE.json',{'time':time.time(),'sources':source_map,'server_only':True,'patient_data':'FORBIDDEN',
            'phases':['train','val'],'new_downloads':False,'upstream_ROUND01_read_only':True,
            'existing_round01_array_weight_bytes':existing,'new_best_plus_temporary_upper_bytes':allowance,
            'combined_payload_budget_bytes':25*1024**3,'free_bytes':shutil.disk_usage(ROOT).free,
            'new_array_cache':False,'new_fits_upper_bound':18,'total_updates_upper_bound':355050,
            'total_window_presentations_upper_bound':22720860,'test_operations':False})
        print('WARM_INTAKE_PASS',flush=True);return
    engineering=read(ROOT/'ENGINEERING_COMPLETED.json');assert engineering['passed'] and engineering['real_data_optimizer_steps']==0
    review=read(ROOT/'INDEPENDENT_CODE_REVIEW.json');assert review['decision']=='PASS_FOR_TRAIN_VAL'
    assert review['sources']==source_map,'reviewed sources changed'
    assert review['engineering_sha256']==sha(ROOT/'ENGINEERING_COMPLETED.json')
    reuse=read(ROOT/'LEGACY_FT_REUSE_DECISION.json');assert reuse['decision_before_any_new_fit'] and not reuse['score_triggered']
    # This implementation's actual engineering decision chooses 18 matched fits.
    assert reuse['legacy_FT_reused'] is False and reuse['new_fits']==18
    proofs={str(ROOT/name):sha(ROOT/name) for name in ['INTAKE.json','ENGINEERING_COMPLETED.json',
        'LEGACY_FT_REUSE_DECISION.json','INDEPENDENT_CODE_REVIEW.json']}
    for p,d in engineering['proofs'].items():
        from pathlib import Path
        assert Path(p).is_relative_to(ROOT) and sha(p)==d;proofs[p]=d
    dump(ROOT/'WARM_PROTOCOL.json',{'frozen':True,'time':time.time(),'sources':{**source_map,**proofs},
        'seeds':list(SEEDS),'new_fit_conditions':list(CONDITIONS),'new_fits':18,'epochs':5,
        'global_batch':64,'updates_per_epoch':3945,'train_windows_per_epoch':252454,'validation_windows':30500,
        'maximum_updates':355050,'maximum_supervised_window_presentations':22720860,
        'checkpoint_selection':'original allVAL3s negativePearson minimum, epoch-1 incumbent eligible, strict earliest improvement; no early stopping within fixed5',
        'reporting':'allcells incumbent and selected VAL60s/person and3s; no test; oldFT only historical reference',
        'optimizer':'freshAdam,encoder1e-5/readout1e-3,no wd/no clip/no shuffle','precision':read(ROUND01/'INCREMENTAL_PROTOCOL.json')['precision'],
        'row_interaction_caveat':'encoder joint vs frozen also changes train/eval dropout, not pure encoder-gradient intervention',
        'history':'originalM4H16,tau30,shifts4/8/16,.1;nonoverlapparity;global64encoderbeforestatesplits;detachafterstep;resetrecord/epoch/phase',
        'budget_failure_policy':'no retry, no new cell or budget extension; preserve all evidence and stop affected cell',
        'test_operations':False,'patient_data':'FORBIDDEN','scientific_goal_achieved_claimed':False})
    print('WARM_PROTOCOL_FROZEN',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();main(a.freeze)
