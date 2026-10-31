"""Synthetic completion metadata checks. NO scientific checkpoint accepted.

The architecture/key/seed checkpoint loader has a separate real/unit check;
here a stub isolates the completion/ranking/cost manifest traversal itself.
"""
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch
import torch
from pub_01_s03_ss07_baseline import ROOT,OLD,read,dump,sha
import pub_01_s03_ss07_final_manifest as lock


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);dump(path,value)


def main():
    torch.set_num_threads(2);sys.path.insert(0,str(OLD/'input'))
    root=Path(tempfile.mkdtemp(prefix='synthetic_final_lock_',dir=ROOT/'engineering_history'))
    for name in lock.PROTOCOLS:
        if name in ['FULL_SELECTION.json','FULL_PRIMARY_SELECTION.json','EVALUATION_EXECUTOR_PROTOCOL.json']:continue
        shutil.copy2(ROOT/name,root/name)
    write(root/'EVALUATION_EXECUTOR_PROTOCOL.json',{'frozen':True,'sources':{},'synthetic_only':True})
    science=read(root/'INCREMENTAL_PROTOCOL.json');catalog={c['id']:c for c in science['full_conditions']};seeds=lock.SEEDS
    means={};selected=[f'm{i}_h16' for i in range(1,6)];primary='m2_h16'
    controls=['m2_current_h16','static_m2_h16','output_prior_h16']
    def completed(config,seed,stage,value):
        return {'seed':seed,'condition':config,'stage':stage,'test_read':False,'epochs':1,'updates':3945,
            'selected_epoch':0,'best_validation_negative_Pearson':-.1,
            'validation_macro':{'macro_r':value,'subjects':49,'blocks':752},
            'protocol_sha256':sha(root/'INCREMENTAL_PROTOCOL.json'),
            'engineering_amendment_sha256':sha(root/'FULL_ENGINEERING_AMENDMENT.json')}
    for c in science['conditions']:
        value=.2 if c['id'] in selected else .1;means[c['id']]=math.fsum([value]*3)/3
        for seed in seeds[:3]:write(root/f'screen/{c["id"]}_seed{seed}/completed.json',completed(c,seed,'screen',value))
    full={'frozen':True,'parent_protocol_sha256':sha(root/'INCREMENTAL_PROTOCOL.json'),
        'selected_family_condition_ids':selected,'ranking_means_all18':means,'test_read':False}
    write(root/'FULL_SELECTION.json',full)
    family_values={id_:(.3 if id_==primary else .2) for id_ in selected}
    family_means={id_:math.fsum([value]*3)/3 for id_,value in family_values.items()}
    final={'frozen':True,'final_seeds':seeds,'parent_full_selection_sha256':sha(root/'FULL_SELECTION.json'),
        'primary':primary,'same_full_budget_control_ids':controls,'full_family_validation_means':family_means,'test_read':False}
    write(root/'FULL_PRIMARY_SELECTION.json',final)
    write(root/'FULL_SELECTED_CONDITIONS_COMPLETED.json',{'conditions':[primary]+controls,'paired_seeds':seeds,'test_read':False})
    for seed in seeds:
        pre=root/f'baseline/pretrain_seed{seed}';pre.mkdir(parents=True)
        (pre/'best_validation.pt').write_text('SYNTHETIC_PLACEHOLDER_NOT_WEIGHTS '+str(seed))
        write(pre/'completed.json',{'seed':seed,'epochs':1,'updates':820,'selected_epoch':0,'best_validation':3.,
            'test_payload_loaded':False,'fresh_initialization':True,'source_protocol_sha256':sha(root/'BASELINE_PROTOCOL.json')})
        write(pre/'history.json',[{'epoch':0,'updates':820,'validation_loss':3.}])
        head=root/f'baseline/head_seed{seed}'
        write(head/'completed.json',{'seed':seed,'epochs':1,'updates':3945,'selected_epoch':0,'best_validation':-.1,
            'test_payload_loaded':False,'encoder_sha256':sha(pre/'best_validation.pt')})
        write(head/'history.json',[{'epoch':0,'updates':3945,'validation_negative_Pearson':-.1}])
        write(head/'cache_manifest.json',{'encoder_sha256':sha(pre/'best_validation.pt')})
        (head/'best_validation.pt').write_text('SYNTHETIC_PLACEHOLDER_NOT_WEIGHTS '+str(seed))
        ft=root/f'finetune/native_P_seed{seed}'
        write(ft/'completed.json',{'seed':seed,'epochs':1,'updates':3945,'selected_epoch':0,'test_read':False,
            'initial_validation_negative_Pearson':-.1,'best_validation_negative_Pearson':-.11,
            'protocol_sha256':sha(root/'FINETUNE_PROTOCOL.json')})
        write(ft/'history.json',[{'epoch':0,'updates':3945,'validation_negative_Pearson':-.11}])
        write(ft/'INCUMBENT_EQUIVALENCE.json',{'passed':True,'optimizer_steps_before_check':0,'absolute_difference':0.,'prespecified_absolute_tolerance':1e-4})
        (ft/'best_validation.pt').write_text('SYNTHETIC_PLACEHOLDER_NOT_WEIGHTS '+str(seed))
    cells={(id_,s) for id_ in [primary]+controls for s in seeds}
    cells.update((id_,s) for id_ in selected for s in seeds[:3])
    for id_,seed in cells:
        folder=root/f'full/{id_}_seed{seed}'
        value=family_values.get(id_,.1);write(folder/'completed.json',completed(catalog[id_],seed,'full',value))
        write(folder/'history.json',[{'epoch':0,'updates':3945,'validation_negative_Pearson':-.1}])
        write(folder/'resolved_config.json',{'encoder_sha256':sha(root/f'baseline/pretrain_seed{seed}/best_validation.pt')})
        (folder/'best_validation.pt').write_text('SYNTHETIC_PLACEHOLDER_NOT_WEIGHTS '+str(seed))
    write(root/'SOURCE_FAMILY_JOIN.json',{'synthetic_only':True})
    write(root/'DONOR_METADATA_PLAN.json',{'synthetic_only':True})
    def stub(path,model,state_key,seed,epoch):
        assert path.read_text().startswith('SYNTHETIC_PLACEHOLDER_NOT_WEIGHTS')
        return {'path':str(path),'sha256':sha(path),'seed':seed,'selected_epoch':epoch,
            'state_key':state_key,'registered_parameters':sum(p.numel() for p in model.parameters())}
    with patch.object(lock,'ROOT',root),patch.object(lock,'checkpoint',stub):lock.build()
    result=read(root/'FINAL_CHECKPOINT_MANIFEST.json')
    ledger=result['training_costs'];assert sum(c['stage']=='screen' for c in ledger)==54
    assert sum(c['stage']=='shared_pretrain' for c in ledger)==5
    assert sum(c['stage']=='full' for c in ledger)==len(cells)==32
    assert result['primary']==primary and result['test_signal_or_target_or_score_read'] is False
    # Copy without the output manifest; retain original positive fixture.
    bad=root.parent/(root.name+'_wrong_full_mean')
    shutil.copytree(root,bad,ignore=shutil.ignore_patterns('FINAL_CHECKPOINT_MANIFEST.json'))
    altered=read(bad/'FULL_PRIMARY_SELECTION.json');altered['full_family_validation_means']['m1_h16']=.9
    (bad/'FULL_PRIMARY_SELECTION.json').write_text(json.dumps(altered,indent=2))
    with patch.object(lock,'ROOT',bad),patch.object(lock,'checkpoint',stub):
        try:lock.build()
        except AssertionError:pass
        else:raise AssertionError('false recorded full-validation means accepted')
    assert not (bad/'FINAL_CHECKPOINT_MANIFEST.json').exists()
    dump(ROOT/'FINAL_MANIFEST_ENGINEERING_CHECK.json',{'passed':True,'synthetic_metadata_only':True,
        'checkpoint_loader_stub_explicit_not_scientific_weight_validation':True,
        'positive_fixture':str(root),'negative_fixture':str(bad),
        'screen54_pretrain5_full32_cost_cells_accounted_once':True,
        'screen_width_full_primary_and_EMA_ranking_recomputed':True,
        'wrong_recorded_full_validation_means_rejected':True,
        'real_test_or_DTU_payload_read':False,'real_final_checkpoint_manifest_created':False})
    print(json.dumps({'PASS':True,'synthetic_only':True,'full_cells':len(cells)}))


if __name__=='__main__':main()
