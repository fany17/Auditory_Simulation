"""Lock completed train/val decisions and EXACT checkpoints, no test reads.

This is the previously specified final checkpoint child, not new approval.
Signal/target masks and inference remain separate until this succeeds.
"""
import json
import math
from pathlib import Path
import time
import torch
from pub_01_s03_ss07_baseline import ROOT,read,dump,sha,native,cfg
from pub_01_s03_ss07_train import make
from pub_01_s03_ss07_finetune import Decoder


SEEDS=[1024,2026,31415,42,12345]
PROTOCOLS=['BASELINE_PROTOCOL.json','INCREMENTAL_PROTOCOL.json','FULL_SELECTION.json',
    'FULL_PRIMARY_SELECTION.json','FULL_ENGINEERING_AMENDMENT.json','FULL_QUEUE_V2_PROTOCOL.json',
    'EVALUATION_RULES.json','DTU_PROTOCOL.json','FINETUNE_PROTOCOL.json','EMA_PROTOCOL.json',
    'EMA_PROTOCOL_V2.json','EMA_NUMERICAL_AMENDMENT.json','EMA_SELECTION.json','EVALUATION_EXECUTOR_PROTOCOL.json']


def check_sources(protocol):
    assert protocol.get('frozen',False)
    for field in ['sources','new_sources','proofs']:
        for path,digest in protocol.get(field,{}).items():assert sha(path)==digest,'changed frozen source '+path


def checkpoint(path,model,state_key,seed,epoch):
    assert path.exists()
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['seed']==seed and saved['epoch']==epoch
    # Strict key/shape checking on the REAL architecture, no forward needed.
    model.load_state_dict(saved[state_key],strict=True)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if v.dtype.is_floating_point)
    return {'path':str(path),'sha256':sha(path),'seed':seed,'selected_epoch':epoch,
        'state_key':state_key,'registered_parameters':sum(v.numel() for v in model.parameters())}


def history_check(folder,completed,updates_per_epoch,score_field,incumbent=None):
    history=read(folder/'history.json');assert len(history)==completed['epochs']>0
    for i,row in enumerate(history):
        assert row['epoch']==i and row['updates']==(i+1)*updates_per_epoch
        assert math.isfinite(row[score_field])
    assert completed['updates']==len(history)*updates_per_epoch
    candidates=[(-1,incumbent)] if incumbent is not None else []
    candidates.extend((row['epoch'],row[score_field]) for row in history)
    best=min(candidates,key=lambda pair:(pair[1],pair[0]))
    assert best[0]==completed['selected_epoch'],'checkpoint not earliest minimum validation point'
    return best[1]


def build():
    torch.set_num_threads(2)
    # Only JSON, train/val histories and train-derived checkpoints above here.
    protocols={name:read(ROOT/name) for name in PROTOCOLS}
    for name,p in protocols.items():check_sources(p)
    science=protocols['INCREMENTAL_PROTOCOL.json'];full=protocols['FULL_SELECTION.json'];final=protocols['FULL_PRIMARY_SELECTION.json']
    assert science['final_seeds']==final['final_seeds']==SEEDS
    assert full['parent_protocol_sha256']==sha(ROOT/'INCREMENTAL_PROTOCOL.json')
    assert final['parent_full_selection_sha256']==sha(ROOT/'FULL_SELECTION.json')
    assert final['test_read'] is False and full['test_read'] is False
    assert final['primary'] in full['selected_family_condition_ids']
    assert science['screening_seeds']==[1024,2026,31415]
    controls=final['same_full_budget_control_ids'];main_full=[final['primary']]+controls
    assert len(set(main_full))==len(main_full)
    catalog={c['id']:c for c in science['full_conditions']};primary=catalog[final['primary']]
    f=primary['family'];h=primary['width']
    expected=[f'{f}_current_h{h}',f'static_{f}_h{h}',f'output_prior_h{h}']
    extra={'m3':f'm3_reconstruct_h{h}','m4':'static_alignment','m5':f'm5_prior_h{h}'}.get(f)
    if extra:expected.append(extra)
    assert controls==expected,'control set drifted from frozen family rule'
    completed_all=read(ROOT/'FULL_SELECTED_CONDITIONS_COMPLETED.json')
    assert completed_all['conditions']==main_full and completed_all['paired_seeds']==SEEDS
    assert completed_all['test_read'] is False
    ema=protocols['EMA_SELECTION.json'];assert ema['test_read'] is False
    assert ema['screening_seeds']==science['screening_seeds']
    assert ema['apply_same_tau_to_all_final5seeds']==SEEDS
    assert ema['parent_protocol_sha256']==sha(ROOT/'INCREMENTAL_PROTOCOL.json')
    assert ema['EMA_protocol_sha256']==sha(ROOT/'EMA_PROTOCOL_V2.json')
    assert ema['numerical_amendment_sha256']==sha(ROOT/'EMA_NUMERICAL_AMENDMENT.json')
    assert ema['tau_seconds'] in science['EMA_control']['taus_seconds']
    tau_means={tau:math.fsum(ema['paired_scores'][str(tau)])/3 for tau in science['EMA_control']['taus_seconds']}
    assert all(len(ema['paired_scores'][str(tau)])==3 and all(math.isfinite(v) for v in ema['paired_scores'][str(tau)]) for tau in tau_means)
    assert all(tau_means[tau]==ema['validation_means'][str(tau)] for tau in tau_means)
    assert sorted(tau_means,key=lambda tau:(-tau_means[tau],tau))[0]==ema['tau_seconds']
    # Validate every screen cell and full initial family cell before locking.
    screen_means={};costs=[]
    for config in science['conditions']:
        scores=[]
        for seed in science['screening_seeds']:
            c=read(ROOT/f'screen/{config["id"]}_seed{seed}/completed.json')
            assert c['condition']==config and c['seed']==seed and c['stage']=='screen' and c['test_read'] is False
            assert c['updates']==c['epochs']*3945 and c['validation_macro']['subjects']==49 and c['validation_macro']['blocks']==752
            assert c['epochs']<=12 and math.isfinite(c['validation_macro']['macro_r'])
            assert c['protocol_sha256']==sha(ROOT/'INCREMENTAL_PROTOCOL.json')
            scores.append(c['validation_macro']['macro_r'])
            costs.append({'stage':'screen','kind':config['id'],'seed':seed,'training':c,
                'shared_pretrain_seed':seed,'includes_unselected_conditions':True})
        screen_means[config['id']]=math.fsum(scores)/3
    assert screen_means==full['ranking_means_all18']
    expected_family_ids=[]
    for family in ['m1','m2','m3','m4','m5']:
        alternatives=[catalog[f'{family}_h{width}'] for width in [16,32]]
        expected_family_ids.append(sorted(alternatives,key=lambda c:(-screen_means[c['id']],c['width'],c['id']))[0]['id'])
    assert full['selected_family_condition_ids']==expected_family_ids
    entries={};encoders={};mod=native()
    for seed in SEEDS:
        pre=ROOT/f'baseline/pretrain_seed{seed}';pc=read(pre/'completed.json')
        assert pc['seed']==seed and pc['test_payload_loaded'] is False and pc['fresh_initialization']
        assert pc['source_protocol_sha256']==sha(ROOT/'BASELINE_PROTOCOL.json')
        assert history_check(pre,pc,820,'validation_loss')==pc['best_validation']
        original=mod.TransformerCLIP(cfg())
        encoders[str(seed)]=checkpoint(pre/'best_validation.pt',original,'model',seed,pc['selected_epoch'])
        assert encoders[str(seed)]['registered_parameters']==3800713
        folder=ROOT/f'baseline/head_seed{seed}';c=read(folder/'completed.json')
        assert c['seed']==seed and c['test_payload_loaded'] is False and c['encoder_sha256']==sha(pre/'best_validation.pt')
        assert read(folder/'cache_manifest.json')['encoder_sha256']==sha(pre/'best_validation.pt')
        assert history_check(folder,c,3945,'validation_negative_Pearson')==c['best_validation']
        head=make({'family':'native'},seed)
        entry=checkpoint(folder/'best_validation.pt',head.head,'head',seed,c['selected_epoch'])
        entry.update({'kind':'native_P','condition':'native_P','config':{'family':'native'},
            'encoder_sha256':encoders[str(seed)]['sha256']})
        entries[f'native_P:{seed}']=entry
        costs.append({'stage':'shared_pretrain','kind':'TransformerCLIP','seed':seed,'training':pc,
            'charged_once_in_search_total':True,'shared_by_all_same_seed_frozen_encoder_conditions':True})
        costs.append({'stage':'native_head','kind':'native_P','seed':seed,'training':c,'shared_pretrain_seed':seed})
        ft=ROOT/f'finetune/native_P_seed{seed}';fc=read(ft/'completed.json')
        assert fc['seed']==seed and fc['test_read'] is False and fc['epochs']<=5
        assert fc['protocol_sha256']==sha(ROOT/'FINETUNE_PROTOCOL.json')
        equivalent=read(ft/'INCUMBENT_EQUIVALENCE.json')
        assert equivalent['passed'] and equivalent['optimizer_steps_before_check']==0
        assert equivalent['prespecified_absolute_tolerance']==1e-4
        assert equivalent['absolute_difference']<=1e-4
        assert history_check(ft,fc,3945,'validation_negative_Pearson',fc['initial_validation_negative_Pearson'])==fc['best_validation_negative_Pearson']
        decoder=Decoder(original.eeg)
        entry=checkpoint(ft/'best_validation.pt',decoder,'model',seed,fc['selected_epoch'])
        assert entry['registered_parameters']==3372617
        entry.update({'kind':'P_FT','condition':'P_FT','config':{'family':'native'}})
        entries[f'P_FT:{seed}']=entry;costs.append({'stage':'finetune','kind':'P_FT','seed':seed,'training':fc,
            'warmstart_pretrain_and_native_head_not_recharged':True})
    full_cells={(id_,seed) for id_ in main_full for seed in SEEDS}
    full_cells.update((id_,seed) for id_ in full['selected_family_condition_ids'] for seed in science['screening_seeds'])
    full_scores={id_:[] for id_ in full['selected_family_condition_ids']}
    for id_,seed in sorted(full_cells):
        config=catalog[id_];folder=ROOT/f'full/{id_}_seed{seed}';c=read(folder/'completed.json')
        assert c['seed']==seed and c['condition']==config and c['stage']=='full' and c['test_read'] is False
        assert c['protocol_sha256']==sha(ROOT/'INCREMENTAL_PROTOCOL.json')
        assert c['engineering_amendment_sha256']==sha(ROOT/'FULL_ENGINEERING_AMENDMENT.json')
        assert read(folder/'resolved_config.json')['encoder_sha256']==encoders[str(seed)]['sha256']
        assert c['epochs']<=250 and c['validation_macro']['subjects']==49 and c['validation_macro']['blocks']==752
        assert history_check(folder,c,3945,'validation_negative_Pearson')==c['best_validation_negative_Pearson']
        entry=checkpoint(folder/'best_validation.pt',make(config,seed),'model',seed,c['selected_epoch'])
        entry.update({'kind':'incremental','condition':id_,'config':config,
            'encoder_sha256':encoders[str(seed)]['sha256'],'engineering_amendment_sha256':c['engineering_amendment_sha256']})
        entries[f'{id_}:{seed}']=entry
        role='selected_primary_final5' if id_==final['primary'] else 'matched_control_final5' if id_ in controls else 'family_candidate_initial3'
        costs.append({'stage':'full','kind':id_,'seed':seed,'training':c,'role':role,'shared_pretrain_seed':seed})
        if id_ in full_scores and seed in science['screening_seeds']:full_scores[id_].append(c['validation_macro']['macro_r'])
    full_means={id_:math.fsum(values)/3 for id_,values in full_scores.items()}
    assert all(len(values)==3 and all(math.isfinite(v) for v in values) for values in full_scores.values())
    assert full_means==final['full_family_validation_means']
    family_order={family:i for i,family in enumerate(['m1','m2','m3','m4','m5'])}
    winner=sorted(expected_family_ids,key=lambda id_:(-full_means[id_],catalog[id_]['width'],family_order[catalog[id_]['family']],id_))[0]
    assert winner==final['primary'],'primary must remain first3-seed validation winner, no extra-seed reselection'
    rules=protocols['EVALUATION_RULES.json'];contrasts=[('primary-native_P','native_P'),('primary-current',controls[0]),
        ('primary-static',controls[1]),('primary-output_prior',controls[2]),('primary-P_EMA','P_EMA'),('primary-P_FT','P_FT')]
    if extra:contrasts.append((rules['additional_contrast_rule'][f],extra))
    assert len(contrasts) in (6,7)
    result={'frozen':True,'time':time.time(),'seeds':SEEDS,'screening_seeds':science['screening_seeds'],
        'primary':final['primary'],'main_model_ids':['native_P','P_EMA','P_FT']+main_full,
        'exploratory_family_ids':full['selected_family_condition_ids'],'contrast_ids_and_comparators':contrasts,
        'EMA_alias':{'condition':'native_P','tau_seconds':ema['tau_seconds']},
        'encoders':encoders,'entries':entries,'training_costs':costs,
        'protocols':{str(ROOT/name):sha(ROOT/name) for name in PROTOCOLS},
        'source_family_join_sha256':sha(ROOT/'SOURCE_FAMILY_JOIN.json'),
        'donor_metadata_plan_sha256':sha(ROOT/'DONOR_METADATA_PLAN.json'),
        'all_expected_train_val_cells_verified':True,'test_signal_or_target_or_score_read':False,
        'next':'freeze prediction-independent target/coverage masks, then ready child, then inference',
        'interpretation':'internal oldtest exposed; external fixed-adaptation/no-target-domain-model-fit'}
    dump(ROOT/'FINAL_CHECKPOINT_MANIFEST.json',result)
    print(json.dumps({'checkpoint_manifest':'FROZEN','entries':len(entries),'test_read':False}))


def require_checkpoints():
    manifest=read(ROOT/'FINAL_CHECKPOINT_MANIFEST.json');assert manifest['frozen']
    assert manifest['seeds']==SEEDS and manifest['all_expected_train_val_cells_verified']
    for path,digest in manifest['protocols'].items():
        assert sha(path)==digest;check_sources(read(path))
    for entry in list(manifest['entries'].values())+list(manifest['encoders'].values()):
        assert sha(entry['path'])==entry['sha256']
    assert sha(ROOT/'SOURCE_FAMILY_JOIN.json')==manifest['source_family_join_sha256']
    assert sha(ROOT/'DONOR_METADATA_PLAN.json')==manifest['donor_metadata_plan_sha256']
    return manifest


def require_ready():
    manifest=require_checkpoints();ready=read(ROOT/'EVALUATION_READY.json');assert ready['frozen']
    assert ready['final_checkpoint_manifest_sha256']==sha(ROOT/'FINAL_CHECKPOINT_MANIFEST.json')
    for path,digest in ready['mask_files'].items():assert sha(path)==digest
    assert ready['executor_protocol_sha256']==sha(ROOT/'EVALUATION_EXECUTOR_PROTOCOL.json')
    assert ready['actual_final_weight_VAL_bridge_sha256']==sha(ROOT/'FINAL_WEIGHT_VAL_BRIDGE.json')
    assert read(ROOT/'FINAL_WEIGHT_VAL_BRIDGE.json')['passed']
    return manifest,ready


if __name__=='__main__':build()
