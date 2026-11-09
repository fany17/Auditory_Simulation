"""Isolated synthetic caller/aggregation fixture. NO real test/DTU reads.

One seed runs the complete caller with synthetic signals and small models;
five-seed aggregation uses explicitly cloned synthetic metric metadata.
This tests wiring/coverage, NOT five independent scientific experiments.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from pub_01_s03_ss07_baseline import ROOT,read,dump
from pub_01_s03_ss07_train import NativeHead
from pub_01_s03_ss07_incremental import NativeResidual
from pub_01_s03_ss07_eval_masks import record_mask
import pub_01_s03_ss07_eval_seed as actor
import pub_01_s03_ss07_eval_aggregate as aggregate


def main():
    torch.set_num_threads(2);device='cuda:0';torch.manual_seed(335)
    fixture=Path(tempfile.mkdtemp(prefix='synthetic_final_pipeline_',dir=ROOT/'engineering_history'))
    seeds=[1024,2026,31415,42,12345];primary='m2_h16'
    controls=['m2_current_h16','static_m2_h16','output_prior_h16']
    ids=['native_P','P_EMA','P_FT',primary]+controls
    exploratory=[f'm{i}_h16' for i in range(1,6)]
    comparisons=[('primary-native_P','native_P'),('primary-current',controls[0]),('primary-static',controls[1]),
        ('primary-output_prior',controls[2]),('primary-P_EMA','P_EMA'),('primary-P_FT','P_FT')]
    manifest={'seeds':seeds,'screening_seeds':seeds[:3],'main_model_ids':ids,'primary':primary,
        'exploratory_family_ids':exploratory,'EMA_alias':{'tau_seconds':0},
        'contrast_ids_and_comparators':comparisons,'training_costs':[
            {'stage':stage,'kind':'synthetic_only','seed':seeds[i%5],
             'training':{'epochs':1,'updates':820 if stage=='shared_pretrain' else 3945,'seconds':.01,
                'supervised_window_presentations':0 if stage=='shared_pretrain' else 252454,'teacher_presentations':0}}
            for stage,count in [('screen',54),('shared_pretrain',5),('native_head',5),('finetune',5),('full',32)]
            for i in range(count)],
        'entries':{f'{condition}:{seed}':{'config':{}} for seed in seeds for condition in set(ids+exploratory)}}
    dump(fixture/'FINAL_CHECKPOINT_MANIFEST.json',{'synthetic_fixture_only':True})
    dump(fixture/'EVALUATION_READY.json',{'synthetic_fixture_only':True})
    scalp=[f'c{i}' for i in range(64)]
    dump(fixture/'DTU_PROTOCOL.json',{'synthetic_fixture_only':True,'physical_scalp_names':scalp,
        'windows':{'starts':list(range(0,2977,96))}})
    target=np.sin(np.arange(3840)/12);rows=[];masks={};latents={}
    rng=np.random.default_rng(232)
    for i,phase in enumerate(['test_seen','test_heldout']):
        path=fixture/f'synthetic_target_{i}.npy'
        with path.open('xb') as f:np.save(f,target)
        row={'record':f'r{i}','phase':phase,'phase_offset':0,'common_samples':3840,
            'subject':f'p{i}','story':f's{i}','source_family':f'f{i}','envelope_path':str(path)}
        rows.append(row);masks[row['record']]=record_mask(row,target)
        latents[row['record']]=torch.from_numpy(rng.normal(size=(39,8,192)).astype('float32')).to(device)
        latents[f'd{i}']=torch.from_numpy(rng.normal(size=(39,8,192)).astype('float32')).to(device)
    donors=[]
    for i in range(2):
        donors.append({'target_record':f'r{i}','same_person_wrong_family':{'eligible':i==0,'donor_record':'d0'},
            'same_story_wrong_person':{'eligible':True,'donor_record':f'd{i}'}})
    dump(fixture/'DONOR_METADATA_PLAN.json',{'records':donors})
    dump(fixture/'SHUFFLE_METADATA_PLAN.json',{'permutations':{r['record']:list(range(38,-1,-1)) for r in rows}})
    dump(fixture/'INTERVENTION_PAIR_MASK.json',{'pairs':{
        'same_person_wrong_family':{'eligible_keys':[['r0',0]]},
        'same_story_wrong_person':{'eligible_keys':[['r0',0],['r1',0]]}}})
    marker=rng.normal(size=(32,192,64)).astype('float32');calls={'adapter':0,'P_encode':0,'FT_external':0,'FT_internal':0}
    digest={};last_models=[]

    class Models:
        def __init__(self,seed,dev):
            self.seed=seed;self.device=dev;self.manifest=manifest;self.ready={};self.loaded={};self.encoder=nn.Identity()
            last_models.append(self)
        def load(self,condition):
            assert condition!='P_EMA','EMA must alias exact native predictions'
            if condition not in self.loaded:
                model=NativeResidual('m2',16) if condition==primary else NativeHead()
                if condition==primary:
                    with torch.no_grad():model.residual.weight.normal_(0,.03)
                self.loaded[condition]=model.to(device).eval()
            return self.loaded[condition]
        def encode(self,raw):
            assert np.array_equal(raw,marker),'external input changed after unique adapter'
            calls['P_encode']+=1;return raw[:,:,:8].transpose(0,2,1).copy()
        def finetuned_windows(self,raw,global_offset=0):
            assert np.array_equal(raw,marker);calls['FT_external']+=1
            return np.cos(np.arange(192)[None]/11+np.arange(len(raw))[:,None]*.03).astype('float32')

    class Records:
        def __init__(self,models):self.models=models;self.rows=rows;self.masks=masks
        def encode_test(self):pass
        def neutral_train_mean(self):return torch.ones(48,device=device)*.2
        def latent(self,record):return latents[record]
        def finetuned_predictions(self):
            calls['FT_internal']+=1
            return {r['record']:np.cos(np.arange(192)[None]/11+np.arange(39)[:,None]*.03).astype('float32') for r in rows}

    dummy_eeg=np.zeros((3200,66));wa=np.sin(np.arange(3200)*.013);wb=np.cos(np.arange(3200)*.019)
    def fake_loadmat(path,**kwargs):
        assert Path(path).is_relative_to(fixture/'external_dtu/extracted')
        return {'data':{'eeg':[dummy_eeg]*60,'wavA':[wa]*60,'wavB':[wb]*60,
            'dim':{'chan':{'eeg':[np.array(scalp+['mastoid1','mastoid2'],dtype=object)]*60}}}}
    def adapter(eeg,labels,names):
        assert eeg.shape==(3200,66) and names==scalp
        calls['adapter']+=1;return marker,np.arange(0,2977,96),[0]*32
    def capture(path,value):
        assert Path(path).is_relative_to(fixture) and np.isfinite(value).all()
        stage,condition,name=path.parts[-3:];digest[(stage,condition,name)]=hashlib.sha256(value.tobytes()).hexdigest()
    with patch.object(actor,'ROOT',fixture),patch.object(actor,'require_ready',lambda:(manifest,{})),\
        patch.object(actor,'FinalModels',Models),patch.object(actor,'SparrRecords',Records),\
        patch.object(actor,'loadmat',fake_loadmat),patch.object(actor,'native_windows',adapter),\
        patch.object(actor,'save_array',capture):
        actor.main(1024,device)
    assert calls=={'adapter':1080,'P_encode':1080,'FT_external':1080,'FT_internal':1}
    normal_calls=dict(calls)
    assert 'P_EMA' not in last_models[0].loaded
    for (stage,condition,name),value in digest.items():
        if condition=='P_EMA':assert value==digest[(stage,'native_P',name)]
    assert any(value!=digest[(stage,'native_P',name)] for (stage,condition,name),value in digest.items() if condition=='P_FT')
    source=fixture/'final_results/seed1024';assert read(source/'COMPLETED.json')['all_conditions_complete']
    for seed in seeds[1:]:
        folder=fixture/f'final_results/seed{seed}';folder.mkdir()
        for path in source.glob('*.json'):
            data=read(path)
            if 'seed' in data:data['seed']=seed
            dump(folder/path.name,data)
    # Five-seed metadata clones are synthetic; they test matrix wiring only.
    with patch.object(aggregate,'ROOT',fixture),patch.object(aggregate,'require_ready',lambda:(manifest,{})):
        aggregate.main()
    report=read(fixture/'final_summary/INTERNAL_MAIN.json')
    assert report['M']==6 and report['contrasts'][0]['weighted_mean_difference']==report['contrasts'][4]['weighted_mean_difference']
    diagnostic=read(fixture/'final_summary/INTERNAL_HISTORY_DIAGNOSTICS.json')['diagnostics']
    one=next(r for r in diagnostic if r['intervention']=='same_person_wrong_family')
    assert one['coverage']['test_seen']['blocks']==1 and one['coverage']['test_heldout']['blocks']==0
    assert 'weighted_endpoint_unavailable_reason' in one
    external=read(fixture/'final_summary/EXTERNAL_DTU_MAIN.json');assert external['M']==6
    assert read(fixture/'final_summary/COMPLETED.json')['scientific_goal_achieved_claimed'] is False
    ledger=read(fixture/'final_summary/COSTS.json')['stage_totals']
    assert ledger['screen']['cells']==54 and ledger['shared_pretrain']['cells']==5
    failure_fixture=Path(tempfile.mkdtemp(prefix='synthetic_internal_failure_',dir=ROOT/'engineering_history'))
    for name in ['FINAL_CHECKPOINT_MANIFEST.json','EVALUATION_READY.json','DONOR_METADATA_PLAN.json','SHUFFLE_METADATA_PLAN.json','INTERVENTION_PAIR_MASK.json']:
        dump(failure_fixture/name,read(fixture/name))
    class BadRecords(Records):
        def finetuned_predictions(self):raise RuntimeError('synthetic technical I/O failure, not a low score')
    attempted=[]
    def no_external(*args,**kwargs):attempted.append(True);raise AssertionError('external must not open after internal technical failure')
    with patch.object(actor,'ROOT',failure_fixture),patch.object(actor,'require_ready',lambda:(manifest,{})),\
        patch.object(actor,'FinalModels',Models),patch.object(actor,'SparrRecords',BadRecords),\
        patch.object(actor,'loadmat',no_external),patch.object(actor,'native_windows',adapter),\
        patch.object(actor,'save_array',lambda path,value:None):
        try:actor.main(1024,device)
        except RuntimeError:pass
        else:raise AssertionError('internal technical failure did not stop seed')
    stopped=read(failure_fixture/'final_results/seed1024/COMPLETED.json')
    assert not stopped['all_conditions_complete'] and stopped['external_stage_opened'] is False and not attempted
    assert not list((failure_fixture/'final_results/seed1024').glob('EXTERNAL_*.json'))
    dump(ROOT/'FINAL_EVAL_PIPELINE_ENGINEERING_CHECK_v2.json',{'passed':True,'entirely_synthetic_fixture':str(fixture),
        'real_test_or_DTU_payload_read':False,'actual_model_scientific_performance_computed':False,
        'one_seed_full_caller_executed_with_synthetic_signals_small_models':True,
        'five_seed_aggregation_used_explicit_synthetic_metadata_clones':True,
        'calls':normal_calls,'native_EMA_tau0_waveform_digests_identical':True,
        'FT_distinct_prediction_source_used':True,'all1080_external_keys_each_condition':True,
        'different_donor_subset_true_keys_matched':True,'missing_stratum_not_silently_reweighted':True,
        'separate_main_M6_exploratory_M5_external_M6_families':True,
        'all54_screen_and5_shared_pretraining_costs_accounted_once':True,
        'internal_technical_failure_preserves_partial_and_blocks_external':True,
        'external_payload_calls_after_injected_internal_failure':len(attempted),
        'production_ready_gate_tested_separately_with_zero_premature_payload_reads':True})
    print(json.dumps({'PASS':True,'synthetic_only':True,'calls':calls}))


if __name__=='__main__':main()
