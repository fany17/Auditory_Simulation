"""Freeze the authorized pilot-verified run02 protocol; no audit payload."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pub_01_s03_ss06_matched import ROOT,OLD,read,dump,sha,SEEDS,Inputs,schedule


def freeze():
    assert read(ROOT/'ENGINEERING_CHECK.json')['passed']
    assert 'ENGINEERING_PIPELINE_COMPLETED' in (ROOT/'engineering.log').read_text()
    for name in ('C','O','P_pretrain','P_head'):assert (ROOT/f'pilot01/{name}_seed1024/completed.json').exists()
    intake=read(ROOT/'INTAKE.json')
    original=OLD/'input';upstream=[p for p in original.rglob('*.py') if '__pycache__' not in str(p)]
    protocol={'frozen':True,'created_epoch':time.time(),'seeds':list(SEEDS),'task':'PUB-01-S03_SS06',
        'kind':'POST-HOC INTERNAL EXPLORATORY; audit already viewed in run01',
        'coordinator_authorization':'2026-10-26 C/O/P matrix accepted; after required tests and ledger freeze/start3seeds without waiting another reply',
        'source_sha256':{p.name:sha(p) for p in (ROOT/'input').glob('*.py')},
        'upstream_source_sha256':{str(p):sha(p) for p in upstream},
        'split_sha256':intake['split_sha256'],'normalization_contract_sha256':intake['normalization_contract_sha256'],
        'feature_manifest_sha256':intake['feature_manifest_sha256'],
        'C_O':{'steps':3262,'target_samples_per_step':3840,'total_target_samples_per_seed':12526080,
            'record_order':'numpy.default_rng(seed).permutation(run01 train.rows); every chronological score block once',
            'optimizer':'AdamW','lr':.001,'weight_decay':.01,'gradient_clip_norm':1.,'loss':'unchanged run01 dense_objective(1-r+.1MSE)',
            'validation_steps':[1000,2000,3000,3262],'selection':'highest full val record→person macro r; earliest tie',
            'validation_records':63,'validation_blocks':980,'warmup_samples_per_record':3840,
            'precision':'float32; TF32 disabled; no mixed precision','normalizer':'exact run01 newtrain-only stats',
            'old_EEG_audio_alignment_or_head_checkpoints':'FORBIDDEN; fresh parameters for every formal condition/seed'},
        'C':{'label':'strong causal adaptation','parameters_including_head':3128137,
            'attention_window_samples_including_current':192,'per_layer_KV_past':191,'per_layer_conv_past':63,'head_past':31,
            'position':'no positional encoding, as native SS02; sample-relative sliding mask; integer position for cache diagnostics',
            'normalization':'pointwise channel LayerNorm; removes244480 time-affine parameters from native',
            'microchunk':192,'internal_detach':False,'between_steps':'detach caches, preserve values across60s; reset record only',
            'future_permission':'EEG raw coordinate <= target+8samples (125ms) at every prediction'},
        'O':{'label':'native architecture, direct supervised offline control','parameters_including_head':3372617,
            'windows_per60s':39,'window_samples':192,'hop':96,'starts_within_block':list(range(0,3649,96)),
            'reconstruction':'nonperiodic Hann; first/last exactly-zero endpoint gets sole window prediction; all3840 target points scored',
            'block_edges':'native windows wholly within each common60s block, no cross-block overlap windows; same for P audit',
            'gradient':'all39 predictions overlap-add, single60s loss/step; whole3s timeLN/sameConv/unmasked attention retained',
            'EEG_sample_window_presentations_excluding_warmup':24425856,
            'future_permission':'offline whole3s window; never claim same temporal permission as C or pure latency effect'},
        'P':{'label':'complete two-stage functional workflow at reduced one-sweep budget; not fully trained SS02/paper ceiling',
            'pretrain':intake['P_pretrain'],'pretrain_loss':'native symmetric InfoNCE EEG↔SpeechLSTM(existing XLSR19)',
            'pretrain_validation_steps':[159,319,479,639],'selection':'minimum fixed newval symmetric InfoNCE; earliest tie',
            'validation_schedule_seed':20261026,'batch_rule':'same story/distinct timestamps; deterministic exact coverage; declared repeat-anchor fallback, actual repeats0',
            'pretrain_optimizer':'AdamW lr.001 wd.01 clip1; original StepLR10epochs has no decay within1sweep',
            'head':intake['P_head'],'head_validation_steps':[508,1017,1526,2035],
            'head_loss_selection':'negative3s Pearson, mean over all fixed newval overlapping windows; earliest tie',
            'head_optimizer':'AdamW lr.001 wd.01 clip1; frozen this seed newtrain-pretrained encoder',
            'head_training_grid':'each record starts3840; hop96 through end of last complete60s block minus192',
            'head_training_order':'numpy.default_rng(seed).permutation(concatenated train windows), batches64, last55',
            'inference':'same O39window/Hann reconstruction, EEG only; no audio/candidates; MSE uncalibrated descriptive',
            'budget_warning':'P adds contrastive pretraining and overlapped supervision, not matched total training budget; M3run01 extra EEG SSL also disclosed'},
        'assessment':{'after_all_training_and_validation_selection':True,'audit_records':41,'audit_blocks':538,
            'aggregate':'same per60s Pearson/MSE → record → person; equal seed average',
            'primary_this_round_exploratory':['C-m1_multi','C-m2','C-m3_finetune','C-m4_adaptive','C-m5_full','C-gru'],
            'bootstrap':'10000 crossed subject/family weights, same seed-averaging and macro weighting as run01;6 Bonferroni comparisons',
            'reference_conditions':'O/P descriptive, not pure component ablations','no_audit_guided_extension':True},
        'pilot_weights_reused':False,'audit_signal_read_in_run02_before_freeze':False,'patient_data':'FORBIDDEN'}
    dump(ROOT/'FORMAL_PROTOCOL.json',protocol);print(json.dumps({'frozen':True,'source_files':len(protocol['source_sha256']),'upstream_files':len(upstream)}),flush=True)


def review():
    train=Inputs('train');rows=[]
    for seed in SEEDS:
        expected=[(train.rows[i]['record'],b) for i in __import__('numpy').random.default_rng(seed).permutation(len(train.rows)) for b in range(train.rows[i]['score_blocks'])]
        for name in ('C','O','P_pretrain','P_head'):
            folder=ROOT/f'formal01/{name}_seed{seed}';d=read(folder/'completed.json');assert (folder/'best_validation.pt').is_file()
            if name in ('C','O'):
                assert d['steps']==3262 and d['targets']==12526080
                assert [(a['record'],a['block']) for a in d['actual_sequence']]==expected
                assert [h['step'] for h in read(folder/'history.json')]==[1000,2000,3000,3262]
            elif name=='P_pretrain':
                actual=[(a['record'],a['start']) for a in d['actual_sequence'] if not a['repeat_anchor']]
                assert d['steps']==639 and len(actual)==len(set(actual))==65240
                assert d['EEG_and_audio_sample_presentations']==12526080
            else:
                assert d['steps']==2035 and d['unique_window_indices']==130231 and d['target_sample_presentations']==25004352
                assert d['cache_windows']=={'train':130231,'validation':39137}
            assert d['audit_read'] is False
            rows.append({'condition':name,'seed':seed,'selected_step':d['selected_step'],'checkpoint_sha256':sha(folder/'best_validation.pt')})
    dump(ROOT/'ALL_TRAINING_REVIEW.json',{'passed':True,'rows':rows,'all_conditions_completed':True,
        'exact_matched_record_block_order_verified':True,'P_actual_coverage_and_budget_verified':True,
        'audit_payload_read':False,'review_kind':'executor separate post-training technical review; coordinator review also requested'})
    print(json.dumps({'ALL_TRAINING_REVIEW':'PASS','conditions':len(rows)}),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:review()
