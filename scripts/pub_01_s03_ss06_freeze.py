"""One prospective formal protocol freeze after actual bounded pilot evidence."""
import json
import time
from pathlib import Path
from pub_01_s03_ss06_data import ROOT,LENGTH,Records,dump,read
from pub_01_s03_ss06_contract import digest
from m6a_public.continuity.execution import VARIANTS


def main():
    assert read(ROOT/'pilot_completed.json')['steps_each']==100
    assert read(ROOT/'preflight_final.json')['technical']=='PASS'
    assert read(ROOT/'adapter_verification.json')['technical']=='PASS'
    assert not (ROOT/'audit_opened.json').exists() and not (ROOT/'cache_audit').exists()
    train=Records('train');steps=sum(r['score_blocks'] for r in train.rows)
    pretrain_steps=sum(r['samples']//LENGTH for r in train.rows)
    assert steps==3262
    sources=sorted((ROOT/'input/src/m6a_public/continuity').glob('*.py'))
    sources+=sorted((ROOT/'input').glob('pub_01_s03_ss06_*.py'))
    sources+=[ROOT/'input/tests/test_ss06_execution.py']
    protocol={'task':'PUB-01-S03_SS06','run':'eeg_run01','freeze_epoch':time.time(),
        'scope':'prospective INTERNAL / EXPLORATORY source-family audit; all sources historically used in SS03/SS05 train',
        'source_base_commit':'55c87a65a3b90e7448ece2709cf1e038463b244e',
        'source_sha256':{str(p.relative_to(ROOT/'input')):digest(p) for p in sources},
        'split_sha256':digest(ROOT/'SPLIT_PROVENANCE.json'),
        'normalization_sha256':digest(ROOT/'newtrain_EEG_target_normalization.npz'),
        'variants':list(VARIANTS),'seeds':[1024,2026,31415],'optimizer_steps':steps,
        'budget_reason':'100-step pilots all finite, most validation curves rise; choose bounded ONE full pass through every eligible train record/time block, same sample order per seed across candidates; no test-based extension',
        'pretraining_steps':pretrain_steps,'pretraining':'all complete60s chunks across every newtrain record, fresh-state complete trajectory; EMA.99 strictly after update; no val/audit EEG; same formal seed checkpoint for frozen/finetune; no pilot reuse',
        'pretraining_selection':'fixed final step only, train-only; no performance-based SSL checkpoint search',
        'optimizer':'AdamW','learning_rate':.001,'weight_decay':.01,'gradient_clip':1.,
        'batch':1,'score_samples':3840,'warmup_samples':3840,'sampling_hz':64,'lag_samples':8,
        'lag_basis':'fixed prior train-validation-supported lag, no SS06 search; EEG[t+8] maps envelope[t]',
        'stream_amendment':'supersedes initial SPLIT_PROVENANCE.training_stream PILOT two-block limit for FORMAL only: complete all blocks of each record before changing record; same RNG(seed) record permutation; detach state after each60s update, retain values; no random noncontiguous spans',
        'objective':'1-Pearson(concatenated continuous60s)+.1*MSE of NEW train standardized target',
        'm4_weight_smoothness':.001,'m5_stopgrad_observation_MSE':.01,
        'prior_objective':'M5 autonomous prior trained by MSE only, selected by validation MSE; nearconstant Pearson undefined rather than0',
        'pearson_numeric_rule':'prediction or target std<=1e-6 normalized units -> null/undefined; no dropping nonconstant model records',
        'validation_every_steps':1000,'validation':'all980 complete60s val blocks; recordequal then personequal r; highest r earliest tie, no early stop; prior lowest MSE',
        'primary_endpoint':'all538 audit60s blocks after eachrecord60s warmup -> recordequal -> personequal, then average seeds',
        'history_endpoint':'FIRST scored60s perrecord only, equal60s true/donor prefix; same frozen weights; true/state_reset/state_wrong with SAME encoder cache, plus prefix_reset/prefix_wrong separate',
        'donor':'deterministic sorted-phase records; different record/person/source_family; no crossphase or target-based donor selection',
        'm2_periodic_reset':'same trained persistent weights, reset only postencoder state every192samples; encoder remains continuous; full-record score',
        'm1_controls':'four ACTIVE independent branches per candidate; multi1/8/64/256, short all1, long all256; same active parameter count/MAC, long shares maxspan; original idle-branch toggles not used',
        'm3_no_pretrain_alias':'gru is exactly M3 no_pretrain downstream baseline; no double-counted run',
        'output_controls':'local+causal EMA alpha[0,.9,.98,.995]; local+predicted-envelope AR lag[1,2,4,8,16,32], trainridge.001, corrective gain[.05,.2,.5,1]; choose by full validation only AFTER all training BEFORE audit; inference past predictions only, no true audio',
        'primary_contrasts':['m1_multi_minus_m1_long','m2_minus_gru','m2_minus_periodic_reset3s','m3_finetune_minus_gru',
            'm3_finetune_minus_m3_frozen','m4_adaptive_minus_m4_fixed','m5_full_minus_m5_eeg','m5_full_minus_local_ar'],
        'statistics':'seedmeans within record; subject macro;10000 subject-only and crossed independent subject/sourcefamily cluster bootstrap;8 Bonferroni primary intervals each99.375percent; secondary95percent descriptive;5 families limited',
        'precision':'float32, no mixed precision; TF32 disabled in FORMAL. Pilot default TF32 retained as exploratory evidence; final CPU/GPU parity checked separately',
        'fairness':'equal supervised target samples/order/steps/objective except autonomous prior; params/active grads/walltime/device and SSL extra samples disclosed; not equal wallclock or capacity across all frameworks',
        'audit_gate':'all39 downstream +3 formal SSL complete; source/split/norm/checkpoint selection/content/order independently checked; output controls selected, then atomic write-once audit opening; one assessment per selected model with all frozen conditions',
        'stop':'valid positive/null/negative results all deliver; no new architecture/budget/search after audit, no SS04/S04/parentS05/artifact/merge'}
    dump(ROOT/'FORMAL_PROTOCOL.json',protocol)
    dump(ROOT/'freeze_review.json',{'technical':'PASS','audit_payload_not_opened':True,
        'independent_coordinator_source_review':'formal coverage, mode isolation, identities, gating and CUDA parity findings corrected before freeze',
        'same_agent_separate_freeze_review':True,'formal_steps':steps,'SSL_steps':pretrain_steps})
    print(json.dumps({'formal_frozen':True,'steps':steps,'SSL_steps':pretrain_steps,'variants':len(VARIANTS),'seeds':3}),flush=True)


if __name__=='__main__':main()
