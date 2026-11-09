"""Freeze reviewed inference/statistics RULES and metadata-only donor join.

No raw signal, target, model checkpoint or test-score reads. A later separate
checkpoint/mask child is mandatory before any evaluation output is produced.
"""
import hashlib
import json
from pathlib import Path
import csv
import time

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_continuity_20261101')
OLD=ROOT.parent/'s03_strong_baselines_20261015/transformer_available01'
SPLIT=ROOT.parent/'s02_acoustic_baseline_20260924/split_manifest.csv'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,indent=2)


def main():
    reference=OLD/'envelope_readout_seed1024/manifest_frozen.json'
    native=json.loads(reference.read_text())['records'];split=list(csv.DictReader(SPLIT.open()))
    mapping={r['source_recording']:r for r in split};assert len(mapping)==len(split)==666
    records=[]
    for r in native:
        source=mapping[r['record']];assert source['subject']==r['subject'] and source['raw_story']==r['story']
        family=source['source_family'];assert family and family!='UNKNOWN'
        records.append({k:r[k] for k in ['record','subject','story','phase','common_samples']}|
            {'source_family':family,'windows':(r['common_samples']-192)//96+1,
            'family_mapping_basis':source['family_mapping_basis']})
    assert len(records)==619
    dump('SOURCE_FAMILY_JOIN.json',{'metadata_only':True,'records':records,
        'family_is_shared_stimulus_not_subject_recording':True,
        'sources':{str(reference):sha(reference),str(SPLIT):sha(SPLIT)}})
    donors=[]
    for target in records:
        if target['phase'] not in ('test_seen','test_heldout'):continue
        record={'target_record':target['record'],'target_phase':target['phase'],'target_windows':target['windows'],
            'geometry_60s_block_starts':list(range(0,target['common_samples']-3840+1,3840))}
        for kind in ['same_person_wrong_family','same_story_wrong_person']:
            candidates=[r for r in records if r['windows']>=target['windows'] and
                ((r['subject']==target['subject'] and r['source_family']!=target['source_family'])
                if kind=='same_person_wrong_family' else (r['story']==target['story'] and r['subject']!=target['subject']))]
            candidates.sort(key=lambda r:(r['windows']-target['windows'],r['record']))
            donor=candidates[0] if candidates else None
            record[kind]={'eligible':donor is not None,'donor_record':donor['record'] if donor else None,
                'donor_phase':donor['phase'] if donor else None,'candidate_count':len(candidates),
                'reason':None if donor else 'no_metadata_eligible_full_length_donor_no_fallback'}
        donors.append(record)
    assert len(donors)==144
    dump('DONOR_METADATA_PLAN.json',{'frozen':True,'prediction_or_target_read':False,'records':donors,
        'pool':'original619 public records; cross-phase donors OFFLINE distribution intervention only',
        'selection':'full target window length required, minimum excess length then lexical record; no fallback',
        'true_and_intervention_must_share_final_target_mask_keys':True,
        'eligible_block_keys':'geometry starts shown; target/coverage eligibility frozen later BEFORE predictions',
        'source_family_join_sha256':sha(ROOT/'SOURCE_FAMILY_JOIN.json')})
    source_names=['statistics','statistics_check','history','history_check','freeze_eval_rules']
    sources={str(ROOT/f'input/pub_01_s03_ss07_{n}.py'):sha(ROOT/f'input/pub_01_s03_ss07_{n}.py') for n in source_names}
    rules={'frozen':True,'time':time.time(),'sources':sources,
        'review':'coordinator accepted rules with5 clarifications before new test results',
        'exposure':'ALL existing test POST-HOC INTERNAL EXPLORATORY; no independent confirmation',
        'primary_endpoint':'mean r over blocks per person, equal people within group, 2/3 seen+1/3 heldout, mean paired5seeds',
        'primary_contrast_ids':['primary-native_P','primary-current','primary-static','primary-output_prior','primary-P_EMA','primary-P_FT'],
        'additional_contrast_rule':{'m1':None,'m2':None,'m3':'primary-current_teacher','m4':'primary-static_alignment','m5':'primary-prior_only'},
        'multiplicity':'M6/7 applies ONLY original weighted endpoint; Bonferroni-adjusted APPROXIMATE bootstrap intervals; no finite-sample/confirmatory guarantee',
        'supporting_endpoints':'seen/heldout95% exploratory intervals, not covered by weighted M; no joint3M claim',
        'bootstrap':{'draws':20000,'rng':20261101,'method':'CROSSED seed/person, one seed vector and one person vector per stratum shared across ALL sampled seeds/models/contrasts'},
        'family_sensitivity':{'draws':20000,'rng':20261103,'method':'positive Exp1 person and shared stimulus-family weights; not calibrated frequentist CI','LOFO_and_per_family':'descriptive with exact eligible people/blocks; no silent coverage change'},
        'family3seed_exploration':'separate five-contrast M5 family, no jointFWER/independent replication claim with main5seed family',
        'shuffle':{'rng':20261102,'record_order':'lexical','offline_only':True,'future_observation_possible':True},
        'older_history':'ctx_before(i-1), last source i-3; both recency and parity change; audit actual Hann support, no pure-delay causal claim',
        'content_blind':'same update count and elapsed time, train-only neutral mean, distinct from reset',
        'donor_plan_sha256':sha(ROOT/'DONOR_METADATA_PLAN.json'),'source_family_join_sha256':sha(ROOT/'SOURCE_FAMILY_JOIN.json'),
        'required_later_child':'immutable checkpoint/selection/mask/source manifest AFTER all main5seed/controls/nativeFT/EMA complete and BEFORE any prediction',
        'test_read_or_inference_authorized_by_this_file':False,'external_DTU':'separate adaptation/endpoint/split protocol, never pooled with existing tests'}
    dump('EVALUATION_RULES.json',rules)
    print(json.dumps({'frozen':'rules and metadata only','test_read':False,'donor_targets':len(donors),
        'source_families':sorted({r['source_family'] for r in records}),
        'eligible_donors':{kind:sum(r[kind]['eligible'] for r in donors) for kind in ['same_person_wrong_family','same_story_wrong_person']}}))


if __name__=='__main__':main()
