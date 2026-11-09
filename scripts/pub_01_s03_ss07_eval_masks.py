"""Prediction-independent target/geometry masks AFTER checkpoint lock.

This performs NO neural inference. Real test/DTU target reads are forbidden
until require_checkpoints succeeds. All masks freeze BEFORE any predictions.
"""
import json
import time
import numpy as np
from scipy.io import loadmat
from pub_01_s03_ss07_baseline import ROOT,OLD,read,dump,sha
from pub_01_s03_ss07_final_manifest import require_checkpoints


def record_mask(row,target):
    n=row['common_samples'];target=np.asarray(target).reshape(-1);assert len(target)>=n
    count=(n-192)//96+1;weight=np.zeros(n);h=np.hanning(192)
    for i in range(count):weight[i*96:i*96+192]+=h
    covered=weight>1e-8;blocks=[]
    for s in range(0,n-3840+1,3840):
        keep=covered[s:s+3840];truth=target[s:s+3840][keep]
        reason='insufficient_overlap_support' if len(truth)<3800 else 'nonfinite_target' if not np.isfinite(truth).all() else 'constant_target' if np.std(truth)==0 else None
        blocks.append({'start':s,'samples':len(truth),'eligible':reason is None,'reason':reason})
    return {'record':row['record'],'common_samples':n,'windows':count,'blocks':blocks}


def main():
    require_checkpoints()  # MUST precede every target/signal/MAT read below.
    bridge=read(ROOT/'FINAL_WEIGHT_VAL_BRIDGE.json')
    assert bridge['passed'] and len(bridge['records'])==10
    assert bridge['checkpoint_manifest_sha256']==sha(ROOT/'FINAL_CHECKPOINT_MANIFEST.json')
    family={r['record']:r for r in read(ROOT/'SOURCE_FAMILY_JOIN.json')['records']}
    rows=read(OLD/'envelope_readout_seed1024/manifest_frozen.json')['records'];records=[];masks=[]
    for source in rows:
        if source['phase'] not in ('test_seen','test_heldout'):continue
        row=dict(source,source_family=family[source['record']]['source_family'])
        assert row['subject']==family[row['record']]['subject'] and row['story']==family[row['record']]['story']
        target=np.load(row['envelope_path'],mmap_mode='r').reshape(-1)
        masks.append(record_mask(row,target));records.append(row)
    assert len(records)==144 and sum(r['phase']=='test_seen' for r in records)==69
    dump(ROOT/'SPARR_TEST_MASK.json',{'frozen':True,'prediction_independent':True,
        'checkpoint_manifest_sha256':sha(ROOT/'FINAL_CHECKPOINT_MANIFEST.json'),
        'records':records,'record_masks':masks,
        'eligible_blocks':sum(b['eligible'] for m in masks for b in m['blocks']),
        'excluded_blocks':[dict(record=m['record'],**b) for m in masks for b in m['blocks'] if not b['eligible']],
        'test_predictions_computed':False})
    # Each intervention owns its target-key subset; no replacement donors.
    by_record={m['record']:m for m in masks};donors=read(ROOT/'DONOR_METADATA_PLAN.json')['records']
    pairs={}
    for kind in ['same_person_wrong_family','same_story_wrong_person']:
        keys=[]
        for row in donors:
            if row[kind]['eligible']:
                keys.extend([[row['target_record'],b['start']] for b in by_record[row['target_record']]['blocks'] if b['eligible']])
        pairs[kind]={'eligible_keys':keys,'records':sum(r[kind]['eligible'] for r in donors),
            'true_history_keys':keys,'must_pair_same_subset':True}
    dump(ROOT/'INTERVENTION_PAIR_MASK.json',{'frozen':True,'source_test_mask_sha256':sha(ROOT/'SPARR_TEST_MASK.json'),
        'donor_metadata_sha256':sha(ROOT/'DONOR_METADATA_PLAN.json'),'pairs':pairs,
        'other_interventions':'all fixed eligible test keys; same mask for corresponding true-history'})
    rng=np.random.default_rng(read(ROOT/'EVALUATION_RULES.json')['shuffle']['rng'])
    permutations={r['record']:rng.permutation(by_record[r['record']]['windows']).tolist()
        for r in sorted(records,key=lambda r:r['record'])}
    dump(ROOT/'SHUFFLE_METADATA_PLAN.json',{'frozen':True,'rng':20261102,
        'permutations':permutations,'same_permutation_all_seeds':True,
        'offline_distribution_intervention_future_observation_possible':True})
    protocol=read(ROOT/'DTU_PROTOCOL.json');dtu=ROOT/'external_dtu';trials=[]
    weight=np.zeros(3200);h=np.hanning(192)
    for s in protocol['windows']['starts']:weight[s:s+192]+=h
    covered=weight>1e-8;assert np.flatnonzero(covered).tolist()==list(range(1,3167))
    for person in range(1,19):
        data=loadmat(dtu/f'extracted/S{person}_data_preproc.mat',simplify_cells=True)['data']
        assert len(data['wavA'])==len(data['wavB'])==60
        for i,(a,b) in enumerate(zip(data['wavA'],data['wavB'])):
            assert a.shape==b.shape==(3200,)
            # Retention of all1080 was frozen. Any invalid target is a protocol
            # inconsistency requiring review, NEVER a selective new exclusion.
            assert np.isfinite(a[covered]).all() and np.std(a[covered])>0
            assert np.isfinite(b[covered]).all() and np.std(b[covered])>0
            trials.append({'subject':f'S{person}','trial':i+1,'samples':3166,'eligible':True})
    assert len(trials)==1080
    dump(ROOT/'DTU_TARGET_MASK.json',{'frozen':True,'prediction_independent':True,'trials':trials,
        'covered_half_open_range':[1,3167],'all1080_retained':True,'predictions_computed':False,
        'protocol_sha256':sha(ROOT/'DTU_PROTOCOL.json')})
    dump(ROOT/'EVALUATION_READY.json',{'frozen':True,'time':time.time(),
        'final_checkpoint_manifest_sha256':sha(ROOT/'FINAL_CHECKPOINT_MANIFEST.json'),
        'mask_files':{str(ROOT/name):sha(ROOT/name) for name in ['SPARR_TEST_MASK.json','INTERVENTION_PAIR_MASK.json','SHUFFLE_METADATA_PLAN.json','DTU_TARGET_MASK.json']},
        'executor_protocol_sha256':sha(ROOT/'EVALUATION_EXECUTOR_PROTOCOL.json'),
        'actual_final_weight_VAL_bridge_sha256':sha(ROOT/'FINAL_WEIGHT_VAL_BRIDGE.json'),
        'model_predictions_so_far':False,'all_masks_created_before_predictions':True})
    print(json.dumps({'ready':True,'Sparr_records':144,'DTU_trials':1080,'predictions':False}))


if __name__=='__main__':main()
