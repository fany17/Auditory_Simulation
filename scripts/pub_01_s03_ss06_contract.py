"""Fail-closed SS06 formal identity and all-training-before-audit gates."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_continuity_20261023/eeg_run01')


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities():
    return {'protocol_sha256':digest(ROOT/'FORMAL_PROTOCOL.json'),
        'split_sha256':digest(ROOT/'SPLIT_PROVENANCE.json'),
        'normalization_sha256':digest(ROOT/'newtrain_EEG_target_normalization.npz')}


def verify_contract(stage=None,seed=None,steps=None,variant=None):
    protocol=read(ROOT/'FORMAL_PROTOCOL.json')
    for path,value in protocol['source_sha256'].items():assert digest(ROOT/'input'/path)==value,'formal source changed: '+path
    assert digest(ROOT/'SPLIT_PROVENANCE.json')==protocol['split_sha256']
    assert digest(ROOT/'newtrain_EEG_target_normalization.npz')==protocol['normalization_sha256']
    if stage is not None:
        assert not (ROOT/'audit_opened.json').exists(),'no fitting after audit opened'
        assert seed in protocol['seeds']
        assert steps==protocol['pretraining_steps' if stage=='pretrain' else 'optimizer_steps']
        if variant is not None:assert variant in protocol['variants']
    return protocol


def check_identity(record):
    for key,value in identities().items():assert record[key]==value,key+' mismatch'


def verify_pretrained(seed,path):
    protocol=verify_contract();expected=ROOT/'formal01'/f'pretrain_seed{seed}'/'pretrained.pt'
    assert Path(path).resolve()==expected.resolve(),'pilot/other-seed weights forbidden'
    meta=read(expected.parent/'pretrain_completed.json');check_identity(meta)
    assert meta['seed']==seed and meta['steps']==protocol['pretraining_steps'] and meta['stage']=='formal'
    assert meta['checkpoint_sha256']==digest(expected)
    assert set(meta['actual_records'])==set(meta['eligible_records'])
    assert meta['actual_spans']==protocol['pretraining_steps']
    return {'seed':seed,'steps':meta['steps'],'checkpoint_sha256':meta['checkpoint_sha256'],'path':str(expected)}


def verify_all_training(allow_audit=False):
    protocol=verify_contract();checked=[]
    manifest=read(ROOT/'cache_trainval/manifest.json');train=[r for r in manifest['records'] if r['phase']=='train' and r['score_blocks']]
    expected={r['record']:r['score_blocks'] for r in train}
    from collections import Counter
    from numpy.random import default_rng
    import torch
    for seed in protocol['seeds']:
        pre=verify_pretrained(seed,ROOT/'formal01'/f'pretrain_seed{seed}'/'pretrained.pt')
        order=default_rng(seed).permutation(len(train));sequence=[train[i]['record'] for i in order for _ in range(train[i]['score_blocks'])]
        for variant in protocol['variants']:
            folder=ROOT/'formal01'/f'{variant}_seed{seed}'
            completed=read(folder/'train_completed.json');check_identity(completed)
            assert completed['variant']==variant and completed['seed']==seed
            assert completed['steps']==protocol['optimizer_steps'] and Counter(completed['training_record_sequence'])==Counter(expected)
            assert completed['training_record_sequence']==sequence,'different record/block training sequence'
            assert completed['checkpoint_sha256']==digest(folder/'best_validation.pt')
            history=read(folder/'history.json');key=lambda r:-r['validation']['mse'] if variant=='m5_prior' else r['validation']['r']
            best=max(history,key=key)
            assert completed['selected_step']==best['step'] and completed['selected_score']==key(best)
            assert history[-1]['step']==protocol['optimizer_steps']
            checkpoint=torch.load(folder/'best_validation.pt',map_location='cpu',weights_only=True)
            assert checkpoint['step']==best['step'] and checkpoint['variant']==variant and checkpoint['seed']==seed
            assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
            if not allow_audit:assert not (folder/'audit01').exists(),'audit assessment already exists before all-training gate'
            if variant in ('m3_frozen','m3_finetune'):assert completed['pretrained_identity']==pre
            checked.append({'variant':variant,'seed':seed,'selected_step':best['step'],'checkpoint_sha256':completed['checkpoint_sha256']})
    return {'technical':'PASS','checked_runs':checked,'formal_pretrain_seeds':protocol['seeds'],
        'full_record_coverage_equal_order':True,'all_training_precedes_audit':True,**identities()}
