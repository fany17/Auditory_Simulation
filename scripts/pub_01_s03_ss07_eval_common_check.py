"""Real VAL numerical checks + synthetic pairing/gate checks, no test reads."""
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch import nn
from pub_01_s03_ss07_baseline import ROOT,OLD,read,dump,native,cfg
from pub_01_s03_ss07_train import NativeHead,LatentData,macro_validation,make
from pub_01_s03_ss07_history import intervene
from pub_01_s03_ss07_eval_common import predict_windows,predict_raw_phase,intervene_aligned,score_record,paired_arrays,descriptive_family_slices
from pub_01_s03_ss07_final_manifest import checkpoint,history_check,build
from pub_01_s03_ss07_finetune import RawWindows,Decoder
from pub_01_s03_ss07_incremental import NativeResidual
from pub_01_s03_ss07_eval_masks import record_mask


def rejected(call):
    try:call()
    except (AssertionError,RuntimeError):return
    raise AssertionError('invalid input accepted')


def main():
    torch.set_num_threads(2);device='cuda:0'
    # Native completed seed1024 only; VAL remains legal for engineering.
    data=LatentData(1024,'val',device);model=NativeHead().to(device).eval()
    model.head.load_state_dict(torch.load(ROOT/'baseline/head_seed1024/best_validation.pt',map_location=device,weights_only=True)['head'])
    summary,old,people=macro_validation(model,data);new=[]
    for row in data.rows:
        first=row['offset'];last=first+row['windows'];z=data.x[first:last]
        windows=predict_windows(model,z,row['record'],global_offset=row['offset'])
        mask={'record':row['record'],'blocks':[b for b in data.fixed_mask['blocks'] if b['record']==row['record']]}
        target=np.load(row['envelope_path'],mmap_mode='r')
        result,_=score_record(windows,target,dict(row,source_family='engineering_only'),mask)
        new.extend(result)
    assert [(r['record'],r['start']) for r in old]==[(r['record'],r['start']) for r in new]
    difference=max(abs(a['r']-b['r']) for a,b in zip(old,new));assert difference==0
    # An ACTUAL fitted residual, not zero initialization, reproduces exactly.
    folder=ROOT/'screen/m2_h16_seed1024';c=read(folder/'completed.json')['condition']
    residual=make(c,1024).to(device).eval()
    residual.load_state_dict(torch.load(folder/'best_validation.pt',map_location=device,weights_only=True)['model'])
    assert torch.count_nonzero(residual.residual.weight)>0
    first=data.rows[1];z=data.x[first['offset']:first['offset']+first['windows']];saved=z.clone()
    assert first['offset']%64!=0
    expected=predict_windows(residual,z,first['record'],global_offset=first['offset'])
    actual=intervene_aligned(residual,z,first['record'],global_offset=first['offset'])
    history_error=float(np.max(np.abs(expected-actual)));assert history_error<1e-6
    wrong=intervene_aligned(residual,z,first['record'],'wrong_history',global_offset=first['offset'],donor_observations=torch.zeros(len(z),48,device=device))
    assert torch.equal(z,saved) and not np.array_equal(expected,wrong)
    # All six residual forms, nonzero projections, an odd first chunk.
    torch.manual_seed(9823);synthetic=torch.randn(137,8,192,device=device);aligned_errors={}
    for family in ['m1','m2','m3','m4','m5','output_prior']:
        candidate=NativeResidual(family,16).to(device).eval()
        with torch.no_grad():candidate.residual.weight.normal_(0,.1);candidate.residual.bias.normal_(0,.1)
        expected=predict_windows(candidate,synthetic,'synthetic',global_offset=63)
        actual=intervene_aligned(candidate,synthetic,'synthetic',global_offset=63)
        error=float(np.max(np.abs(expected-actual)));assert error<1e-6;aligned_errors[family]=error
    # FT preserves whole GLOBAL64 raw batches across record boundaries.
    # Frozen real baseline weights emulate incumbent, no active FT file read.
    import sys
    sys.path.insert(0,str(OLD/'input'))
    original=native().TransformerCLIP(cfg())
    original.load_state_dict(torch.load(ROOT/'baseline/pretrain_seed1024/best_validation.pt',map_location='cpu',weights_only=True)['model'])
    decoder=Decoder(original.eeg).to(device).eval()
    decoder.head.load_state_dict(torch.load(ROOT/'baseline/head_seed1024/best_validation.pt',map_location=device,weights_only=True)['head'])
    raw=RawWindows('val');limit=sum(r['windows'] for r in raw.rows[:2]);reference=[]
    with torch.no_grad():
        for first in range(0,limit,64):
            xx,_=raw.batch(first,min(first+64,limit),device);reference.append(decoder(xx)[:,0].cpu().numpy())
    records=((r['record'],raw.batch(r['offset'],r['offset']+r['windows'],'cpu')[0].numpy()) for r in raw.rows[:2])
    streamed=predict_raw_phase(decoder,records,device)
    joined=np.concatenate([streamed[r['record']] for r in raw.rows[:2]])
    raw_error=float(np.max(np.abs(np.concatenate(reference)-joined)));assert raw_error==0
    truth=np.sin(np.arange(8000)/11);mask=record_mask({'record':'synthetic','common_samples':8000},truth)
    assert len(mask['blocks'])==2 and all(b['eligible'] for b in mask['blocks'])
    truth[:3840]=2;mask=record_mask({'record':'synthetic','common_samples':8000},truth)
    assert not mask['blocks'][0]['eligible'] and mask['blocks'][0]['reason']=='constant_target'
    assert mask['blocks'][1]['eligible']
    # Different donor coverage must select SAME keys from true and intervention.
    base=[{'record':f'r{i}','start':0,'phase':'test_seen' if i<2 else 'test_heldout',
        'subject':f'p{i}','story':f's{i%2}','source_family':f'f{i%2}','samples':3840,'r':v}
        for i,v in enumerate([.1,.9,.2,.8])]
    subset=[('r0',0),('r2',0)];results={}
    for seed in [1,2]:
        results[('true',seed)]=copy.deepcopy(base)
        results[('wrong',seed)]=[dict(r,r=r['r']-.05) for r in base if (r['record'],r['start']) in subset]
    values,blocks=paired_arrays(results,['true','wrong'],[1,2],subset)
    assert values.shape==(2,2,2) and np.allclose(values[0]-values[1],.05)
    rejected(lambda:paired_arrays(results,['true','wrong'],[1,2]))
    bad=copy.deepcopy(results);bad[('wrong',2)]=bad[('wrong',2)][:1]
    rejected(lambda:paired_arrays(bad,['true','wrong'],[1,2],subset))
    bad=copy.deepcopy(results);bad[('wrong',2)][0]['subject']='mismatched_person'
    rejected(lambda:paired_arrays(bad,['true','wrong'],[1,2],subset))
    fixture=Path(tempfile.mkdtemp(prefix='final_manifest_',dir=ROOT/'engineering_history'))
    linear=nn.Linear(3,2);path=fixture/'synthetic.pt'
    torch.save({'seed':5,'epoch':1,'model':linear.state_dict()},path)
    checkpoint(path,nn.Linear(3,2),'model',5,1)
    rejected(lambda:checkpoint(path,nn.Linear(3,2),'model',6,1))
    rejected(lambda:checkpoint(path,nn.Linear(3,2),'model',5,2))
    dump(fixture/'history.json',[{'epoch':i,'updates':(i+1)*3945,'loss':v} for i,v in enumerate([.2,.1,.1,.3])])
    completed={'epochs':4,'updates':4*3945,'selected_epoch':1}
    assert history_check(fixture,completed,3945,'loss')==.1
    rejected(lambda:history_check(fixture,dict(completed,selected_epoch=2),3945,'loss'))
    assert history_check(fixture,dict(completed,selected_epoch=-1),3945,'loss',.1)==.1
    # No checkpoint/target access is possible from a premature builder call.
    assert not (ROOT/'FULL_SELECTION.json').exists()
    assert not (ROOT/'FINAL_CHECKPOINT_MANIFEST.json').exists()
    calls=[];old_load=torch.load;old_np=np.load
    def forbidden(*args,**kwargs):calls.append('payload_load');raise AssertionError('premature payload read')
    try:
        torch.load=forbidden;np.load=forbidden
        try:build()
        except FileNotFoundError:pass
        else:raise AssertionError('premature final manifest built')
    finally:torch.load=old_load;np.load=old_np
    assert not calls and not (ROOT/'FINAL_CHECKPOINT_MANIFEST.json').exists()
    dump(ROOT/'FINAL_EVAL_COMMON_ENGINEERING_CHECK_v2.json',{'passed':True,'test_payload_or_scores':False,
        'actual_frozen_native_fullval49people752blocks_r_max_difference':difference,
        'actual_nonzero_fitted_M2_val_predict_vs_true_history_max_difference':history_error,
        'six_family_nonzero_aligned_chunk_identity_errors':aligned_errors,
        'FT_incumbent_real_val_two_record_global64_streaming_max_difference':raw_error,
        'prediction_independent_geometry_and_constant_target_masks_verified':True,
        'current_val_z_unchanged_by_wrong_history':True,'donor_subset_same_keys_verified':True,
        'mismatched_keys_or_person_rejected':True,'checkpoint_seed_epoch_mismatch_rejected':True,
        'earliest_checkpoint_tie_and_incumbent_verified':True,
        'premature_manifest_call_payload_loads':len(calls),'no_incomplete_scientific_manifest_created':True})
    print(json.dumps({'PASS':True,'fullval_metric_diff':difference,'fitted_history_diff':history_error}))


if __name__=='__main__':main()
