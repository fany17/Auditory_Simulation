"""Actual FINAL weights on legal VAL before any test/DTU predictions."""
import json
import time
import numpy as np
import torch
from pub_01_s03_ss07_baseline import ROOT,read,dump,sha
from pub_01_s03_ss07_final_manifest import require_checkpoints
from pub_01_s03_ss07_train import LatentData,NativeHead,make
from pub_01_s03_ss07_eval_common import predict_windows,intervene_aligned,fuse


def main(device='cuda:0'):
    manifest=require_checkpoints();torch.set_num_threads(2);torch.cuda.set_device(device)
    precision=read(ROOT/'INCREMENTAL_PROTOCOL.json')['precision']
    torch.backends.cuda.matmul.allow_tf32=precision['TF32_matmul'];torch.backends.cudnn.allow_tf32=precision['TF32_cudnn']
    records=[];errors=[]
    for seed in manifest['seeds']:
        try:
            spec=manifest['entries'][f'{manifest["primary"]}:{seed}']
            model=make(spec['config'],seed).to(device).eval()
            saved=torch.load(spec['path'],map_location=device,weights_only=True)
            assert saved['seed']==seed;model.load_state_dict(saved['model'],strict=True)
            p=manifest['entries'][f'native_P:{seed}'];head=NativeHead().to(device).eval()
            head.head.load_state_dict(torch.load(p['path'],map_location=device,weights_only=True)['head'])
            data=LatentData(seed,'val',device)
            chosen=[data.rows[0],next(r for r in data.rows if r['offset']%64!=0)]
            assert len({r['record'] for r in chosen})==2
            for row in chosen:
                z=data.x[row['offset']:row['offset']+row['windows']];original=z.clone()
                normal=predict_windows(model,z,row['record'],global_offset=row['offset'])
                bridge=intervene_aligned(model,z,row['record'],global_offset=row['offset'])
                difference=float(np.max(np.abs(normal-bridge)))
                assert difference<1e-6 and torch.equal(z,original)
                native=predict_windows(head,z,row['record'],global_offset=row['offset'])
                direct,covered=fuse(native,row['common_samples'],0)
                ema_zero,keep=fuse(native,row['common_samples'],0)
                assert np.array_equal(direct,ema_zero) and np.array_equal(covered,keep)
                records.append({'seed':seed,'record':row['record'],'phase':'val','global_offset':row['offset'],
                    'primary_checkpoint_sha256':spec['sha256'],'normal_true_bridge_max_abs':difference,
                    'prespecified_tolerance':1e-6,'native_EMA_tau0_exact_identity':True,'current_z_unchanged':True})
        except Exception as exc:errors.append({'seed':seed,'error':repr(exc)})
    passed=not errors and len(records)==10
    dump(ROOT/'FINAL_WEIGHT_VAL_BRIDGE.json',{'frozen':True,'passed':passed,'time':time.time(),
        'checkpoint_manifest_sha256':sha(ROOT/'FINAL_CHECKPOINT_MANIFEST.json'),
        'actual_final_primary_five_seeds':manifest['seeds'],'records':records,'errors':errors,
        'no_test_or_DTU_inputs_or_scores':True,'model_selection_or_threshold_changes':False})
    assert passed,'actual final-weight VAL bridge failed; preserve and review before test'
    print(json.dumps({'final_weight_VAL_bridge':'PASS','seeds':5,'records':10,'test_read':False}))


if __name__=='__main__':main()
