"""Actual train/validation indexing, stream and CPU/GPU checks before freeze."""
import json
import time
import numpy as np
import torch
from m6a_public.continuity.execution import build_variant
from m6a_public.continuity.common import detach_state
from pub_01_s03_ss06_data import ROOT,PUBLIC,LENGTH,Records,oriented,dump,read
from pub_01_s03_ss06_train import tensor,metrics


def main():
    torch.set_num_threads(1);torch.manual_seed(1024)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    split=read(ROOT/'SPLIT_PROVENANCE.json');norm=read(ROOT/'normalization_contract.json')
    train_ids={r['record'] for r in split['records'] if r['phase']=='train'}
    assert set(norm['fit_records'])==train_ids
    assert all(r['old_SS03_phase']==r['old_SS05_phase']=='train' for r in split['records'])
    scale=np.load(ROOT/'newtrain_EEG_target_normalization.npz');index_checks=[]
    for phase in ('train','validation'):
        data=Records(phase);r=data.rows[0];x,y=data.arrays(r);raw=oriented(r)
        env=np.load(r['envelope_path'],mmap_mode='r').reshape(-1)
        xx=(np.asarray(raw[8:1032],np.float64)-scale['mean'][:64])/scale['std'][:64]
        yy=(np.asarray(env[:1024],np.float64)-scale['mean'][-1])/scale['std'][-1]
        errorx=float(np.max(np.abs(x[:1024]-xx)));errory=float(np.max(np.abs(y[:1024,0]-yy)))
        assert errorx<2e-6 and errory<2e-6
        index_checks.append({'phase':phase,'record':r['record'],'EEG_normalized_error':errorx,'target_normalized_error':errory})
    r=Records('train').rows[0];x,y=Records('train').arrays(r);xx=tensor(x[:LENGTH],'cpu');yy=tensor(y[:LENGTH],'cpu')
    comparisons=[]
    for name in ('local','m1_multi','m2','gru','m4_adaptive','m5_full'):
        torch.manual_seed(1024);model=build_variant(name).eval()
        with torch.no_grad():
            full=model(xx);first=model(xx[:,:127]);rest=model(xx[:,127:],detach_state(first.state))
            joined=torch.cat((first.prediction,rest.prediction),1);chunk_error=float((full.prediction-joined).abs().max())
            gpu=build_variant(name).cuda().eval();gpu.load_state_dict(model.state_dict());out=gpu(xx.cuda())
            gpu_error=float((full.prediction-out.prediction.cpu()).abs().max())
        assert chunk_error<2e-5 and gpu_error<2e-4
        model.train();model.zero_grad(set_to_none=True);pred=model(xx).prediction
        loss=(pred-yy).square().mean();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        comparisons.append({'variant':name,'whole_chunk_max_error':chunk_error,'CPU_GPU_max_error':gpu_error,
                            'real_60s_gradients_finite':True})
    result={'technical':'PASS','actual_train_validation_index_checks':index_checks,'real_60s_comparisons':comparisons,
        'fit_ids_exact_newtrain':True,'old_exposed_test_excluded':True,'audit_payload_read':False,
        'torch':torch.__version__,'numpy':np.__version__,'epoch':time.time()}
    dump(ROOT/'preflight_final.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
