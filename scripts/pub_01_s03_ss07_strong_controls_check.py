"""Real train/val engineering, random discarded models/predictions only."""
import json
import sys
import numpy as np
import torch
from torch import nn
from pub_01_s03_ss07_baseline import ROOT,OLD,PREPARED,read,dump,native,cfg,sha
from pub_01_s03_ss07_finetune import RawWindows,Decoder
from pub_01_s03_ss07_train import LatentData,objective,macro_validation
from pub_01_s03_ss07_ema import score_windows


def main():
    sys.path.insert(0,str(OLD/'input'));torch.set_num_threads(2);torch.manual_seed(6161)
    data=RawWindows('train');n=data.rows[0]['windows'];x,y=data.batch(n-2,n+2,'cpu')
    assert data.ids[n-2]!=data.ids[n+1] and x.shape==(4,192,64)
    # Independent direct raw slices on both sides of the GLOBAL batch boundary.
    stats=np.load(PREPARED/'train_only_normalization.npz');expected=[]
    for index in range(n-2,n+2):
        row=data.rows[data.ids[index]];raw=np.load(row['eeg_path'],mmap_mode='r');raw=raw.T if raw.shape[0]==64 else raw
        s=(index-row['offset'])*96
        expected.append((np.asarray(raw[s:s+192],np.float32)-stats['mean'])/stats['std'])
    assert np.array_equal(x.numpy(),np.stack(expected).astype(np.float32))
    mod=native();backbone=mod.TransformerCLIP(cfg());model=Decoder(backbone.eeg).eval()
    with torch.no_grad():
        original=model.head(backbone.eeg(x).transpose(1,2));actual=model(x)
        assert torch.equal(actual,original)
    model.train();opt=torch.optim.Adam([{'params':model.encoder.parameters(),'lr':1e-5},
        {'params':model.head.parameters(),'lr':1e-3}],weight_decay=0.)
    loss=objective(model(x),y);loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    enc_grad=sum(int(torch.count_nonzero(v.grad)) for v in model.encoder.parameters())
    head_grad=sum(int(torch.count_nonzero(v.grad)) for v in model.head.parameters())
    assert enc_grad>0 and head_grad>0;opt.step()
    # Original macro evaluator vs EMA tau0, full real VAL geometry/truth but
    # artificial predictions: compares numerical definitions, not performance.
    val=RawWindows('val');val.x=torch.arange(len(val))[:,None,None]
    val.fixed_mask=read(ROOT/'block_eligibility/val.json')
    val.spans=lambda first,last:LatentData.spans(val,first,last)
    rng=np.random.default_rng(914);pred=rng.normal(size=(len(val),192)).astype('float32')
    class FixedPredictions(nn.Module):
        def forward(self,z,*args,**kwargs):return torch.from_numpy(pred)[z[:,0,0].long(),None,:],None,{}
    first,segments,people=macro_validation(FixedPredictions(),val)
    second,people2,segments2=score_windows(pred,val,0)
    error=max(abs(a['r']-b['r']) for a,b in zip(segments,segments2));assert error==0 and first==second
    assert people==people2
    dump(ROOT/'STRONG_CONTROLS_ENGINEERING_CHECK.json',{'status':'PASS','test_read':False,
        'random_full_encoder_discarded':True,'random_validation_predictions_only':True,
        'raw_zero_lag_mixed_record_batch_matches_independent_slices':True,
        'native_EEG_head_composition_initial_forward_max_difference':0,
        'encoder_nonzero_gradient_scalars':enc_grad,'head_nonzero_gradient_scalars':head_grad,
        'Adam_group_rates':[1e-5,1e-3],'full49person752block_EMA0_vs_original_macro_max_difference':error,
        'no_performance_selection':True,
        'sources':{str(ROOT/'input'/n):sha(ROOT/'input'/n) for n in
            ['pub_01_s03_ss07_finetune.py','pub_01_s03_ss07_ema.py','pub_01_s03_ss07_strong_controls_check.py']}})
    print(json.dumps({'status':'PASS','encoder_nonzero_grad':enc_grad,'head_nonzero_grad':head_grad,'EMA0_macro_difference':error}))


if __name__=='__main__':main()
