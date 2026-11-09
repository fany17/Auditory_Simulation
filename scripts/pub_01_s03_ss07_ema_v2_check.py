"""Actual trained native-P validation bridge, no test data or tau selection."""
import json
import numpy as np
import torch
from pub_01_s03_ss07_baseline import ROOT,dump,sha
from pub_01_s03_ss07_train import NativeHead,LatentData,macro_validation
from pub_01_s03_ss07_ema import score_windows
from pub_01_s03_ss07_ema_v2 import predict_validation


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    data=LatentData(1024,'val','cuda:0');model=NativeHead().cuda().eval()
    model.head.load_state_dict(torch.load(ROOT/'baseline/head_seed1024/best_validation.pt',map_location='cuda:0',weights_only=True)['head'])
    reference,old_segments,old_people=macro_validation(model,data)
    corrected=predict_validation(model,data)
    corrected_macro,corrected_people,corrected_segments=score_windows(corrected,data,0)
    error=max(abs(a['r']-b['r']) for a,b in zip(old_segments,corrected_segments))
    assert error==0 and reference==corrected_macro and old_people==corrected_people
    legacy=[]
    with torch.no_grad():
        for first in range(0,len(data.x),64):legacy.append(model(data.x[first:first+64])[0][:,0].cpu().numpy())
    legacy=np.concatenate(legacy);_,_,legacy_segments=score_windows(legacy,data,0)
    legacy_error=max(abs(a['r']-b['r']) for a,b in zip(old_segments,legacy_segments))
    dump(ROOT/'EMA_V2_ACTUAL_WEIGHT_CHECK.json',{'passed':True,'actual_completed_native_P_seed':1024,
        'whole_val_windows':30500,'people':49,'blocks':752,
        'corrected_tau0_perblock_r_max_difference_from_original_macro':error,
        'legacy_unsplit_B64_perblock_r_max_difference':legacy_error,
        'correction':'phase-global64 forward_global_batch with record boundary splits/reset; all tau reuse SAME predictions',
        'no_tau_selection_in_this_check':True,'test_read':False,'neural_parameters_updated':False,
        'sources':{str(ROOT/'input'/name):sha(ROOT/'input'/name) for name in
            ['pub_01_s03_ss07_ema.py','pub_01_s03_ss07_ema_v2.py','pub_01_s03_ss07_ema_v2_check.py','pub_01_s03_ss07_train.py']}})
    print(json.dumps({'passed':True,'new_tau0_diff':error,'old_unsplit_diff':legacy_error}))


if __name__=='__main__':main()
