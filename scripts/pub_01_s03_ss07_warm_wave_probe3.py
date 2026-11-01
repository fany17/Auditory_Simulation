"""Final bounded three-repeat, original-recipe input stride localization."""
import numpy as np
import torch
from pub_01_s03_ss07_warm import ROOT,ROUND01,OLD,make,native,cfg,read,dump,precision
from pub_01_s03_ss07_warm_verify import FreshVAL

@torch.no_grad()
def main():
    out=ROOT/'verification_failure_probe03';out.mkdir(exist_ok=False);device='cuda:0'
    model=make(1024,'frozen_p',device);model.eval();data=FreshVAL()
    original=native().TransformerCLIP(cfg()).to(device)
    cp=torch.load(ROUND01/'baseline/pretrain_seed1024/best_validation.pt',map_location=device,weights_only=True)
    original.load_state_dict(cp['model']);original.eval();precision()
    cache=np.load(ROUND01/'baseline/head_seed1024/val_latent.npy',mmap_mode='r')
    rows=[r for r in read(OLD/'prepared/manifest_frozen.json')['records'] if r['phase']=='val']
    stats=np.load(OLD/'prepared/train_only_normalization.npz');mean,std=stats['mean'],stats['std'];results=[]
    for global_window in (0,9445,22967):
        row=data.ids[global_window];offset=row['phase_offset'];local=global_window-offset;first=local//256*256;last=min(first+256,row['readout_windows'])
        source=next(r for r in rows if r['record']==row['record']);raw=np.load(source['eeg_path'],mmap_mode='r');eeg=raw.T if raw.shape[0]==64 else raw
        values=[(np.asarray(eeg[s:s+192],np.float32)-mean)/std for s in np.arange(first,last)*96]
        original_input=torch.from_numpy(np.stack(values).astype(np.float32)).to(device)
        c_order_input,_=data.batch(offset+first,offset+last,device)
        cached=torch.from_numpy(np.array(cache[offset+first:offset+last],copy=True)).to(device)
        encoded=[model.encoder(original_input).transpose(1,2) for _ in range(3)]
        direct=[original.eeg(original_input).transpose(1,2) for _ in range(3)]
        results.append({'record':row['record'],'batch':last-first,'dtype':str(original_input.dtype),
            'original_input_stride':list(original_input.stride()),'old_review_input_stride':list(c_order_input.stride()),
            'numeric_input_max_difference':float((original_input-c_order_input).abs().max()),
            'original_recipe_selected_wrapper_vs_cache':[float((z-cached).abs().max()) for z in encoded],
            'original_recipe_original_wrapper_vs_cache':[float((z-cached).abs().max()) for z in direct],
            'same_input_selected_three_repeat_max':max(float((z-encoded[0]).abs().max()) for z in encoded),
            'same_input_original_three_repeat_max':max(float((z-direct[0]).abs().max()) for z in direct)})
    result={'probes':results,'precision':{'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cudnn':torch.backends.cudnn.allow_tf32},
        'all_original_recipe_exact_cache':all(max(r['original_recipe_selected_wrapper_vs_cache'])==0 and max(r['original_recipe_original_wrapper_vs_cache'])==0 for r in results),
        'source_of_layout_mismatch':'old FreshVAL used np.asarray(list,dtype=float32); original cache and RawWindows use np.stack(list).astype(float32)',
        'changed_checkpoint_training_or_gate':False,'test_read':False}
    dump(out/'PROBE.json',result);print(__import__('json').dumps(result))

if __name__=='__main__':main()
