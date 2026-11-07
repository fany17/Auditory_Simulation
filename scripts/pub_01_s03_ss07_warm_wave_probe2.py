"""Read-only exact cache-construction input/wrapper/layout localization."""
import json
import numpy as np
import torch
from pub_01_s03_ss07_warm import ROOT,ROUND01,OLD,make,read,dump,native,cfg,precision
from pub_01_s03_ss07_warm_verify import FreshVAL

@torch.no_grad()
def main():
    out=ROOT/'verification_failure_probe02';out.mkdir(exist_ok=False)
    device='cuda:0';model=make(1024,'frozen_p',device);model.eval();data=FreshVAL()
    original_rows=[r for r in read(OLD/'prepared/manifest_frozen.json')['records'] if r['phase']=='val']
    mismatch=[]
    for a,b in zip(original_rows,data.rows):
        for key in ('record','eeg_path','envelope_path','common_samples'):
            if a.get(key)!=b.get(key):mismatch.append({'record':a['record'],'key':key,'original':a.get(key),'replay':b.get(key)})
    assert len(original_rows)==len(data.rows)==49
    original=native().TransformerCLIP(cfg()).to(device)
    cp=torch.load(ROUND01/'baseline/pretrain_seed1024/best_validation.pt',map_location=device,weights_only=True)
    original.load_state_dict(cp['model']);original.eval();precision()
    cache=np.load(ROUND01/'baseline/head_seed1024/val_latent.npy',mmap_mode='r')
    stats=np.load(OLD/'prepared/train_only_normalization.npz');mean,std=stats['mean'],stats['std']
    results=[]
    for global_window in (0,9445,22967):
        row=data.ids[global_window];offset=row['phase_offset'];local=global_window-offset;first=local//256*256;last=min(first+256,row['readout_windows'])
        src=next(r for r in original_rows if r['record']==row['record'])
        raw=np.load(src['eeg_path'],mmap_mode='r');eeg=raw.T if raw.shape[0]==64 else raw
        starts=np.arange(first,last)*96
        xx=np.stack([(np.asarray(eeg[s:s+192],np.float32)-mean)/std for s in starts]).astype(np.float32)
        x,_=data.batch(offset+first,offset+last,device)
        original_input=torch.from_numpy(xx).to(device)
        cached=torch.from_numpy(np.array(cache[offset+first:offset+last],copy=True)).to(device)
        z1=model.encoder(x).transpose(1,2);z2=model.encoder(x).transpose(1,2)
        zorig=original.eeg(original_input).transpose(1,2)
        model.encoder.requires_grad_(True);zgrad=model.encoder(x).transpose(1,2);model.encoder.requires_grad_(False)
        results.append({'record':row['record'],'local_start':first,'batch':last-first,'input_exact_original_recipe_max_diff':float((x-original_input).abs().max()),
            'selected_wrapper_repeated_forward_max_diff':float((z1-z2).abs().max()),
            'selected_wrapper_vs_cache':float((z1-cached).abs().max()),
            'original_full_model_vs_cache':float((zorig-cached).abs().max()),
            'original_full_model_vs_selected_wrapper':float((zorig-z1).abs().max()),
            'reenabled_parameter_grad_vs_cache':float((zgrad-cached).abs().max()),
            'source_encoder_parameters_equal':all(torch.equal(p,q) for p,q in zip(model.encoder.parameters(),original.eeg.parameters())),
            'input_stride':list(x.stride()),'original_input_stride':list(original_input.stride()),'latent_stride':list(z1.stride()),'cached_stride':list(cached.stride())})
    result={'metadata_differences':mismatch,'probes':results,'reference_waveform_peak_float32_spacing':float(np.spacing(np.float32(313.4454650878906))),
        'original_wave_absolute_gate':1e-6,'no_gate_or_checkpoint_or_training_change':True,'test_read':False}
    dump(out/'PROBE.json',result);print(json.dumps(result))

if __name__=='__main__':main()
