"""One read-only localization of the retained wholeVAL replay failure."""
from pathlib import Path
import json
import shutil
import torch
import numpy as np
from pub_01_s03_ss07_warm import ROOT,ROUND01,make,read,dump,sha
from pub_01_s03_ss07_warm_verify import FreshVAL,readout
from pub_01_s03_ss07_finetune import RawWindows

class Native256:
    """One original per-record B256 block retained in RAM, no file cache."""
    def __init__(self,model,data,device):self.model=model;self.data=data;self.device=device;self.key=None;self.z=None
    def batch(self,first,last):
        parts=[];cursor=first
        while cursor<last:
            row=self.data.ids[cursor];local=cursor-row['phase_offset'];block=local//256*256
            stop=min(block+256,row['readout_windows']);key=(row['record'],block)
            if key!=self.key:
                start=row['phase_offset']+block;end=row['phase_offset']+stop
                x,_=self.data.batch(start,end,self.device)
                self.z=self.model.encoder(x).transpose(1,2);self.key=key
            count=min(last-cursor,stop-local);parts.append(self.z[local-block:local-block+count]);cursor+=count
        return torch.cat(parts)

@torch.no_grad()
def main():
    out=ROOT/'verification_failure_probe01';out.mkdir(exist_ok=False)
    for name in ['FINAL_REVIEW_FAILURE.json','FINAL_REVIEW_STARTED.json','final_review01.log']:
        shutil.copy2(ROOT/name,out/name)
    shutil.copy2(ROOT/'report_input/pub_01_s03_ss07_warm_verify.py',out/'failed_verify_source.py')
    seed=1024;condition='frozen_p';device='cuda:0'
    cp_path=ROOT/f'fits/{condition}_seed{seed}/best_validation.pt'
    cp=torch.load(cp_path,map_location='cpu',weights_only=True);model=make(seed,condition,device);model.load_state_dict(cp['model']);model.eval()
    data=FreshVAL();reference=RawWindows('val');cache=np.load(ROUND01/f'baseline/head_seed{seed}/val_latent.npy',mmap_mode='r')
    boundary=data.rows[0]['readout_windows'];starts=[0,boundary//64*64]
    loader_checks=[]
    for first in starts:
        x,spans=data.batch(first,first+64,device);old,y=reference.batch(first,first+64,device)
        loader_checks.append({'global_start':first,'neural_input_max_diff':float((x-old).abs().max()),
            'target_max_diff':float(np.abs(data.y[first:first+64]-y.cpu().numpy()).max())})
    results=[]
    for mode in ['global64','per_record256']:
        native=Native256(model,data,device);state=None;active=None;cs=None;ca=None;resets=0;maxlatent=0.;maxwave=0.;scale=0.;sumwave2=0.;scalars=0;witness=None
        for first in range(0,30500,64):
            last=min(first+64,30500);x,spans=data.batch(first,last,device)
            z=model.encoder(x).transpose(1,2) if mode=='global64' else native.batch(first,last)
            cached=torch.from_numpy(np.array(cache[first:last],copy=True)).to(device)
            maxlatent=max(maxlatent,float((z-cached).abs().max()))
            p,state,active=readout(model,z,spans,state,active);q,cs,ca=readout(model,cached,spans,cs,ca)
            diff=(p-q).abs();current=float(diff.max())
            if current>maxwave:
                index=np.unravel_index(int(diff.argmax()),tuple(diff.shape));witness={'global_window':first+int(index[0]),'time_index':int(index[2]),'record':data.ids[first+int(index[0])]['record']};maxwave=current
            scale=max(scale,float(q.abs().max()));sumwave2+=float(q.double().square().sum());scalars+=q.numel()
            resets+=sum(local==0 for _,_,_,local in spans)
        results.append({'encoder_geometry':mode,'head_geometry':'fixed_global64','state_record_starts':resets,
            'latent_max_absolute_difference':maxlatent,'waveform_max_absolute_difference':maxwave,
            'reference_waveform_max_absolute':scale,'reference_waveform_RMS':(sumwave2/scalars)**.5,
            'waveform_max_difference_over_reference_max':maxwave/scale,'max_difference_witness':witness,
            'unchanged_wave_gate_1e6_pass':maxwave<=1e-6})
    result={'seed':seed,'condition':condition,'selected_checkpoint':str(cp_path),'checkpoint_sha256':sha(cp_path),
        'failed_verify_source_sha256':sha(out/'failed_verify_source.py'),'loader_against_original_RawWindows':loader_checks,
        'cache_creation_source':str(ROUND01/'input/pub_01_s03_ss07_baseline.py'),
        'cache_creation_geometry_from_frozen_source':'per-record blocks of256, partial record tail; original encoder eval',
        'results':results,'changed_training_or_checkpoint_or_tolerance':False,'new_fit':False,'test_read':False}
    dump(out/'PROBE.json',result);print(json.dumps(result))

if __name__=='__main__':main()
