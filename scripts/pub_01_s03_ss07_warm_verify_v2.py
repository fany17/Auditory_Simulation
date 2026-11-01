"""Independent v2 whole-VAL review; v1 failure remains immutable.

Independent branch reuses frozen neural modules, not primary loader/readout.
Reference branch explicitly uses the unchanged primary Data.batch for all18.
No fit, model selection, test, download or new cache. Waits for all18 reports.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from pub_01_s03_ss07_warm import ROOT,ROUND01,OLD,SEEDS,CONDITIONS,make,read,dump,update,sha,Data as CanonicalData,native,cfg

def pearson(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert a.max()!=a.min() and b.max()!=b.min(),'exact constant'
    ac=a-a.mean();bc=b-b.mean();den=np.linalg.norm(ac)*np.linalg.norm(bc)
    assert den>0 and np.isfinite(den)
    r=float(np.dot(ac,bc)/den);assert np.isfinite(r)
    return r

def checks():
    constant=np.full(3839,.1);normal=np.sin(np.arange(3839)*.03)
    assert np.std(constant)>0
    for a,b in [(constant,normal),(normal,constant)]:
        try:pearson(a,b)
        except AssertionError:pass
        else:raise AssertionError('nonbinary constant accepted')
    other=np.cos(np.arange(3839)*.017)
    difference=abs(pearson(normal,other)-float(np.corrcoef(normal,other)[0,1]))
    assert difference<=1e-10
    return {'constant_pred_and_target_rejected':True,'normal_formula_difference':difference}

class FreshVAL:
    def __init__(self):
        allrows=read(OLD/'envelope_readout_seed1024/manifest_frozen.json')['records']
        self.rows=[r for r in allrows if r['phase']=='val'];assert len(self.rows)==49
        self.ids=[]
        for row in self.rows:
            assert row['phase_offset']==len(self.ids)
            n=(row['common_samples']-192)//96+1;assert n==row['readout_windows']
            self.ids.extend([row]*n)
        assert len(self.ids)==30500
        stats=np.load(OLD/'prepared/train_only_normalization.npz');self.mean,self.std=stats['mean'],stats['std']
        self.y=np.load(OLD/'envelope_readout_seed1024/val_envelope.npy',mmap_mode='r');assert self.y.shape==(30500,1,192)
        self.record=None;self.eeg=None
    def batch(self,first,last,device):
        values=[];spans=[];cursor=first
        while cursor<last:
            row=self.ids[cursor];local=cursor-row['phase_offset'];end=min(last,row['phase_offset']+row['readout_windows'])
            if self.record!=row['record']:
                loaded=np.load(row['eeg_path'],mmap_mode='r');self.eeg=loaded.T if loaded.shape[0]==64 else loaded
                assert self.eeg.ndim==2 and self.eeg.shape[1]==64;self.record=row['record']
            for index in range(local,local+end-cursor):
                start=index*96;values.append((np.asarray(self.eeg[start:start+192],np.float32)-self.mean)/self.std)
            spans.append((cursor-first,end-first,row,local));cursor=end
        # Preserve original RawWindows/cache recipe AND its tensor strides.
        x=torch.from_numpy(np.stack(values).astype(np.float32)).to(device)
        assert x.shape==(last-first,192,64)
        return x,spans

class Native256:
    """Original per-record B256 encoder blocks, one block in RAM only."""
    def __init__(self,encoder,data,device):self.encoder=encoder;self.data=data;self.device=device;self.key=None;self.z=None
    def batch(self,first,last):
        parts=[];cursor=first
        while cursor<last:
            row=self.data.ids[cursor];local=cursor-row['phase_offset'];block=local//256*256
            stop=min(block+256,row['readout_windows']);key=(row['record'],block)
            if key!=self.key:
                x,_=self.data.batch(row['phase_offset']+block,row['phase_offset']+stop,self.device)
                self.z=self.encoder(x).transpose(1,2);self.key=key
            count=min(last-cursor,stop-local);parts.append(self.z[local-block:local-block+count]);cursor+=count
        return torch.cat(parts)

def readout(model,z,spans,state,active):
    if model.kind=='p':return model.readout.head(z),None,None
    transformed=[]
    for first,last,row,local in spans:
        if active!=row['record']:state=None
        value=z[first:last]
        if model.kind=='current':context=model.readout.core.current_only(value);state=None
        else:
            pooled=F.adaptive_avg_pool1d(value,6).flatten(1)
            context,_,state=model.readout.core.safe_history(pooled,state,local,row['record'])
        coefficients=model.readout.residual(context).reshape(-1,8,3)
        result=value
        for j,lag in enumerate((4,8,16)):
            result=result+.1*coefficients[:,:,j,None]*(F.pad(value[:,:,:-lag],(lag,0))-value)
        transformed.append(result);active=row['record']
    return model.readout.head(torch.cat(transformed)),state,active

def continuous(pred,data):
    assert pred.shape==(30500,192) and np.isfinite(pred).all() and pred.max()!=pred.min()
    plan=read(ROUND01/'block_eligibility/val.json');assert plan['records']==[r['record'] for r in data.rows]
    keys={(r['record'],r['start']):r for r in plan['blocks']};segments={};people={};mathdiff=0.
    taper=np.hanning(192)
    for row in data.rows:
        n=row['common_samples'];start=row['phase_offset'];count=row['readout_windows']
        coordinates=np.arange(count)[:,None]*96+np.arange(192)[None,:]
        numerator=np.zeros(n);denominator=np.zeros(n)
        np.add.at(numerator,coordinates.reshape(-1),(pred[start:start+count]*taper).reshape(-1))
        np.add.at(denominator,coordinates.reshape(-1),np.broadcast_to(taper,(count,192)).reshape(-1))
        covered=denominator>1e-8;numerator[covered]/=denominator[covered]
        truth=np.load(row['envelope_path'],mmap_mode='r').reshape(-1);assert len(truth)>=n
        for begin in range(0,n-3840+1,3840):
            mask=covered[begin:begin+3840];key=(row['record'],begin);rule=keys[key]
            assert int(mask.sum())==rule['samples']
            if not rule['eligible']:continue
            p=numerator[begin:begin+3840][mask];t=np.asarray(truth[begin:begin+3840])[mask]
            r=pearson(p,t);mathdiff=max(mathdiff,abs(r-float(np.corrcoef(p,t)[0,1])))
            segments[key]=r;people.setdefault(row['subject'],[]).append(r)
    assert len(segments)==plan['eligible_blocks']==752 and len(people)==49 and mathdiff<=1e-10
    persons={p:float(np.mean(v)) for p,v in people.items()}
    return float(np.mean(list(persons.values()))),segments,persons,mathdiff

def weight_changes(model,cp):
    initial=model.state_dict();assert set(initial)==set(cp['model'])
    groups={}
    for name,trained in cp['model'].items():
        before=initial[name].detach().cpu();after=trained.detach().cpu()
        assert before.shape==after.shape and before.dtype==after.dtype and torch.isfinite(after).all()
        group='encoder' if name.startswith('encoder.') else 'head' if name.startswith('readout.head.') else 'residual' if name.startswith('readout.residual.') else 'context_core'
        row=groups.setdefault(group,{'scalars':0,'changed_scalars':0,'max_absolute_change':0.,'tensors':0})
        row['scalars']+=after.numel();row['changed_scalars']+=int(torch.count_nonzero(after!=before));row['tensors']+=1
        row['max_absolute_change']=max(row['max_absolute_change'],float((after.double()-before.double()).abs().max()))
    if not model.joint:assert groups['encoder']['changed_scalars']==0
    if cp['epoch']==-1:assert all(v['changed_scalars']==0 for v in groups.values())
    # No nonzero-update assertion on every parameter: legitimate exact-zero
    # gradients and incumbent selection are evidence, not reasons to refit.
    return groups

@torch.no_grad()
def replay(condition,seed,device,out):
    folder=ROOT/f'fits/{condition}_seed{seed}';done=read(folder/'completed.json');reference=read(folder/'SELECTED_VALIDATION.json')
    assert sha(folder/'best_validation.pt')==done['checkpoint_sha256']
    cp=torch.load(folder/'best_validation.pt',map_location='cpu',weights_only=True)
    assert cp['seed']==seed and cp['condition']==condition and cp['epoch']==done['selected_epoch']
    model=make(seed,condition,device);changes=weight_changes(model,cp);model.load_state_dict(cp['model']);model.eval()
    data=FreshVAL();canonical=CanonicalData(seed,'val',device,model.joint)
    original=None;native_reader=None
    if not model.joint:
        # Exact original cache wrapper/context; selected encoder remains frozen.
        # Original parameter grad flags TRUE, eval and enclosing no_grad.
        original=native().TransformerCLIP(cfg()).to(device)
        pre=torch.load(ROUND01/f'baseline/pretrain_seed{seed}/best_validation.pt',map_location=device,weights_only=True)
        assert pre['seed']==seed;original.load_state_dict(pre['model']);original.eval()
        a=original.eeg.state_dict();b=model.encoder.state_dict()
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
        assert all(p.requires_grad for p in original.eeg.parameters()) and not original.eeg.training
        native_reader=Native256(original.eeg,data,device)
    state=None;active=None;ref_state=None;ref_active=None;parts=[];maxwave=0.;resets=0;ref_scale=0.;fresh_scale=0.
    for first in range(0,30500,64):
        last=min(first+64,30500);x,spans=data.batch(first,last,device)
        assert x.shape[0]==last-first
        z=model.encoder(x).transpose(1,2) if model.joint else native_reader.batch(first,last)
        pred,state,active=readout(model,z,spans,state,active)
        resets+=sum(local==0 for _,_,_,local in spans)
        reference_wave,_,ref_state,ref_active=canonical.batch(model,first,last,ref_state,ref_active)
        maxwave=max(maxwave,float((pred-reference_wave).abs().max()))
        ref_scale=max(ref_scale,float(reference_wave.abs().max()));fresh_scale=max(fresh_scale,float(pred.abs().max()))
        parts.append(pred[:,0].cpu().numpy())
    assert resets==49
    normalized_wave=maxwave/max(1.,ref_scale,fresh_scale)
    pred=np.concatenate(parts);a=pred.astype(np.float64);b=np.asarray(data.y[:,0],np.float64)
    ac=a-a.mean(1,keepdims=True);bc=b-b.mean(1,keepdims=True)
    r3=np.sum(ac*bc,axis=1)/(np.maximum(np.linalg.norm(ac,axis=1),1e-6)*np.maximum(np.linalg.norm(bc,axis=1),1e-6))
    fresh3=float(r3.mean());assert np.isfinite(r3).all()
    macro,segments,persons,mathdiff=continuous(pred,data)
    original={(r['record'],r['start']):r['r'] for r in reference['continuous']['segments']}
    origpeople={r['subject']:r['r'] for r in reference['continuous']['people']}
    assert original.keys()==segments.keys() and origpeople.keys()==persons.keys()
    blockdiff=max(abs(v-original[k]) for k,v in segments.items());persondiff=max(abs(v-origpeople[k]) for k,v in persons.items())
    mean3diff=abs(fresh3+reference['negative_Pearson']);mean60diff=abs(macro-reference['continuous']['macro_r'])
    result={'condition':condition,'seed':seed,'selected_epoch':cp['epoch'],'windows':30500,'records':49,'persons':49,'blocks':752,
        'checkpoint_sha256':done['checkpoint_sha256'],'actual_weight_changes':changes,
        'fresh_minus_primary_reference_wave_max_absolute':maxwave,'reference_wave_max_absolute':ref_scale,
        'fresh_wave_max_absolute':fresh_scale,'normalized_wave_error':normalized_wave,
        'original_absolute_wave_gate_1e6_passed':maxwave<=1e-6,
        'normalized_wave_gate_1e6_passed':normalized_wave<=1e-6,
        'encoder_geometry':'joint original global64' if model.joint else 'original cache per-record256 including tails',
        'encoder_eval_and_no_grad':True,'encoder_parameter_grad_flags_match_original_forward':True,
        'fresh_VAL3s_r':fresh3,'fresh_VAL60s_person_r':macro,'VAL3s_absolute_difference':mean3diff,
        'VAL60s_absolute_difference':mean60diff,'max_block_difference':blockdiff,'max_person_difference':persondiff,
        'same_array_independent_formula_max_difference':mathdiff,
        'primary_wave_acceptance':'original absolute1e-6, normalized reported only',
        'passed':maxwave<=1e-6 and mean3diff<=1e-7 and mean60diff<=1e-7 and blockdiff<=1e-6 and persondiff<=1e-6,'test_read':False}
    dump(out/f'{condition}_seed{seed}.json',result)
    assert result['passed'],'preserve complete discrepancy; no automatic retry, fit or threshold change'
    return result

def main(device,wait,expected):
    assert sha(Path(__file__))==expected
    protocol=read(ROOT/'WARM_PROTOCOL.json');assert protocol['frozen'] and protocol['test_operations'] is False
    review_protocol=read(ROOT/'FINAL_REVIEW_V2_PROTOCOL.json');assert review_protocol['frozen']
    for path,digest in review_protocol['sources'].items():assert sha(path)==digest,path
    started=time.time();checks_result=checks()
    dump(ROOT/'FINAL_REVIEW_V2_STARTED.json',{'time':started,'pid':__import__('os').getpid(),'source_sha256':expected,
        'protocol_sha256':sha(ROOT/'WARM_PROTOCOL.json'),'target':'all18 selected wholeVAL,original derivative and targets only',
        'tolerances':{'same_array_r':1e-10,'original_absolute_wave_reported':1e-6,'normalized_wave':1e-6,'mean3s_60s_r':1e-7,'block_person_r':1e-6},
        'v1_failure_retained':True,'review_protocol_sha256':sha(ROOT/'FINAL_REVIEW_V2_PROTOCOL.json'),
        'frozen_neural_modules_reused_not_independent_neural_math':True,'test_operations':False,'new_fits':False})
    while not (ROOT/'results01/COMPLETED.json').exists():
        assert wait,'all18 report incomplete'
        assert time.time()-started<2*86400 and not (ROOT/'QUEUE_FAILURE.json').exists() and not (ROOT/'REPORT_QUEUE_FAILURE.json').exists()
        # Wait on verified report-controller handle; never restart it.
        lease=read(ROOT/'REPORT_QUEUE_LEASE.json');p=Path(f'/proc/{lease["pid"]}')
        try:argv=p.joinpath('cmdline').read_bytes()
        except FileNotFoundError:
            if (ROOT/'results01/COMPLETED.json').exists():break
            raise AssertionError('report handle gone without complete marker; inspect, do not restart')
        if not argv:
            if (ROOT/'results01/COMPLETED.json').exists():break
            raise AssertionError('report handle terminal without complete marker')
        assert b'pub_01_s03_ss07_warm_report_queue.py' in argv
        update(ROOT/'FINAL_REVIEW_V2_STATUS.json',{'state':'WAIT_COMPLETE_18CELL_REPORT','verified_report_pid':lease['pid'],'time':time.time()})
        time.sleep(20)
    assert read(ROOT/'results01/COMPLETED.json')['all18_reported']
    assert read(ROOT/'TRAIN_VAL_COMPLETED.json')['new_fits']==18
    for path,digest in read(ROOT/'results01/SUMMARY.json')['inputs'].items():assert sha(path)==digest,path
    for p,d in protocol['sources'].items():assert sha(p)==d,p
    assert sha(Path(__file__))==expected
    out=ROOT/'final_review02';out.mkdir(exist_ok=False);results=[]
    for condition in CONDITIONS:
        for seed in SEEDS:
            results.append(replay(condition,seed,device,out))
            update(ROOT/'FINAL_REVIEW_V2_STATUS.json',{'state':'REPLAY_WHOLE_VAL','completed_cells':len(results),'time':time.time()})
    dump(out/'COMPLETED.json',{'time':time.time(),'cells':18,'wholeVAL_window_replays':18*30500,'continuous_blocks':18*752,
        'all_passed':all(r['passed'] for r in results),'max_mean3s_difference':max(r['VAL3s_absolute_difference'] for r in results),
        'max_mean60s_difference':max(r['VAL60s_absolute_difference'] for r in results),'max_block_difference':max(r['max_block_difference'] for r in results),
        'max_person_difference':max(r['max_person_difference'] for r in results),'independent_math_checks':checks_result,
        'max_absolute_wave_difference':max(r['fresh_minus_primary_reference_wave_max_absolute'] for r in results),
        'max_normalized_wave_error':max(r['normalized_wave_error'] for r in results),
        'corrected_replay_absolute_gate_all18_passed':all(r['original_absolute_wave_gate_1e6_passed'] for r in results),
        'v1_failed_gate_remains_failed_historical_evidence':True,'v2_normalized_all18_passed':all(r['normalized_wave_gate_1e6_passed'] for r in results),
        'review_protocol_sha256':sha(ROOT/'FINAL_REVIEW_V2_PROTOCOL.json'),
        'source_sha256':expected,'test_read':False,'new_fits':False,'scientific_goal_achieved_claimed':False,
        'limitation':'frozen neural modules reused; canonical derivative inputs, not original BDF/audio preprocessing replay; TRAIN_VAL development only'})
    update(ROOT/'FINAL_REVIEW_V2_STATUS.json',{'state':'COMPLETED_ALL18_WHOLE_VAL_REPLAY','time':time.time(),'scientific_goal_achieved_claimed':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--wait',action='store_true');p.add_argument('--expected-sha',required=True);a=p.parse_args()
    try:main(a.device,a.wait,a.expected_sha)
    except Exception as e:
        update(ROOT/'FINAL_REVIEW_V2_FAILURE.json',{'time':time.time(),'error':repr(e),'automatic_retries':False});raise
