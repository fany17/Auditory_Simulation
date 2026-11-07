"""Bounded SOURCE-TRAIN CPU diagnostics; no VAL/test, fit or GPU use."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_train_diagnostics_20261101')
P=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_continuity_20261101')
OLD=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261015/transformer_available01')
SEEDS=(1024,2026,31415)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    assert Path(p).is_relative_to(ROOT)
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def status(v):
    path=ROOT/'STATUS.json';tmp=ROOT/'STATUS.writing';tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,path)
def ratio(n,d):
    value=n/d if d>0 else np.nan
    return float(value) if np.isfinite(value) else None
def desc(v,prefix):
    a=np.asarray(v,dtype=np.float64).reshape(-1);a=a[np.isfinite(a)]
    if len(a):values=[len(a),float(a.mean()),float(a.std()),*map(float,np.quantile(a,[.05,.5,.95]))]
    else:values=[0,None,None,None,None,None]
    return {prefix+'_'+k:v for k,v in zip(['n','mean','std','q05','median','q95'],values)}
def csvfile(path,rows):
    if not rows:return
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def alignment(a,b,fit_first=True):
    assert a.shape==b.shape and a.shape[1]==96
    fit=slice(0,48) if fit_first else slice(48,96);held=slice(48,96) if fit_first else slice(0,48)
    x=a[:,fit];y=b[:,fit];xm=x.mean(1);ym=y.mean(1);xc=x-xm[:,None];yc=y-ym[:,None]
    den=(xc*xc).sum(1);cov=(xc*yc).sum(1)
    valid=(x.max(1)!=x.min(1))&(den>0)
    scale=np.full(len(x),np.nan);np.divide(cov,den,out=scale,where=valid)
    offset=ym-scale*xm;valid &= np.isfinite(scale)&np.isfinite(offset)&(scale>0)
    raw=((a[valid,held]-b[valid,held])**2).mean(1)
    aligned=((a[valid,held]*scale[valid,None]+offset[valid,None]-b[valid,held])**2).mean(1)
    return {'valid':valid,'scale':scale[valid],'offset':offset[valid],'raw':raw,'aligned':aligned}

def energy(x):
    centered=x.astype(np.float64)-x.astype(np.float64).mean(2,keepdims=True)
    bins=centered.reshape(len(x),8,6,32).mean(3);recon=np.repeat(bins,32,axis=2)
    e0=(centered*centered).sum(2);e6=(recon*recon).sum(2);er=((centered-recon)**2).sum(2)
    error=np.abs(e0-e6-er)/np.maximum(e0,1e-300)
    assert np.max(error)<=1e-10
    values=np.full(e0.shape,np.nan);np.divide(e6,e0,out=values,where=e0>0)
    assert np.all(values[np.isfinite(values)]>=-1e-12) and np.all(values[np.isfinite(values)]<=1+1e-12)
    return e0,e6,er,values,float(error.max())

def math_check():
    rng=np.random.default_rng(20261101);a=rng.normal(size=(4,96));b=2*a+.3
    for direction in (True,False):
        r=alignment(a,b,direction);assert r['valid'].all() and r['aligned'].max()<1e-28
    assert not alignment(a,-a,True)['valid'].any()
    assert not alignment(np.full((1,96),.1),np.ones((1,96)),True)['valid'].any()
    x=np.repeat(rng.normal(size=(2,8,6)),32,axis=2).astype(np.float32)
    e0,e6,er,_,_=energy(x);assert np.max(er)<1e-27 and np.allclose(e0,e6,rtol=1e-12,atol=1e-12)
    alternating=np.tile(np.array([-1.,1.]),96)[None,None,:].repeat(8,axis=1)
    _,e6,_,_,_=energy(alternating);assert e6.max()==0
    return {'positive_affine_both_half_directions_exact':True,'negative_scale_rejected':True,
        'nonbinary_constant_fit_rejected':True,'piecewise_constant_energy_preserved':True,'within_bin_zero_mean_energy_discarded':True}

def prepare():
    assert not (ROOT/'DIAGNOSTIC_PROTOCOL.json').exists()
    manifest=OLD/'envelope_readout_seed1024/manifest_frozen.json'
    rows=[r for r in read(manifest)['records'] if r['phase']=='train'];assert len(rows)==426
    offset=0
    for r in rows:
        assert r['phase_offset']==offset;offset+=r['readout_windows']
    assert offset==252454
    dump(ROOT/'TRAIN_RECORDS.json',rows)
    sources={str(Path(__file__)):sha(Path(__file__)),str(ROOT/'DEFINITION.md'):sha(ROOT/'DEFINITION.md'),
        str(manifest):sha(manifest),str(ROOT/'TRAIN_RECORDS.json'):sha(ROOT/'TRAIN_RECORDS.json')}
    inputs={}
    for seed in SEEDS:
        folder=P/f'baseline/head_seed{seed}';cache=read(folder/'cache_manifest.json')
        assert cache['only_train_val'] and cache['encoder_sha256']==sha(P/f'baseline/pretrain_seed{seed}/best_validation.pt')
        for path in [folder/'cache_manifest.json',folder/'best_validation.pt',P/f'baseline/pretrain_seed{seed}/best_validation.pt']:
            sources[str(path)]=sha(path)
        inputs[str(seed)]={'train_latent':str(folder/'train_latent.npy'),'bytes':(folder/'train_latent.npy').stat().st_size,
            'checkpoint':str(folder/'best_validation.pt'),'cache_manifest':str(folder/'cache_manifest.json')}
    dump(ROOT/'DIAGNOSTIC_PROTOCOL.json',{'frozen':True,'time':time.time(),'sources':sources,'inputs':inputs,
        'seeds':list(SEEDS),'records':426,'windows_per_seed':252454,'phase':'TRAIN_ONLY','batch':256,'compute_threads':1,
        'CUDA_VISIBLE_DEVICES':'','D1':'all adjacent96 overlaps; two fixed48 fit/held directions; strict positive scale; paired raw denominator; no labels',
        'D2':'center each channel/window; six nonoverlapping32 means, piecewise reconstruction; E6/E0 and zero-energy coverage',
        'D3':'original sharedTRAIN target lags192/384/768 once, centered pair Pearson within record',
        'new_NN_fits':False,'encoder_forward':False,'VAL_test_DTU_reads':False,'new_cache_or_payload_download':False,
        'bad_record_policy':'record error/coverage and continue other records; no source repair, replacement or automatic retry'})
    print('TRAIN_DIAGNOSTIC_DEFINED_FROZEN',flush=True)

@torch.inference_mode()
def process_record(seed,row,cache,head):
    begin=row['phase_offset'];count=row['readout_windows'];prev=None
    means=[];stds=[];mse=[];nrmse=[];corr=[];mean_delta=[];scale_ratio=[]
    parts={d:{'scale':[],'offset':[],'raw':[],'aligned':[],'positive':0} for d in ('forward','reverse')}
    e0sum=np.zeros(8);e6sum=np.zeros(8);ersum=np.zeros(8);retentions=[];zero=0;balance=0.
    for first in range(begin,begin+count,256):
        last=min(first+256,begin+count);x=np.array(cache[first:last],dtype=np.float32,copy=True)
        assert x.shape==(last-first,8,192) and np.isfinite(x).all()
        pred=head(torch.from_numpy(x))[:,0].numpy().astype(np.float64)
        assert np.isfinite(pred).all();means.append(pred.mean(1));stds.append(pred.std(1))
        if prev is None:left=pred[:-1];right=pred[1:]
        else:left=np.concatenate([prev[None],pred[:-1]],0);right=pred
        prev=pred[-1].copy();a=left[:,96:];b=right[:,:96]
        if len(a):
            raw=((a-b)**2).mean(1);va=a.var(1);vb=b.var(1);norm=(va+vb)/2
            nrm=np.full(len(a),np.nan);np.divide(raw,norm,out=nrm,where=norm>0);nrm=np.sqrt(nrm)
            ac=a-a.mean(1,keepdims=True);bc=b-b.mean(1,keepdims=True);den=np.sqrt((ac*ac).sum(1)*(bc*bc).sum(1))
            r=np.full(len(a),np.nan);np.divide((ac*bc).sum(1),den,out=r,where=(den>0)&(a.max(1)!=a.min(1))&(b.max(1)!=b.min(1)))
            sr=np.full(len(a),np.nan);np.divide(np.sqrt(vb),np.sqrt(va),out=sr,where=va>0)
            mse.append(raw);nrmse.append(nrm);corr.append(r);mean_delta.append(np.abs(a.mean(1)-b.mean(1)));scale_ratio.append(sr)
            for direction in parts:
                fitted=alignment(a,b,direction=='forward');parts[direction]['positive']+=int(fitted['valid'].sum())
                for k in ('scale','offset','raw','aligned'):parts[direction][k].append(fitted[k])
        e0,e6,er,ret,error=energy(x);e0sum+=e0.sum(0);e6sum+=e6.sum(0);ersum+=er.sum(0)
        retentions.append(ret.reshape(-1));zero+=int((e0==0).sum());balance=max(balance,error)
    cat=lambda values:np.concatenate(values) if values else np.array([],dtype=np.float64)
    wm=cat(means);ws=cat(stds);raw=cat(mse);assert len(wm)==count and len(raw)==max(0,count-1)
    d1={'seed':seed,'record':row['record'],'subject':row['subject'],'windows':count,'overlap_pairs':len(raw),
        **desc(wm,'window_mean'),**desc(ws,'window_std'),'window_mean_drift_over_median_std':ratio(float(wm.std()),float(np.median(ws))),
        'window_std_CV':ratio(float(ws.std()),float(ws.mean())),**desc(raw,'raw_overlap_MSE'),
        **desc(cat(nrmse),'raw_overlap_NRMSE'),**desc(cat(corr),'overlap_r'),**desc(cat(mean_delta),'overlap_abs_mean_delta'),
        **desc(cat(scale_ratio),'overlap_std_ratio')}
    for direction,data in parts.items():
        aligned=cat(data['aligned']);pairedraw=cat(data['raw']);scales=cat(data['scale']);offsets=cat(data['offset'])
        d1.update({direction+'_positive_fit_pairs':data['positive'],direction+'_positive_fit_fraction':ratio(data['positive'],len(raw)),
            direction+'_paired_raw_MSE_sum':float(pairedraw.sum()),direction+'_aligned_held_MSE_sum':float(aligned.sum()),
            direction+'_held_aligned_raw_ratio':ratio(float(aligned.sum()),float(pairedraw.sum())),
            **desc(scales,direction+'_scale'),**desc(offsets,direction+'_offset')})
    d2={'seed':seed,'record':row['record'],'subject':row['subject'],'windows':count,'window_channel_cases':count*8,
        'zero_energy_cases':zero,'E0':float(e0sum.sum()),'E6':float(e6sum.sum()),'Er':float(ersum.sum()),
        'energy_retained_ratio':ratio(float(e6sum.sum()),float(e0sum.sum())),
        'energy_balance_relative_max_error':balance,**desc(cat(retentions),'window_channel_retention')}
    channels=[{'seed':seed,'record':row['record'],'channel':c,'E0':float(e0sum[c]),'E6':float(e6sum[c]),'Er':float(ersum[c]),
        'energy_retained_ratio':ratio(float(e6sum[c]),float(e0sum[c]))} for c in range(8)]
    return d1,d2,channels

def target_autocorrelation(rows):
    output=[];errors=[]
    for row in rows:
        try:
            y=np.asarray(np.load(row['envelope_path'],mmap_mode='r')).reshape(-1)[:row['common_samples']].astype(np.float64)
            assert len(y)==row['common_samples'] and np.isfinite(y).all()
            for seconds in (3,6,12):
                lag=seconds*64;r=None;n=max(0,len(y)-lag)
                if n>1:
                    a=y[:-lag];b=y[lag:]
                    if a.max()!=a.min() and b.max()!=b.min():
                        a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
                        if den>0:r=float(np.dot(a,b)/den);assert np.isfinite(r)
                output.append({'record':row['record'],'subject':row['subject'],'lag_seconds':seconds,'paired_samples':n,'r':r})
        except Exception as e:errors.append({'diagnostic':'D3','record':row['record'],'error':repr(e)})
    return output,errors

def ecdf(ax,values,**kwargs):
    a=np.array([v for v in values if v is not None and np.isfinite(v)],dtype=np.float64);a.sort()
    if len(a):ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',**kwargs)

def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and torch.cuda.device_count()==0
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=read(ROOT/'DIAGNOSTIC_PROTOCOL.json');assert p['frozen'] and p['phase']=='TRAIN_ONLY'
    for path,d in p['sources'].items():assert sha(path)==d,path
    started=time.time();dump(ROOT/'MATH_CHECK.json',math_check());rows=read(ROOT/'TRAIN_RECORDS.json')
    d1=[];d2=[];channels=[];errors=[]
    for seed in SEEDS:
        try:
            item=p['inputs'][str(seed)];cache=np.load(item['train_latent'],mmap_mode='r');assert cache.shape==(252454,8,192)
            assert Path(item['train_latent']).name=='train_latent.npy' and Path(item['train_latent']).stat().st_size==item['bytes']
            head=nn.Sequential(nn.Conv1d(8,1,32,padding='same'),nn.LeakyReLU(.01))
            cp=torch.load(item['checkpoint'],map_location='cpu',weights_only=True);head.load_state_dict(cp['head']);head.eval();del cp
        except Exception as e:
            errors.append({'diagnostic':'D1_D2_seed','seed':seed,'error':repr(e)});continue
        for i,row in enumerate(rows):
            try:
                a,b,c=process_record(seed,row,cache,head);d1.append(a);d2.append(b);channels.extend(c)
            except Exception as e:errors.append({'diagnostic':'D1_D2','seed':seed,'record':row['record'],'error':repr(e)})
            if (i+1)%25==0:status({'state':'CPU_TRAIN_STREAM','seed':seed,'records_visited':i+1,'time':time.time(),'GPU_used':False})
        print(json.dumps({'seed':seed,'records_visited':len(rows),'errors':len(errors)}),flush=True)
    d3,e=target_autocorrelation(rows);errors.extend(e)
    out=ROOT/'results01';out.mkdir(exist_ok=False)
    csvfile(out/'D1_PER_RECORD.csv',d1);csvfile(out/'D2_PER_RECORD.csv',d2);csvfile(out/'D2_PER_CHANNEL.csv',channels);csvfile(out/'D3_PER_RECORD.csv',d3)
    summaries=[]
    for seed in SEEDS:
        a=[r for r in d1 if r['seed']==seed];b=[r for r in d2 if r['seed']==seed];n=sum(r['overlap_pairs'] for r in a)
        summary={'seed':seed,'records':len(a),'windows':sum(r['windows'] for r in a),'overlap_pairs':n,
            'D2_energy_retained_ratio':ratio(sum(r['E6'] for r in b),sum(r['E0'] for r in b)),
            'D2_zero_energy_window_channels':sum(r['zero_energy_cases'] for r in b)}
        for direction in ('forward','reverse'):
            summary[direction+'_positive_fit_fraction']=ratio(sum(r[direction+'_positive_fit_pairs'] for r in a),n)
            summary[direction+'_held_aligned_raw_ratio']=ratio(sum(r[direction+'_aligned_held_MSE_sum'] for r in a),sum(r[direction+'_paired_raw_MSE_sum'] for r in a))
        summaries.append(summary)
    dump(out/'SUMMARY.json',{'time':time.time(),'seconds':time.time()-started,'seeds':summaries,'expected_records_per_seed':426,
        'errors':errors,'D3_records':len({r['record'] for r in d3}),'D3_shared_target_computed_once':True,
        'phase':'TRAIN_ONLY','threads':torch.get_num_threads(),'GPU_used':False,'new_NN_fits':False,'VAL_test_DTU_reads':False,
        'source_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'DIAGNOSTIC_PROTOCOL.json'),
        'scientific_goal_achieved_claimed':False,'interpretation':'descriptive affine consistency and latent energy compression; not predictive information, mechanism or generalization proof'})
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});colors=['#0072B2','#D55E00','#009E73']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for seed,color in zip(SEEDS,colors):
        a=[r for r in d1 if r['seed']==seed]
        ecdf(axes[0],[r['window_mean_drift_over_median_std'] for r in a],color=color,label=str(seed))
        ecdf(axes[1],[r['raw_overlap_NRMSE_mean'] for r in a],color=color,label=str(seed))
        for direction,ls in [('forward','-'),('reverse','--')]:
            ecdf(axes[2],[r[direction+'_held_aligned_raw_ratio'] for r in a],color=color,ls=ls,label=f'{seed} {direction}')
    axes[0].set_xlabel('Within-record SD(window mean) / median window SD');axes[0].set_xscale('symlog',linthresh=.1)
    axes[1].set_xlabel('Mean raw overlap normalized RMSE');axes[2].set_xlabel('Held-half aligned / paired raw MSE')
    axes[2].set_xscale('symlog',linthresh=.1);axes[2].axvline(1,color='grey',lw=1)
    for ax in axes:ax.set_ylabel('Record ECDF (not a confidence interval)')
    axes[0].legend(title='P seed');axes[2].legend(fontsize=8)
    fig.suptitle('D1: SOURCE-TRAIN P predictions only; alignment uses no labels; positive-fit coverage in SUMMARY');fig.tight_layout();fig.savefig(out/'D1_OVERLAP_AFFINE.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for seed,color in zip(SEEDS,colors):ecdf(ax,[r['energy_retained_ratio'] for r in d2 if r['seed']==seed],color=color,label=str(seed))
    ax.set_xlim(-.005,1.005);ax.set_xlabel('Centered latent energy retained by six-bin piecewise means');ax.set_ylabel('Record ECDF');ax.legend(title='P encoder seed')
    ax.set_title('D2: energy compression does not measure useful predictive information');fig.tight_layout();fig.savefig(out/'D2_SIX_BIN_ENERGY.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for seconds,color in zip((3,6,12),colors):ecdf(ax,[r['r'] for r in d3 if r['lag_seconds']==seconds],color=color,label=f'{seconds}s')
    ax.set_xlabel('Within-record target autocorrelation');ax.set_ylabel('Record ECDF');ax.legend(title='Fixed lag');ax.set_title('D3: shared TRAIN target once; not a conditional-history usefulness test')
    fig.tight_layout();fig.savefig(out/'D3_TARGET_AUTOCORRELATION.png',dpi=170);plt.close(fig)
    dump(out/'COMPLETED.json',{'time':time.time(),'records_D1_D2':len(d1),'records_D3':len({r['record'] for r in d3}),
        'errors':len(errors),'full_coverage':len(d1)==3*426 and len(d3)==3*426 and not errors,'GPU_used':False,'new_fits':False,'scientific_goal_complete':False})
    status({'state':'DIAGNOSTICS_COMPLETE','time':time.time(),'errors':len(errors),'GPU_used':False});print('CPU_TRAIN_DIAGNOSTICS_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    else:run()
