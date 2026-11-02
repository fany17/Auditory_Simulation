"""Coordinator verification on 2203: real TRAIN gradients via autograd, no NN.

No production imports. Reconstruct fixed Ccap from original caches, synthesize
whole records with torch.fold, differentiate scalar losses, and check every
record/seed. Also audit block identities/aggregation and prior independent VAL.
"""
import csv, hashlib, json, resource, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

BASE = Path('/home/fanyu/auditory_simulation_m6a/pub_01')
ROOT = BASE / 's03_full_ft_error_decomposition_20261102'
D8 = BASE / 's03_finetuned_readout_transfer_20261101'
D12 = BASE / 's03_full_ft_history_20261102'
OUT = ROOT / 'coordinator_review'
SEEDS = (1024, 2026, 31415)
ARMS = ('H_full', 'C_match', 'P_prior')
CHECKS = {}

def read(p): return json.loads(Path(p).read_text())
def rows(p): return list(csv.DictReader(Path(p).open()))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, x):
    with Path(p).open('x') as f: json.dump(x, f, indent=2, allow_nan=False)
def close(group, a, b, atol=1e-10, rtol=1e-8):
    a, b = float(a), float(b)
    assert np.isfinite([a,b]).all(), (group,a,b)
    error = abs(a-b)
    c = CHECKS.setdefault(group, dict(count=0, max_abs=0.))
    c['count'] += 1; c['max_abs'] = max(c['max_abs'], error)
    assert error <= atol + rtol*abs(b), (group,a,b,error)
def transform(x, s):
    x = (x-s['mean'])/s['std']; x[...,s['constant']] = 0
    return x
def normdot(gs):
    return {a+b:float(torch.sum(gs[a]*gs[b])) for a,b in
            [('M','M'),('F','F'),('R','R'),('M','F'),('M','R'),('F','R')]}

def audit_gradients(seed, protocol, support):
    folder=D8/f'seed{seed}'; rr=protocol['records']['train']
    cal=read(folder/'P_FT_BASELINE.json')['calibration']
    with np.load(folder/'TRAIN_TRANSFORMS.npz') as z:
        stats=[{k:z[f'{k}_{i}'].copy() for k in ['mean','std','constant']} for i in range(3)]
    with np.load(folder/'Ccap_lambda0.01_COEFFICIENTS.npz') as c:
        coef,bias=c['coef'].copy(),c['bias'].copy()
    latent=np.load(folder/'train_latent.npy',mmap_mode='r')
    pred=np.load(folder/'train_P.npy',mmap_mode='r')
    reported={r['record']:r for r in rows(ROOT/f'results/seed{seed}/GRADIENT_RECORDS.csv')}
    mask=support['train']; total=sum(r['windows'] for r in rr)
    hann=torch.tensor(np.hanning(192),dtype=torch.float64)
    result=[]
    for row in rr:
        off,n=row['offset'],row['windows']; length=row['common_samples']
        z=transform(np.asarray(latent[off:off+n],dtype='f8').reshape(n,1536),stats[0])
        a=cal['a']*np.asarray(pred[off:off+n],dtype='f8')+cal['b']
        features=np.concatenate((z[4:]/np.sqrt(1536),transform(a[4:],stats[1])/np.sqrt(192),
            transform(z[4:]*z[4:],stats[2])/np.sqrt(1536)),axis=1)
        a[4:] += features@coef+bias
        raw=np.load(row['envelope_path'],mmap_mode='r').reshape(-1)
        y=torch.tensor(np.asarray(raw,dtype='f4').astype('f8')[:length])
        xx=torch.tensor(a,requires_grad=True)
        synth=lambda t:F.fold(t.T.unsqueeze(0),(1,length),(1,192),stride=(1,96)).reshape(-1)
        weight=synth(hann.repeat(n,1)); valid=weight>1e-8
        wave=synth(xx*hann)/torch.where(valid,weight,torch.ones_like(weight))
        ywin=y.unfold(0,192,96)
        losses={'M':torch.sum((xx-ywin)**2)/(total*192)}
        termsF=[];termsR=[]
        bs=[b for b in mask['blocks'] if b['record']==row['record'] and b['eligible']]
        for b in bs:
            s,t=b['start'],b['stop']; k=valid[s:t]
            v=wave[s:t][k]; target=y[s:t][k]
            vc=v-v.mean(); yc=target-target.mean()
            termsF.append(torch.mean((v-target)**2))
            termsR.append(-torch.dot(vc,yc)/(torch.linalg.vector_norm(vc)*torch.linalg.vector_norm(yc)))
        divisor=mask['people']*mask['person_blocks'][row['subject']]
        losses['F']=torch.stack(termsF).sum()/divisor
        losses['R']=torch.stack(termsR).sum()/divisor
        gs={}
        for label in ['M','F','R']:
            g=torch.autograd.grad(losses[label],xx,retain_graph=label!='R')[0]
            g[:9]=0; gs[label]=g
        stat=normdot(gs); expected=reported[row['record']]
        for key,v in stat.items():close('autograd_record_moments',v,expected[key],atol=1e-15,rtol=2e-8)
        for left,right in [('M','F'),('M','R'),('F','R')]:
            c=stat[left+right]/np.sqrt(stat[left+left]*stat[right+right])
            close('autograd_record_cosines',c,expected[f'cos_{left}_{right}'],atol=1e-9,rtol=1e-8)
        result.append(stat)
    sums={k:sum(r[k] for r in result) for k in result[0]}
    reported=read(ROOT/f'results/seed{seed}/GRADIENT_SUMMARY.json')
    for k,v in sums.items():close('autograd_seed_moments',v,reported[k],atol=1e-15,rtol=2e-8)
    for l,r in [('M','F'),('M','R'),('F','R')]:
        close('autograd_seed_cosines',sums[l+r]/np.sqrt(sums[l+l]*sums[r+r]),reported[f'cos_{l}_{r}'])
    return dict(seed=seed,records=len(rr),windows=total,learnable_coordinates=(total-9*len(rr))*192)

def audit_rows(protocol):
    count=0
    for seed in SEEDS:
        for arm in ARMS:
            for phase in ['train','val']:
                folder=ROOT/f'results/seed{seed}'
                bb=rows(folder/f'{phase}_{arm}_BLOCKS.csv')
                assert len(bb)==protocol['support'][phase]['blocks']
                for row in bb:
                    assert row['projection_defined']=='True' and row['Pearson_defined']=='True'
                    d={k:float(v) for k,v in row.items() if k in ['var_a','var_e','var_y','cov_a_y','cov_e_y','mean_a','mean_d','mean_y','alpha','r_anchor','r_new','MSE_anchor','MSE_new','MSE_bias','MSE_parallel','MSE_orthogonal']}
                    a=d['alpha'];va=d['var_a'];ve=d['var_e'];vy=d['var_y'];ce=d['cov_e_y'];ca=d['cov_a_y']
                    r=((1+a)*ca+ce)/np.sqrt(((1+a)**2*va+ve)*vy)
                    mse=2*d['mean_d']*(d['mean_a']-d['mean_y'])+d['mean_d']**2+(a*a+2*a)*va-2*a*ca+ve-2*ce
                    close('block_r_identity',r,d['r_new']);close('block_MSE_identity',mse,d['MSE_new']-d['MSE_anchor'])
                    count+=1
                people=rows(folder/f'{phase}_{arm}_PERSON.csv')
                for person in people:
                    own=[r for r in bb if r['subject']==person['subject']]
                    for k in ['r_anchor','r_new','delta_r','MSE_anchor','MSE_new','delta_MSE','MSE_bias','MSE_parallel','MSE_orthogonal','r_anchor_normalization_delta','r_orthogonal_delta']:
                        close('person_aggregation',np.mean([float(r[k]) for r in own]),person[k])
                summary=next(r for r in read(folder/'SUMMARY.json') if r['phase']==phase and r['arm']==arm)
                for k in ['r_anchor','r_new','delta_r','MSE_anchor','MSE_new','delta_MSE','MSE_bias','MSE_parallel','MSE_orthogonal','r_anchor_normalization_delta','r_orthogonal_delta']:
                    close('phase_aggregation',np.mean([float(r[k]) for r in people]),summary[k])
                if phase=='val':
                    old=read(D12/f'coordinator_last_review/{arm}_seed{seed}.json')
                    close('prior_independent_VAL_r',summary['r_new'],old['phases']['val']['r'],atol=2e-6)
                    close('prior_independent_VAL_MSE',summary['MSE_new'],old['phases']['val']['MSE'],atol=2e-6)
    return count

def main():
    start,cpu=time.time(),time.process_time(); torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    write(OUT/'STARTED.json',dict(source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'PROTOCOL.json'),
        new_NN=0,optimizer_updates=0,statistical_recomputation='All3TRAINseed scalar losses and autograd; all fixed block identities/aggregation',
        prediction_comparison='Reuses independent D12 last19 VAL replay; no repeat NN',device='CPU',threads=2))
    p=read(ROOT/'PROTOCOL.json');m=read(ROOT/'SUPPORT_MASK.json')
    assert read(ROOT/'COMPLETED.json')['effective_residual_NN_windows']==2508111
    gradients=[]
    for seed in SEEDS:
        gradients.append(audit_gradients(seed,p,m));print(json.dumps(gradients[-1]),flush=True)
    blocks=audit_rows(p)
    result=dict(passed=True,source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'PROTOCOL.json'),checks=CHECKS,
        gradients=gradients,blocks=blocks,new_NN=0,optimizer_updates=0,learning_model_fits=0,
        limits='Independent real gradient and reported-moment identities; no second TRAIN nonzero network replay, no parameter-gradient conclusion',
        CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        goal_complete=False)
    write(OUT/'REVIEW.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
