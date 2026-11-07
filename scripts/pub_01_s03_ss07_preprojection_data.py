"""D18 server-only original raw windows/random access and immutable Ccap anchor.

Only existing TRAIN/VAL assets, no downloads, new disk arrays or target fitting.
"""
import json,time,resource
from pathlib import Path
import numpy as np
import torch
from pub_01_s03_ss07_preprojection_model import BASE,OLD,CcapReadout
D8=BASE/'s03_finetuned_readout_transfer_20261101'

def read(p):return json.loads(Path(p).read_text())
def numpy_normalize(x,stat):
    v=(x-stat['mean'])/stat['std'];v[...,stat['constant']]=0;return v
def frozen_readout_assets(seed):
    folder=D8/f'seed{seed}'
    with np.load(folder/'TRAIN_TRANSFORMS.npz',allow_pickle=False) as f:
        statistics=[{k:f[f'{k}_{i}'].copy() for k in ('mean','std','constant')} for i in range(3)]
    with np.load(folder/'Ccap_lambda0.01_COEFFICIENTS.npz',allow_pickle=False) as f:
        coefficient=f['coef'].copy();bias=f['bias'].copy()
    calibration=read(folder/'P_FT_BASELINE.json')['calibration']
    return statistics,coefficient,bias,calibration

class RawPhase:
    def __init__(self,seed,phase,ram_raw=True,lazy_anchor=False):
        start=time.time();cpu=time.process_time();assert phase in ('train','val')
        self.seed=seed;self.phase=phase;self.rows=read(D8/'PROTOCOL.json')['records'][phase]
        joined=[r for r in read(BASE/'s03_incremental_continuity_20261101/SOURCE_FAMILY_JOIN.json')['records'] if r['phase']==phase]
        join={(r['phase'],r['record']):r for r in joined};assert len(join)==len(joined)==len(self.rows)
        enriched=[]
        for row in self.rows:
            item=join[(phase,row['record'])]
            assert all(item[k]==row[k] for k in ('subject','story','common_samples','windows'))
            enriched.append(dict(row,source_family=item['source_family'],family_mapping_basis=item['family_mapping_basis']))
        self.rows=enriched;assert len({r['source_family'] for r in self.rows})=={'train':37,'val':5}[phase]
        self.n=sum(r['windows'] for r in self.rows);assert self.n=={'train':252454,'val':30500}[phase]
        self.mask=read(D8/'PROTOCOL.json')['mask'] if phase=='val' else None
        self.local=np.empty(self.n,dtype='i4');self.record_id=np.empty(self.n,dtype='i4')
        offset=0
        for index,row in enumerate(self.rows):
            assert row['phase']==phase and row['offset']==offset and row['windows']==(row['common_samples']-192)//96+1
            count=row['windows'];self.local[offset:offset+count]=np.arange(count);self.record_id[offset:offset+count]=index;offset+=count
        self.statistics,self.coefficient,self.bias,self.calibration=frozen_readout_assets(seed)
        with np.load(OLD/'prepared/train_only_normalization.npz',allow_pickle=False) as f:
            # Preserve original dtype/broadcasting, including subtraction before
            # final stack.astype(float32). Do not silently normalize differently.
            self.mean=f['mean'].copy();self.std=f['std'].copy()
        assert np.isfinite(self.mean).all() and np.all(self.std>0)
        self.y=np.load(OLD/f'envelope_readout_seed1024/{phase}_envelope.npy',mmap_mode='r')
        assert len(self.y)==self.n
        self.anchor=np.empty((self.n,192),dtype='f8');self._lazy_anchor=lazy_anchor
        self._anchor_ready=np.zeros(self.n,dtype=bool) if lazy_anchor else None
        if not lazy_anchor:self._anchor_from_existing_cache()
        self.raw={};self.raw_RAM_bytes=0
        for index,row in enumerate(self.rows):
            raw=np.load(row['eeg_path'],mmap_mode='r');value=raw.T if raw.shape[0]==64 else raw
            assert value.ndim==2 and value.shape[1]==64 and len(value)>=row['common_samples']
            if ram_raw:
                value=np.array(value[:row['common_samples']],dtype='f4',copy=True,order='K');self.raw_RAM_bytes+=value.nbytes
            self.raw[index]=value
        self.setup=dict(phase=phase,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,raw_RAM_bytes=self.raw_RAM_bytes,anchor_bytes=self.anchor.nbytes,records=len(self.rows),windows=self.n,scientific_new_disk_arrays=0)
    def ensure_anchor(self,indices):
        if not self._lazy_anchor:return
        ids=np.unique(np.asarray(indices,dtype='i8').reshape(-1));todo=ids[~self._anchor_ready[ids]]
        if not len(todo):return
        folder=D8/f'seed{self.seed}';z=np.load(folder/f'{self.phase}_latent.npy',mmap_mode='r');p=np.load(folder/f'{self.phase}_P.npy',mmap_mode='r')
        st=self.statistics;cal=self.calibration
        u=numpy_normalize(np.array(z[todo],dtype='f8').reshape(len(todo),1536),st[0])
        base=cal['a']*np.array(p[todo],dtype='f8')+cal['b'];answer=base.copy();eligible=self.local[todo]>=4
        if eligible.any():
            uu=u[eligible];v=numpy_normalize(base[eligible],st[1]);w=numpy_normalize(uu*uu,st[2])
            features=np.concatenate((uu/np.sqrt(1536),v/np.sqrt(192),w/np.sqrt(1536)),1)
            answer[eligible]+=features@self.coefficient+self.bias
        assert np.isfinite(answer).all();self.anchor[todo]=answer;self._anchor_ready[todo]=True
        del z,p
    def _anchor_from_existing_cache(self):
        folder=D8/f'seed{self.seed}';z=np.load(folder/f'{self.phase}_latent.npy',mmap_mode='r');p=np.load(folder/f'{self.phase}_P.npy',mmap_mode='r')
        assert z.shape==(self.n,8,192) and p.shape==(self.n,192) and z.dtype==np.float32
        st=self.statistics;cal=self.calibration
        for row in self.rows:
            o=row['offset'];end=o+row['windows']
            for first in range(o,end,1024):
                last=min(first+1024,end);raw=np.asarray(z[first:last],dtype='f8').reshape(last-first,1536)
                u=numpy_normalize(raw,st[0]);base=cal['a']*np.asarray(p[first:last],dtype='f8')+cal['b'];self.anchor[first:last]=base
                valid_first=max(first,o+4)
                if valid_first<last:
                    q=valid_first-first;uu=u[q:];v=numpy_normalize(base[q:],st[1]);w=numpy_normalize(uu*uu,st[2])
                    features=np.concatenate((uu/np.sqrt(1536),v/np.sqrt(192),w/np.sqrt(1536)),1)
                    self.anchor[valid_first:last]+=features@self.coefficient+self.bias
        assert np.isfinite(self.anchor).all()
        del z,p
    def torch_readout(self):
        return CcapReadout(self.statistics,self.coefficient,self.bias,self.calibration)
    def sample_raw(self,indices,device):
        ids=np.asarray(indices,dtype='i8').reshape(-1);assert len(ids)>0 and np.all((ids>=0)&(ids<self.n))
        pieces=[]
        for idx in ids:
            record=int(self.record_id[idx]);local=int(self.local[idx]);start=local*96
            value=np.asarray(self.raw[record][start:start+192],np.float32)
            assert value.shape==(192,64)
            pieces.append((value-self.mean)/self.std)
        # Same original stack/astype construction, not np.asarray(list).
        value=np.stack(pieces).astype(np.float32)
        assert np.isfinite(value).all()
        return torch.from_numpy(value).to(device)
    def batch(self,indices,arm,device,older=False,need_target=True):
        ids=np.asarray(indices,dtype='i8');assert ids.ndim==1 and 0<len(ids)<=64
        valid=self.local[ids]>=9;effective=ids[valid]
        self.ensure_anchor(ids);x=self.sample_raw(ids,device);anchor=torch.from_numpy(self.anchor[ids].copy()).to(device)
        mask=torch.from_numpy(valid).to(device);history=None;prior=None
        if len(effective) and arm in ('H_history','P_prediction_history'):
            lags=(9,7) if older else (5,3);ii=np.stack((effective-lags[0],effective-lags[1]),1)
            assert np.all(self.record_id[ii]==self.record_id[effective,None])
            assert np.all(self.local[ii]*96+192<=self.local[effective,None]*96-96)
            if arm=='H_history':history=self.sample_raw(ii.reshape(-1),device).reshape(len(effective),2,192,64)
            else:
                self.ensure_anchor(ii);prior=torch.from_numpy(self.anchor[ii].copy()).to(device)
        target=torch.from_numpy(np.array(self.y[ids],dtype='f4').astype('f8').reshape(len(ids),192)).to(device) if need_target else None
        return dict(x=x,anchor=anchor,valid=mask,history_raw=history,prior=prior),target
    def older_memory(self,indices,device):
        # For selected-H paired evaluation reuse the SAME current query/x/anchor.
        ids=np.asarray(indices,dtype='i8');effective=ids[self.local[ids]>=9]
        ii=np.stack((effective-9,effective-7),1)
        assert np.all(self.record_id[ii]==self.record_id[effective,None])
        assert np.all(self.local[ii]*96+192<=self.local[effective,None]*96-96)
        return self.sample_raw(ii.reshape(-1),device).reshape(len(effective),2,192,64) if len(effective) else None
    def describe_indices(self,indices):
        return [dict(global_index=int(i),record=self.rows[int(self.record_id[i])]['record'],subject=self.rows[int(self.record_id[i])]['subject'],story=self.rows[int(self.record_id[i])]['story'],local_index=int(self.local[i]),eligible=bool(self.local[i]>=9)) for i in indices]
