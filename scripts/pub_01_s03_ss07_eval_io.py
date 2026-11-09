"""Gated final readers/model loaders. Nothing executes at import time."""
from pathlib import Path
import shutil
import numpy as np
import torch
from pub_01_s03_ss07_baseline import ROOT,OLD,PREPARED,read,dump,sha,native,cfg,disk_guard
from pub_01_s03_ss07_final_manifest import require_ready
from pub_01_s03_ss07_train import NativeHead,make
from pub_01_s03_ss07_finetune import Decoder
from pub_01_s03_ss07_incremental import summary


def save_array(path,value):
    """Server-only arrays, exclusive writes and disk reserve. No overwrite."""
    assert Path(path).is_relative_to(ROOT);disk_guard()
    assert shutil.disk_usage(ROOT).free-value.nbytes>=20*1024**3
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as output:np.save(output,value,allow_pickle=False)


def planned_array_budget(manifest,rows):
    windows=sum((r['common_samples']-192)//96+1 for r in rows);samples=sum(r['common_samples'] for r in rows)
    main=len(manifest['main_model_ids']);extra=len(set(manifest['exploratory_family_ids'])-{manifest['primary']})
    # Five P test feature caches; float64 fused main/6 diagnostics; three-seed
    # non-primary exploration; five-seed external fused1080x3200 predictions.
    planned=5*windows*8*192*4+5*(main+6)*samples*8+3*extra*samples*8+5*main*1080*3200*8
    existing=sum(p.stat().st_size for p in ROOT.rglob('*.npy'))
    assert existing+planned<=25*1024**3,'new array cache/output plan exceeds25GiB'
    assert shutil.disk_usage(ROOT).free-planned>=20*1024**3,'plan would violate20GiB reserve'
    return {'current_arrays_bytes':existing,'planned_additional_upper_bound_bytes':planned,
        'array_cap_bytes':25*1024**3,'reserve_bytes':20*1024**3}


class FinalModels:
    def __init__(self,seed,device):
        self.manifest,self.ready=require_ready()  # before checkpoint/EEG loads
        assert seed in self.manifest['seeds'];self.seed=seed;self.device=device
        precision=read(ROOT/'INCREMENTAL_PROTOCOL.json')['precision']
        torch.set_num_threads(2);torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=precision['TF32_matmul']
        torch.backends.cudnn.allow_tf32=precision['TF32_cudnn']
        self.mod=native();original=self.mod.TransformerCLIP(cfg())
        source=self.manifest['encoders'][str(seed)]
        saved=torch.load(source['path'],map_location='cpu',weights_only=True)
        assert saved['seed']==seed;original.load_state_dict(saved['model'],strict=True)
        self.encoder=original.eeg.to(device).eval();self.loaded={}

    def load(self,condition):
        assert condition!='P_EMA','P_EMA must reuse exact native predictions'
        if condition in self.loaded:return self.loaded[condition]
        spec=self.manifest['entries'][f'{condition}:{self.seed}']
        saved=torch.load(spec['path'],map_location='cpu',weights_only=True)
        assert saved['seed']==self.seed and saved['epoch']==spec['selected_epoch']
        if spec['kind']=='native_P':
            model=NativeHead();model.head.load_state_dict(saved['head'],strict=True)
        elif spec['kind']=='P_FT':
            original=self.mod.TransformerCLIP(cfg());model=Decoder(original.eeg)
            model.load_state_dict(saved['model'],strict=True)
        else:
            model=make(spec['config'],self.seed);model.load_state_dict(saved['model'],strict=True)
        model.to(self.device).eval();self.loaded[condition]=model;return model

    @torch.no_grad()
    def encode(self,windows):
        assert windows.ndim==3 and windows.shape[1:]==(192,64)
        output=[]
        for first in range(0,len(windows),256):
            x=torch.from_numpy(windows[first:first+256]).to(self.device)
            z=self.encoder(x).transpose(1,2)
            assert z.shape==(len(x),8,192) and torch.isfinite(z).all()
            output.append(z.cpu().numpy())
        return np.concatenate(output)

    @torch.no_grad()
    def finetuned_windows(self,windows,global_offset=0):
        from pub_01_s03_ss07_eval_common import aligned_chunks
        decoder=self.load('P_FT');output=[]
        for first,last in aligned_chunks(len(windows),global_offset):
            x=torch.from_numpy(windows[first:last]).to(self.device);prediction=decoder(x)
            assert prediction.shape==(last-first,1,192)
            if not torch.isfinite(prediction).all():raise RuntimeError('nonfinite fine-tuned output')
            output.append(prediction[:,0].cpu().numpy())
        return np.concatenate(output)


class SparrRecords:
    def __init__(self,models):
        require_ready();self.models=models;self.seed=models.seed;self.device=models.device
        mask=read(ROOT/'SPARR_TEST_MASK.json');self.rows=mask['records']
        self.masks={m['record']:m for m in mask['record_masks']}
        all_rows=read(OLD/'envelope_readout_seed1024/manifest_frozen.json')['records']
        self.all_rows={r['record']:r for r in all_rows};assert len(self.all_rows)==619
        baseline=ROOT/f'baseline/head_seed{self.seed}'
        cache=read(baseline/'cache_manifest.json');self.baseline=baseline;self.train_val=cache['records']
        assert cache['encoder_sha256']==models.manifest['encoders'][str(self.seed)]['sha256']
        scaling=np.load(PREPARED/'train_only_normalization.npz');self.mean,self.std=scaling['mean'],scaling['std']
        self.folder=ROOT/f'final_latent/seed{self.seed}';self.folder.mkdir(parents=True,exist_ok=False)
        self.paths={};self.phase_arrays={};self.errors={}

    def raw_windows(self,row):
        assert row['record'] in self.all_rows
        raw=np.load(row['eeg_path'],mmap_mode='r');eeg=raw.T if raw.shape[0]==64 else raw
        assert eeg.shape[1]==64 and len(eeg)>=row['common_samples']
        count=(row['common_samples']-192)//96+1
        return np.stack([(np.asarray(eeg[i*96:i*96+192],np.float32)-self.mean)/self.std for i in range(count)]).astype(np.float32)

    def encode_test(self):
        for row in self.rows:
            try:
                assert Path(row['record']).name==row['record']
                windows=self.raw_windows(row);latent=self.models.encode(windows)
                path=self.folder/(row['record']+'.npy');save_array(path,latent);self.paths[row['record']]=str(path)
            except Exception as exc:self.errors[row['record']]=repr(exc)
        dump(self.folder/'cache_manifest.json',{'records':self.paths,
            'encoder_sha256':self.models.manifest['encoders'][str(self.seed)]['sha256'],
            'checkpoint_manifest_sha256':sha(ROOT/'FINAL_CHECKPOINT_MANIFEST.json'),
            'only_original144_test_records':True,'arrays_stay_on_server':True,
            'read_failures':self.errors,'complete':len(self.paths)==144})

    def finetuned_predictions(self):
        from pub_01_s03_ss07_eval_common import predict_raw_phase
        result={}
        for phase in ['test_seen','test_heldout']:
            rows=[r for r in self.rows if r['phase']==phase];offset=0
            for row in rows:
                assert row['phase_offset']==offset;offset+=(row['common_samples']-192)//96+1
            inputs=((row['record'],self.raw_windows(row)) for row in rows)
            result.update(predict_raw_phase(self.models.load('P_FT'),inputs,self.device))
        return result

    def latent(self,record):
        if record in self.paths:array=np.load(self.paths[record],mmap_mode='r')
        else:
            meta=self.train_val[record];phase=meta['phase'];assert phase in ['train','val']
            if phase not in self.phase_arrays:self.phase_arrays[phase]=np.load(self.baseline/f'{phase}_latent.npy',mmap_mode='r')
            array=self.phase_arrays[phase][meta['offset']:meta['offset']+meta['windows']]
        assert array.ndim==3 and array.shape[1:]==(8,192) and np.isfinite(array).all()
        return torch.from_numpy(np.array(array,copy=True)).to(self.device)

    def neutral_train_mean(self):
        array=np.load(self.baseline/'train_latent.npy',mmap_mode='r');assert len(array)==252454
        total=np.zeros(48,dtype=np.float64)
        for first in range(0,len(array),1024):
            x=torch.from_numpy(np.array(array[first:first+1024],copy=True)).to(self.device)
            with torch.no_grad():total+=summary(x).cpu().numpy().astype(np.float64).sum(0)
        result=total/len(array)
        save_array(self.folder/'train_only_neutral_mean.npy',result)
        return torch.tensor(result,dtype=torch.float32,device=self.device)
