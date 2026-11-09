"""Frozen ds004408 external migration after shared offline gain adaptation, 2203 only.

Actions inventory/freeze never run a real NN. Seed inference requires explicit
coordinator approval binding the complete EVALUATION_PROTOCOL fingerprint.
"""
import argparse,csv,hashlib,json,math,os,resource,shutil,sys,time,traceback
import importlib.metadata
from pathlib import Path
import numpy as np
import torch
from torch import nn
from pub_01_s03_ss07_openneuro_transport import ROOT
import pub_01_s03_ss07_openneuro_adapter_v2 as adapter
import pub_01_s03_ss07_openneuro_metrics as metric
import pub_01_s03_ss07_openneuro_scoring as scoring

BASE=ROOT.parent;P=BASE/'s03_incremental_continuity_20261101'
D8=BASE/'s03_finetuned_readout_transfer_20261101';D9=BASE/'s03_physical_objective_20261101'
sys.path.insert(0,str(P/'input'))
import pub_01_s03_ss07_finetune as ft
sys.path.insert(0,str(ft.OLD/'input'))  # Native train_v2 imports its sibling model definitions.
sys.path.insert(0,str(BASE/'s03_current_readout_internal_audit_20261101/input'))
import pub_01_s03_ss07_generalization_core as old_core
SEEDS=(1024,2026,31415);ROLES=('P_FT_star','Ccap','W','F');START=time.time()


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):
    p=Path(p);assert p.resolve().is_relative_to(ROOT.resolve())
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def guard(full=False):
    assert time.time()-START<14400,'4h inference/scoring stage cap'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3
    assert shutil.disk_usage(ROOT).free>=20*1024**3
    if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<=4*1024**3
    if full:
        root_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
        arrays=sum(p.stat().st_size for directory in list(BASE.glob('s03_*_20261101'))+[ROOT] for p in directory.rglob('*.npy'))
        assert root_bytes<=3*1024**3 and arrays<=28*1024**3
        return dict(root_bytes=root_bytes,all_SS07_NPY_bytes=arrays,free_bytes=shutil.disk_usage(ROOT).free)


def paths(seed):
    folder=D8/f'seed{seed}'
    return dict(norm=adapter.NORM,transforms=folder/'TRAIN_TRANSFORMS.npz',calibration=folder/'P_FT_BASELINE.json',
        Ccap=folder/'Ccap_lambda0.01_COEFFICIENTS.npz',W=D9/f'seed{seed}/W_COEFFICIENTS.npz',
        F=D9/f'seed{seed}/F_COEFFICIENTS.npz')


def frozen(seed):
    pp=paths(seed);s=np.load(pp['transforms'],allow_pickle=False)
    stats=[{key:s[f'{key}_{i}'] for key in ('mean','std','constant')} for i in range(3)]
    calibration=read(pp['calibration'])['calibration'];assert calibration['a']>0
    coefficients={}
    for role in ROLES[1:]:
        f=np.load(pp[role],allow_pickle=False);co={k:np.asarray(f[k]) for k in ['coef','bias']}
        assert co['coef'].shape==(3264,192) and co['bias'].shape==(192,)
        assert all(np.isfinite(v).all() for v in co.values());coefficients[role]=co
    return calibration,stats,coefficients


def source_assets():
    assert not torch.cuda.is_initialized()
    old=read(D8/'PROTOCOL.json');pins={};checkpoints={};checks=[];rng=np.random.default_rng(20261102)
    for seed in SEEDS:
        cp=old['checkpoints'][str(seed)];path=Path(cp['checkpoint'])
        assert sha(path)==cp['checkpoint_sha256'] and cp['selected_epoch']==4
        checkpoints[str(seed)]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,epoch=4)
        pins.update({str(p):sha(p) for p in paths(seed).values()});pins[str(path)]=sha(path)
        cal,stats,co=frozen(seed)
        z=(stats[0]['mean']+rng.normal(size=(11,1536))*stats[0]['std']).reshape(11,8,192).astype('f4')
        raw=((stats[1]['mean']+rng.normal(size=(11,192))*stats[1]['std']-cal['b'])/cal['a']).astype('f4')
        oldstats=[stats[0],stats[1],None,None,None,stats[2]]
        for role in ROLES[1:]:
            coef=co[role]['coef'];split=dict(BZ=coef[:1536],BP=coef[1536:1728],BQ=coef[1728:],bias=co[role]['bias'])
            got=scoring.readout(z,raw,cal,stats,co[role]);reference=old_core.predict(z,raw,cal,oldstats,split)
            error=float(abs(got-reference).max());assert error<=1e-8
            checks.append(dict(seed=seed,role=role,synthetic_only=True,dense_vs_frozen_split_max_abs=error))
    legacy=read(ROOT/'LEGACY_ASSET_IDENTITY_CHECK.json');assert legacy['passed'] and len(legacy['assets_checked'])==19
    assert legacy['SOURCE_ASSETS_sha256']==sha(ROOT/'SOURCE_ASSETS.json')
    for item in legacy['assets_checked']:
        assert pins[item['path']]==item['sha256'] and all(r['sha256']==item['sha256'] for r in item['historical_references'])
    for path,digest in legacy['historical_manifests'].items():assert sha(path)==digest
    pins.update(legacy['historical_manifests']);pins[str(ROOT/'LEGACY_ASSET_IDENTITY_CHECK.json')]=sha(ROOT/'LEGACY_ASSET_IDENTITY_CHECK.json')
    native=ft.native()
    pins[str(Path(native.__file__))]=sha(native.__file__)
    pins[str(ft.OLD/'training_seed1024_attempt04/config_frozen.json')]=sha(ft.OLD/'training_seed1024_attempt04/config_frozen.json')
    for _,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path and str(path).endswith('.py') and Path(path).resolve().is_relative_to(BASE):pins[str(Path(path).resolve())]=sha(path)
    _,_,amplitude=adapter.source_normalization()
    return dict(checkpoints=checkpoints,source_pins=pins,precision=old['precision'],readout_bridges=checks,
        historical_asset_identities_verified=19,runtime=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,mne=importlib.metadata.version('mne'),scipy=importlib.metadata.version('scipy'),brain_pipe=importlib.metadata.version('brain-pipe')),
        original_TRAIN_norm_sha256=adapter.NORM_SHA,original_TRAIN_RMS=amplitude,
        new_learning_model_fits=0,new_target_data_statistics='One unlabeled scalar gain per successful record',real_NN_forwards=0)


def inventory():
    value=source_assets();value['scoring_synthetic_check']=scoring.synthetic_check()
    dump(ROOT/'SOURCE_ASSETS_V2.json',value);print('SOURCE_ASSETS_AND_NINE_READOUT_BRIDGES_PASS_NO_REAL_NN',flush=True)


def freeze():
    assert not torch.cuda.is_initialized() and not (ROOT/'EVALUATION_PROTOCOL.json').exists()
    completed=read(ROOT/'EEG_PREPARE_COMPLETED_V2.json')
    assert completed['passed'] and completed['records']==380 and completed['failures']==[]
    assert not (ROOT/'EEG_PREPARE_FAILURE_V2.json').exists()
    assets=read(ROOT/'SOURCE_ASSETS_V2.json')
    for path,digest in assets['source_pins'].items():assert sha(path)==digest,path
    headers=read(ROOT/'HEADERS_AND_SIZES_V2.json')
    for digest in {r['git_blob_sha'] for r in headers['channel_table_mapping']}:
        data=(ROOT/'metadata/channel_tables'/f'{digest}.tsv').read_bytes()
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==digest
    audio=read(ROOT/'AUDIO_PREPARE_COMPLETED.json')['audios'];targets={}
    for item in audio:
        path=ROOT/'targets'/Path(item['path']).with_suffix('.npy').name
        assert sha(path)==item['target_sha256'];targets[path.stem[-2:]]=path
    old=read(ROOT/'PREPROCESSING_TRANSPORT_HANDOFF.json')['old_record_rows']
    new=[json.loads(line) for line in (ROOT/'EEG_PREPARE_RECORDS_V2.jsonl').read_text().splitlines()]
    allrows=sorted(old+new,key=lambda r:r['record']);assert len(allrows)==380 and len({r['record'] for r in allrows})==380
    rows=[];offset=0
    for item in allrows:
        path=Path(item['processed_path']);assert sha(path)==item['processed_sha256']
        x=np.load(path,mmap_mode='r',allow_pickle=False);target=targets[item['run']]
        assert x.shape==(64,item['mask']['length']) and x.dtype==np.float32 and x.flags.c_contiguous and np.isfinite(x).all()
        actual=metric.mechanical_mask(np.load(target,allow_pickle=False)[:x.shape[1]])
        assert actual==item['mask']
        row=dict(subject=item['subject'],record=item['record'],run=item['run'],EEG_path=str(path),EEG_sha256=item['processed_sha256'],
            target_path=str(target),target_sha256=sha(target),offset=offset,**actual)
        rows.append(row);offset+=row['windows']
    assert offset==45429 and sum(sum(b['eligible'] for b in r['blocks']) for r in rows)==988
    pins=dict(assets['source_pins'])
    for p in [Path(__file__),Path(scoring.__file__),ROOT/'SOURCE_ASSETS_V2.json',ROOT/'input/pub_01_s03_ss07_openneuro_statistics.py',ROOT/'STATISTICS_PROTOCOL.json',ROOT/'PLAN.md',ROOT/'AMENDMENT_GAIN_CALIBRATION.md',
        ROOT/'TRANSPORT_AMENDMENT.md',ROOT/'PREPROCESSING_PROTOCOL.json',ROOT/'PREPROCESSING_PROTOCOL_V2.json',
        ROOT/'EEG_PREPARE_COMPLETED_V2.json',ROOT/'COORDINATOR_GAIN_CALIBRATION_DECISION_20261102.json',
        ROOT/'HUMAN_AUTHORIZATION.json',ROOT/'SCORING_SYNTHETIC_CHECK.json',ROOT/'SOURCE_INDEPENDENCE.json']:
        pins[str(p)]=sha(p)
    dump(ROOT/'EVALUATION_PROTOCOL.json',dict(frozen_before_any_real_NN=True,rows=rows,source_pins=pins,
        seeds=SEEDS,roles=ROLES,windows=offset,people=19,eligible_blocks=988,checkpoints=assets['checkpoints'],
        precision=assets['precision'],raw_Decoder_bridge_absolute=1e-6,readout_bridge_absolute=1e-8,
        primary=[['Ccap','P_FT_star'],['F','W']],statistics=dict(people_unit=True,equal_block_then_person_then_seed=True,
            signflip_statistic='abs(mean(d_person))',all_2_to_n=True,inclusive_tie_absolute=1e-15,Holm=2,
            bootstrap_replicates=20000,bootstrap_seed=20261102,quantiles=[.0125,.9875]),
        offline_full_record_gain=True,physical_unit_status='UNVERIFIED',new_model_fits=0,
        unlabeled_record_gain_statistics=380,boundary='first4 windows use calibrated P_FT_star',
        main_readout_and_Hann='FP64',diagnostic='W/F full FP32 readout plus FP32 Hann, same Z/P; not selected',
        mechanical_input_payload_checks=dict(EEG=380,channel_TSV=119,audio_target=20),resource=guard(True)))
    print('FINAL_DATA_MODEL_MASK_STATISTICS_FROZEN_NO_REAL_NN',flush=True)


class RawRecords:
    def __init__(self,rows):
        self.rows=rows;self.mean,self.std,_=adapter.source_normalization()
    def windows(self,row):
        raw=np.load(row['EEG_path'],mmap_mode='r');assert raw.shape==(64,row['length'])
        return adapter.fixed_norm_windows(raw,self.mean,self.std,np.arange(row['windows'])*96)
    def __iter__(self):
        for row in self.rows:yield row['record'],self.windows(row)


class Capture(nn.Module):
    def __init__(self,decoder,zout,pout,check_batches):
        super().__init__();self.decoder=decoder;self.zout=zout;self.pout=pout;self.cursor=0
        self.check_batches=set(check_batches);self.checks=[];self.batches=0;self.extra=0;self.signature=None
    def forward(self,x):
        assert not torch.is_grad_enabled() and not torch.is_inference_mode_enabled() and not self.training
        start=self.cursor;stop=start+len(x);z=self.decoder.encoder(x).transpose(1,2);p=self.decoder.head(z)
        assert z.shape==(len(x),8,192) and p.shape==(len(x),1,192) and torch.isfinite(z).all() and torch.isfinite(p).all()
        if self.signature is None:self.signature=dict(input_stride=list(x.stride()),latent_stride=list(z.stride()),
            prediction_stride=list(p.stride()),params_require_grad=all(v.requires_grad for v in self.parameters()),
            eval=True,no_grad=True,inference_mode=False,dtype=str(x.dtype))
        if start in self.check_batches:
            original=self.decoder(x);difference=float((original-p).abs().max());assert difference<=1e-6
            self.checks.append(dict(global_start=start,stop=stop,original_Decoder_max_abs=difference));self.extra+=len(x)
        self.zout[start:stop]=z.cpu().numpy();self.pout[start:stop]=p[:,0].cpu().numpy();self.cursor=stop;self.batches+=1
        if self.batches%100==0:guard()
        return p


@torch.no_grad()
def encode(model,data,zout,pout,check_batches,device):
    capture=Capture(model,zout,pout,check_batches).eval();pending=[];destinations=[];scattered=0
    def flush():
        nonlocal scattered
        if not pending:return
        x=torch.from_numpy(np.stack(pending)).to(device);pred=capture(x)[:,0].cpu().numpy()
        assert np.array_equal(pred,np.asarray(pout[scattered:scattered+len(pred)]))
        scattered+=len(pred);pending.clear();destinations.clear()
    for record,windows in data:
        for local,value in enumerate(windows):
            pending.append(value);destinations.append((record,local))
            if len(pending)==64:flush()
    flush();assert scattered==capture.cursor==len(pout)
    return dict(windows=capture.cursor,batches=capture.batches,signature=capture.signature,
        original_Decoder_checks=capture.checks,extra_bridge_windows=capture.extra)


def assert_protocol():
    protocol=read(ROOT/'EVALUATION_PROTOCOL.json');review=read(ROOT/'COORDINATOR_EVALUATION_REVIEW.json')
    assert protocol['frozen_before_any_real_NN'] and review['approved_for_execution']
    assert review['protocol_sha256']==sha(ROOT/'EVALUATION_PROTOCOL.json')
    for path,digest in protocol['source_pins'].items():assert sha(path)==digest,path
    return protocol


def seed_run(seed,device):
    protocol=assert_protocol();assert seed in SEEDS;guard(True)
    folder=ROOT/f'evaluation_seed{seed}';folder.mkdir(exist_ok=False)
    dump(folder/'STARTED.json',dict(seed=seed,pid=os.getpid(),time=time.time(),protocol_sha256=sha(ROOT/'EVALUATION_PROTOCOL.json')))
    torch.set_num_threads(2);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32=protocol['precision']['TF32_matmul'];torch.backends.cudnn.allow_tf32=protocol['precision']['TF32_cudnn']
    original=ft.native().TransformerCLIP(ft.cfg());model=ft.Decoder(original.eeg);del original
    checkpoint=torch.load(protocol['checkpoints'][str(seed)]['path'],map_location='cpu',weights_only=True)
    assert checkpoint['seed']==seed and checkpoint['epoch']==4;model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    model=model.to(device).eval();assert all(p.requires_grad for p in model.parameters())
    n=protocol['windows'];zpath=folder/'latent.npy';ppath=folder/'P.npy'
    assert not zpath.exists() and not ppath.exists()
    zout=np.lib.format.open_memmap(zpath,mode='w+',dtype=np.float32,shape=(n,8,192))
    pout=np.lib.format.open_memmap(ppath,mode='w+',dtype=np.float32,shape=(n,192));guard(True)
    boundary=next(r['offset'] for r in protocol['rows'][1:] if r['offset']%64)
    check_batches={0,((n-1)//64)*64,(boundary//64)*64}
    info=encode(model,RawRecords(protocol['rows']),zout,pout,check_batches,device)
    assert len(info['original_Decoder_checks'])==len(check_batches)
    zout.flush();pout.flush();dump(folder/'CACHE_BRIDGE.json',info)
    del model,zout,pout;torch.cuda.empty_cache()
    z=np.load(zpath,mmap_mode='r');raw=np.load(ppath,mmap_mode='r');cal,stats,co=frozen(seed)
    for role in ROLES:
        results=[];precision=[];failures=[];precision_failures=[]
        for index,row in enumerate(protocol['rows']):
            offset=row['offset'];count=row['windows'];zz=z[offset:offset+count];pp=raw[offset:offset+count]
            try:
                pred=scoring.readout(zz,pp,cal,stats,co.get(role));target=np.load(row['target_path'],allow_pickle=False)
                result,_,_=scoring.measure(pred,target,row,seed,role);results.append(result)
                if role in ('W','F'):
                    try:
                        p32=scoring.readout(zz,pp,cal,stats,co[role],np.float32)
                        precision.append(scoring.precision_diagnostic(pred,p32,target,row))
                    except Exception:
                        precision_failures.append(dict(record=row['record'],traceback=traceback.format_exc()))
            except Exception:
                failures.append(dict(record=row['record'],traceback=traceback.format_exc()))
            if (index+1)%50==0:guard()
        passed=not failures and len(results)==380 and all(r['all_primary_blocks_defined'] for r in results)
        dump(folder/f'{role}_METRICS.json',dict(seed=seed,role=role,passed=passed,records=results,failures=failures,
            fixed_support_dropped=False,undefined_primary_blocks=sum(sum(b['r'] is None for b in r['blocks']) for r in results),
            precision_diagnostics=precision,precision_failures=precision_failures,physical_unit_status='UNVERIFIED',offline_gain_calibrated=True))
        print(json.dumps(dict(stage='ROLE_SCORED',seed=seed,role=role,passed=passed,records=len(results),failures=len(failures))),flush=True)
    dump(folder/'COMPLETED.json',dict(seed=seed,roles=ROLES,wall_seconds=time.time()-START,CPU_seconds=time.process_time(),
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
        main_encoder_head_windows=n,extra_bridge_windows=info['extra_bridge_windows'],new_model_fits=0,
        input_record_gain_statistics_reused=380,resource=guard(True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['inventory','freeze','run'])
    parser.add_argument('--seed',type=int);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    try:
        if args.mode=='inventory':inventory()
        elif args.mode=='freeze':freeze()
        else:seed_run(args.seed,args.device)
    except BaseException:
        path=ROOT/f'EVALUATION_V2_{args.mode}_{args.seed}_FAILURE.json'
        if not path.exists():dump(path,dict(time=time.time(),traceback=traceback.format_exc(),automatic_retry=False))
        raise
