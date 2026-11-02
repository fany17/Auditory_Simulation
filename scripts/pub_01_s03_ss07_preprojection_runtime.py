"""D18 bounded execution/identity/accounting helpers. No process launches on import."""
import os,sys,json,time,hashlib,resource,shutil,subprocess,stat,gzip
from pathlib import Path
import numpy as np
import torch
from pub_01_s03_ss07_preprojection_model import BASE,ARMS,AdaptedDecoder
from pub_01_s03_ss07_preprojection_data import D8,RawPhase,read
ROOT=BASE/'s03_preprojection_memory_20261103'
D16=BASE/'s03_full_ft_token_normalization_20261102'
PYTHON=BASE/'s03_strong_baselines_20261013/env01/bin/python'
SEEDS=(1024,2026,31415)
METRIC_FILE_LIMIT=2*1024**2
def read_result(path):
    """Read lossless D18 metrics (.json.gz); also accepts provenance JSON."""
    p=Path(path)
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def dump_metrics(path,value):
    p=Path(path);assert p.is_relative_to(ROOT) and p.name.endswith('.json.gz')
    raw=json.dumps(value,separators=(',',':'),allow_nan=False).encode()
    packed=gzip.compress(raw,compresslevel=6,mtime=0)
    assert len(packed)<=METRIC_FILE_LIMIT,('Metric-file storage contract',p,len(packed))
    assert json.loads(gzip.decompress(packed))==value,'Lossless JSON roundtrip'
    with p.open('xb') as f:f.write(packed)
    return dict(uncompressed_bytes=len(raw),gzip_bytes=len(packed),all_fields_preserved=True)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def dump(path,value):
    p=Path(path);assert p.is_relative_to(ROOT)
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def update(path,value):
    p=Path(path);assert p.is_relative_to(ROOT);tmp=Path(str(p)+'.writing')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,p)
def root_bytes():
    total=0
    for p in ROOT.rglob('*'):
        try:s=p.stat()
        except FileNotFoundError:continue
        if stat.S_ISREG(s.st_mode):total+=s.st_size
    return total
def guard():
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=48*1024**2,'48GiB RSS'
    if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<=32*1024**3,'32GiB allocated GPU'
    assert min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/sata5/fanyu/auditory_simulation_m6a').free)>=20*1024**3,'20GiB each disk'
    assert root_bytes()+64*1024**2<768*1024**2,'768MiB root includes64MiB concurrent write reserve'
def seed_all(seed,device):
    torch.manual_seed(seed);np.random.seed(seed);torch.cuda.set_device(device);torch.cuda.manual_seed_all(seed);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
def permutation(seed,epoch,n):
    return np.random.default_rng(np.random.SeedSequence([seed,20261102,epoch])).permutation(n)
def make_model(seed,arm,data,device):
    cfg=read(data_config_path())['transformer'];pin=read(D8/'PROTOCOL.json')['checkpoints'][str(seed)]
    cp=torch.load(pin['checkpoint'],map_location='cpu',weights_only=True)
    assert cp['seed']==seed and cp['epoch']==pin['selected_epoch']==4
    model=AdaptedDecoder(seed,arm,cfg,cp['model'],data.torch_readout()).to(device)
    assert all(not p.requires_grad for p in model.reference_encoder.parameters())
    return model
def data_config_path():
    return BASE/'s03_strong_baselines_20261015/transformer_available01/training_seed1024_attempt04/config_frozen.json'
def admission(device):
    i=device.split(':')[1]
    r=subprocess.run(['nvidia-smi','--id='+i,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True);uuid,free=r.stdout.strip().split(',');free=int(free)
    r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    pids=[int(x.split(',')[1]) for x in r.stdout.splitlines() if ',' in x and x.split(',')[0].strip()==uuid.strip()]
    mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    return dict(device=device,GPU_uuid=uuid.strip(),free_GPU_MiB=free,compute_pids=pids,MemAvailable_bytes=mem,admitted=not pids and free>=36*1024 and mem>=64*1024**3)
def verify(p):
    for path,digest in p['source_pins'].items():assert sha(path)==digest,path
    for path,item in p['inputs'].items():
        s=Path(path).stat();assert s.st_size==item['bytes'] and s.st_mtime_ns==item['mtime_ns'],path
        if item.get('recheck_sha256',False):assert sha(path)==item['sha256'],path
def gradient_summary(model):
    modules={'encoder':model.encoder,'head':model.head,'original_attention0':model.encoder.attend[0]}
    if model.arm!='T_continue':modules.update(memory=model.memory,query_norm=model.memory.query_norm,memory_out_projection=model.memory.attention.out_proj)
    if model.arm=='P_prediction_history':modules['P_projection']=model.prior_projection
    result={}
    for name,module in modules.items():
        parameters=module if isinstance(module,list) else module.parameters()
        grads=[p.grad.detach() for p in parameters if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        values=torch.stack([sum(torch.count_nonzero(g).double() for g in grads),sum(g.double().square().sum() for g in grads).sqrt()]).cpu().tolist()
        result[name]=dict(nonzero_scalars=int(values[0]),L2_norm=values[1])
    if model.arm!='T_continue':
        weight=model.memory.attention.in_proj_weight.grad;bias=model.memory.attention.in_proj_bias.grad
        assert weight is not None and bias is not None
        for index,name in enumerate(('Q','K','V')):
            for suffix,g in [('weight',weight[index*64:(index+1)*64]),('bias',bias[index*64:(index+1)*64])]:
                assert torch.isfinite(g).all();values=torch.stack([torch.count_nonzero(g).double(),g.double().square().sum().sqrt(),g.double().abs().max()]).cpu().tolist()
                result[f'memory_{name}_{suffix}']=dict(nonzero_scalars=int(values[0]),L2_norm=values[1],max_abs=values[2])
        result['K_bias_note']='Common key bias is softmax-unidentifiable; tiny float32 nonzero is not evidence of useful capacity.'
    assert all(p.grad is None for p in model.reference_encoder.parameters())
    return result
def counters_delta(after,before):return {k:after[k]-before[k] for k in after}
def save_checkpoint(path,model,seed,epoch,validation_r,protocol_sha):
    p=Path(path);assert p.is_relative_to(ROOT);tmp=Path(str(p)+'.writing')
    torch.save(dict(trainable_state=model.trainable_state(),seed=seed,arm=model.arm,epoch=epoch,validation_r=validation_r,protocol_sha256=protocol_sha,frozen_assets='Source pins only; no repeated E0/Ccap/position/optimizer'),tmp)
    os.replace(tmp,p)
