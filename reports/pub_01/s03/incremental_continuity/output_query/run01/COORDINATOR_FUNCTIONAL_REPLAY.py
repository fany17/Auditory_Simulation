"""D22 bounded CPU functional replay, fixed metadata endpoints, no fit/test selection."""
import os,time,pathlib,json,hashlib,math,resource,sys
begin=time.time();cpu=time.process_time()
os.environ["CUDA_VISIBLE_DEVICES"]=""
ROOT=pathlib.Path("/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_output_query_20261106")
PROTOCOL="40de86545cdcd69264a308a68cc02c653b211f9688511b2cea9df2976dbcc74e"
read=lambda p:json.loads(pathlib.Path(p).read_text())
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
protocol=read(ROOT/"PROTOCOL.json")
assert sha(ROOT/"PROTOCOL.json")==PROTOCOL
assert (ROOT/"fit/COMPLETED.json").exists() and not (ROOT/"fit/FAILURE.json").exists()
assert sha(ROOT/'input/pub_01_s03_ss07_output_query.py')==protocol['source_sha256']
for path,h in protocol['frozen_pins'].items():assert sha(path)==h
D19=ROOT.parent/'s03_physical_stream_20261106'
D20=ROOT.parent/'s03_recent_prediction_context_20261106'
import numpy as np
import torch
torch.set_num_threads(2)
sys.path.insert(0,str(ROOT/'input'))
sys.path.insert(0,str(D19/'input'))
sys.path.insert(0,str(D20/'input'))
import pub_01_s03_ss07_physical_stream_core as core
import pub_01_s03_ss07_physical_stream_meta as meta
import pub_01_s03_ss07_physical_stream_run as legacy
import pub_01_s03_ss07_recent_prediction_context as recent
import pub_01_s03_ss07_output_query as query
Table,Decoder,_,_,_=query.adapters(recent,core,legacy)
metadata=read(D19/'PREPARATION_METADATA.json')
chosen=[metadata["phases"]["val"]["records"][i] for i in (0,-1)]
assert chosen[0]["record"]!=chosen[1]["record"]
subset=dict(metadata,phases=dict(metadata["phases"]))
subset["phases"]["val"]=dict(metadata["phases"]["val"],records=chosen)
counters={"producer_model_points":0,"independent_numpy_model_points":0,"frozen_readout_windows_independent":0,"frozen_readout_windows_producer":0,"raw_EEG_encoder_windows":0,"backward":0,"optimizer_updates":0}
def failure_hook(kind,error,trace):
 import traceback
 failure={"passed":False,"error":repr(error),"source_sha256":sha(__file__),"counters_completed_operations":counters,
          "failing_operation_cost_if_incomplete":"unknown","CPU_seconds":time.process_time()-cpu,"wall_seconds":time.time()-begin,
          "peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 with (ROOT/"COORDINATOR_FUNCTIONAL_REPLAY_FAILURE.json").open("x") as f:json.dump(failure,f,indent=2);f.write("\n")
 traceback.print_exception(kind,error,trace)
sys.excepthook=failure_hook
def check(a,b,atol,rtol=0):
 a=np.asarray(a);b=np.asarray(b)
 assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
 difference=float(np.max(np.abs(a-b))) if a.size else 0.
 assert np.allclose(a,b,atol=atol,rtol=rtol),(a.shape,difference,atol,rtol)
 return difference
def normalize(value,s):
 mean=np.asarray(s["mean"],dtype=np.float32);std=np.asarray(s["std"],dtype=np.float32)
 constant=np.asarray(s["constant"],dtype=bool)
 return np.where(constant,np.float32(0),(value.astype(np.float32)-mean)/std).astype(np.float32)
def make_reference(seed,row):
 files=meta.assets(seed);lo=row["offset"];hi=lo+row["windows"]
 z=np.load(files["folder"]/"val_latent.npy",mmap_mode="r")[lo:hi]
 p=np.load(files["folder"]/"val_P.npy",mmap_mode="r")[lo:hi].astype(np.float64)
 with np.load(files["stats"]) as s:
  stats=[{key:s[f"{key}_{i}"].copy() for key in ("mean","std","constant")} for i in range(3)]
 with np.load(files["coefficient"]) as c:coef=c["coef"].copy();bias=c["bias"].copy()
 cal=read(files["calibration"])["calibration"]
 baseline=cal["a"]*p+cal["b"];windows=baseline.copy()
 def oldnorm(x,s):return np.where(s["constant"],0.,(x-s["mean"])/s["std"])
 u=oldnorm(z[4:].astype(np.float64).reshape(-1,1536),stats[0])
 v=oldnorm(baseline[4:],stats[1]);square=oldnorm(u*u,stats[2])
 features=np.column_stack((u/np.sqrt(1536),v/np.sqrt(192),square/np.sqrt(1536)))
 windows[4:]+=features@coef+bias
 n=row["common_samples"];w=np.hanning(192)
 weight=np.zeros(n);g=np.zeros((n,8));b=np.zeros(n)
 # Scatter addition separately from production's slice accumulator.
 positions=(96*np.arange(len(z))[:,None]+np.arange(192)[None,:]).reshape(-1)
 np.add.at(weight,positions,np.tile(w,len(z)))
 np.add.at(b,positions,(windows*w).reshape(-1))
 for channel in range(8):np.add.at(g[:,channel],positions,(z[:,channel,:]*w).reshape(-1))
 support=weight>1e-8
 b[support]/=weight[support];g[support]/=weight[support,None]
 counters["frozen_readout_windows_independent"]+=len(z)
 return z,g.astype(np.float32),b
def context_reference(role,points,z,g,b,lag):
 answers=[]
 for point in points:
  if role=="C_current":
   ids=[i for i in range(len(z)) if 0<point-96*i<191]
   assert len(ids) in (1,2)
   if len(ids)==1:ids*=2
   frames=np.asarray(z[ids],dtype=np.float64)
   value=frames.reshape(2,8,48,4).mean(axis=3).transpose(0,2,1).reshape(96,8)
  else:
   wave=g if role=="H_past_eeg" else b
   interval=wave[point-lag-383:point-lag+1]
   assert len(interval)==384
   value=interval.reshape((96,4)+wave.shape[1:]).mean(axis=1,dtype=np.float64)
  answers.append(value)
 return np.asarray(answers,dtype=np.float32)
def gelu(x):
 erf=np.fromiter((math.erf(float(v)/math.sqrt(2)) for v in x.flat),dtype=np.float32,count=x.size).reshape(x.shape)
 return np.float32(.5)*x*(np.float32(1)+erf)
def numpy_model(x,v,b,state):
 d={k:t.numpy() for k,t in state.items()}
 lin=lambda a,name:a@d[name+".weight"].T+d[name+".bias"]
 q=gelu(lin(x,"current"));context=lin(gelu(lin(v.reshape(len(v),-1),"context_in")),"context_out")
 mixed=gelu(lin(q*(np.float32(1)+np.tanh(context)),"mix"))
 return b+lin(mixed,"output").reshape(-1).astype(np.float64)
results=[]
for seed in meta.SEEDS:
 records=core.load_phase(subset,seed,"val",lambda:None)
 counters["frozen_readout_windows_producer"]+=sum(r.metadata["windows"] for r in records)
 table=Table(records,"cpu",lambda:None)
 rawstats=read(ROOT/"fit"/f"seed{seed}_TRAIN_STATS.json")
 stats={k:(torch.tensor(s["mean"],dtype=torch.float32),torch.tensor(s["std"],dtype=torch.float32),torch.tensor(s["constant"],dtype=torch.bool)) for k,s in rawstats.items()}
 reference=[make_reference(seed,row) for row in chosen]
 for role in ("P_past_prediction",):
  folder=ROOT/"fit"/f"seed{seed}_{role}";fit=read(folder/"RESULT.json")
  checkpoints=[("last",folder/"last.pt")]
  if fit["selected_epoch"]>=0 and fit["selected_epoch"]!=19:checkpoints.append(("selected",folder/"selected.pt"))
  for label,path in checkpoints:
   state=torch.load(path,map_location="cpu",weights_only=True)
   model=Decoder(seed,role);model.load_state_dict(state);model.eval()
   max_inputs=max_output=max_anchor=0.
   for i,(record,ref) in enumerate(zip(records,reference)):
    z,g,b=ref
    points=np.r_[np.arange(meta.FIRST_ACTIVE,meta.FIRST_ACTIVE+64),np.arange(record.stop-64,record.stop)]
    assert len(set(points))==128
    spans=[(i,int(t),int(t)+1) for t in points]
    ids=table.indices(spans)
    xp,vp,bp,_=core.batch_inputs(table,ids,role,stats)
    xr=normalize(b[points,None],rawstats["current"])
    vr=normalize(context_reference(role,points,z,g,b,1),rawstats[role])
    max_anchor=max(max_anchor,check(bp.numpy(),b[points],1e-8))
    max_inputs=max(max_inputs,check(xp.numpy(),xr,2e-6),check(vp.numpy(),vr,2e-6))
    with torch.no_grad():prod=model(xp,vp,bp).numpy()
    counters["producer_model_points"]+=len(points)
    independent=numpy_model(xr,vr,b[points],state)
    counters["independent_numpy_model_points"]+=len(points)
    max_output=max(max_output,check(prod,independent,2e-5,2e-5))
   results.append({"seed":seed,"role":role,"checkpoint":label,"checkpoint_sha256":sha(path),"passed":True,"input_max_abs":max_inputs,"anchor_max_abs":max_anchor,"prediction_max_abs":max_output,"points":256})
 assert counters["producer_model_points"]<=1536 and counters["independent_numpy_model_points"]<=1536
 del table,records,reference
assert 3<=len(results)<=6
out={"passed":True,"protocol_sha256":PROTOCOL,"source_sha256":sha(__file__),"selected_records":[r["record"] for r in chosen],"selection_rule":"Original VAL metadata first/last records; first64 active and last64 positive-covered points. Fixed before replay, no performance selection.","tolerances":{"normalized_input_atol":2e-6,"anchor_atol":1e-8,"prediction_atol":2e-5,"prediction_rtol":2e-5},"results":results,"counters":counters,"CPU_seconds":time.process_time()-cpu,"wall_seconds":time.time()-begin,"peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"scope":"Bounded independent direct-bin/scatter fusion/NumPy GELU functional replay from existing VAL latent/P. No raw EEG encoder replay, no new scientific scoring, no fit or model selection. Producer and independent point forwards charged separately.","gpu_used":False,"goal_complete":False}
with (ROOT/"COORDINATOR_FUNCTIONAL_REPLAY.json").open("x") as f:json.dump(out,f,indent=2,allow_nan=False);f.write("\n")
print(json.dumps(out))
