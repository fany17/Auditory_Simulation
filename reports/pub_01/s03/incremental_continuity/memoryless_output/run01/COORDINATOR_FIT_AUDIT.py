"""Independent saved-result/checkpoint audit. CPU only on2203; no EEG/NN forward."""
import argparse, pathlib, json, hashlib, time, math, collections, os
begin=time.time(); cpu_begin=time.process_time()
import torch
ROOT=pathlib.Path("/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_memoryless_output_20261106")
ROLES=("P_past_prediction",)
SEEDS=(1024,2026,31415)
NTRAIN,NVAL,STEPS=23703510,2866799,1447
PROTOCOL="339f41a6b0490dbdcef73b64ce9ad17c24f9e372f6304f5050309952129018a0"
SOURCE=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def state_sha(d):
 h=hashlib.sha256()
 for k,v in d.items():h.update(k.encode());h.update(v.cpu().numpy().tobytes())
 return h.hexdigest()
def near(a,b,tol=1e-12):
 assert math.isfinite(float(a)) and math.isfinite(float(b))
 assert abs(a-b)<=tol,(a,b)
 return abs(a-b)
def aggregate(metric,mask):
 blocks=metric["blocks"];people=metric["people"]
 assert len(blocks)==752 and len(people)==49
 assert {(b["record"],b["start"]) for b in blocks}==set(mask)
 grouped=collections.defaultdict(list);maxerr=0.
 for b in blocks:
  ref=mask[(b["record"],b["start"])]
  assert b["samples"]==ref["samples"]
  assert b["subject"]==ref["subject"] and b["source_family"]==ref["source_family"]
  assert math.isfinite(b["r"]) and -1.0000001<=b["r"]<=1.0000001 and b["MSE"]>=0 and math.isfinite(b["MSE"])
  grouped[b["subject"]].append(b)
 byperson={p["subject"]:p for p in people}
 assert len(byperson)==49 and set(byperson)==set(grouped)
 for subject,rows in grouped.items():
  p=byperson[subject];assert p["blocks"]==len(rows)
  for field in ("r","MSE"):
   maxerr=max(maxerr,near(p[field],sum(x[field] for x in rows)/len(rows)))
 for field in ("r","MSE"):
  maxerr=max(maxerr,near(metric[field],sum(x[field] for x in people)/len(people)))
 assert metric["active_points"]==NVAL
 return maxerr
def audit(folder,mask):
 r=read(folder/"RESULT.json");cfg=read(folder/"CONFIG.json")
 seed,role=r["seed"],r["role"];ordinal=SEEDS.index(seed)*len(ROLES)+ROLES.index(role)
 assert folder.name==f"seed{seed}_{role}" and cfg["protocol_sha256"]==PROTOCOL
 assert cfg["batch"]==16384 and cfg["epochs"]==20 and cfg["shuffle"]=="global_without_replacement_int32"
 assert cfg["optimizer"]=="Adam" and cfg["lr"]==.001 and cfg["weight_decay"]==0
 ep=r["epochs"];assert [x["epoch"] for x in ep]==list(range(-1,20))
 assert [x["epoch"] for x in ep if "train" in x]==[-1,0,4,9,19]
 best_epoch=-1;best_r=ep[0]["val"]["r"]
 for x in ep:
  assert x["val"]["active_points"]==NVAL and math.isfinite(x["val"]["r"])
  if x["epoch"]>=0:assert math.isfinite(x["online_TRAIN_MSE"]) and x["online_TRAIN_MSE"]>=0
  if "train" in x:assert x["train"]["active_points"]==NTRAIN
  if x["val"]["r"]>best_r:best_r=x["val"]["r"];best_epoch=x["epoch"]
 assert r["selected_epoch"]==best_epoch
 near(r["selected"]["r"],best_r)
 maxerr=aggregate(r["selected"],mask)
 if role!="C_current":
  pair=r["val_pair"];assert set(pair)=={"near","older"}
  for m in pair.values():maxerr=max(maxerr,aggregate(m,mask))
  near(pair["near"]["r"],best_r)
  assert pair["near"]==pair["older"], 'Memoryless repeated slots must agree exactly'
 else:assert r["val_pair"] is None
 firsts=[read(folder/f"epoch{i:02}_FIRST_BATCH.json") for i in range(20)]
 assert len(firsts[0]["gradients"])==10
 for i,x in enumerate(firsts):
  assert x["epoch"]==i and x["active_points"]==16384
  assert math.isfinite(x["physical_MSE"]) and math.isfinite(x["residual_RMS"])
  assert all(math.isfinite(v) and v>=0 for v in x["gradients"].values())
 assert firsts[0]["residual_RMS"]==0 and firsts[0]["gradients"]["output.weight"]>0
 assert all(v==0 for k,v in firsts[0]["gradients"].items() if not k.startswith("output."))
 recovery=sorted((folder/"recovery").glob("epoch*.pt"));assert len(recovery)==20
 assert [p.name for p in recovery]==[f"epoch{i:02}.pt" for i in range(20)]
 last=torch.load(folder/"last.pt",map_location="cpu",weights_only=True)
 assert len(last)==10 and all(torch.isfinite(t).all() and t.dtype==torch.float32 for t in last.values())
 expected=3361
 assert last['current.weight'].shape==(32,1)
 assert last['context_in.weight'].shape==(64,1)
 assert sum(t.numel() for t in last.values())==cfg["trainable_parameters"]==expected
 assert state_sha(last)!=cfg["initialization_sha256"]
 cp=torch.load(recovery[-1],map_location="cpu",weights_only=True)
 assert cp["epoch"]==19 and cp["next_epoch"]==20 and cp["seed"]==seed and cp["role"]==role and cp["best_epoch"]==best_epoch
 assert all(torch.equal(t,cp["model"][k]) for k,t in last.items())
 assert cp["scores"]==ep
 assert len(cp["optimizer"]["state"])==10
 assert all(int(s["step"])==20*STEPS for s in cp["optimizer"]["state"].values())
 assert cp["numpy_bit_generator"]["bit_generator"]=="PCG64"
 assert cp["torch_rng"].numel()>0 and cp["torch_cuda_rng"].numel()>0
 st=cp["stage"];assert st["protocol_sha256"]==PROTOCOL
 assert st["counters"]["optimizer_updates"]==(ordinal+1)*20*STEPS
 assert st["counters"]["backward_points"]==(ordinal+1)*20*NTRAIN
 for key in ('pid','uid','linux_startticks','stage_started_unix','deadline_unix','cuda_uuid'):
  assert st['process'][key]==completed['process'][key]
 near(st['process']['deadline_unix']-st['process']['stage_started_unix'],21600)
 assert read(folder/"LAST.json")["epoch"]==19
 if best_epoch>=0:
  assert read(folder/"BEST.json")["epoch"]==best_epoch
  selected=torch.load(folder/"selected.pt",map_location="cpu",weights_only=True)
  bestcp=torch.load(folder/"recovery"/f"epoch{best_epoch:02}.pt",map_location="cpu",weights_only=True)
  assert all(torch.equal(t,bestcp["model"][k]) for k,t in selected.items())
 else:assert not (folder/"selected.pt").exists() and not (folder/"BEST.json").exists()
 stats=read(folder.parent/f"seed{seed}_TRAIN_STATS.json")
 assert set(stats)=={"current",*ROLES}
 oldroot=ROOT.parent/"s03_recent_prediction_context_20261106/fit"
 oldcfg=read(ROOT.parent/'s03_output_query_20261106/fit'/f"seed{seed}_{role}/CONFIG.json")
 original=core.PhysicalDecoder(seed,role).state_dict()
 original['current.weight']=original['current.weight'][:,:1]*(8**.5)
 assert state_sha(original)==oldcfg['initialization_sha256']
 original['context_in.weight']=original['context_in.weight'][:,:1]*(96**.5)
 assert cfg['initialization_sha256']==state_sha(original)
 pstats=read(oldroot/f"seed{seed}_TRAIN_STATS.json")[role]
 assert stats[role]==pstats and stats['current']=={k:[v] for k,v in pstats.items()}
 def vals(v):return v if isinstance(v,list) else [v]
 for k,s in stats.items():
  assert set(s)=={"mean","std","constant"}
  assert len(vals(s["mean"]))==1
  assert all(math.isfinite(x) for x in vals(s["mean"]))
  assert all(math.isfinite(x) and x>0 for x in vals(s["std"]))
  assert all(isinstance(x,bool) for x in vals(s["constant"]))
 return {"fit":folder.name,"seed":seed,"role":role,"passed":True,"selected_epoch":best_epoch,"selected_r":best_r,"last_VAL_r":ep[-1]["val"]["r"],"last_TRAIN_r":ep[-1]["train"]["r"],"initialization_sha256":cfg["initialization_sha256"],"result_sha256":sha(folder/"RESULT.json"),"last_checkpoint_sha256":sha(folder/"last.pt"),"recovery_last_sha256":sha(recovery[-1]),"max_saved_aggregate_error":maxerr,"first_batch_logs_checked":20,"checkpoint_files":20,"parameters":expected}
parser=argparse.ArgumentParser();parser.add_argument("--batch",required=True);args=parser.parse_args()
assert args.batch.isdigit()
assert sha(ROOT/"PROTOCOL.json")==PROTOCOL
protocol=read(ROOT/'PROTOCOL.json')
assert sha(ROOT/'input/pub_01_s03_ss07_memoryless_output.py')==protocol['source_sha256']
for path,digest in protocol['frozen_pins'].items():assert sha(path)==digest
import sys
sys.path.insert(0,str(ROOT.parent/'s03_physical_stream_20261106/input'))
import pub_01_s03_ss07_physical_stream_core as core
completed=read(ROOT/"fit/COMPLETED.json")
assert completed["passed"] and not (ROOT/"fit/FAILURE.json").exists()
assert completed['history_intervention'] is False
proc=pathlib.Path('/proc')/str(completed['process']['pid'])
if proc.exists():assert int((proc/'stat').read_text().rsplit(') ',1)[1].split()[19])!=completed['process']['linux_startticks']
assert completed["counters"]["formal_fits_completed"]==3
assert completed["counters"]["optimizer_updates"]==86820 and completed["counters"]["backward_points"]==1422210600
assert completed['counters']['TRAIN_records_loaded']==1278 and completed['counters']['VAL_records_loaded']==147
assert completed['counters']['TRAIN_cached_windows_loaded']==757362 and completed['counters']['VAL_cached_windows_loaded']==91500
paired=sum(v['selected_epoch']>=0 for v in completed['seeds'].values())*NVAL
base=1422210600+284442120+172007940
assert completed['counters']['current_stem_points']==base+paired and completed['counters']['context_output_points']==base+2*paired
previous=set()
for p in ROOT.glob("COORDINATOR_FIT_AUDIT_BATCH*.json"):
 old=read(p);assert old["source_sha256"]==SOURCE
 previous.update(x["fit"] for x in old["fits"])
metadata=read(ROOT.parent/"s03_physical_stream_20261106/PREPARATION_METADATA.json")
records={r["record"]:r for r in metadata["phases"]["val"]["records"]}
mask={(b["record"],b["start"]):dict(b,subject=records[b["record"]]["subject"],source_family=records[b["record"]]["source_family"]) for b in metadata["phases"]["val"]["blocks"]}
fits=[]
for folder in sorted((ROOT/"fit").glob("seed*")):
 if (folder/"RESULT.json").exists() and folder.name not in previous:fits.append(audit(folder,mask))
assert fits,"No new completed fits; do not save an empty repeated audit"
out={"passed":True,"source_sha256":SOURCE,"protocol_sha256":PROTOCOL,"completion_sha256":sha(ROOT/'fit/COMPLETED.json'),"fits":fits,"new_neural_forward":0,"new_backward":0,"new_optimizer_updates":0,"EEG_or_prediction_arrays_loaded":False,"checkpoint_parameter_arrays_loaded_on_CPU":True,"CPU_seconds":time.process_time()-cpu_begin,"wall_seconds":time.time()-begin,"scope":"Independent saved scalar aggregation/selection/cost and actual last/selected parameter/Adam/RNG checks; twenty recovery filenames checked per fit, only last/selected recovery contents loaded. Independent initial-state reconstruction, no neural forward.","pid":os.getpid()}
path=ROOT/f"COORDINATOR_FIT_AUDIT_BATCH{int(args.batch):02}.json"
with path.open("x") as f:json.dump(out,f,indent=2,allow_nan=False);f.write("\n")
print(json.dumps(out))
