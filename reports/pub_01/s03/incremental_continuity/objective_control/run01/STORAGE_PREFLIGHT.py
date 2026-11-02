import json,time,sys,resource
sys.path.insert(0,"input")
from pub_01_s03_ss07_objective_control_core import *
start=time.time();data=Data(1024,"train");sample=train_score(data,data.anchor)
def inflate(value):
 if isinstance(value,float):return "X"*32
 if isinstance(value,dict):return {k:inflate(v) for k,v in value.items()}
 if isinstance(value,list):return [inflate(v) for v in value]
 return value
actual=len(json.dumps(sample,indent=2,allow_nan=False).encode());bound=len(json.dumps(inflate(sample),indent=2,allow_nan=False).encode())
fits=D14/"fits"
checkpoints=sum(p.stat().st_size for p in fits.glob("*/*.pt"))
other=sum(p.stat().st_size for p in fits.glob("*/*") if p.is_file() and p.suffix in (".json",".jsonl") and not p.name.startswith(("OFFLINE_TRAIN_","OLD_LAST19_")))
review=sum(p.stat().st_size for p in (D14/"coordinator_review").rglob("*") if p.is_file())
# Inflate every offline float up to32 chars; add20% to current VAL/review
# schema sizes and3MiB for sources/protocol/plots/logs; direct SCP, no tar.
components=dict(offline45_worst=45*bound,checkpoints=checkpoints,other_fit_outputs_with_margin=int(other*1.2)+131072,independent_review_with_margin=int(review*1.2),source_protocol_plots_logs_reserve=3*1024**2)
estimate=sum(components.values())
out=dict(sample_actual_bytes=actual,sample_every_float32char_bound_bytes=bound,components=components,conservative_total_bytes=estimate,root_cap_bytes=128*1024**2,direct_whitelist_SCP_no_server_tar=True,NN=0,fit=0,optimizer_updates=0,wall_seconds=time.time()-start,CPU_seconds=time.process_time(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(ROOT/"STORAGE_ESTIMATE.json").write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
