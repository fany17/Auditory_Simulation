import os
os.environ.update(OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUDA_VISIBLE_DEVICES='')
import json,hashlib,time,resource
from pathlib import Path
import numpy as np
t0=time.time();cpu0=time.process_time()
root=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_finetuned_internal_audit_20261101')
base=root.parent;d8=base/'s03_finetuned_readout_transfer_20261101';d9=base/'s03_physical_objective_20261101';old=base/'s03_incremental_continuity_20261101'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (root/'COMPLETED.json').exists() and not (root/'FAILURE.json').exists()
p=read(root/'PROTOCOL.json');scope=read(root/'COORDINATOR_REVIEW_SCOPE.json');done=read(root/'COMPLETED.json');summary=read(root/'SUMMARY.json');cost=read(root/'COST.json')
assert sha(root/'PROTOCOL.json')=='14b2cedd0df026637624afbc63004c3803644d420e7966bad234cab33babb78e'
assert done['role_split_scores']==36 and not done['new_fit_selection'] and not done['DTU']
assert p['fit_selection']==False and p['DTU']==False and p['roles']==['P_FT','Ponly','C','Ccap','W','F']
for f,h in p['source_pins'].items():assert sha(f)==h,f
for f,v in p['inputs'].items():
 q=Path(f);st=q.stat();assert st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],f
 if 'shape' in v:
  a=np.load(q,mmap_mode='r',allow_pickle=False);assert list(a.shape)==v['shape'] and str(a.dtype)==v['dtype'],f
assert read(root/'MATH_CHECK.json')['passed']
mapping=read(root/'VAL_MAPPING_CHECK.json');assert mapping['passed'] and len(mapping['cases'])==36
assert max(c['prediction_max_abs'] for c in mapping['cases'])<=1e-8 and max(c['block_r_max_abs'] for c in mapping['cases'])<=1e-10
key=lambda b:(b['record'],b['phase'],b['subject'],b['story'],b['source_family'],b['start'],b['samples'])
maskkeys=[(r['record'],r['phase'],r['subject'],r['story'],r['source_family'],b['start'],b['samples']) for r in p['records'] for b in p['masks'][r['record']] if b['eligible']]
assert len(maskkeys)==2044 and len(p['records'])==144
record_map={r['record']:r for r in p['records']}
case_rows=[rr for ph in ['test_seen','test_heldout'] for rr in ([r for r in p['records'] if r['phase']==ph][0],[r for r in p['records'] if r['phase']==ph][-1])]
maxagg=0.;maxr=0.;maxmse=0.;max3=0.;maxfb=0.;maxhann=0.;maxoldr=0.;maxwave=0.;maxdecoder=0.;maxcapture=0.
mainN=0;extraDecoder=0;extraVal=0;batchN=0;cachebytes=0;cases=[];undefined={};values=[];bridgerecords=0
signature=dict(input_stride=[12288,1,192],latent_view_stride=[1536,1,8],pred_stride=[192,192,1],params_require_grad=True,eval=True,grad_enabled=False,inference_mode=False,dtype='torch.float32')
h=np.hanning(192)
for seed in p['seeds']:
 folder=root/f'seed{seed}';s=read(folder/'COMPLETED.json');assert s['roles']==6 and s['role_split_scores']==12 and s['new_fits']==0
 cap=read(folder/'VAL_CAPTURE_CHECK.json');assert cap['passed'] and cap['encoder_head_windows']==100
 assert [(q['global_first'],q['windows']) for q in cap['cases']]==[(0,64),(30464,36)]
 for q in cap['cases']:
  assert q['signature']==signature
  maxcapture=max(maxcapture,q['Z_max_abs'],q['P_raw_max_abs'])
 extraVal+=cap['encoder_head_windows']
 bridge=read(folder/'RAW_FT_ALL144_PASSED.json')
 assert len(bridge['records'])==144 and {q['record'] for q in bridge['records']}==set(record_map)
 maxwave=max(maxwave,bridge['maximum_raw_original_units']);maxoldr=max(maxoldr,bridge['maximum_original_block_r'])
 for ph,e in p['counts'].items():
  b=read(folder/f'{ph}_CACHE_BRIDGE.json');assert b['signature']==signature
  assert len(b['records'])==e['records'] and [q['record'] for q in b['records']]==[r['record'] for r in p['records'] if r['phase']==ph]
  assert b['main_encoder_head_windows']==e['windows'] and b['batches']==e['global64_batches']
  assert b['extra_original_Decoder_windows']==64+e['tail_batch']
  assert [(q['first'],q['last']) for q in b['Decoder_checks']]==[(0,64),(e['windows']-e['tail_batch'],e['windows'])]
  mainN+=b['main_encoder_head_windows'];batchN+=b['batches'];extraDecoder+=b['extra_original_Decoder_windows'];bridgerecords+=len(b['records'])
  maxdecoder=max(maxdecoder,max(q['raw_units_max_abs'] for q in b['Decoder_checks']))
  maxwave=max(maxwave,max(q['raw_original_fused_max_abs'] for q in b['records']));maxoldr=max(maxoldr,max(q['calibrated_block_r_max_abs'] for q in b['records']))
  for typ,shape in [('latent',(e['windows'],8,192)),('P',(e['windows'],192))]:
   f=folder/f'{ph}_{typ}.npy';a=np.load(f,mmap_mode='r',allow_pickle=False);assert a.shape==shape and a.dtype==np.float32 and np.isfinite(a).all()
   cachebytes+=f.stat().st_size
 original=read(old/f'final_results/seed{seed}/INTERNAL_P_FT.json');originalblocks={(b['record'],b['start']):b['r'] for b in original['segments']}
 cal=read(d8/f'seed{seed}/P_FT_BASELINE.json')['calibration'];assert cal['a']>0
 stats=np.load(d8/f'seed{seed}/TRAIN_TRANSFORMS.npz')
 def norm(x,j):
  v=(x-stats[f'mean_{j}'])/stats[f'std_{j}'];v[:,stats[f'constant_{j}']]=0;return v
 for role in p['roles']:
  m=read(folder/f'{role}_METRICS.json');assert [key(b) for b in m['segments']]==maskkeys
  assert len(m['records'])==144 and {b['record'] for b in m['records']}==set(record_map)
  if role=='P_FT':
   maxoldr=max(maxoldr,max(abs(b['r']-originalblocks[(b['record'],b['start'])]) for b in m['segments']))
  expected_phases={}
  for ph,e in p['counts'].items():
   bb=[b for b in m['segments'] if b['phase']==ph];pp=[q for q in m['people'] if q['phase']==ph];rr=[q for q in m['records'] if q['phase']==ph]
   assert len(pp)==e['people'] and len(bb)==e['eligible_blocks'] and len(rr)==e['records'] and sum(q['windows'] for q in rr)==e['windows']
   agg={}
   for metric in ['r','MSE','weighted_single_MSE','disagreement']:
    people={q['subject']:float(np.mean([b[metric] for b in bb if b['subject']==q['subject']])) for q in pp}
    maxagg=max(maxagg,max(abs(q[metric]-people[q['subject']]) for q in pp))
    agg[metric]=float(np.mean(list(people.values())))
    maxagg=max(maxagg,abs(agg[metric]-m['phases'][ph][metric]),abs(agg[metric]-summary['results'][str(seed)][role]['phases'][ph][metric]))
   un=sum(q['window_r_undefined'] for q in rr);undefined[f'{seed}:{role}:{ph}']=un
   assert un==m['phases'][ph]['window_r_undefined'] and m['phases'][ph]['window_r_defined']==e['windows']-un
   if un:assert m['phases'][ph]['r3s'] is None
   else:maxagg=max(maxagg,abs(sum(q['r3s']*q['windows'] for q in rr)/e['windows']-m['phases'][ph]['r3s']))
   maxagg=max(maxagg,abs(sum(q['window_MSE']*q['windows'] for q in rr)/e['windows']-m['phases'][ph]['window_MSE']))
   for b in bb:maxhann=max(maxhann,abs(b['weighted_single_MSE']-b['MSE']-b['disagreement']))
   expected_phases[ph]=agg
  for metric in ['r','MSE','weighted_single_MSE','disagreement']:
   v=2/3*expected_phases['test_seen'][metric]+1/3*expected_phases['test_heldout'][metric]
   maxagg=max(maxagg,abs(v-m['weighted'][metric]),abs(v-summary['results'][str(seed)][role]['weighted'][metric]))
  values.append(dict(seed=seed,role=role,seen=m['phases']['test_seen']['r'],heldout=m['phases']['test_heldout']['r'],weighted=m['weighted']['r'],weighted_MSE=m['weighted']['MSE']))
  coef=None
  if role!='P_FT':
   cp=(d9/f'seed{seed}/{role}_COEFFICIENTS.npz') if role in ['W','F'] else d8/f'seed{seed}/{role}_lambda{(.0001 if role=="C" else .01):g}_COEFFICIENTS.npz'
   assert str(cp) in p['source_pins'] and sha(cp)==p['source_pins'][str(cp)]
   coef=np.load(cp)
  for row in case_rows:
   record=row['record'];w=(row['common_samples']-192)//96+1;n=row['common_samples'];o=row['phase_offset'];ph=row['phase']
   z=np.asarray(np.load(folder/f'{ph}_latent.npy',mmap_mode='r')[o:o+w],float)
   raw=np.asarray(np.load(folder/f'{ph}_P.npy',mmap_mode='r')[o:o+w],float)
   anchor=cal['a']*raw+cal['b'];pred=anchor.copy()
   if coef is not None:
    xp=norm(anchor[4:],1)/np.sqrt(192)
    if role=='Ponly':X=xp
    else:
     zz=norm(z[4:].reshape(w-4,1536),0);parts=[zz/np.sqrt(1536),xp]
     if role!='C':parts.append(norm(zz*zz,2)/np.sqrt(1536))
     X=np.concatenate(parts,1)
    pred[4:]+=X@coef['coef']+coef['bias']
   maxfb=max(maxfb,float(abs(pred[:4]-anchor[:4]).max()))
   total=np.zeros(n);den=np.zeros(n);sq=np.zeros(n);werr=np.zeros(n)
   target=np.asarray(np.load(row['envelope_path'],mmap_mode='r'),float).reshape(-1)
   for j in range(w):
    a=96*j;b=a+192;total[a:b]+=h*pred[j];den[a:b]+=h;sq[a:b]+=h*pred[j]**2;werr[a:b]+=h*(pred[j]-target[a:b])**2
   covered=den>1e-8;fused=np.zeros(n);weighted=np.zeros(n);dis=np.zeros(n)
   np.divide(total,den,out=fused,where=covered);np.divide(werr,den,out=weighted,where=covered);np.divide(sq,den,out=dis,where=covered);dis[covered]-=fused[covered]**2
   saved={(b['record'],b['start']):b for b in m['segments']};lr=0.;le=0.
   for block in p['masks'][record]:
    a=block['start'];b=a+3840;k=covered[a:b];assert int(k.sum())==block['samples']
    if not block['eligible']:continue
    xx=fused[a:b][k];yy=target[a:b][k];ref=saved[(record,a)]
    assert np.ptp(xx)>0 and np.ptp(yy)>0
    lr=max(lr,abs(float(np.corrcoef(xx,yy)[0,1])-ref['r']))
    le=max(le,abs(float(np.mean((xx-yy)**2))-ref['MSE']),abs(float(weighted[a:b][k].mean())-ref['weighted_single_MSE']),abs(float(dis[a:b][k].mean())-ref['disagreement']))
   yt=np.stack([target[j*96:j*96+192] for j in range(w)]).astype(np.float32).astype(float)
   valid=(np.ptp(pred,axis=1)>0)&(np.ptp(yt,axis=1)>0)
   rec=next(x for x in m['records'] if x['record']==record);assert rec['window_r_undefined']==int((~valid).sum())
   if valid.all():
    px=pred-pred.mean(1,keepdims=True);yy=yt-yt.mean(1,keepdims=True)
    r3=float(np.mean(np.sum(px*yy,1)/np.sqrt(np.sum(px*px,1)*np.sum(yy*yy,1))));max3=max(max3,abs(r3-rec['r3s']))
   else:assert rec['r3s'] is None
   le=max(le,abs(float(np.mean((pred-yt)**2))-rec['window_MSE']))
   maxr=max(maxr,lr);maxmse=max(maxmse,le);cases.append(dict(seed=seed,role=role,record=record,r_max_abs=lr,MSE_Hann_max_abs=le))
assert len(cases)==72 and mainN==254076 and batchN==3972 and extraVal==300 and extraDecoder==636 and bridgerecords==432 and cachebytes==1756174848
assert cost['main_encoder_head_windows']==mainN and cost['extra_VAL_capture_windows']==extraVal and cost['extra_original_Decoder_windows']==extraDecoder
assert done['all_SS07_NPY_bytes']==26225606304 and done['all_SS07_NPY_bytes']<=25*1024**3 and done['new_root_bytes']<=2*1024**3 and done['free_disk_bytes']>=20*1024**3
assert maxcapture<=1e-6 and maxdecoder<=1e-6 and maxwave<=1e-6 and maxoldr<=1e-10
assert maxfb==0 and maxagg<=1e-12 and maxr<=1e-10 and maxmse<=1e-10 and max3<=1e-10 and maxhann<=1e-10
means={role:{field:float(np.mean([r[field] for r in values if r['role']==role])) for field in ['seen','heldout','weighted','weighted_MSE']} for role in p['roles']}
assert abs(means['P_FT']['weighted']-.168944726368672)<=1e-12
lookup={(r['seed'],r['role']):r for r in values};contrasts=[]
for a,b in p['primary_pairs']+p['secondary_pairs']:
 contrasts.append(dict(pair=f'{a}-{b}',**{field:dict(mean=float(np.mean([lookup[(s,a)][field]-lookup[(s,b)][field] for s in p['seeds']])),by_seed=[lookup[(s,a)][field]-lookup[(s,b)][field] for s in p['seeds']]) for field in ['seen','heldout','weighted','weighted_MSE']}))
out=dict(status='PASS_ACTUAL_NUMERICAL_RESULTS',goal_complete=False,scientific_status='POST-HOC INTERNAL EXPLORATORY',protocol_sha256=sha(root/'PROTOCOL.json'),scope_sha256=sha(root/'COORDINATOR_REVIEW_SCOPE.json'),source_pins_verified=len(p['source_pins']),input_fingerprints_verified=len(p['inputs']),all36_original_masks_exact=True,score_cells=36,aggregation_max_abs=maxagg,cached_reconstructions=72,cached_block_r_max_abs=maxr,cached_MSE_Hann_max_abs=maxmse,cached_3s_r_max_abs=max3,first4_fallback_max_abs=maxfb,Hann_aggregate_identity_max_abs=maxhann,original_FT_raw_bridge_records=bridgerecords,original_FT_raw_wave_max_abs=maxwave,original_FT_all_block_r_max_abs=maxoldr,VAL_gold_capture_max_abs=maxcapture,original_Decoder_max_abs=maxdecoder,actual_main_NN_windows=mainN,actual_extra_VAL_windows=extraVal,actual_extra_Decoder_windows=extraDecoder,actual_global64_batches=batchN,new12cache_bytes=cachebytes,undefined3s_by_cell=undefined,means=means,contrasts=contrasts,per_seed=values,reconstruction_cases=cases,completion=done,review_cost=dict(wall_seconds=time.time()-t0,CPU_seconds=time.process_time()-cpu0,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,encoder_or_head_forwards=0,newfits=0,new_NPY=0,cached_readout_windows=sum((r['common_samples']-192)//96+1 for r in case_rows)*18))
with (root/'COORDINATOR_NUMERICAL_AUDIT.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in out.items() if k not in ['reconstruction_cases','undefined3s_by_cell','per_seed','contrasts']},indent=2))
