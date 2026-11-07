import os
os.environ.update(OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUDA_VISIBLE_DEVICES='')
from pathlib import Path
import json,hashlib,time,resource
import numpy as np
t0=time.time();cpu0=time.process_time()
r=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_finetuned_readout_transfer_20261101')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (r/'COMPLETED.json').exists() and not (r/'FAILURE.json').exists()
p=read(r/'PROTOCOL.json');done=read(r/'COMPLETED.json');summary=read(r/'SUMMARY.json');scope=read(r/'COORDINATOR_REVIEW_SCOPE.json')
assert sha(r/'PROTOCOL.json')=='54d424d5c26f1ab4aee4712c2efb149739a9e70ec876d2698391e4e5b6d0c980'
assert done['new_ridge_fits']==27 and not done['test_DTU'] and not done['goal_complete']
for f,h in p['source_pins'].items():assert sha(f)==h,f
for f,v in p['inputs'].items():
 st=Path(f).stat();assert st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],f
rows=p['records']['val'];rm={x['record']:x for x in rows};masks={x['record']:[b for b in p['mask']['blocks'] if b['record']==x['record']] for x in rows}
key=lambda b:(b['record'],b['phase'],b['subject'],b['story'],b['source_family'],b['start'],b['samples'])
expected=[(x['record'],x['phase'],x['subject'],x['story'],x.get('source_family',x['story']),b['start'],b['samples']) for x in rows for b in masks[x['record']] if b['eligible']]
assert len(expected)==752 and len(rows)==49
maxagg=0.;maxnorm=0.;maxbias=0.;maxr=0.;maxmse=0.;max3=0.;maxhann=0.;maxfb=0.;cases=[];sol=[];basevals={};bridge_rows=[];undef={}
def checkmetric(m,label):
 global maxagg,maxhann
 assert [key(b) for b in m['segments']]==expected
 assert [x['record'] for x in m['records']]==[x['record'] for x in rows]
 assert len(m['people'])==49 and len(m['segments'])==752 and sum(x['windows'] for x in m['records'])==30500
 for field in ['r','MSE','weighted_single_MSE','disagreement']:
  people={x['subject']:float(np.mean([b[field] for b in m['segments'] if b['subject']==x['subject']])) for x in m['people']}
  maxagg=max(maxagg,max(abs(x[field]-people[x['subject']]) for x in m['people']),abs(float(np.mean(list(people.values())))-m['phases']['val'][field]))
 un=sum(x['window_r_undefined'] for x in m['records']);undef[label]=un
 assert un==m['phases']['val']['window_r_undefined'] and m['phases']['val']['window_r_defined']==30500-un
 if un:assert m['phases']['val']['r3s'] is None
 else:maxagg=max(maxagg,abs(sum(x['r3s']*x['windows'] for x in m['records'])/30500-m['phases']['val']['r3s']))
 maxagg=max(maxagg,abs(sum(x['window_MSE']*x['windows'] for x in m['records'])/30500-m['phases']['val']['window_MSE']))
 for x in m['segments']:maxhann=max(maxhann,abs(x['weighted_single_MSE']-x['MSE']-x['disagreement']))
for seed in p['seeds']:
 folder=r/f'seed{seed}';br=read(folder/'FT_VAL_BRIDGE.json');tr=read(folder/'FT_TRAIN_CACHE.json');cc=read(folder/'CACHE_COMPLETED.json')
 assert br['passed'] and br['encoder_head_windows']==30500 and br['global64_batches']==477 and br['extra_original_Decoder_windows']==100
 assert br['loss_absolute_difference']<=1e-7 and all(x['original_units_max_abs']<=1e-6 for x in br['original_Decoder_wave_checks'])
 assert [(x['first'],x['last']) for x in br['original_Decoder_wave_checks']]==[(0,64),(30464,30500)]
 assert tr['encoder_head_windows']==252454 and tr['global64_batches']==3945 and tr['extra_original_Decoder_windows']==0
 assert cc['checkpoint_sha']==p['checkpoints'][str(seed)]['checkpoint_sha256'] and cc['new_NN_updates']==0
 bridge_rows.append(dict(seed=seed,loss_difference=br['loss_absolute_difference'],wave_max=max(x['original_units_max_abs'] for x in br['original_Decoder_wave_checks'])))
 for phase,n in [('train',252454),('val',30500)]:
  for kind,shape in [('latent',(n,8,192)),('P',(n,192))]:
   a=np.load(folder/f'{phase}_{kind}.npy',mmap_mode='r');assert a.shape==shape and a.dtype==np.float32
 baseline=read(folder/'P_FT_BASELINE.json');cal=baseline['calibration'];assert cal['a']>0 and cal['TRAIN_physical_points']==sum(96*x['windows']+94 for x in p['records']['train'])==24275628
 assert baseline['positive_affine_block_r_max_abs']<=1e-10
 checkmetric(baseline,f'{seed}:P_FT');basevals[seed]=baseline['phases']['val']['r'];maxagg=max(maxagg,abs(basevals[seed]-summary['baseline'][str(seed)]['phases']['val']['r']))
 st=np.load(folder/'TRAIN_TRANSFORMS.npz');mom=np.load(folder/'TRAIN_MOMENTS.npz');assert int(mom['n'])==250750
 for j,d in enumerate([1536,192,1536]):
  assert st[f'mean_{j}'].shape==st[f'std_{j}'].shape==st[f'constant_{j}'].shape==(d,)
  assert np.isfinite(st[f'mean_{j}']).all() and np.isfinite(st[f'std_{j}']).all() and (st[f'std_{j}']>0).all()
  assert (st[f'std_{j}'][st[f'constant_{j}']]==1).all()
 G=mom['G'];H=mom['H'];xm=mom['x_mean'];rr=mom['residual_mean'];assert G.shape==(3264,3264) and H.shape==(3264,192)
 for arm in p['arms']:
  ids=np.arange(1536,1728) if arm=='Ponly' else np.arange(1728 if arm=='C' else 3264)
  ga=G[np.ix_(ids,ids)];ha=H[ids]
  for lam in p['lambdas']:
   tag=f'{arm}_lambda{lam:g}';q=read(folder/f'{tag}_METRICS.json');c=np.load(folder/f'{tag}_COEFFICIENTS.npz')
   B=c['coef'];bias=c['bias'];assert B.shape==(len(ids),192) and bias.shape==(192,) and np.isfinite(B).all() and np.isfinite(bias).all()
   normal=float(abs(ga@B+lam*B-ha).max());biaserr=float(abs(bias-(rr-xm[ids]@B)).max())
   assert normal<=1e-7*max(1.,float(abs(ha).max())) and biaserr<1e-12
   maxnorm=max(maxnorm,normal);maxbias=max(maxbias,biaserr)
   assert q['parameters']==B.size+bias.size and q['seed']==seed and q['arm']==arm and q['lam']==lam
   checkmetric(q['metrics'],f'{seed}:{tag}');maxagg=max(maxagg,abs(q['r60s']-q['metrics']['phases']['val']['r']))
   old=next(x for x in summary['all_solutions'] if x['seed']==seed and x['arm']==arm and x['lam']==lam)
   assert q['r60s']==old['r60s'];sol.append(dict(seed=seed,arm=arm,lam=lam,r=q['r60s']))
 selections={x['arm']:x for x in summary['selection']}
 z=np.load(folder/'val_latent.npy',mmap_mode='r');rawP=np.load(folder/'val_P.npy',mmap_mode='r')
 def norm(x,j):
  out=(x-st[f'mean_{j}'])/st[f'std_{j}'];out[:,st[f'constant_{j}']]=0;return out
 for arm in p['arms']:
  sel=selections[arm];lam=sel['selected_lambda']
  m=baseline if sel['P_FT_baseline_selected'] else read(folder/f'{arm}_lambda{lam:g}_METRICS.json')['metrics']
  coef=None if sel['P_FT_baseline_selected'] else np.load(folder/f'{arm}_lambda{lam:g}_COEFFICIENTS.npz')
  for record in scope['records']:
   row=rm[record];o,w=row['offset'],row['windows'];n=row['common_samples']
   anchor=cal['a']*np.asarray(rawP[o:o+w],float)+cal['b'];pred=anchor.copy()
   if coef is not None:
    zn=norm(np.asarray(z[o+4:o+w],float).reshape(w-4,1536),0);pn=norm(anchor[4:],1);qn=norm(zn*zn,2)
    x=pn/np.sqrt(192) if arm=='Ponly' else np.concatenate([zn/np.sqrt(1536),pn/np.sqrt(192)]+([qn/np.sqrt(1536)] if arm=='Ccap' else []),1)
    pred[4:]+=x@coef['coef']+coef['bias']
   maxfb=max(maxfb,float(abs(pred[:4]-anchor[:4]).max()))
   h=np.hanning(192);total=np.zeros(n);weight=np.zeros(n);sqpred=np.zeros(n);werr=np.zeros(n)
   target=np.asarray(np.load(row['envelope_path'],mmap_mode='r'),float).reshape(-1)
   for j in range(w):
    a=j*96;b=a+192;total[a:b]+=h*pred[j];weight[a:b]+=h;sqpred[a:b]+=h*pred[j]**2;werr[a:b]+=h*(pred[j]-target[a:b])**2
   covered=weight>1e-8;fused=np.zeros(n);weighted=np.zeros(n);dis=np.zeros(n)
   np.divide(total,weight,out=fused,where=covered);np.divide(werr,weight,out=weighted,where=covered)
   np.divide(sqpred,weight,out=dis,where=covered);dis[covered]-=fused[covered]**2
   saved={(x['record'],x['start']):x for x in m['segments']};lr=0.;le=0.
   for block in masks[record]:
    a=block['start'];b=a+3840;k=covered[a:b];assert int(k.sum())==block['samples']
    if not block['eligible']:continue
    x=fused[a:b][k];y=target[a:b][k];ref=saved[(record,a)]
    assert np.ptp(x)>0 and np.ptp(y)>0
    lr=max(lr,abs(float(np.corrcoef(x,y)[0,1])-ref['r']))
    le=max(le,abs(float(np.mean((x-y)**2))-ref['MSE']),abs(float(weighted[a:b][k].mean())-ref['weighted_single_MSE']),abs(float(dis[a:b][k].mean())-ref['disagreement']))
   yt=np.stack([target[j*96:j*96+192] for j in range(w)]).astype(np.float32).astype(float)
   ok=(np.ptp(pred,axis=1)>0)&(np.ptp(yt,axis=1)>0);rec=next(x for x in m['records'] if x['record']==record)
   assert rec['window_r_undefined']==int((~ok).sum())
   if ok.all():
    pp=pred-pred.mean(1,keepdims=True);yy=yt-yt.mean(1,keepdims=True)
    s=float(np.mean(np.sum(pp*yy,1)/np.sqrt(np.sum(pp*pp,1)*np.sum(yy*yy,1))));max3=max(max3,abs(s-rec['r3s']))
   else:assert rec['r3s'] is None
   le=max(le,abs(float(np.mean((pred-yt)**2))-rec['window_MSE']))
   maxr=max(maxr,lr);maxmse=max(maxmse,le);cases.append(dict(seed=seed,arm=arm,lam=lam,record=record,r_max=lr,MSE_and_Hann_max=le))
bm=float(np.mean(list(basevals.values())));selected={}
for arm in p['arms']:
 candidates=[(float(np.mean([q['r'] for q in sol if q['arm']==arm and q['lam']==lam])),lam) for lam in p['lambdas']]
 value,lam=max(candidates);use=value>bm;s=next(x for x in summary['selection'] if x['arm']==arm)
 assert s['selected_lambda']==(lam if use else None) and s['P_FT_baseline_selected']==(not use)
 maxagg=max(maxagg,abs(s['mean_VAL60s']-(value if use else bm)))
 selected[arm]={seed:(next(q['r'] for q in sol if q['arm']==arm and q['seed']==seed and q['lam']==lam) if use else basevals[seed]) for seed in p['seeds']}
assert len(sol)==27 and len(cases)==18 and maxfb==0 and maxagg<1e-12 and maxr<1e-12 and maxmse<1e-12 and max3<1e-12 and maxhann<1e-12
vals={'P_FT':basevals,**selected};means={a:float(np.mean(list(v.values()))) for a,v in vals.items()}
contrasts={}
for a,b in [('Ccap','P_FT'),('Ccap','Ponly'),('C','Ponly'),('Ccap','C')]:
 dd=[vals[a][seed]-vals[b][seed] for seed in p['seeds']];contrasts[f'{a}-{b}']=dict(by_seed=dd,mean=float(np.mean(dd)))
out=dict(status='PASS_ACTUAL_NUMERICAL_RESULTS',goal_complete=False,protocol_sha256=sha(r/'PROTOCOL.json'),review_scope_sha256=sha(r/'COORDINATOR_REVIEW_SCOPE.json'),source_pins_verified=len(p['source_pins']),input_fingerprints_verified=len(p['inputs']),learned_scores=27,baseline_scores=3,all_original_VAL_masks_exact=True,raw_FT_bridges=bridge_rows,all27_coefficient_normal_max=maxnorm,bias_constraint_max=maxbias,aggregation_max=maxagg,cached_cases=18,cached_block_r_max=maxr,cached_MSE_Hann_max=maxmse,cached_3s_r_max=max3,first4_max=maxfb,Hann_aggregate_identity_max=maxhann,undefined3s=undef,selection_verified=True,means=means,contrasts=contrasts,reconstruction_cases=cases,completion=done,review_cost=dict(wall_seconds=time.time()-t0,CPU_seconds=time.process_time()-cpu0,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,encoder_or_head_forward=0,newfits=0))
with (r/'COORDINATOR_NUMERICAL_AUDIT.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in out.items() if k not in ['reconstruction_cases','undefined3s']},indent=2))
print('undefined3s_total',sum(undef.values()))
