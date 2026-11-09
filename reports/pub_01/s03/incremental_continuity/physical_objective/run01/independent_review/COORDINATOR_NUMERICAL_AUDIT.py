import os
os.environ.update(OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
from pathlib import Path
import json,hashlib,time,resource,gc
import numpy as np
import torch
torch.set_num_threads(8)
torch.cuda.set_device(0)
torch.cuda.reset_peak_memory_stats()
t0=time.time();cpu0=time.process_time()
root=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_physical_objective_20261101')
d8=root.parent/'s03_finetuned_readout_transfer_20261101'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (root/'COMPLETED.json').exists() and not (root/'FAILURE.json').exists()
p=read(root/'PROTOCOL.json');done=read(root/'COMPLETED.json');summary=read(root/'SUMMARY.json')
assert sha(root/'PROTOCOL.json')=='867042734282eeb7f9c7bf70e381996cb50072893afaedabd7075c30ee4f1499'
assert done['solves']==6 and not done['old_test_DTU'] and done['encoder_head_forward']==0
for f,h in p['source_pins'].items():assert sha(f)==h,f
for f,v in p['inputs'].items():
 st=Path(f).stat();assert st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],f
rows=p['records']['val'];trainrows=p['records']['train']
masks={x['record']:[b for b in p['mask']['blocks'] if b['record']==x['record']] for x in rows}
key=lambda b:(b['record'],b['phase'],b['subject'],b['story'],b['source_family'],b['start'],b['samples'])
expected=[(x['record'],x['phase'],x['subject'],x['story'],x.get('source_family',x['story']),b['start'],b['samples']) for x in rows for b in masks[x['record']] if b['eligible']]
assert len(expected)==752 and len(rows)==49
maxagg=0.;maxhann=0.;maxr=0.;maxmse=0.;max3=0.;maxfb=0.;maxobj=0.;maxres=0.;undef={};cases=[];numerical=[];vals={}
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
def features(z,base,row,st):
 o,w=row['offset'],row['windows']
 def norm(x,j):
  out=(x-st[f'mean_{j}'])/st[f'std_{j}'];out[:,st[f'constant_{j}']]=0;return out
 zn=norm(np.asarray(z[o+4:o+w],float).reshape(w-4,1536),0)
 pn=norm(base[o+4:o+w],1);qn=norm(zn*zn,2)
 return np.concatenate([zn/np.sqrt(1536),pn/np.sqrt(192),qn/np.sqrt(1536)],1)
h=np.hanning(192);T=p['counts']['common_physical_target_points'];lam=p['lambda_scalar']
assert T==sum(96*(r['windows']-4)+94 for r in trainrows)==24112044
def preconditioned_energy(B,b,moment,role):
 nfit=250750;nr=426;diag=np.zeros(192);bias_energy=0.
 for j in range(96):
  a=h[j]/(h[j]+h[j+96]);bprev=1-a;t=float(j<95)
  if role=='F':S=np.array([[nfit*a*a,(nfit-nr)*a*bprev],[(nfit-nr)*a*bprev,(nfit-nr)*bprev*bprev+nr*t]])/T
  else:S=np.diag([nfit*a,(nfit-nr)*bprev+nr*t])/T
  pair=[j,j+96];ids=[k for k,c in enumerate(pair) if c not in [0,191]];cols=[pair[k]-1 for k in ids]
  diag[pair]=np.diag(S);v=b[cols]
  bias_energy+=float(v@np.linalg.solve(S[np.ix_(ids,ids)],v))
 denom=moment[:,None]*diag[None,1:191]+lam
 return float(np.sum(B*B/denom)+bias_energy)
def train_direct(pred,base,role):
 # Independent full physical-axis overlap-add; never invokes D9 core or its pair adjoint.
 V=np.zeros(pred.shape);Vbase=np.zeros(pred.shape);totalW=0.;totalF=0.;totalD=0.;total_points=0
 for row in trainrows:
  o,w=row['offset'],row['windows'];n=(w-1)*96+192
  y=np.asarray(np.load(row['envelope_path'],mmap_mode='r')[:n],np.float32).astype(float).reshape(-1)
  assert len(y)==n
  den=np.zeros(n);summed=np.zeros(n);base_sum=np.zeros(n);werr=np.zeros(n);sumsq=np.zeros(n)
  for i in range(w):
   a=i*96;b=a+192;v=pred[o+i];q=base[o+i]
   den[a:b]+=h;summed[a:b]+=h*v;base_sum[a:b]+=h*q
   werr[a:b]+=h*(v-y[a:b])**2;sumsq[a:b]+=h*v*v
  keep=np.zeros(n,bool);keep[385:n-1]=True
  assert keep.sum()==96*(w-4)+94 and np.all(den[keep]>0)
  fused=np.zeros(n);fusedbase=np.zeros(n);weight=np.zeros(n)
  np.divide(summed,den,out=fused,where=den>0);np.divide(base_sum,den,out=fusedbase,where=den>0)
  np.divide(keep.astype(float),T*den,out=weight,where=den>0)
  total_points+=int(keep.sum());totalW+=float(np.sum(werr[keep]/den[keep]))
  totalF+=float(np.sum((fused[keep]-y[keep])**2))
  totalD+=float(np.sum(sumsq[keep]/den[keep]-fused[keep]**2))
  for i in range(w):
   a=i*96;b=a+192;wt=h*weight[a:b]
   if role=='F':
    V[o+i]=wt*(fused[a:b]-y[a:b]);Vbase[o+i]=wt*(fusedbase[a:b]-y[a:b])
   else:
    V[o+i]=wt*(pred[o+i]-y[a:b]);Vbase[o+i]=wt*(base[o+i]-y[a:b])
 assert total_points==T
 return V,Vbase,dict(W_data=totalW/T,F_data=totalF/T,disagreement=totalD/T)
for seed in p['seeds']:
 folder=root/f'seed{seed}';original=d8/f'seed{seed}'
 st=np.load(original/'TRAIN_TRANSFORMS.npz');cal=read(original/'P_FT_BASELINE.json')['calibration']
 anchor=np.load(original/'Ccap_lambda0.01_COEFFICIENTS.npz')
 momentfile=np.load(original/'TRAIN_MOMENTS.npz');moment=np.diag(momentfile['G']).copy()+momentfile['x_mean']**2
 z=np.load(original/'train_latent.npy',mmap_mode='r');rp=np.load(original/'train_P.npy',mmap_mode='r')
 base=cal['a']*np.asarray(rp,float)+cal['b']
 X=torch.zeros((252454,3264),dtype=torch.float64,device='cuda:0')
 active=np.zeros(252454,bool)
 for row in trainrows:
  o,w=row['offset'],row['windows'];X[o+4:o+w]=torch.as_tensor(features(z,base,row,st),device='cuda:0');active[o+4:o+w]=True
 assert active.sum()==250750
 for role in p['roles']:
  q=read(folder/f'{role}_METRICS.json');solver=read(folder/f'{role}_SOLVER.json')
  c=np.load(folder/f'{role}_COEFFICIENTS.npz');B=c['coef'];bias=c['bias']
  assert B.shape==(3264,192) and bias.shape==(192,) and np.isfinite(B).all() and np.isfinite(bias).all()
  assert np.array_equal(B[:,[0,191]],anchor['coef'][:,[0,191]]) and np.array_equal(bias[[0,191]],anchor['bias'][[0,191]])
  assert solver['iterations']<=200 and solver['hessian_actions']<=214
  assert solver['converged']==(solver['true_preconditioned_relative']<=1e-8)
  checkmetric(q['metrics'],f'{seed}:{role}');vals[(seed,role)]=q['r']
  maxagg=max(maxagg,abs(q['r']-q['metrics']['phases']['val']['r']))
  with torch.no_grad():
   pred=(X@torch.as_tensor(B,device='cuda:0')).cpu().numpy()
  pred+=active[:,None]*bias;pred+=base
  V,Vbase,objectives=train_direct(pred,base,role)
  def grad(v):
   with torch.no_grad():
    vb=torch.as_tensor(np.ascontiguousarray(v[:,1:191]),device='cuda:0')
    gg=(X.T@vb).cpu().numpy()
   return gg,v[active,1:191].sum(0)
  gb,gc0=grad(V);rb,rc=grad(Vbase);gb+=lam*B[:,1:191];rb=-rb;rc=-rc
  ratio=preconditioned_energy(gb,gc0,moment,role)**.5/max(preconditioned_energy(rb,rc,moment,role)**.5,1e-30)
  err=abs(ratio-solver['true_preconditioned_relative']);maxres=max(maxres,err);assert err<=1e-10
  penalty=float(lam*np.sum(B[:,1:191]**2))
  objectives.update(regularizer=penalty,W_total=objectives['W_data']+penalty,F_total=objectives['F_data']+penalty)
  objective_err=max(abs(value-solver['train_objectives'][field]) for field,value in objectives.items())
  maxobj=max(maxobj,objective_err);assert objective_err<=1e-9
  numerical.append(dict(seed=seed,role=role,reported_converged=solver['converged'],iterations=solver['iterations'],Hessian_actions=solver['hessian_actions'],reported_true_residual=solver['true_preconditioned_relative'],independent_true_residual=ratio,independent_below_formal_threshold=ratio<=1e-8,residual_absolute_difference=err,TRAIN_objective_max_abs=objective_err,TRAIN_objectives=objectives))
  del pred,V,Vbase,gb,rb
 del X,z,rp,base;gc.collect();torch.cuda.empty_cache()
 zv=np.load(original/'val_latent.npy',mmap_mode='r');rawP=np.load(original/'val_P.npy',mmap_mode='r')
 base=cal['a']*np.asarray(rawP,float)+cal['b']
 for role in p['roles']:
  c=np.load(folder/f'{role}_COEFFICIENTS.npz');m=read(folder/f'{role}_METRICS.json')['metrics']
  for row in (rows[0],rows[-1]):
   record=row['record'];o,w=row['offset'],row['windows'];n=row['common_samples'];pred=base[o:o+w].copy()
   pred[4:]+=features(zv,base,row,st)@c['coef']+c['bias']
   maxfb=max(maxfb,float(abs(pred[:4]-base[o:o+4]).max()))
   total=np.zeros(n);weight=np.zeros(n);sqpred=np.zeros(n);werr=np.zeros(n)
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
    xx=fused[a:b][k];yy=target[a:b][k];ref=saved[(record,a)]
    lr=max(lr,abs(float(np.corrcoef(xx,yy)[0,1])-ref['r']))
    le=max(le,abs(float(np.mean((xx-yy)**2))-ref['MSE']),abs(float(weighted[a:b][k].mean())-ref['weighted_single_MSE']),abs(float(dis[a:b][k].mean())-ref['disagreement']))
   yt=np.stack([target[j*96:j*96+192] for j in range(w)]).astype(np.float32).astype(float)
   ok=(np.ptp(pred,axis=1)>0)&(np.ptp(yt,axis=1)>0);rec=next(x for x in m['records'] if x['record']==record)
   assert rec['window_r_undefined']==int((~ok).sum())
   if ok.all():
    pp=pred-pred.mean(1,keepdims=True);yy=yt-yt.mean(1,keepdims=True)
    r3=float(np.mean(np.sum(pp*yy,1)/np.sqrt(np.sum(pp*pp,1)*np.sum(yy*yy,1))));max3=max(max3,abs(r3-rec['r3s']))
   else:assert rec['r3s'] is None
   le=max(le,abs(float(np.mean((pred-yt)**2))-rec['window_MSE']))
   maxr=max(maxr,lr);maxmse=max(maxmse,le);cases.append(dict(seed=seed,role=role,record=record,r_max=lr,MSE_and_Hann_max=le))
for role in p['roles']:
 maxagg=max(maxagg,abs(float(np.mean([vals[(s,role)] for s in p['seeds']]))-summary['mean_r'][role]))
for i,s in enumerate(p['seeds']):maxagg=max(maxagg,abs(vals[(s,'F')]-vals[(s,'W')]-summary['paired_F_minus_W'][i]))
assert len(numerical)==6 and len(cases)==12 and maxfb==0 and maxagg<1e-12 and maxr<1e-12 and maxmse<1e-12 and max3<1e-12 and maxhann<1e-12
out=dict(status='PASS_ACTUAL_NUMERICAL_RESULTS',goal_complete=False,protocol_sha256=sha(root/'PROTOCOL.json'),scope_sha256=sha(root/'COORDINATOR_REVIEW_SCOPE.json'),source_pins_verified=len(p['source_pins']),input_fingerprints_verified=len(p['inputs']),all6_VAL_original_masks_exact=True,aggregation_max=maxagg,TRAIN_independent_full_physical_gradient_checks=numerical,TRAIN_objective_max=maxobj,true_residual_absolute_difference_max=maxres,cached_cases=12,cached_block_r_max=maxr,cached_MSE_Hann_max=maxmse,cached_3s_r_max=max3,first4_max=maxfb,Hann_aggregate_identity_max=maxhann,undefined3s=undef,reconstruction_cases=cases,mean_r=summary['mean_r'],paired_F_minus_W=summary['paired_F_minus_W'],review_cost=dict(wall_seconds=time.time()-t0,CPU_seconds=time.process_time()-cpu0,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),TRAIN_readout_window_predictions=252454*6,encoder_or_head_forward=0,newfits=0,new_NPY=0))
with (root/'COORDINATOR_NUMERICAL_AUDIT.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in out.items() if k not in ['reconstruction_cases','undefined3s']},indent=2))

