import os
os.environ.update(OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUDA_VISIBLE_DEVICES='')
import json,hashlib,time,resource,shutil
from pathlib import Path
import numpy as np
START=time.time(); CPU=time.process_time()
root=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_observation_state_probe_20261101')
base=root.parent; d8=base/'s03_finetuned_readout_transfer_20261101'; old=base/'s03_strong_baselines_20261015/transformer_available01'
read=lambda x:json.loads(Path(x).read_text())
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
assert (root/'COMPLETED.json').exists() and not (root/'FAILURE.json').exists()
p=read(root/'PROTOCOL.json'); done=read(root/'COMPLETED.json'); cost=read(root/'COST.json')
assert p['seeds']==[1024,2026,31415] and p['scores']==24 and p['new_ridge_fits']==15
assert set(p['records'])=={'train','val'} and not p['test_DTU'] and not p['new_download']
for name,h in p['source_pins'].items(): assert sha(name)==h,name
for name,v in p['inputs'].items():
 st=Path(name).stat(); assert st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],name
rows=p['records']; tr=rows['train']; val=rows['val']
assert len(tr)==426 and len(val)==49 and sum(r['windows'] for r in val)==30500
assert sum(r['windows']-41 for r in tr)==234988 and sum(r['windows']-41 for r in val)==28491
seeds=p['seeds']; roles=['Ccap','Add','CurrentCov','RecordPrefix','PersonTRAIN','PastCov','PastOlder','PastNeutral']
states=['CurrentCov','RecordPrefix','PersonTRAIN','PastCov','OlderCov']
assert p['score_roles']==roles and p['cache_roles']==states
h=np.hanning(192); tri=np.triu_indices(64,1)
def norm(x,m,s,c):
 y=(x-m)/s; y[...,c]=0.; return y
def saved_norm(x,f,prefix):
 return norm(x,f[prefix+'mean'],f[prefix+'std'],f[prefix+'constant'])
def corr(x,y):
 x=x-x.mean();y=y-y.mean();return float(x@y/np.sqrt((x@x)*(y@y)))
def desc(x):
 center=x-x.mean(0); k=center.T@center/len(x);return desc_cov(k)
def desc_cov(k):
 d=np.maximum(np.diag(k),1e-8);c=k[tri]/np.sqrt(d[tri[0]]*d[tri[1]])
 assert np.max(np.maximum(abs(c)-1.,0.))<=1e-10
 return np.r_[np.log(d),np.clip(c,-1.,1.)]
nrm=np.load(old/'prepared/train_only_normalization.npz')
def raw(row,a,b):
 ar=np.load(row['eeg_path'],mmap_mode='r')
 return ((np.asarray(ar[:,a:b].T,np.float32)-nrm['mean'])/nrm['std']).astype(np.float32).astype(np.float64)
def covariance(x):
 y=x-x.mean(0);return y.T@y/len(y)
cache={phase:{role:np.load(root/f'{phase}_{role}.npy',mmap_mode='r') for role in states} for phase in rows}
provenance=read(root/'PERSON_REFERENCE_PROVENANCE.json')['TRAIN_records_by_subject']
actual={s:[r['record'] for r in tr if r['subject']==s] for s in sorted({r['subject'] for r in tr})}
assert provenance==actual
for phase,rr in rows.items():
 offset=0
 for row in rr:
  w=row['windows']; o=row['offset']; assert o==offset and w==(row['common_samples']-192)//96+1
  assert 96*(41-41)==0 and 96*(w-2)<=96*(w-1) and row['subject'] in provenance
  for role in states:
   c=cache[phase][role]
   assert c.dtype==np.float32 and c.shape==(sum(r['windows'] for r in rr),8)
   assert np.isfinite(c[o:o+w]).all() and np.array_equal(c[o:o+41],np.zeros((41,8),np.float32))
   if role in ['RecordPrefix','PersonTRAIN']: assert np.all(c[o+41:o+w]==c[o+41])
  offset+=w
statef=np.load(root/'STATE_TRAIN_TRANSFORMS.npz')
assert int(statef['n'])==4*234988
direct=[]; pref={}
for phase in ['train','val']:
 for row in [rows[phase][0],rows[phase][-1]]:
  reference_rows=[r for r in tr if r['subject']==row['subject'] and (phase=='val' or r['record']!=row['record'])]
  assert len(reference_rows)>=2 and all(r['record']!=row['record'] for r in reference_rows)
  for rr in reference_rows:
   if rr['record'] not in pref:pref[rr['record']]=covariance(raw(rr,0,1920))
  pc=sum(pref[rr['record']] for rr in reference_rows)/len(reference_rows)
  w=row['windows'];o=row['offset']
  for i in [41,(41+w-1)//2,w-1]:
   s=96*(i-1)
   intervals={'CurrentCov':(96*i,96*i+192),'RecordPrefix':(0,1920),'PastCov':(s-1920,s),'OlderCov':(s-3840,s-1920)}
   assert intervals['PastCov'][1]==s and intervals['OlderCov'][1]==intervals['PastCov'][0]
   for role in states:
    if role=='PersonTRAIN':d=desc_cov(pc)
    else:
     a,b=intervals[role];assert 0<=a<b<=row['common_samples'];d=desc(raw(row,a,b))
    v=saved_norm(d,statef,'')@statef['basis']; actual_v=np.asarray(cache[phase][role][o+i],np.float64)
    delta=float(abs(v-actual_v).max())
    assert np.allclose(v,actual_v,atol=2e-6,rtol=2e-6),(phase,row['record'],i,role,delta)
    direct.append(dict(phase=phase,record=row['record'],window=i,role=role,max_abs=delta))
assert len(direct)==60
key=lambda b:(b['record'],b['phase'],b['subject'],b['start'],b['samples'])
mask=[(r['record'],r['phase'],r['subject'],b['start'],b['samples']) for r in val for b in p['mask']['blocks'] if b['record']==r['record'] and b['eligible']]
assert len(mask)==752
aggmax=0.;normalmax=0.;biasmax=0.;rmax=0.;msemax=0.;r3max=0.;cases=[];values=[];equations=[]
for seed in seeds:
 sdir=root/f'seed{seed}'; prev=d8/f'seed{seed}'
 f=np.load(sdir/'TRAIN_TRANSFORMS.npz'); oldf=np.load(prev/'TRAIN_TRANSFORMS.npz')
 z=np.load(prev/'val_latent.npy',mmap_mode='r');pp=np.load(prev/'val_P.npy',mmap_mode='r')
 cal=read(prev/'P_FT_BASELINE.json')['calibration']; cc=np.load(prev/'Ccap_lambda0.01_COEFFICIENTS.npz')
 def oldnorm(x,j):return norm(x,oldf[f'mean_{j}'],oldf[f'std_{j}'],oldf[f'constant_{j}'])
 coef={}
 for role in roles[1:6]:
  co=np.load(sdir/f'{role}_COEFFICIENTS.npz');mom=np.load(sdir/f'{role}_TRAIN_MOMENTS.npz')
  B,b=co['coef'],co['bias'];G,H=mom['G'],mom['H']
  d=32 if role=='Add' else 296
  assert int(mom['n'])==234988 and B.shape==(d,192) and b.shape==(192,)
  ne=float(abs((G+.01*np.eye(d))@B-H).max());be=float(abs(b-(mom['residual_mean']-mom['x_mean']@B)).max())
  assert ne<=1e-7*max(1.,float(abs(H).max())) and be<=1e-10
  normalmax=max(normalmax,ne);biasmax=max(biasmax,be);coef[role]=(B,b)
  equations.append(dict(seed=seed,role=role,normal_max_abs=ne,intercept_max_abs=be))
 expected_anchor=read(sdir/'Ccap_METRICS.json')['full']
 for role in roles:
  data=read(sdir/f'{role}_METRICS.json');m=data['full']
  assert [key(b) for b in m['segments']]==mask and len(m['records'])==49
  for label,part in [('full',m),('warm',data['subsets']['warm']),('cold',data['subsets']['cold'])]:
   want=[k for k in mask if label=='full' or (label=='warm' and k[3]>=4032) or (label=='cold' and k[3]==0)]
   assert [key(b) for b in part['segments']]==want
   person=part['people'];segments=part['segments'];v=part['phases']['val']
   assert len(person)==49 and v['people']==49 and v['blocks']==len(want) and v['records']==49
   for field in ['r','MSE','weighted_single_MSE','disagreement']:
    scores=[]
    for personrow in person:
     bb=[b for b in segments if b['subject']==personrow['subject']]
     expected=float(np.mean([b[field] for b in bb]));scores.append(expected)
     aggmax=max(aggmax,abs(expected-personrow[field]))
     assert personrow['blocks']==len(bb) and personrow['records']==len({b['record'] for b in bb})
    aggmax=max(aggmax,abs(float(np.mean(scores))-v[field]))
   for b in segments:
    assert abs(b['weighted_single_MSE']-b['MSE']-b['disagreement'])<=1e-8
   if label=='cold':
    anchorblocks={key(b):b for b in expected_anchor['segments']}
    assert all(b['r']==anchorblocks[key(b)]['r'] and b['MSE']==anchorblocks[key(b)]['MSE'] for b in segments)
  rr=m['records'];v=m['phases']['val'];wn=sum(r['windows'] for r in rr)
  assert wn==30500 and v['windows']==wn
  for field in ['r3s','window_MSE']:
   assert all(r[field] is not None for r in rr)
   aggmax=max(aggmax,abs(sum(r[field]*r['windows'] for r in rr)/wn-v[field]))
  pairs=sum(r['overlap_pairs'] for r in rr)
  aggmax=max(aggmax,abs(sum(r['overlap_SSE_mean_sum'] for r in rr)/pairs-v['overlap_MSE']))
  assert v['overlap_pairs']==pairs
  values.append(dict(seed=seed,role=role,**{k:v[k] for k in ['r','MSE','r3s','window_MSE','weighted_single_MSE','disagreement']},warm_r=data['subsets']['warm']['phases']['val']['r'],cold_r=data['subsets']['cold']['phases']['val']['r']))
  for row in [val[0],val[-1]]:
   o,w=row['offset'],row['windows']; n=row['common_samples']
   anchor=cal['a']*np.asarray(pp[o:o+w],np.float64)+cal['b']
   zz=oldnorm(np.asarray(z[o+4:o+w],np.float64).reshape(w-4,1536),0)
   x=np.concatenate([zz/np.sqrt(1536),oldnorm(anchor[4:],1)/np.sqrt(192),oldnorm(zz**2,2)/np.sqrt(1536)],axis=1)
   anchor[4:]+=x@cc['coef']+cc['bias'];pred=anchor.copy()
   if role!='Ccap':
    q=saved_norm(oldnorm(np.asarray(z[o+41:o+w],np.float64).reshape(w-41,1536),0)@f['current_basis'],f,'q_')
    if role=='Add':x=q/np.sqrt(32);model='Add'
    else:
     model='PastCov' if role in ['PastCov','PastOlder','PastNeutral'] else role
     state='OlderCov' if role=='PastOlder' else role
     hh=np.zeros((w-41,8)) if role=='PastNeutral' else saved_norm(np.asarray(cache['val'][state][o+41:o+w],np.float64),f,'h_')
     products=saved_norm((q[:,:,None]*hh[:,None,:]).reshape(w-41,256),f,'product_')
     x=np.concatenate([q/np.sqrt(32),hh/np.sqrt(8),products/16.],axis=1)
    B,b=coef[model];pred[41:]+=x@B+b
   assert np.array_equal(pred[:41],anchor[:41])
   target=np.asarray(np.load(row['envelope_path'],mmap_mode='r'),np.float64).reshape(-1)
   summed=np.zeros(n);weight=np.zeros(n);weighted=np.zeros(n);disagreement=np.zeros(n)
   for i,pr in enumerate(pred):
    a=96*i;summed[a:a+192]+=h*pr;weight[a:a+192]+=h;weighted[a:a+192]+=h*(pr-target[a:a+192])**2
   keep=weight>1e-8;summed[keep]/=weight[keep];weighted[keep]/=weight[keep]
   alpha=h[96:]/(h[96:]+h[:96])
   for i in range(w-1):disagreement[(i+1)*96:(i+2)*96]=alpha*(1-alpha)*(pred[i,96:]-pred[i+1,:96])**2
   fusederr=(summed-target[:n])**2
   lr=0.;le=0.
   for block in [b for b in m['segments'] if b['record']==row['record']]:
    a=block['start'];sl=slice(a,a+3840);k=keep[sl]
    lr=max(lr,abs(corr(summed[sl][k],target[sl][k])-block['r']))
    for name,arr in [('MSE',fusederr),('weighted_single_MSE',weighted),('disagreement',disagreement)]:
     le=max(le,abs(float(arr[sl][k].mean())-block[name]))
   yw=np.stack([target[i*96:i*96+192] for i in range(w)]).astype(np.float32).astype(np.float64)
   xx=pred-pred.mean(1,keepdims=True);yy=yw-yw.mean(1,keepdims=True)
   r3=float(np.mean(np.sum(xx*yy,axis=1)/np.sqrt(np.sum(xx*xx,axis=1)*np.sum(yy*yy,axis=1))))
   record=next(r for r in m['records'] if r['record']==row['record'])
   e3=abs(r3-record['r3s']);le=max(le,abs(float(((pred-yw)**2).mean())-record['window_MSE']))
   assert lr<=1e-10 and e3<=1e-10 and le<=1e-8
   rmax=max(rmax,lr);msemax=max(msemax,le);r3max=max(r3max,e3)
   cases.append(dict(seed=seed,role=role,record=row['record'],block_r_max_abs=lr,MSE_max_abs=le,r3s_max_abs=e3))
assert len(equations)==15 and len(cases)==48 and len(values)==24 and aggmax<=1e-10
summary=read(root/'SUMMARY.json');contrasts={}
for a,b in [('PastCov','Ccap'),('PastCov','CurrentCov'),('PastCov','RecordPrefix'),('PastCov','PersonTRAIN'),('PastCov','PastOlder'),('PastCov','PastNeutral')]:
 ds=[next(v['r'] for v in values if v['seed']==s and v['role']==a)-next(v['r'] for v in values if v['seed']==s and v['role']==b) for s in seeds]
 expected=summary['primary_contrasts'][f'{a}-{b}'];assert np.array_equal(ds,expected['seed_differences']) and bool(all(v>0 for v in ds))==expected['all3positive']
 contrasts[f'{a}-{b}']=dict(seed_differences=ds,mean=float(np.mean(ds)),all3positive=all(v>0 for v in ds))
condition=all(v['all3positive'] for v in contrasts.values());assert condition==summary['condition_met']
files=[x for x in root.rglob('*') if x.is_file()]
npybytes=sum(x.stat().st_size for dd in base.glob('s03_*_20261101') for x in dd.rglob('*.npy'))
arraybytes=sum(x.stat().st_size for x in files if x.suffix in ['.npy','.npz'])
assert npybytes<=25*1024**3 and arraybytes<=128*1024**2 and sum(x.stat().st_size for x in files)<=256*1024**2
assert cost['new_ridge_fits']==15 and cost['nofit_interventions']==6 and cost['VAL_role_seed_scores']==24 and cost['NN_forwards']==cost['NN_updates']==0
assert done['wall_seconds']<3600 and done['peak_RSS_bytes']<8*1024**3 and shutil.disk_usage(root).free>=20*1024**3
out=dict(status='PASS_ACTUAL_NUMERICAL_RESULTS',goal_complete=False,scientific_status=p['scientific_status'],protocol_sha256=sha(root/'PROTOCOL.json'),scope_sha256=sha(root/'COORDINATOR_NUMERICAL_REVIEW_SCOPE.json'),source_hashes=len(p['source_pins']),input_size_mtime_checks=len(p['inputs']),input_content_rehashed=False,all475_cache_support_checks=True,normal_equations=equations,normal_equation_max_abs=normalmax,intercept_max_abs=biasmax,raw_direct_state_checks=direct,raw_state_cache_max_abs=max(c['max_abs'] for c in direct),cached_reconstructions=cases,block_r_max_abs=rmax,MSE_max_abs=msemax,r3s_max_abs=r3max,aggregation_max_abs=aggmax,score_sets=24,condition_met=condition,contrasts=contrasts,per_seed=values,means={role:{k:float(np.mean([v[k] for v in values if v['role']==role])) for k in ['r','MSE','r3s','window_MSE','weighted_single_MSE','disagreement','warm_r','cold_r']} for role in roles},new_numeric_array_bytes=arraybytes,all_SS07_NPY_bytes=npybytes,completion=done,review_cost=dict(wall_seconds=time.time()-START,CPU_seconds=time.process_time()-CPU,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_NPY=0,new_fits=0,neural_forwards=0))
with (root/'COORDINATOR_NUMERICAL_AUDIT.json').open('x') as stream:json.dump(out,stream,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in out.items() if k not in ['normal_equations','raw_direct_state_checks','cached_reconstructions','per_seed']},indent=2))
