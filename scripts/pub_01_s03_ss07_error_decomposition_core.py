"""Fixed descriptive projections and output-space gradients; no fitting or NN."""
import json,hashlib,math,resource,shutil,time
from pathlib import Path
import numpy as np

BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01')
ROOT=BASE/'s03_full_ft_error_decomposition_20261102'
D12=BASE/'s03_full_ft_history_20261102'
SEEDS=(1024,2026,31415);ARMS=('H_full','C_match','P_prior')
HANN=np.hanning(192);EPS=np.finfo(np.float64).eps
NUMERIC=('mean_a','mean_d','mean_y','alpha','mean_e','var_a','var_e','var_p','var_y','cov_a_y','cov_e_y',
 'orthogonality_cov','r_anchor','r_new','r_reconstructed','delta_r','r_anchor_normalization_delta','r_orthogonal_delta',
 'r_identity_error','MSE_anchor','MSE_new','delta_MSE','MSE_bias','MSE_parallel','MSE_orthogonal','MSE_identity_error','r_denominator_ratio')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def dump(p,j):
    with Path(p).open('x') as f:json.dump(j,f,indent=2,allow_nan=False)
def guard():
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8*1024**2
    assert shutil.disk_usage(ROOT).free>=20*1024**3
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<128*1024**2
    assert time.time()<read(ROOT/'PROTOCOL.json')['deadline_unix']
def weights(windows,n):
    weight=np.zeros(n,dtype='f8')
    for i in range(windows):weight[96*i:96*i+192]+=HANN
    return weight,weight>1e-8
def fuse(pred,n):
    weight,keep=weights(len(pred),n);out=np.zeros(n,dtype='f8')
    for i,p in enumerate(pred):out[96*i:96*i+192]+=HANN*p
    out[keep]/=weight[keep]
    return out,weight,keep
def low_variance(x,var):return bool(var<=64*EPS*float(np.mean(x*x)))
def decompose(a,p,y):
    a=np.asarray(a,dtype='f8');p=np.asarray(p,dtype='f8');y=np.asarray(y,dtype='f8')
    row={k:None for k in NUMERIC};row.update(n=len(a),projection_defined=False,Pearson_defined=False,flags='')
    if len(a)<2:row['flags']='INSUFFICIENT_POINTS';return row
    if not all(np.isfinite(x).all() for x in (a,p,y)):row['flags']='NONFINITE_INPUT';return row
    d=p-a;ma,md,my=map(float,(a.mean(),d.mean(),y.mean()));u=a-ma;v=d-md;yc=y-my;pc=p-p.mean()
    va=float(np.mean(u*u));vy=float(np.mean(yc*yc));vp=float(np.mean(pc*pc));cay=float(np.mean(u*yc))
    msea=float(np.mean((a-y)**2));msep=float(np.mean((p-y)**2));dmse=msep-msea
    scalar=[ma,md,my,va,vy,vp,cay,msea,msep,dmse]
    if not np.isfinite(scalar).all():row['flags']='NONFINITE_MOMENTS';return row
    flags=[]
    for name,x,var in [('A',a,va),('Y',y,vy),('P',p,vp)]:
        if low_variance(x,var):flags.append('LOW_VARIANCE_'+name)
    row.update(mean_a=ma,mean_d=md,mean_y=my,var_a=va,var_p=vp,var_y=vy,cov_a_y=cay,
        MSE_anchor=msea,MSE_new=msep,delta_MSE=dmse,MSE_bias=2*md*(ma-my)+md*md)
    if va>0:
        alpha=float(np.dot(u,v)/np.dot(u,u));e=v-alpha*u;ec=e-e.mean();ve=float(np.mean(ec*ec));cey=float(np.mean(ec*yc))
        parallel=(alpha*alpha+2*alpha)*va-2*alpha*cay;orth=ve-2*cey
        row.update(alpha=alpha,mean_e=float(e.mean()),var_e=ve,cov_e_y=cey,orthogonality_cov=float(np.mean(u*ec)),
            MSE_parallel=parallel,MSE_orthogonal=orth,MSE_identity_error=dmse-row['MSE_bias']-parallel-orth,projection_defined=True)
        assert abs(row['MSE_identity_error'])<=1e-10*max(1,msea,msep,abs(parallel),abs(orth))
        if vy>0:row['r_anchor']=cay/np.sqrt(va*vy)
        if vy>0 and vp>0:
            ratio=float(np.sqrt((1+alpha)**2+ve/va));ra=row['r_anchor'];rn=float(np.mean(pc*yc)/np.sqrt(vp*vy))
            if ratio>0 and np.isfinite(ratio):
                anc=(1+alpha)*ra/ratio;other=cey/(np.sqrt(va*vy)*ratio);recon=anc+other
                row.update(r_new=rn,r_reconstructed=recon,delta_r=rn-ra,r_anchor_normalization_delta=anc-ra,
                    r_orthogonal_delta=other,r_identity_error=rn-recon,r_denominator_ratio=ratio,Pearson_defined=True)
                assert abs(rn-recon)<=1e-10
    if not row['projection_defined']:flags.append('UNDEFINED_PROJECTION')
    if not row['Pearson_defined']:flags.append('UNDEFINED_PEARSON')
    row['flags']=';'.join(flags) or 'OK'
    return row

def physical_gradients(a,y,blocks,person_weight):
    """Full physical gradients, before window adjoint and learnability mask."""
    gf=np.zeros(len(a));gr=np.zeros(len(a));support=np.zeros(len(a),dtype=bool);badF=0;badR=0;max_angular=0.;low=0
    for b in blocks:
        s,t=b['start'],b['stop'];keep=np.asarray(b['keep_local'],dtype=int);idx=s+keep
        aa=a[idx];yy=y[idx];n=len(aa);w=person_weight
        if not np.isfinite(aa).all() or not np.isfinite(yy).all():badF+=1;badR+=1;continue
        f=2*w*(aa-yy)/n;gf[idx]=f;support[idx]=True
        u=aa-aa.mean();yc=yy-yy.mean();uu=float(u@u);tt=float(yc@yc)
        if uu<=0 or tt<=0:badR+=1;continue
        r=float((u@yc)/np.sqrt(uu*tt));g=w*(r*u/uu-yc/np.sqrt(uu*tt));gr[idx]=g
        fc=f-f.mean();perp=fc-u*(u@fc)/uu;identity=perp-(2*np.sqrt(uu*tt)/n)*g
        max_angular=max(max_angular,float(np.max(np.abs(identity))))
        low+=int(low_variance(aa,uu/n) or low_variance(yy,tt/n))
    return gf,gr,support,dict(undefined_F_blocks=badF,undefined_R_blocks=badR,low_variance_blocks=low,
        angular_identity_max_abs=max_angular)

def cosine(dot,aa,bb):return float(dot/np.sqrt(aa*bb)) if aa>0 and bb>0 else None
def gradient_record(anchor_windows,y_windows,truth,row,eligible,people_count,subject_blocks,total_windows):
    a,weight,keep=fuse(anchor_windows,row['common_samples']);blocks=[]
    for b in eligible:
        s,t=b['start'],b['stop'];blocks.append(dict(b,keep_local=np.flatnonzero(keep[s:t]).tolist()))
    gf,gr,primary,flags=physical_gradients(a,truth,blocks,1/(people_count*subject_blocks))
    sums=dict(MM=0.,FF=0.,RR=0.,MF=0.,MR=0.,FR=0.,M2_outside_primary=0.,M2_zero_Hann=0.,learnable_coordinates=0)
    for first in range(9,len(anchor_windows),128):
        last=min(first+128,len(anchor_windows));ids=np.arange(first,last)[:,None]*96+np.arange(192)[None,:]
        gm=2*(anchor_windows[first:last]-np.asarray(y_windows[first:last],dtype='f4').astype('f8').reshape(last-first,192))/(total_windows*192)
        den=weight[ids];ratio=np.zeros_like(den);np.divide(HANN[None,:],den,out=ratio,where=den>1e-8)
        f=gf[ids]*ratio;r=gr[ids]*ratio
        assert np.isfinite(gm).all() and np.isfinite(f).all() and np.isfinite(r).all()
        for key,value in [('MM',gm*gm),('FF',f*f),('RR',r*r),('MF',gm*f),('MR',gm*r),('FR',f*r)]:sums[key]+=float(np.sum(value))
        sums['M2_outside_primary']+=float(np.sum((gm*gm)[~primary[ids]]))
        sums['M2_zero_Hann']+=float(np.sum((gm*gm)[:,HANN==0]))
        sums['learnable_coordinates']+=(last-first)*192
    sums.update(physical_FF=float(gf@gf),physical_RR=float(gr@gr),physical_FR=float(gf@gr),**flags)
    sums['cos_M_F']=cosine(sums['MF'],sums['MM'],sums['FF']) if flags['undefined_F_blocks']==0 else None
    sums['cos_M_R']=cosine(sums['MR'],sums['MM'],sums['RR']) if flags['undefined_R_blocks']==0 else None
    sums['cos_F_R']=cosine(sums['FR'],sums['FF'],sums['RR']) if flags['undefined_F_blocks']+flags['undefined_R_blocks']==0 else None
    return sums

def aggregate_gradients(rows):
    additive=['MM','FF','RR','MF','MR','FR','M2_outside_primary','M2_zero_Hann','learnable_coordinates','physical_FF','physical_RR','physical_FR',
        'undefined_F_blocks','undefined_R_blocks','low_variance_blocks']
    out={k:sum(r[k] for r in rows) for k in additive}
    out['angular_identity_max_abs']=max(r['angular_identity_max_abs'] for r in rows)
    for a,b,key in [('M','F','MF'),('M','R','MR'),('F','R','FR')]:
        undefined=(b=='F' or a=='F') and out['undefined_F_blocks']>0 or (a=='R' or b=='R') and out['undefined_R_blocks']>0
        out[f'cos_{a}_{b}']=None if undefined else cosine(out[key],out[a+a],out[b+b])
    out['physical_cos_F_R_before_Hann']=None if out['undefined_R_blocks'] else cosine(out['physical_FR'],out['physical_FF'],out['physical_RR'])
    return out

def aggregate_blocks(rows):
    out=dict(blocks=len(rows),undefined_projection_blocks=sum(not r['projection_defined'] for r in rows),
        undefined_Pearson_blocks=sum(not r['Pearson_defined'] for r in rows),flagged_blocks=sum(r['flags']!='OK' for r in rows))
    for key in NUMERIC:
        v=[r[key] for r in rows];out[key]=float(np.mean(v)) if all(x is not None for x in v) else None
    return out

def synthetic_checks():
    rng=np.random.default_rng(20261102);a=rng.normal(size=1000)+.7;y=.2*a+rng.normal(size=1000)
    raw=rng.normal(size=1000);u=a-a.mean();raw-=raw.mean();e=raw-u*(u@raw)/(u@u)
    p=a+.2-.1*u+.15*e;s=decompose(a,p,y)
    assert abs(s['alpha']+.1)<1e-14 and abs(s['mean_d']-.2)<1e-14
    negative=decompose(a,a-2*u,y);assert abs(negative['r_new']+negative['r_anchor'])<1e-14
    assert not decompose(np.ones(20),np.ones(20),np.arange(20.))['projection_defined']
    assert not decompose(a,np.ones_like(a),y)['Pearson_defined']
    assert decompose(np.array([np.nan,1]),np.ones(2),np.ones(2))['flags']=='NONFINITE_INPUT'
    records=[];total_windows=0;subject_blocks={'A':4,'B':2}
    for ri,subject in enumerate(['A','A','B']):
        windows=90+ri;n=96*(windows-1)+192;t=np.arange(n);truth=(.4+np.sin(t*.013)+.3*np.cos(t*.029)).astype('f4').astype('f8')
        pred=np.stack([.3+.7*np.sin((i*96+np.arange(192))*.012)+.1*np.cos(np.arange(192)*.06) for i in range(windows)])
        wave,weight,keep=fuse(pred,n);blocks=[dict(start=s,stop=s+3840,keep_local=np.flatnonzero(keep[s:s+3840]).tolist()) for s in [0,3840]]
        gf,gr,support,flag=physical_gradients(wave,truth,blocks,1/(2*subject_blocks[subject]));assert flag['angular_identity_max_abs']<1e-15
        ids=np.arange(windows)[:,None]*96+np.arange(192)[None,:];ratio=np.zeros((windows,192));np.divide(HANN[None,:],weight[ids],out=ratio,where=weight[ids]>1e-8)
        adj=gr[ids]*ratio;adj[:9]=0;fadj=gf[ids]*ratio;fadj[:9]=0
        records.append(dict(pred=pred,truth=truth,blocks=blocks,gr=adj,gf=fadj,subject=subject,keep=keep,n=n,ids=ids,ratio=ratio))
        total_windows+=windows
    for rec in records:
        rec['gm']=2*(rec['pred']-rec['truth'][rec['ids']])/(total_windows*192);rec['gm'][:9]=0
    def loss(records,delta,step,kind):
        out=0.
        for rec,d in zip(records,delta):
            pp=rec['pred']+step*d;wave,_,_=fuse(pp,rec['n'])
            if kind=='M':out+=float(np.sum((pp-rec['truth'][rec['ids']])**2))/(total_windows*192);continue
            for b in rec['blocks']:
                idx=b['start']+np.asarray(b['keep_local']);xx=wave[idx];yy=rec['truth'][idx];xc=xx-xx.mean();yc=yy-yy.mean()
                if kind=='R':out-=float(xc@yc/np.sqrt((xc@xc)*(yc@yc)))/(2*subject_blocks[rec['subject']])
                else:out+=float(np.mean((xx-yy)**2))/(2*subject_blocks[rec['subject']])
        return out
    finite=[]
    for mode in ['random','cross_block_boundary','zero_Hann_endpoints','unscored_tail','fallback_fixed']:
        directions=[]
        for rec in records:
            d=np.zeros_like(rec['pred'])
            if mode=='random':d=rng.normal(size=d.shape);d[:9]=0
            elif mode=='cross_block_boundary':d[39,80:112]=rng.normal(size=32)
            elif mode=='zero_Hann_endpoints':d[9:,[0,191]]=1.
            elif mode=='unscored_tail':d[-1,100:140]=1.
            # fallback_fixed: perturbation of non-learnable parameters maps to0.
            directions.append(d)
        for kind,key in [('M','gm'),('F','gf'),('R','gr')]:
            analytic=sum(float(np.sum(rec[key]*d)) for rec,d in zip(records,directions));step=1e-5
            actual=(loss(records,directions,step,kind)-loss(records,directions,-step,kind))/(2*step)
            assert abs(analytic-actual)<1e-8,(mode,kind,analytic,actual)
            finite.append(dict(mode=mode,gradient=kind,analytic=analytic,finite_difference=actual,abs_error=abs(analytic-actual)))
    return dict(passed=True,projection_r_abs=abs(s['r_identity_error']),projection_MSE_abs=abs(s['MSE_identity_error']),
        negative_gain_and_constant_nonfinite_flags=True,physical_angular_identity_checked=True,Hann_adjoint_finite_differences=finite,
        records=3,people=2,person_blocks={'A':4,'B':2},synthetic_NN=0,optimizer_updates=0)
