"""D6 constrained float64 ridge and fixed-Hann diagnostics; no I/O side effects."""
import numpy as np
import scipy.linalg as la

def local_map(channels,length,offsets):
    offsets=np.asarray(offsets,dtype=int);assert np.all(abs(offsets)<length)
    indices=[];valid=[]
    for t in range(length):
        s=t+offsets;v=np.tile((s>=0)&(s<length),channels)
        ids=(np.arange(channels)[:,None]*length+s[None,:]).reshape(-1)
        indices.append(ids);valid.append(v)
    return np.array(indices),np.array(valid),np.tile(length-abs(offsets),channels)

def shared_setup(G,H,channels=8,length=192,offsets=tuple(range(-15,17))):
    zdim=channels*length;pdim=len(G)-zdim;ids,valid,m=local_map(channels,length,offsets);k=len(m)
    A=np.zeros((k,k));cross=np.zeros((length,k,pdim));rhs=np.zeros(k);scale=1/np.sqrt(m)
    for t in range(length):
        v=np.flatnonzero(valid[t]);zi=ids[t,v];s=scale[v]
        A[np.ix_(v,v)]+=G[np.ix_(zi,zi)]*s[:,None]*s[None,:]
        cross[t,v]=G[zi,zdim:]*s[:,None];rhs[v]+=H[zi,t]*s
    eig,Q=la.eigh((G[zdim:,zdim:]+G[zdim:,zdim:].T)*.5,driver='evr');assert eig.min()>-1e-8
    cq=cross@Q;flat=cq.transpose(1,0,2).reshape(k,length*pdim).copy()
    return dict(A=A,cq=cq,flat=flat,rhs=rhs,eig=eig,Q=Q,hpq=Q.T@H[zdim:],ids=ids,valid=valid,m=m,zdim=zdim,pdim=pdim,length=length)

def shared_solve(setup,lam):
    a=setup;k=len(a['m']);T=a['length'];P=a['pdim'];inv=1/(a['eig']+lam);weights=np.tile(inv,T)
    weighted=a['flat']*np.sqrt(weights)[None,:]
    schur=a['A']+lam*np.eye(k)-weighted@weighted.T
    rhs=a['rhs']-a['flat']@(weights*a['hpq'].T.reshape(-1))
    factor=la.cho_factor((schur+schur.T)*.5,lower=True);theta=la.cho_solve(factor,rhs)
    bpq=(a['hpq']-np.einsum('tap,a->pt',a['cq'],theta))*inv[:,None];bp=a['Q']@bpq
    bz=np.zeros((a['zdim'],T));w=theta/np.sqrt(a['m'])
    for t in range(T):bz[a['ids'][t,a['valid'][t]],t]=w[a['valid'][t]]
    assert abs(float((bz*bz).sum()-theta@theta))<=1e-10*max(1.,float(theta@theta))
    gradtheta=(a['A']+lam*np.eye(k))@theta+np.einsum('tap,pt->a',a['cq'],bpq)-a['rhs']
    gradp=(a['eig'][:,None]+lam)*bpq+np.einsum('tap,a->pt',a['cq'],theta)-a['hpq']
    normal=max(float(abs(gradtheta).max()),float(abs(gradp).max()))
    sinv=la.cho_solve(factor,np.eye(k));weighted2=a['flat']*weights[None,:];extra=weighted2@weighted2.T
    inverse_trace=float(np.trace(sinv)+T*inv.sum()+np.einsum('ij,ji',sinv,extra))
    df=float(k+T*P-lam*inverse_trace);assert df>=-1e-7
    return dict(BZ=bz,BP=bp,effective_df_total=max(0.,df),normal_equation_max_abs=normal,theta=theta)

def unshared_solve(G,H,lambdas,channels=8,length=192,offsets=tuple(range(-15,17))):
    zdim=channels*length;ids,valid,m=local_map(channels,length,offsets);pdim=len(G)-zdim
    result={lam:dict(BZ=np.zeros((zdim,length)),BP=np.zeros((pdim,length)),effective_df_total=0.,normal_equation_max_abs=0.) for lam in lambdas}
    for t in range(length):
        zi=ids[t,valid[t]];allids=np.r_[zi,np.arange(zdim,len(G))];g=G[np.ix_(allids,allids)];h=H[allids,t]
        eig,Q=la.eigh((g+g.T)*.5,driver='evr');assert eig.min()>-1e-8;qh=Q.T@h
        for lam in lambdas:
            b=Q@(qh/(eig+lam));r=result[lam];r['BZ'][zi,t]=b[:len(zi)];r['BP'][:,t]=b[len(zi):]
            r['effective_df_total']+=float(np.sum(eig/(eig+lam)))
            err=float(abs((g+lam*np.eye(len(g)))@b-h).max());r['normal_equation_max_abs']=max(r['normal_equation_max_abs'],err)
    return result

def math_check():
    rng=np.random.default_rng(66101);N=23;T=7;C=2;P=3;offsets=(-1,0,1);lam=.013
    x=rng.normal(size=(N,C*T+P));y=rng.normal(size=(N,T));xm=x.mean(0);ym=y.mean(0);xc=x-xm;yc=y-ym
    G=xc.T@xc/N;H=xc.T@yc/N;setup=shared_setup(G,H,C,T,offsets);sol=shared_solve(setup,lam)
    ids,valid,m=local_map(C,T,offsets);k=len(m);regularized=k+T*P
    design=np.zeros((N*T,regularized+T));centered=np.zeros((N*T,regularized))
    for t in range(T):
        at=slice(t*N,(t+1)*N);v=np.flatnonzero(valid[t]);zi=ids[t,v]
        design[at,v]=x[:,zi]/np.sqrt(m[v]);design[at,k+t*P:k+(t+1)*P]=x[:,C*T:];design[at,regularized+t]=1
        centered[at,v]=xc[:,zi]/np.sqrt(m[v]);centered[at,k+t*P:k+(t+1)*P]=xc[:,C*T:]
    pen=np.column_stack([np.eye(regularized)*np.sqrt(N*lam),np.zeros((regularized,T))])
    direct=la.lstsq(np.vstack([design,pen]),np.r_[y.T.reshape(-1),np.zeros(regularized)])[0]
    bias=ym-xm[:C*T]@sol['BZ']-xm[C*T:]@sol['BP']
    packed=np.r_[sol['theta'],sol['BP'].T.reshape(-1),bias]
    error=float(abs(packed-direct).max());assert error<1e-11
    q=centered.T@centered/N;hat=la.solve(q+lam*np.eye(regularized),centered.T,assume_a='pos')/N
    explicit_df=float(np.trace(centered@hat));dferr=abs(sol['effective_df_total']-explicit_df);assert dferr<1e-10
    u=unshared_solve(G,H,(lam,),C,T,offsets)[lam]
    for t in range(T):
        forbidden=np.setdiff1d(np.arange(C*T),ids[t,valid[t]])
        assert not np.any(sol['BZ'][forbidden,t]) and not np.any(u['BZ'][forbidden,t])
        changed=x.copy();changed[:,forbidden]+=rng.normal(size=(N,len(forbidden)))
        assert np.max(abs(changed[:,:C*T]@u['BZ'][:,t]-x[:,:C*T]@u['BZ'][:,t]))<1e-12
        assert np.max(abs(changed[:,:C*T]@sol['BZ'][:,t]-x[:,:C*T]@sol['BZ'][:,t]))<1e-12
    w=np.hanning(192);alpha=w[96:]/(w[96:]+w[:96]);p1=rng.normal(size=96);p2=rng.normal(size=96);truth=rng.normal(size=96)
    lhs=alpha*(p1-truth)**2+(1-alpha)*(p2-truth)**2
    rhs=(alpha*p1+(1-alpha)*p2-truth)**2+alpha*(1-alpha)*(p1-p2)**2
    herr=float(abs(lhs-rhs).max());assert herr<1e-12
    return dict(passed=True,shared_Schur_vs_augmented_coeff_max_abs=error,coupled_df_vs_explicit_hat_abs=dferr,normal_residual=sol['normal_equation_max_abs'],same_local_padding_band_canaries=True,Hann_identity_max_abs=herr)

def secondary(pred,y,rows,mask):
    """Fixed Hann + all adjacent physical overlaps; never used for selection."""
    eligible={(b['record'],b['start']):b for b in mask['blocks']};window=np.hanning(192);alpha=window[96:]/(window[96:]+window[:96])
    blocks=[];records=[];total_pairs=validpairs=0;overlap_sum=normalized_num=normalized_den=0.;identitymax=0.
    for row in rows:
        o,w=row['offset'],row['windows'];p=pred[o:o+w];truth=np.load(row['envelope_path'],mmap_mode='r').reshape(-1);n=row['common_samples']
        target=np.asarray(truth[:n],np.float64);assert np.isfinite(target).all()
        a=p[:-1,96:];b=p[1:,:96];pair_mse=((a-b)**2).mean(1);den=(a.var(1)+b.var(1))/2;valid=den>0
        total_pairs+=len(a);validpairs+=int(valid.sum());overlap_sum+=float(pair_mse.sum());normalized_num+=float(pair_mse[valid].sum());normalized_den+=float(den[valid].sum())
        total=np.zeros(n);weight=np.zeros(n);single_error=np.zeros(n);disagreement=np.zeros(n)
        for i,value in enumerate(p):
            s=i*96;total[s:s+192]+=value*window;weight[s:s+192]+=window;single_error[s:s+192]+=(value-target[s:s+192])**2*window
        for i in range(w-1):
            s=(i+1)*96;disagreement[s:s+96]=alpha*(1-alpha)*(a[i]-b[i])**2
        keep=weight>1e-8;total[keep]/=weight[keep];single_error[keep]/=weight[keep];fused_error=(total-target)**2
        difference=abs(single_error[keep]-fused_error[keep]-disagreement[keep]);identitymax=max(identitymax,float(difference.max()))
        assert float(difference.max())<=1e-10*max(1.,float(single_error[keep].max()))
        records.append(dict(record=row['record'],subject=row['subject'],pairs=len(a),valid_normalized_pairs=int(valid.sum()),overlap_MSE=float(pair_mse.mean()),overlap_NRMSE=float(np.sqrt(pair_mse[valid].sum()/den[valid].sum())) if valid.any() else None,
            weighted_single_MSE=float(single_error[keep].mean()),fused_MSE=float(fused_error[keep].mean()),disagreement=float(disagreement[keep].mean())))
        for s in range(0,n-3840+1,3840):
            e=eligible[(row['record'],s)];k=keep[s:s+3840];assert int(k.sum())==e['samples']
            if not e['eligible']:continue
            blocks.append(dict(record=row['record'],subject=row['subject'],start=s,weighted_single_MSE=float(single_error[s:s+3840][k].mean()),fused_MSE=float(fused_error[s:s+3840][k].mean()),disagreement=float(disagreement[s:s+3840][k].mean())))
    assert total_pairs==30451 and len(blocks)==752
    people=[dict(subject=s,**{k:float(np.mean([b[k] for b in blocks if b['subject']==s])) for k in ('weighted_single_MSE','fused_MSE','disagreement')}) for s in sorted({b['subject'] for b in blocks})]
    macro={k:float(np.mean([p[k] for p in people])) for k in ('weighted_single_MSE','fused_MSE','disagreement')}
    return dict(window_target_MSE=float(((pred-np.asarray(y[:,0],np.float64))**2).mean()),macro_MSE=macro,overlap_MSE=overlap_sum/total_pairs,overlap_NRMSE=float(np.sqrt(normalized_num/normalized_den)) if normalized_den>0 else None,
        total_overlap_pairs=total_pairs,valid_normalized_pairs=validpairs,Hann_identity_max_abs=identitymax,people=people,blocks=blocks,records=records)
