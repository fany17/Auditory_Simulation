import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np,json
from pathlib import Path
rng=np.random.default_rng(9011);L=6;H=3;D=4;h=np.hanning(L);N=D*L
Jf=np.zeros((N,N));gf=np.zeros(N);Jw=np.zeros((N,N));gw=np.zeros(N)
Mf=np.zeros((N,N));rf=np.zeros(N);Mw=np.zeros((N,N));rw=np.zeros(N)
T=0;identity=0.
theta=rng.normal(size=N)
for w in [7,9]:
 x=np.column_stack([rng.normal(size=(w,D-1)),np.ones(w)]);u=x.copy();u[:4]=0
 p=rng.normal(size=(w,L));n=(w-1)*H+L;y=rng.normal(size=n)
 C=np.zeros((D,D))
 for i in range(5,w):C+=np.outer(u[i],u[i-1])
 A=u.T@u;Z=np.outer(u[-1],u[-1]);B=A-Z
 for j in range(H):
  alpha=h[j]/(h[j]+h[j+H]);beta=1-alpha;tail=float(h[j+H]>1e-8)
  a=slice(j*D,(j+1)*D);b=slice((j+H)*D,(j+H+1)*D)
  Mf[a,a]+=alpha**2*A;Mf[b,b]+=beta**2*B+tail*Z;Mf[a,b]+=alpha*beta*C;Mf[b,a]+=alpha*beta*C.T
  Mw[a,a]+=alpha*A;Mw[b,b]+=beta*B+tail*Z
  for i in range(4,w):
   v=y[i*H+j]-alpha*p[i,j]-beta*p[i-1,j+H]
   rf[a]+=alpha*u[i]*v;rw[a]+=alpha*u[i]*(y[i*H+j]-p[i,j])
   if i>=5:
    rf[b]+=beta*u[i-1]*v;rw[b]+=beta*u[i-1]*(y[i*H+j]-p[i-1,j+H])
  rf[b]+=tail*u[-1]*(y[w*H+j]-p[-1,j+H]);rw[b]+=tail*u[-1]*(y[w*H+j]-p[-1,j+H])
 for t in range(n):
  contributors=[(i,t-i*H,h[t-i*H]) for i in range(w) if 0<=t-i*H<L and h[t-i*H]>1e-8]
  if not contributors or not any(i>=4 for i,k,a in contributors):continue
  total=sum(a for i,k,a in contributors);combined=np.zeros(N);anchor=0.;terms=[]
  for i,k,a in contributors:
   a/=total;q=np.zeros(N);q[k*D:(k+1)*D]=u[i];combined+=a*q;anchor+=a*p[i,k];terms.append((a,q,p[i,k]))
   Jw+=a*np.outer(q,q);gw+=a*q*(y[t]-p[i,k])
  Jf+=np.outer(combined,combined);gf+=combined*(y[t]-anchor);T+=1
  pred=anchor+combined@theta;single=sum(a*(p0+q@theta-y[t])**2 for a,q,p0 in terms)
  disagreement=sum(a*(p0+q@theta-pred)**2 for a,q,p0 in terms)
  identity=max(identity,abs(single-(pred-y[t])**2-disagreement))
off=0.
for a in range(L):
 for b in range(L):
  if a%H!=b%H:off=max(off,float(abs(Jf[a*D:(a+1)*D,b*D:(b+1)*D]).max()))
active=[k*D+d for k in range(1,L-1) for d in range(D)]
reg=np.array([.01 if d%D!=D-1 else 0 for d in range(N)])
minimum=np.linalg.eigvalsh((Jf+np.diag(reg))[np.ix_(active,active)]).min()
out=dict(scope='SYNTHETIC_OPERATOR_DERIVATION_ONLY_NO_EEG_OR_NEW_FORMAL_MODEL',window=L,hop=H,features_including_bias=D,records_windows=[7,9],first4_fixed=True,learnable_physical_points=T,
fused_Gram_max=float(abs(Jf-Mf).max()),fused_RHS_max=float(abs(gf-rf).max()),weighted_Gram_max=float(abs(Jw-Mw).max()),weighted_RHS_max=float(abs(gw-rw).max()),
off_pair_Gram_max=off,Hann_identity_max=identity,active_regularized_Hessian_min_eigenvalue=float(minimum),
interpretation='Paircolumn decomposition is exact in toy; firstlearnable boundary and righttail can identify activebias directions. Interior-only nullspace warning must not be generalized to fullrecord. This is no EEG performance/novelty evidence.')
assert max(out[k] for k in ['fused_Gram_max','fused_RHS_max','weighted_Gram_max','weighted_RHS_max','off_pair_Gram_max','Hann_identity_max'])<1e-12
assert minimum>0
print(json.dumps(out,indent=2))
