"""D9 quadratic actions/PCG; original whole-record Hann, no I/O."""
import numpy as np
import torch

class PhysicalQuadratic:
    def __init__(self,x,active,base,ypair,ytail,left,right,tail,first_pairs,geometry,lam):
        self.x=x;self.active=active;self.base=base;self.ypair=ypair;self.ytail=ytail
        self.left=left;self.right=right;self.tail=tail;self.first=first_pairs;self.geometry=geometry;self.lam=lam
        device=x.device;dtype=x.dtype;assert dtype==torch.float64
        h=np.hanning(192);self.alpha=torch.as_tensor(h[96:]/(h[96:]+h[:96]),dtype=dtype,device=device);self.beta=1-self.alpha
        self.T=len(left)*96-len(first_pairs)+len(tail)*95
        assert len(left)==int(active.sum()) and len(tail)==len(first_pairs)
        assert x.shape==(len(active),3264) and base.shape==(len(active),192)
        assert ypair.shape==(len(left),96) and ytail.shape==(len(tail),95)
        self.counts=dict(Hessian=0,A=0,AT=0,RHS=0,objective=0)

    def A(self,theta):
        self.counts['A']+=1
        out=self.x.new_zeros((len(self.x),192))
        out[:,1:191]=self.x@theta[:-1]+self.active[:,None]*theta[-1]
        return out

    def AT(self,value):
        self.counts['AT']+=1
        v=value[:,1:191];return torch.cat((self.x.T@v,(v*self.active[:,None]).sum(0,keepdim=True)),0)

    def residual_adjoint(self,pred,role,target=False):
        a,b=self.alpha,self.beta
        pl,pr=pred[self.left,96:],pred[self.right,:96]
        targetpair=self.ypair if target else 0.
        targettail=self.ytail if target else 0.
        out=pred.new_zeros(pred.shape)
        if role=='F':
            e=a*pl+b*pr-targetpair;e[self.first,0]=0
            out[self.left,96:]=a*e;out[self.right,:96]=b*e
        else:
            assert role=='W';el=pl-targetpair;er=pr-targetpair;el[self.first,0]=0;er[self.first,0]=0
            out[self.left,96:]=a*el;out[self.right,:96]=b*er
        out[self.tail,96:191]=pred[self.tail,96:191]-targettail
        return out/self.T

    def hessian(self,theta,role):
        self.counts['Hessian']+=1
        out=self.AT(self.residual_adjoint(self.A(theta),role))
        out[:-1]+=self.lam*theta[:-1]
        return out

    def rhs(self,role):
        self.counts['RHS']+=1
        return -self.AT(self.residual_adjoint(self.base,role,target=True))

    def objectives(self,theta):
        self.counts['objective']+=1
        pred=self.base+self.A(theta);a,b=self.alpha,self.beta
        pl,pr=pred[self.left,96:],pred[self.right,:96];yp=self.ypair
        ew=a*(pl-yp)**2+b*(pr-yp)**2
        ef=(a*pl+b*pr-yp)**2;disc=a*b*(pl-pr)**2
        for v in (ew,ef,disc):v[self.first,0]=0
        edge=((pred[self.tail,96:191]-self.ytail)**2).sum()
        W=(ew.sum()+edge)/self.T;F=(ef.sum()+edge)/self.T;D=disc.sum()/self.T
        penalty=self.lam*(theta[:-1]**2).sum()
        out=dict(W_data=float(W),F_data=float(F),disagreement=float(D),regularizer=float(penalty),W_total=float(W+penalty),F_total=float(F+penalty),identity_error=float(abs(W-F-D)))
        assert out['identity_error']<=1e-10*max(1.,out['W_data'])
        return out

    def preconditioner(self,moment,role):
        # The same target-independent construction rule for either objective.
        blocks=torch.as_tensor([r[f'{role}_bias_block'] for r in self.geometry],dtype=self.x.dtype,device=self.x.device)
        diagonal=self.x.new_zeros(192)
        diagonal[:96]=blocks[:,0,0];diagonal[96:]=blocks[:,1,1]
        bdiag=moment[:,None]*diagonal[None,1:191]+self.lam
        assert torch.isfinite(bdiag).all() and torch.all(bdiag>0)
        both=torch.linalg.cholesky(blocks[1:95])
        single_right=torch.sqrt(blocks[0,1,1]);single_left=torch.sqrt(blocks[95,0,0])
        def apply(r):
            rb=self.x.new_zeros(192);rb[1:191]=r[-1]
            pair=torch.stack((rb[1:95],rb[97:191]),1)
            answer=torch.cholesky_solve(pair[:,:,None],both)[:,:,0]
            outbias=self.x.new_zeros(192);outbias[1:95]=answer[:,0];outbias[97:191]=answer[:,1]
            outbias[96]=(rb[96]/single_right)/single_right
            outbias[95]=(rb[95]/single_left)/single_left
            return torch.cat((r[:-1]/bdiag,outbias[None,1:191]),0)
        return apply

def pcg(action,rhs,start,precondition,callback,maximum=200,tolerance=1e-8,maximum_actions=214):
    used=0
    def apply(value):
        nonlocal used
        assert used<maximum_actions
        used+=1;return action(value)
    theta=start.clone();r=rhs-apply(theta);z=precondition(r);p=z.clone();rz=float((r*z).sum())
    rhs_norm=float((rhs*precondition(rhs)).sum());assert np.isfinite(rhs_norm) and rhs_norm>=0
    denom=max(rhs_norm**.5,1e-30)
    assert np.isfinite(rz) and rz>=0
    history=[];converged=False;true_ratio=None
    for step in range(maximum+1):
        recursive=max(rz,0.)**.5/denom
        if step%20==0 or recursive<=tolerance or step==maximum or used>=maximum_actions-1:
            true=rhs-apply(theta);tz=precondition(true);norm=float((true*tz).sum())
            assert np.isfinite(norm) and norm>=0
            true_ratio=norm**.5/denom
            row=dict(iteration=step,hessian_actions=used,recursive_preconditioned_relative=recursive,true_preconditioned_relative=true_ratio)
            history.append(row);callback(row,theta)
            if true_ratio<=tolerance:converged=True;break
        if step==maximum or used>=maximum_actions-1:break
        hp=apply(p);curvature=float((p*hp).sum());assert np.isfinite(curvature) and curvature>0,'nonpositive curvature; do not add hidden jitter'
        amount=rz/curvature;theta+=amount*p;r-=amount*hp;z=precondition(r);new=float((r*z).sum())
        assert np.isfinite(new) and new>=0 and torch.isfinite(theta).all()
        if rz==0:raise ArithmeticError('zero recursive residual but true residual failed; no automatic restart')
        p=z+(new/rz)*p;rz=new
    return theta,dict(converged=converged,iterations=step,hessian_actions=used,true_preconditioned_relative=true_ratio,history=history,status='CONVERGED' if converged else 'FIXED_CAP_NOT_CONVERGED')
