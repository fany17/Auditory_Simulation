"""Scalar-level active-gradient audit; frozen training code is never modified."""
import json
import numpy as np
import torch
from pub_01_s03_ss05_data import ROOT,dump,ContinuousPool
from pub_01_s03_ss05_finite_history import FiniteHistoryDecoder,VARIANTS,pcc_loss


def main():
    torch.set_num_threads(4);pool=ContinuousPool('train','cpu');ids=np.linspace(0,len(pool.segments)-1,5,dtype=int);rows=[]
    for variant in VARIANTS:
        torch.manual_seed(1024);model=FiniteHistoryDecoder(variant);nonzero={n:torch.zeros_like(p,dtype=torch.bool) for n,p in model.named_parameters()};reachable=set()
        for i in ids:
            x,y=pool.batch([i],8);model.zero_grad(set_to_none=True);pcc_loss(model(x),y).backward()
            for name,p in model.named_parameters():
                if p.grad is not None:
                    reachable.add(name);nonzero[name]|=(p.grad!=0)
        counts={n:int(mask.sum()) for n,mask in nonzero.items()}
        params=dict(model.named_parameters());total=sum(p.numel() for p in params.values())
        rows.append({'variant':variant,'instantiated_parameters':total,
            'autograd_reachable_tensor_inventory':sum(params[n].numel() for n in reachable),
            'nonzero_gradient_scalar_union_over5_real_train_batches':sum(counts.values()),
            'observed_allzero_gradient_scalar_count':total-sum(counts.values()),
            'per_tensor_nonzero_gradient_scalar_count':counts})
    dump(ROOT/'scalar_parameter_audit.json',{'profiles':rows,'compute':'server2203 CPU; no optimizer/test use',
        'scope':'scalar gradient union is an empirical activity lowerbound; tensor-level reachability is not an active-scalar count',
        'input_indices':ids.tolist(),'loss':'3s Pearson','seed':1024})
    print(json.dumps([dict((k,v) for k,v in r.items() if not isinstance(v,dict)) for r in rows]),flush=True)


if __name__=='__main__':main()
