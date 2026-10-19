"""Same-current-EEG interventions, with record-equal then subject-equal metrics."""
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F

MODES=('true','reset','wrong_record','wrong_time','shuffled','replace_high_similarity','replace_low_similarity')


def correlations(pred,target):
    a=pred.reshape(len(pred),-1).astype(np.float64);b=target.reshape(len(target),-1).astype(np.float64)
    a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True);den=np.sqrt((a*a).sum(1)*(b*b).sum(1))
    if not np.isfinite(den).all() or (den==0).any():raise RuntimeError('invalid60s Pearson; report failure rather than silently drop')
    return (a*b).sum(1)/den


def encode_pool(model,pool):
    locals_={};tokens={};model.eval()
    with torch.no_grad():
        for rid,row in pool.records.items():
            x=pool.x[rid][:row['window_count']*192].reshape(row['window_count'],192,64);ll=[];tt=[]
            for start in range(0,len(x),128):
                local,token=model.encode(x[start:start+128][None]);ll.append(local[0]);tt.append(token[0])
            locals_[rid]=torch.cat(ll);tokens[rid]=torch.cat(tt)
    return locals_,tokens


def donor_for(pool,rid):
    current=pool.records[rid]
    candidates=[k for k,r in pool.records.items() if k!=rid and r['window_count']>=20 and r['subject']!=current['subject'] and r['source_family']!=current['source_family']]
    if not candidates:candidates=[k for k,r in pool.records.items() if k!=rid and r['window_count']>=20]
    if not candidates:raise RuntimeError('wrong-record intervention needs independent donor record')
    return sorted(candidates)[rid%len(candidates)]


def evaluate(model,pool,lag,modes=('true',),batch=8):
    local,tokens=encode_pool(model,pool);model.eval();rows=[];device=pool.device
    rng=np.random.default_rng(20261019);perm=torch.tensor(rng.permutation(20),device=device)
    with torch.no_grad():
        for first in range(0,len(pool.segments),batch):
            selected=pool.segments[first:first+batch];ll=[];cc=[];hh=[];older=[];wrong=[];target=[]
            for s in selected:
                rid=s['record_id'];q=torch.arange(s['start_window'],s['start_window']+20,device=device)
                history_idx=q[:,None]+torch.arange(-20,0,device=device)
                ll.append(local[rid][q]);cc.append(tokens[rid][q]);hh.append(tokens[rid][history_idx])
                if any(m!='true' and m!='reset' for m in modes):
                    older.append(tokens[rid][history_idx-20])
                    donor=donor_for(pool,rid);n=len(tokens[donor]);starts=(q-20)%(n-20+1)
                    wrong.append(tokens[donor][starts[:,None]+torch.arange(20,device=device)])
                a=s['start_window']*192-lag;target.append(pool.y[rid][a:a+3840].reshape(20,192,1))
            ll=torch.stack(ll);cc=torch.stack(cc);hh=torch.stack(hh);yy=torch.stack(target).cpu().numpy()
            ww=torch.stack(wrong) if wrong else None;oo=torch.stack(older) if older else None
            for mode in modes:
                history=hh
                if mode=='wrong_record':history=ww
                elif mode=='wrong_time':history=oo
                elif mode=='shuffled':history=hh[:,:,perm]
                elif mode.startswith('replace_'):
                    similarity=(F.normalize(hh,dim=-1)*F.normalize(cc,dim=-1)[:,:,None]).sum(-1)
                    index=similarity.argmax(-1) if mode=='replace_high_similarity' else similarity.argmin(-1)
                    history=hh.clone();b=torch.arange(len(hh),device=device)[:,None];q=torch.arange(20,device=device)[None,:]
                    history[b,q,index]=ww[b,q,index]
                prediction=model.queries(ll,cc,history,reset=(mode=='reset')).cpu().numpy()
                scores=correlations(prediction,yy)
                for s,score in zip(selected,scores):rows.append(dict(s,condition=mode,pearson_r=float(score)))
    recordings=[];persons=[];summary={}
    for condition in modes:
        selected=[r for r in rows if r['condition']==condition]
        for rid in sorted({r['record_id'] for r in selected}):
            s=[r for r in selected if r['record_id']==rid];r=pool.records[rid]
            recordings.append({'condition':condition,'record_id':rid,'subject':r['subject'],'source_family':r['source_family'],
                'segments':len(s),'pearson_r':float(np.mean([x['pearson_r'] for x in s]))})
        for subject in sorted({r['subject'] for r in selected}):
            s=[r for r in recordings if r['condition']==condition and r['subject']==subject]
            persons.append({'condition':condition,'subject':subject,'records':len(s),'pearson_r':float(np.mean([x['pearson_r'] for x in s]))})
        values=[r['pearson_r'] for r in persons if r['condition']==condition]
        summary[condition]={'subjects':len(values),'records':len({r['record_id'] for r in selected}),'segments':len(selected),'subject_macro_r':float(np.mean(values))}
    return summary,persons,recordings,rows
