"""D18 record-streaming original physical metrics; no prediction disk arrays."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from pub_01_s03_ss07_preprojection_model import BASE,objective
sys.path.insert(0,str(BASE/'s03_current_readout_internal_audit_20261101/input'))
import pub_01_s03_ss07_generalization_core as physical

class RecordSink:
    def __init__(self,data):
        self.data=data;self.row_index=0;self.parts=[];self.pending=0;self.seen=0;self.records=[]
        self.loss_sum=0.;self.max_window=0.;self.max_fused=0.;self.pred_sum=0.;self.pred_square_sum=0.;self.points=0
        self.counts=dict(windows=0,exact_raw_constant=0,exact_centered_zero=0,centered_norm_le_eps=0)
        if data.phase=='val':blocks=data.mask['blocks']
        else:
            raw=json.loads((BASE/'s03_full_ft_error_decomposition_20261102/SUPPORT_MASK.json').read_text())['train']['blocks']
            assert all(not b['eligible'] or b['stop']-b['start']==3840 for b in raw)
            blocks=[dict(b,samples=b['covered']) for b in raw if b['stop']-b['start']==3840]
        self.blocks={r['record']:[b for b in blocks if b['record']==r['record']] for r in data.rows}
    def push(self,first,prediction):
        p=np.asarray(prediction,dtype='f8');assert first==self.seen and p.ndim==2 and p.shape[1]==192 and np.isfinite(p).all()
        cursor=0
        while cursor<len(p):
            row=self.data.rows[self.row_index];take=min(len(p)-cursor,row['windows']-self.pending)
            self.parts.append(p[cursor:cursor+take].copy());self.pending+=take;self.seen+=take;cursor+=take
            if self.pending==row['windows']:
                values=np.concatenate(self.parts);self._finish_record(row,values);self.parts=[];self.pending=0;self.row_index+=1
    def _finish_record(self,row,p):
        target=np.load(row['envelope_path'],mmap_mode='r')
        # Original TRAIN coordinate precision is f32->f64; VAL uses sourcef64.
        truth=np.asarray(target,dtype='f4').astype('f8') if self.data.phase=='train' else np.asarray(target,dtype='f8')
        record=physical.measure(p,truth,row,self.blocks[row['record']])
        record['family_mapping_basis']=row['family_mapping_basis']
        for segment in record['segments']:segment['family_mapping_basis']=row['family_mapping_basis']
        wave,keep=physical.fuse(p,row['common_samples']);truth=truth.reshape(-1)
        def region(first,last):
            mask=keep[first:last];a=wave[first:last][mask];b=truth[first:last][mask]
            return dict(samples=len(a),MSE=float(np.mean((a-b)**2)) if len(a) else None,r=physical.corr(a,b) if len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0 else None)
        n=row['common_samples'];tail=n//3840*3840
        record['boundary_tail']=dict(prefix=region(0,864),first_learning_overlap=region(864,960),tail=region(tail,n),last_single_window=region((row['windows']-1)*96+96,min(n,(row['windows']-1)*96+192)),maximum_window_abs=float(np.max(abs(p))),maximum_fused_abs=float(np.max(abs(wave[keep]))))
        o=row['offset'];w=row['windows'];y=np.asarray(self.data.y[o:o+w],dtype='f4').astype('f8').reshape(w,192)
        with torch.no_grad():self.loss_sum+=float(objective(torch.from_numpy(p),torch.from_numpy(y)))*w
        centered=y-y.mean(1,keepdims=True);norm=np.linalg.norm(centered,axis=1)
        self.counts['windows']+=w;self.counts['exact_raw_constant']+=int(np.sum(np.ptp(y,axis=1)==0));self.counts['exact_centered_zero']+=int(np.sum(norm==0));self.counts['centered_norm_le_eps']+=int(np.sum(norm<=1e-6))
        self.max_window=max(self.max_window,record['boundary_tail']['maximum_window_abs']);self.max_fused=max(self.max_fused,record['boundary_tail']['maximum_fused_abs'])
        self.pred_sum+=float(p.sum());self.pred_square_sum+=float(np.sum(p*p));self.points+=p.size
        record['window_prediction_mean']=float(p.mean());record['window_prediction_RMS']=float(np.sqrt(np.mean(p*p)))
        self.records.append(record)
    def finish(self):
        assert self.seen==self.data.n and self.row_index==len(self.data.rows) and not self.parts
        result=physical.aggregate(self.records);phase=result['phases'][self.data.phase]
        assert phase['windows']==self.data.n and phase['people']=={'train':70,'val':49}[self.data.phase] and phase['blocks']=={'train':6104,'val':752}[self.data.phase]
        result.update(negative_cosine=self.loss_sum/self.data.n,target_window_counts=self.counts,maximum_window_abs=self.max_window,maximum_fused_abs=self.max_fused,window_prediction_mean=self.pred_sum/self.points,window_prediction_RMS=float(np.sqrt(self.pred_square_sum/self.points)),new_prediction_disk_arrays=0,parameters_fixed=True)
        result['boundary_tail_records']=[dict(record=r['record'],**r['boundary_tail']) for r in self.records]
        source=[]
        for family in sorted({b['source_family'] for b in result['segments']}):
            bb=[b for b in result['segments'] if b['source_family']==family];subjects=sorted({b['subject'] for b in bb})
            source.append(dict(source_family=family,people=len(subjects),blocks=len(bb),r=float(np.mean([np.mean([b['r'] for b in bb if b['subject']==s]) for s in subjects])),scope='family-specific block->person macro, descriptive only'))
        result['source_family_descriptive']=source
        return result

@torch.no_grad()
def evaluate(model,data,device,anchor_only=False,paired_older=False):
    model.eval();sink=RecordSink(data);older_sink=RecordSink(data) if paired_older else None
    assert not paired_older or model.arm=='H_history'
    for first in range(0,data.n,64):
        last=min(first+64,data.n);ids=np.arange(first,last)
        if anchor_only:
            data.ensure_anchor(ids)
            value=data.anchor[first:last];sink.push(first,value)
            if older_sink:older_sink.push(first,value)
            continue
        batch,_=data.batch(ids,model.arm,device,need_target=False)
        if bool(batch['valid'].any()):
            query=model.encode_current(batch['x'])
            prediction=model.predict_from_query(query,**batch)
            sink.push(first,prediction.cpu().numpy())
            if older_sink:
                # Exactly the same already-computed query, current and anchor.
                replaced=dict(batch,history_raw=data.older_memory(ids,device))
                prediction=model.predict_from_query(query,**replaced);older_sink.push(first,prediction.cpu().numpy())
        else:
            value=batch['anchor'].cpu().numpy();sink.push(first,value)
            if older_sink:older_sink.push(first,value)
    return sink.finish(),older_sink.finish() if older_sink else None
