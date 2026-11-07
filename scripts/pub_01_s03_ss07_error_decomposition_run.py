"""One seed, all three frozen last19 arms, TRAIN/VAL exactly once each."""
import argparse,csv,gc,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from pub_01_s03_ss07_error_decomposition_core import *
sys.path.insert(0,str(D12/'input'))
import pub_01_s03_ss07_full_ft_history_core as frozen

def write_rows(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run(seed,device):
    started=time.time();cpu=time.process_time();p=read(ROOT/'PROTOCOL.json');assert read(ROOT/'ENGINEERING_CHECK.json')['passed']
    for path,digest in p['source_pins'].items():assert sha(path)==digest,path
    for path,item in p['inputs'].items():
        st=Path(path).stat();assert st.st_size==item['bytes'] and st.st_mtime_ns==item['mtime_ns'],path
    torch.set_num_threads(2);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    folder=ROOT/f'results/seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    dump(folder/'STARTED.json',dict(seed=seed,pid=os.getpid(),time=started,protocol_sha256=sha(ROOT/'PROTOCOL.json')))
    masks=read(ROOT/'SUPPORT_MASK.json');phase_summaries=[];gradient_rows=[];gradient_people=[];forwards=0;projection_count=0;region_count=0
    for phase in ['train','val']:
        data=frozen.Data(seed,phase);mask=masks[phase];people_count=mask['people'];tally=mask['person_blocks']
        eligible={r['record']:[b for b in mask['blocks'] if b['record']==r['record'] and b['eligible']] for r in data.rows}
        if phase=='train':
            for row in data.rows:
                a,w=row['offset'],row['windows'];raw=np.load(row['envelope_path'],mmap_mode='r').reshape(-1)
                truth=np.asarray(raw,dtype='f4').astype('f8')
                stat=gradient_record(data.anchor[a:a+w],data.y[a:a+w],truth,row,eligible[row['record']],people_count,tally[row['subject']],data.n)
                gradient_rows.append(dict(seed=seed,subject=row['subject'],record=row['record'],**stat))
            for subject in sorted(tally):gradient_people.append(dict(seed=seed,subject=subject,**aggregate_gradients([r for r in gradient_rows if r['subject']==subject])))
            gradient=aggregate_gradients(gradient_rows);dump(folder/'GRADIENT_SUMMARY.json',dict(seed=seed,**gradient,gradient_basis='Frozen anchor, no NN or backward',new_NN=0))
            write_rows(folder/'GRADIENT_RECORDS.csv',gradient_rows);write_rows(folder/'GRADIENT_PERSON.csv',gradient_people)
        for arm in ARMS:
            model=frozen.Conditional(seed,arm).to(device).eval();path=D12/f'fits/{arm}_seed{seed}/last.pt'
            assert sha(path)==p['inputs'][str(path)]['sha256'];cp=torch.load(path,map_location=device,weights_only=True)
            assert cp['epoch']==19 and cp['seed']==seed and cp['arm']==arm;model.load_state_dict(cp['model']);del cp
            prediction,info=frozen.predict(model,data,device);forwards+=info['effective_residual_windows']
            assert torch.cuda.max_memory_allocated()<2*1024**3
            assert info['effective_residual_windows']==p['support'][phase]['effective_windows']
            allblocks=[];record_rows=[];regions=[];persons=[]
            for row in data.rows:
                o,w=row['offset'],row['windows'];n=row['common_samples'];aw,weight,keep=fuse(data.anchor[o:o+w],n);pw,_,_=fuse(prediction[o:o+w],n)
                raw=np.load(row['envelope_path'],mmap_mode='r').reshape(-1)
                truth=np.asarray(raw,dtype='f4').astype('f8') if phase=='train' else np.asarray(raw,dtype='f8')
                own=[]
                for b in eligible[row['record']]:
                    s,t=b['start'],b['stop'];k=keep[s:t];d=decompose(aw[s:t][k],pw[s:t][k],truth[s:t][k])
                    item=dict(seed=seed,arm=arm,phase=phase,subject=row['subject'],record=row['record'],start=s,stop=t,**d);allblocks.append(item);own.append(item);projection_count+=int(d['projection_defined'])
                record_rows.append(dict(seed=seed,arm=arm,phase=phase,subject=row['subject'],record=row['record'],**aggregate_blocks(own)))
                for label,s,t in [('whole_covered',0,n),('prefix',0,864),('first_learning_overlap',864,960),('tail',(n//3840)*3840,n),('last_single_window',(w-1)*96+96,min(n,(w-1)*96+192))]:
                    k=keep[s:t];d=decompose(aw[s:t][k],pw[s:t][k],truth[s:t][k]);regions.append(dict(seed=seed,arm=arm,phase=phase,subject=row['subject'],record=row['record'],region=label,start=s,stop=t,**d));region_count+=int(d['projection_defined'])
            for subject in sorted(tally):persons.append(dict(seed=seed,arm=arm,phase=phase,subject=subject,**aggregate_blocks([b for b in allblocks if b['subject']==subject])))
            summary={key:float(np.mean([r[key] for r in persons])) if all(r[key] is not None for r in persons) else None for key in NUMERIC}
            summary.update(seed=seed,arm=arm,phase=phase,people=len(persons),records=len(record_rows),blocks=len(allblocks),
                undefined_projection_blocks=sum(r['undefined_projection_blocks'] for r in persons),undefined_Pearson_blocks=sum(r['undefined_Pearson_blocks'] for r in persons),
                low_variance_or_other_flagged_blocks=sum(r['flagged_blocks'] for r in persons),aggregation='Equal blocks within person, then equal persons; reconstruct r blockwise first')
            for k in ['MSE_identity_error','r_identity_error']:
                vv=[abs(b[k]) for b in allblocks if b[k] is not None];summary['max_abs_'+k]=max(vv) if vv else None
            phase_summaries.append(summary)
            for name,rows in [('BLOCKS',allblocks),('RECORDS',record_rows),('PERSON',persons),('REGIONS',regions)]:write_rows(folder/f'{phase}_{arm}_{name}.csv',rows)
            print(json.dumps(dict(seed=seed,arm=arm,phase=phase,NN_windows=info['effective_residual_windows'],blocks=len(allblocks),summary=summary),allow_nan=False),flush=True)
            del prediction,model,allblocks,record_rows,regions,persons;gc.collect();torch.cuda.empty_cache();guard()
        del data;gc.collect()
    assert forwards==836037
    dump(folder/'SUMMARY.json',phase_summaries)
    dump(folder/'COMPLETED.json',dict(seed=seed,effective_residual_NN_windows=forwards,block_descriptive_projection_statistics=projection_count,
        auxiliary_region_projection_statistics=region_count,learning_model_fits=0,optimizer_updates=0,new_encoder_head_windows=0,
        wall_seconds=time.time()-started,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),no_gradient_or_history_disk_cache=True,goal_complete=False))

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--seed',type=int,choices=SEEDS,required=True);q.add_argument('--device',required=True);a=q.parse_args()
    try:run(a.seed,a.device)
    except BaseException:
        dump(ROOT/f'FAILURE_seed{a.seed}.json',dict(time=time.time(),traceback=traceback.format_exc(),automatic_retry=False));raise
