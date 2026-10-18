"""Prepare original Dutch XLSR19 and author-fold public inputs on2203.

No author executable imported. Audio backbone Apache2, existing inputs only.
All raw signals, feature tensors and normalization fits remain on2203.
"""
import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path
import numpy as np
from scipy.signal import resample, resample_poly


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def freeze(cfg):
    root,work=Path(cfg['root']),Path(cfg['work'])
    out=work/'scope';out.mkdir(exist_ok=False)
    split=json.loads((work/'input'/cfg['author_fold_json']).read_text())
    index={r['source_recording']:r for r in read_csv(root/cfg['index'])}
    ready=[r for r in read_csv(root/cfg['record_readiness']) if r['record_status']=='READY']
    train_stories=set(s for key,stories in split.items() if key.isdigit() and key!=cfg['validation_fold'] for s in stories)
    val_stories=set(split[cfg['validation_fold']]);test_stories=set(split[cfg['seen_test_fold']])
    if train_stories & val_stories or train_stories & test_stories or val_stories & test_stories:
        raise RuntimeError('exact seen-subject story identity crosses phases')
    records,failures=[],[]
    for row in ready:
        subject,story=row['subject'],row['raw_story']
        if subject in split['train_subs']:
            phase='train' if story in train_stories else 'val' if story in val_stories else 'test_seen' if story in test_stories else None
        else:
            phase='test_heldout' if subject in split['test_subs'] else None
        if phase is None:
            continue
        if phase=='test_seen' and subject in cfg['evaluation']['exclude_seen_subjects']:
            continue
        raw=index[row['source_recording']]
        metadata=root/cfg['source_audio_metadata']/story/'metadata.json'
        if not metadata.is_file():
            failures.append({'record':row['source_recording'],'phase':phase,'reason':'audio metadata unavailable'})
            continue
        audio_meta=json.loads(metadata.read_text())
        audio_path=Path(audio_meta['source_path'])
        if not audio_path.is_file() or not Path(raw['eeg_path']).is_file():
            failures.append({'record':row['source_recording'],'phase':phase,'reason':'existing input unavailable'})
            continue
        records.append({'subject':subject,'story':story,'phase':phase,'record':row['source_recording'],
            'eeg_path':raw['eeg_path'],'audio_path':str(audio_path),'indexed_samples':int(raw['samples']),
            'source_duration_seconds':audio_meta['source_duration_seconds']})
    train_subjects=sorted({r['subject'] for r in records if r['phase']=='train'})
    # Subjects with no training observations cannot enter a seen-subject group.
    records=[r for r in records if r['phase']=='test_heldout' or r['subject'] in train_subjects]
    if set(train_subjects) & set(split['test_subs']):
        raise RuntimeError('held-out subject in training')
    scope={'config':cfg,'records':records,'failures':failures,'train_subjects':train_subjects,
        'heldout_subjects':sorted({r['subject'] for r in records if r['phase']=='test_heldout'}),
        'unavailable_author_train_subjects':sorted(set(split['train_subs'])-set(train_subjects)),
        'phase_records':{p:sum(r['phase']==p for r in records) for p in ('train','val','test_seen','test_heldout')},
        'scope_selected_before_fitting':True,'official_test_packet_used':False}
    dump(out/'manifest_frozen.json',scope)
    dump(out/'config_frozen.json',cfg)
    print(json.dumps({k:scope[k] for k in ('phase_records','train_subjects','heldout_subjects','unavailable_author_train_subjects')}),flush=True)


def load_audio(path):
    opening=gzip.open if path.suffix=='.gz' else open
    with opening(path,'rb') as stream,np.load(stream,allow_pickle=False) as archive:
        audio=np.asarray(archive['audio'],dtype=np.float32);fs=int(np.asarray(archive['fs']).squeeze())
    if audio.ndim!=1 or not np.isfinite(audio).all():
        raise RuntimeError('nonmono or nonfinite source')
    divisor=math.gcd(fs,16000)
    return resample_poly(audio,16000//divisor,fs//divisor).astype(np.float32)


def features(cfg):
    sys.path.insert(0,str(Path(cfg['root'])/'pub_01/s03_strong_baselines_20261013/continuation01/runtime_torchaudio_cu128'))
    import torch
    from transformers import Wav2Vec2Model
    work=Path(cfg['work']);scope=json.loads((work/'scope/manifest_frozen.json').read_text())
    out=work/'features';out.mkdir(exist_ok=True)
    if (out/'completed.json').exists():
        raise FileExistsError('feature run already completed')
    stories={r['story']:r['audio_path'] for r in scope['records']}
    model=Wav2Vec2Model.from_pretrained(cfg['xlsr_path'],local_files_only=True).eval().to('cuda:0')
    torch.set_num_threads(4)
    manifest=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else []
    for story,source in sorted(stories.items()):
        if any(r['story']==story and Path(r['path']).is_file() and r['revision']==cfg['xlsr_revision'] for r in manifest):
            continue
        audio=load_audio(Path(source));padded=np.pad(audio,(16000,0))
        jobs=[]
        for start in range(0,len(audio),8*16000):
            clip=padded[start:start+10*16000]
            if len(clip)<16400:
                continue  # no complete post-context model frame in this tail
            core_samples=min(8*16000,len(audio)-start)
            jobs.append((clip,core_samples))
        chunks=[]
        for offset in range(0,len(jobs),cfg['audio']['inference_batch']):
            group=jobs[offset:offset+cfg['audio']['inference_batch']]
            # Keep only equally sized clips together; last clip is separate so
            # artificial right padding cannot change the representation.
            groups=[]
            for job in group:
                if not groups or len(groups[-1][0][0])!=len(job[0]):
                    groups.append([])
                groups[-1].append(job)
            for subgroup in groups:
                if not subgroup:
                    continue
                x=torch.from_numpy(np.stack([c for c,core in subgroup])).to('cuda:0')
                with torch.inference_mode():
                    hidden=model(x,attention_mask=torch.ones_like(x,dtype=torch.long),output_hidden_states=True).hidden_states[19].cpu().numpy()
                for h,(clip,core_samples) in zip(hidden,subgroup):
                    # Emit at most the actual8s core. A source loop can mark a
                    # penultimate10s clip as end-of-file and duplicate its right
                    # context in the next iteration; keep a monotonic time axis.
                    core_frames=round(core_samples*50/16000)
                    chunks.append(h[50:50+core_frames])
        native=np.concatenate(chunks)
        if abs(len(native)/50-len(audio)/16000)>.05:
            raise RuntimeError('XLSR context crop does not preserve duration')
        feature=resample(native,round(len(native)*64/50),axis=0).astype(np.float32)
        if feature.shape[1]!=1024 or not np.isfinite(feature).all():
            raise RuntimeError('invalid XLSR19 features')
        target=out/(story+'.npy');np.save(target,feature,allow_pickle=False)
        row={'story':story,'path':str(target),'shape':list(feature.shape),'fs':64,'finite':True,
            'source':source,'audio16k_samples':len(audio),'native_frames':len(native),'tuple_index':19,
            'revision':cfg['xlsr_revision'],'zero_origin_after_context_crop':True}
        manifest.append(row);dump(out/'manifest.json',manifest);print(json.dumps(row),flush=True)
    dump(out/'completed.json',{'stories':len(manifest),'all_finite':True})


def prepare(cfg):
    work=Path(cfg['work']);scope=json.loads((work/'scope/manifest_frozen.json').read_text())
    if not (work/'features/completed.json').is_file():
        raise RuntimeError('features still running')
    feature={r['story']:r for r in json.loads((work/'features/manifest.json').read_text())}
    out=work/'prepared';out.mkdir(exist_ok=False)
    n,channel_sum,channel_square=0,np.zeros(64),np.zeros(64)
    records,failures=[],list(scope['failures'])
    for row in scope['records']:
        raw=np.load(row['eeg_path'],mmap_mode='r',allow_pickle=False)
        eeg=raw.T if raw.shape[0]==64 else raw
        common=min(len(eeg),row['indexed_samples'],feature[row['story']]['shape'][0],int(row['source_duration_seconds']*64))
        if eeg.ndim!=2 or eeg.shape[1]!=64 or common<448 or not np.isfinite(eeg[:common]).all():
            failures.append({'record':row['record'],'reason':'bad shape/finite/support'});continue
        item=dict(row,common_samples=common,windows=common//192);records.append(item)
        if row['phase']=='train':
            data=np.asarray(eeg[:common],dtype=np.float64);n+=common
            channel_sum+=data.sum(0);channel_square+=np.square(data).sum(0)
    mean=channel_sum/n;std=np.sqrt(np.maximum(channel_square/n-mean**2,1e-12))
    np.savez(out/'train_only_normalization.npz',mean=mean,std=std)
    # Arrays contain only EEG; original XLSR vectors are shared by story.
    phase_rows={};rng=np.random.default_rng(cfg['training']['seed']);queries=[]
    for phase in ('train','val','test_seen','test_heldout'):
        rows=[]
        for record_id,r in enumerate(records):
            if r['phase']!=phase:
                continue
            for segment in range(r['windows']):
                start=segment*192
                rows.append({'subject':r['subject'],'story':r['story'],'record_id':record_id,'start':start,'phase':phase})
                if phase.startswith('test'):
                    sign=int(rng.choice([-1,1]));negative=start+sign*256
                    if negative<0 or negative+192>r['common_samples']:
                        negative=start-sign*256
                    if negative<0 or negative+192>r['common_samples']:
                        continue
                    queries.append({'phase':phase,'subject':r['subject'],'record_id':record_id,
                        'eeg_window_index':len(rows)-1,'match_start':start,'mismatch_start':negative,'matched_candidate':int(rng.integers(0,2))})
        array=np.lib.format.open_memmap(out/(phase+'_eeg.npy'),mode='w+',dtype='float32',shape=(len(rows),192,64))
        previous,raw=None,None
        for i,w in enumerate(rows):
            r=records[w['record_id']]
            if w['record_id']!=previous:
                raw=np.load(r['eeg_path'],mmap_mode='r');raw=raw.T if raw.shape[0]==64 else raw;previous=w['record_id']
            chunk=(np.asarray(raw[w['start']:w['start']+192],dtype=np.float32)-mean)/std
            array[i]=chunk
        array.flush();phase_rows[phase]=rows
        dump(out/(phase+'_windows.json'),rows)
    with (out/'test_queries_frozen.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(queries[0]));writer.writeheader();writer.writerows(queries)
    dump(out/'manifest_frozen.json',{'config':cfg,'records':records,'source_failures':failures,
        'phase_windows':{p:len(v) for p,v in phase_rows.items()},'test_query_count':len(queries),
        'train_subjects':scope['train_subjects'],'heldout_subjects':scope['heldout_subjects'],
        'normalization_fit_phase':'train_only','no_test_based_input_selection':True,'official_test_packet_used':False})
    dump(out/'completed.json',{'ready':True});print(json.dumps({'phase_windows':{p:len(v) for p,v in phase_rows.items()},'queries':len(queries)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--stage',choices=['freeze','features','prepare'],required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    if cfg['patient_data']!='FORBIDDEN' or not str(Path(cfg['work'])).startswith('/home/fanyu/auditory_simulation_m6a/'):
        raise RuntimeError('2203 public-only')
    {'freeze':freeze,'features':features,'prepare':prepare}[args.stage](cfg)
