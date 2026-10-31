"""One final seed's fixed internal/external evaluation. Strict ready gate.

Invoke only through the reviewed final queue. No training, tuning, discovery
of better checkpoints, dropping prediction failures, or local data export.
"""
import argparse
import json
import time
import numpy as np
import torch
from scipy.io import loadmat
from pub_01_s03_ss07_baseline import ROOT,read,dump,update,sha
from pub_01_s03_ss07_final_manifest import require_ready
from pub_01_s03_ss07_eval_io import FinalModels,SparrRecords,save_array
from pub_01_s03_ss07_eval_common import predict_windows,intervene_aligned,score_record,fuse,correlation,summarize_absolute
from pub_01_s03_ss07_incremental import summary
from pub_01_s03_ss07_dtu_adapter import native_windows


KINDS=['reset','content_blind','shuffled','older_outside_fused_support',
    'same_person_wrong_family','same_story_wrong_person']


def main(seed,device):
    manifest,ready=require_ready()  # before any checkpoint/data load or mkdir
    folder=ROOT/f'final_results/seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    models=FinalModels(seed,device);started=time.time();failures={};timing={}
    dump(folder/'RESOLVED.json',{'seed':seed,'device':device,
        'checkpoint_manifest_sha256':sha(ROOT/'FINAL_CHECKPOINT_MANIFEST.json'),
        'ready_sha256':sha(ROOT/'EVALUATION_READY.json'),'fit_or_selection_performed':False,
        'internal_tests':'POST-HOC INTERNAL EXPLORATORY','external':'fixed adapter, no target-domain model fit'})
    main_ids=manifest['main_model_ids'];ids=list(main_ids)
    if seed in manifest['screening_seeds']:
        ids+=sorted(set(manifest['exploratory_family_ids'])-set(ids))
    diag=['history__'+kind for kind in KINDS];scores={key:[] for key in ids+diag}
    donors={r['target_record']:r for r in read(ROOT/'DONOR_METADATA_PLAN.json')['records']}
    shuffle=read(ROOT/'SHUFFLE_METADATA_PLAN.json')['permutations']
    reader=SparrRecords(models);t=time.perf_counter();reader.encode_test()
    timing['shared_native_P_encoder_testcache']=time.perf_counter()-t
    t=time.perf_counter();neutral=reader.neutral_train_mean();timing['train_only_neutral_summary']=time.perf_counter()-t
    t=time.perf_counter()
    try:finetuned=reader.finetuned_predictions()
    except Exception as exc:
        failed_ft=repr(exc);finetuned={};failures['P_FT']=[{'phase':'raw_phase_prediction','reason':failed_ft}]
    timing['P_FT_full_raw_phase_inference']=time.perf_counter()-t
    controls=[id_ for id_ in ids if id_ not in ['native_P','P_EMA','P_FT']]
    primary=manifest['primary'];primary_model=models.load(primary)

    def failed(condition,record,exc):
        failures.setdefault(condition,[]).append({'record_or_trial':record,'reason':repr(exc)})
        update(folder/'FAILURES.json',failures)

    def store(condition,windows,target,row,tau=0.):
        segments,fused=score_record(windows,target,row,reader.masks[row['record']],tau)
        save_array(folder/'predictions/internal'/condition/(row['record']+'.npy'),fused)
        scores[condition].extend(segments)

    for index,row in enumerate(reader.rows):
        record=row['record'];offset=row['phase_offset'];windows_by_condition={}
        try:z=reader.latent(record);target=np.load(row['envelope_path'],mmap_mode='r')
        except Exception as exc:
            for condition in ids+diag:failed(condition,record,exc)
            continue
        for condition in ids:
            if condition in failures:continue
            t=time.perf_counter()
            try:
                if condition=='P_EMA':
                    if 'native_P' not in windows_by_condition:raise RuntimeError('native prediction failed; EMA alias unavailable')
                    store(condition,windows_by_condition['native_P'],target,row,manifest['EMA_alias']['tau_seconds'])
                elif condition=='P_FT':store(condition,finetuned[record],target,row)
                else:
                    mode='full' if condition=='native_P' else manifest['entries'][f'{condition}:{seed}']['config'].get('mode','full')
                    prediction=predict_windows(models.load(condition),z,record,mode,offset)
                    windows_by_condition[condition]=prediction;store(condition,prediction,target,row)
            except Exception as exc:failed(condition,record,exc)
            timing[condition]=timing.get(condition,0.)+time.perf_counter()-t
        if primary not in failures:
            # Real fitted identity bridge; every intervention below retains z.
            try:
                bridge=intervene_aligned(primary_model,z,record,global_offset=offset)
                assert np.max(np.abs(bridge-windows_by_condition[primary]))<1e-6
            except Exception as exc:
                failed('history__identity_bridge',record,exc)
                break
            for kind in KINDS:
                condition='history__'+kind
                if condition in failures:continue
                t=time.perf_counter()
                try:
                    kwargs={'global_offset':offset};actual_kind=kind
                    if kind in ['same_person_wrong_family','same_story_wrong_person']:
                        spec=donors[record][kind]
                        if not spec['eligible']:continue  # frozen key subset, not a score exclusion
                        donor=reader.latent(spec['donor_record'])[:len(z)];assert len(donor)==len(z)
                        kwargs['donor_observations']=summary(donor);actual_kind='wrong_history'
                    elif kind=='content_blind':kwargs['neutral']=neutral
                    elif kind=='shuffled':kwargs['permutation']=torch.tensor(shuffle[record],device=device,dtype=torch.long)
                    prediction=intervene_aligned(primary_model,z,record,actual_kind,**kwargs)
                    store(condition,prediction,target,row)
                except Exception as exc:failed(condition,record,exc)
                timing[condition]=timing.get(condition,0.)+time.perf_counter()-t
        if index%10==0:update(folder/'STATUS.json',{'phase':'internal','records_attempted':index+1,
            'seconds':time.time()-started,'failed_conditions':list(failures)})
    all_keys={(r['record'],b['start']) for r in reader.rows for b in reader.masks[r['record']]['blocks'] if b['eligible']}
    pair_masks=read(ROOT/'INTERVENTION_PAIR_MASK.json')['pairs']
    for condition,segments in scores.items():
        kind=condition.removeprefix('history__')
        expected={tuple(k) for k in pair_masks[kind]['eligible_keys']} if kind in pair_masks else all_keys
        actual={(r['record'],r['start']) for r in segments}
        complete=condition not in failures and actual==expected and len(segments)==len(expected)
        if not complete and condition not in failures:failed(condition,'matrix',RuntimeError('missing fixed keys; no selective exclusion'))
        dump(folder/f'INTERNAL_{condition}.json',{'seed':seed,'condition':condition,'segments':segments,
            'complete':complete,'failures':failures.get(condition,[]),
            'summary':summarize_absolute(segments),'do_not_pool_donor_subsets_with_full_true':condition.startswith('history__')})
    if failures:
        dump(folder/'COMPLETED.json',{'seed':seed,'all_conditions_complete':False,'failed_conditions':list(failures),
            'external_stage_opened':False,'reason':'internal technical failure; no DTU model inference',
            'checkpoint_manifest_sha256':sha(ROOT/'FINAL_CHECKPOINT_MANIFEST.json'),
            'seconds':time.time()-started,'partial_internal_evidence_preserved':True})
        raise RuntimeError('internal technical failure; external stage blocked for this seed, no score-based stopping')
    # External uses ONLY final main models, all1080 trials, fixed one adapter.
    dtu=read(ROOT/'DTU_PROTOCOL.json');external={id_:[] for id_ in main_ids}
    for person in range(1,19):
        try:data=loadmat(ROOT/f'external_dtu/extracted/S{person}_data_preproc.mat',simplify_cells=True)['data']
        except Exception as exc:
            for condition in main_ids:failed('DTU__'+condition,f'S{person}',exc)
            continue
        assert len(data['eeg'])==len(data['wavA'])==len(data['wavB'])==60
        for trial in range(60):
            key=f'S{person}_trial{trial+1:02d}'
            labels=data['dim']['chan']['eeg'][trial].tolist()
            raw,starts,constants=native_windows(data['eeg'][trial],labels,dtu['physical_scalp_names'])
            assert starts.tolist()==dtu['windows']['starts']
            z=torch.from_numpy(models.encode(raw)).to(device);predictions={}
            for condition in main_ids:
                failure_id='DTU__'+condition
                if failure_id in failures:continue
                t=time.perf_counter()
                try:
                    tau=0.
                    if condition=='P_EMA':pred=predictions['native_P'];tau=manifest['EMA_alias']['tau_seconds']
                    elif condition=='P_FT':pred=models.finetuned_windows(raw)
                    else:
                        mode='full' if condition=='native_P' else manifest['entries'][f'{condition}:{seed}']['config'].get('mode','full')
                        pred=predict_windows(models.load(condition),z,key,mode)
                        predictions[condition]=pred
                    fused,covered=fuse(pred,3200,tau)
                    assert np.flatnonzero(covered).tolist()==list(range(1,3167))
                    ra=correlation(fused[covered],data['wavA'][trial][covered])
                    rb=correlation(fused[covered],data['wavB'][trial][covered])
                    save_array(folder/'predictions/external'/condition/(key+'.npy'),fused)
                    external[condition].append({'subject':f'S{person}','trial':trial+1,'rA':ra,'rB':rb,
                        'aux_accuracy':1. if ra>rb else .5 if ra==rb else 0.,'samples':3166})
                except Exception as exc:failed(failure_id,key,exc)
                timing[failure_id]=timing.get(failure_id,0.)+time.perf_counter()-t
        update(folder/'STATUS.json',{'phase':'external','subjects_attempted':person,
            'seconds':time.time()-started,'failed_conditions':list(failures)})
    for condition,trials in external.items():
        complete='DTU__'+condition not in failures and len(trials)==1080
        if complete:assert all(sum(r['subject']==f'S{p}' for r in trials)==60 for p in range(1,19))
        dump(folder/f'EXTERNAL_{condition}.json',{'seed':seed,'condition':condition,'trials':trials,
            'complete':complete,'failures':failures.get('DTU__'+condition,[]),
            'DTU_protocol_sha256':sha(ROOT/'DTU_PROTOCOL.json')})
    parameter_counts={}
    for condition,model in models.loaded.items():
        spec=manifest['entries'][f'{condition}:{seed}'];prior_only=spec['config'].get('mode')=='prior_only'
        registered=sum(p.numel() for p in model.parameters())
        active=sum(p.numel() for name,p in model.named_parameters()
            if not name.startswith('predictor.') and not (prior_only and name.startswith('residual.')))
        parameter_counts[condition]={'registered_model_parameters':registered,
            'structural_parameters_affecting_prediction':active,
            'shared_P_encoder_parameters_separate':0 if condition=='P_FT' else sum(p.numel() for p in models.encoder.parameters()),
            'unused_prior_only_correction_not_counted_active':prior_only,
            'runtime_still_includes_any_computed_but_unused_branch':True}
    dump(folder/'COMPLETED.json',{'seed':seed,'all_conditions_complete':not failures,
        'failed_conditions':list(failures),'seconds':time.time()-started,
        'cost_components_seconds_including_transfer_scoring_writing':timing,
        'head_condition_costs_exclude_separately_reported_shared_P_encoding':True,
        'FT_encoder_and_head_inference_reported_separately_from_FT_scoring':True,
        'parameter_counts':parameter_counts,
        'peak_GPU_bytes':torch.cuda.max_memory_allocated(device),
        'checkpoint_manifest_sha256':sha(ROOT/'FINAL_CHECKPOINT_MANIFEST.json'),
        'no_training_or_model_reselection':True,'arrays_stay_on_server':True})
    if failures:raise RuntimeError('evaluation conditions failed; retain all evidence, explicit review, no selective averaging')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    main(a.seed,a.device)
