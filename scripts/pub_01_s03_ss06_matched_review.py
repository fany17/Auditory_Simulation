"""Post-execution observer: independent metrics, raw CPU parity and ledger.

Never changes trained files, selection, models, split or conditions. Only run
after all fitting and the common assessment. All scientific computation2203.
"""
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from pub_01_s03_ss06_matched import ROOT,OLD,SEEDS,Inputs,read,dump,sha,setup,write_csv
from pub_01_s03_ss06_strong import CausalStrong,NativeStrong
from pub_01_s03_ss06_review import bootstrap
from pub_01_s03_ss06_data import oriented


def table(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))


def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert read(ROOT/'ALL_TRAINING_REVIEW.json')['passed'] and read(ROOT/'audit01/completed.json')
    out=ROOT/'review01';out.mkdir(exist_ok=False);data=Inputs('audit_holdout');metadata={r['record']:r for r in data.rows}
    assert len(metadata)==41 and sum(r['score_blocks'] for r in data.rows)==538
    persons=table(ROOT/'audit01/per_subject.csv');records=table(ROOT/'audit01/per_record.csv');results=table(ROOT/'audit01/RESULTS.csv')
    errors=[];raw_r_errors=[];raw_mse_errors=[];summary=[];ledger=[]
    stage_starts={}
    for line in (ROOT/'offline_reference_native.log').read_text().splitlines():
        try:item=json.loads(line)
        except (ValueError,TypeError):continue
        cmd=item.get('command',[])
        if cmd and '--pilot' not in cmd and '--seed' in cmd:
            stage_starts[(cmd[2],int(cmd[cmd.index('--seed')+1]))]=item['started']
    for seed in SEEDS:
        for name in ('C','O','P'):
            rr=[r for r in records if r['condition']==name and int(r['seed'])==seed]
            pp=[r for r in persons if r['condition']==name and int(r['seed'])==seed]
            blocks=table(ROOT/f'audit01/{name}_seed{seed}_blocks.csv')
            assert {(r['record'],int(r['block'])) for r in blocks}=={(r['record'],b) for r in data.rows for b in range(r['score_blocks'])}
            for r in rr:
                bb=[b for b in blocks if b['record']==r['record']]
                errors.extend([abs(np.mean([float(b['r']) for b in bb])-float(r['r'])),abs(np.mean([float(b['mse']) for b in bb])-float(r['mse']))])
            for p in pp:
                selected=[r for r in rr if r['subject']==p['subject']]
                errors.append(abs(np.mean([float(r['r']) for r in selected])-float(p['r'])))
            result=next(r for r in results if r['condition']==name and int(r['seed'])==seed)
            errors.append(abs(np.mean([float(p['r']) for p in pp])-float(result['r'])))
            folder=ROOT/f'formal01/{name if name!="P" else "P_head"}_seed{seed}'
            checkpoint=torch.load(folder/'best_validation.pt',map_location='cpu',weights_only=True)
            model=(CausalStrong() if name=='C' else NativeStrong()).eval();model.load_state_dict(checkpoint['model'])
            scaling=np.load(OLD/'newtrain_EEG_target_normalization.npz')
            for r in (data.rows[0],data.rows[-1]):
                raw=oriented(r)
                xx=((np.asarray(raw[8:7688],dtype=np.float64)-scaling['mean'][:64])/scaling['std'][:64]).astype(np.float32)
                env=np.load(r['envelope_path'],mmap_mode='r').reshape(-1)
                yy=((np.asarray(env[3840:7680],dtype=np.float64)-scaling['mean'][-1])/scaling['std'][-1]).astype(np.float32).astype(np.float64)
                with torch.no_grad():
                    kwargs={'chunk':127} if name=='C' else {}
                    _,state=model(torch.from_numpy(xx[:3840])[None],**kwargs)
                    p,_=model(torch.from_numpy(xx[3840:])[None],state,**kwargs)
                p=p.numpy().astype(np.float64).reshape(-1)
                expected=next(b for b in blocks if b['record']==r['record'] and int(b['block'])==0)
                raw_r_errors.append(abs(float(np.corrcoef(p,yy)[0,1])-float(expected['r'])))
                raw_mse_errors.append(abs(float(np.mean((p-yy)**2))-float(expected['mse'])))
            if name=='P':
                h=read(folder/'completed.json');pre=read(ROOT/f'formal01/P_pretrain_seed{seed}/completed.json')
                assert h['actual_window_index_sequence']==list(range(130231))
                assert sha(ROOT/'input/pub_01_s03_ss06_matched_native_p.py')==read(ROOT/'P_NATIVE_AMENDMENT_02.json')['source_sha256']
                ledger.append({'condition':name,'seed':seed,'gradient_steps':pre['steps']+h['steps'],
                    'supervised_target_samples':h['target_sample_presentations'],'pretrain_EEG_samples':12526080,
                    'train_EEG_samples_including_warmup_or_encoder_cache':12526080+25004352,
                    'extra_training':'EEG-audio contrastive639steps;frozen encoder cache;native Adam unshuffled head2035steps',
                    'seconds':pre['seconds']+folder.joinpath('completed.json').stat().st_mtime-stage_starts[('head',seed)],
                    'seconds_note':'includes head cache construction/validation;pretrain measured stage time;sharedGPU walltimes',
                    'selected_step':h['selected_step'],'selected_pretrain_step':pre['selected_step']})
            else:
                d=read(folder/'completed.json');assert d['parameters']['all_parameter_gradients_finite']
                ledger.append({'condition':name,'seed':seed,'gradient_steps':3262,'supervised_target_samples':12526080,
                    'pretrain_EEG_samples':0,'train_EEG_samples_including_warmup_or_encoder_cache':(3262+249)*(3840 if name=='C' else 7488),
                    'extra_training':'none','seconds':d['seconds'],'seconds_note':'includes validation;sharedGPU walltimes',
                    'selected_step':d['selected_step'],'selected_pretrain_step':''})
    for name in ('C','O','P'):
        rows=[r for r in results if r['condition']==name];scores=[float(r['r']) for r in rows]
        summary.append({'condition':name,'mean_r':float(np.mean(scores)),'seed_SD':float(np.std(scores,ddof=1)),
            'mean_MSE':float(np.mean([float(r['mse']) for r in rows])),'seeds':3,'audit_records':41,'audit_persons':36,'families':5,'blocks_per_seed':538})
    old=table(OLD/'review01/per_record_all_seeds.csv');comparators=['m1_multi','m2','m3_finetune','m4_adaptive','m5_full','gru'];contrasts=[]
    c={rid:float(np.mean([float(r['r']) for r in records if r['condition']=='C' and r['record']==rid])) for rid in metadata}
    for variant in comparators:
        b={rid:float(np.mean([float(r['r']) for r in old if r['variant']==variant and r['condition']=='true' and r['record']==rid])) for rid in metadata}
        stat=bootstrap({rid:c[rid]-b[rid] for rid in metadata},metadata,primary_count=6)
        contrasts.append({'contrast':'C_minus_'+variant,'delta_r':stat['mean_delta_r'],
            'crossed_Bonferroni_CI_low':stat['crossed_family95_low'],'crossed_Bonferroni_CI_high':stat['crossed_family95_high'],
            'interval_percent':100*(1-.05/6),'positive_interval':stat['crossed_family95_low']>0,
            'negative_interval':stat['crossed_family95_high']<0,'families':5,'subjects':36,'kind':'predeclared this-round EXPLORATORY; run01 already seen'})
    assert max(errors)<1e-10 and max(raw_r_errors)<1e-4 and max(raw_mse_errors)<1e-4
    write_csv(out/'SUMMARY.csv',summary);write_csv(out/'training_ledger.csv',ledger);write_csv(out/'primary_contrasts.csv',contrasts)
    dump(out/'REVIEW.json',{'technical':'PASS','acceptance':'PASS_WITH_LIMITATION','parent_consistency':'PASS_SS06_ONLY',
        'max_aggregation_error':max(errors),'raw_original_CPU_checks':len(raw_r_errors),'CPU_C_microchunk':127,
        'max_raw_CPU_r_error':max(raw_r_errors),'max_raw_CPU_MSE_error':max(raw_mse_errors),
        'same_agent_separate_observer_pass':True,'third_party_scientific_signoff':False,
        'P_native_optimizer_and_sequential_window_order_verified':True,'old_model_weights_used':False,
        'kind':'POST-HOC INTERNAL EXPLORATORY; only5families; original preprocessed EEG, physical synchronization UNKNOWN',
        'C_causality_scope':'provided preprocessed64Hz EEG sample coordinates; no claim raw preprocessing/hardware causal',
        'all_training_completed_before_common_assessment':True,'protocol_sha256':sha(ROOT/'FORMAL_PROTOCOL.json'),
        'P_amendment_sha256':sha(ROOT/'P_NATIVE_AMENDMENT_02.json'),'observer_source_sha256':sha(__file__)})
    print(json.dumps({'summary':summary,'contrasts':contrasts,'review':read(out/'REVIEW.json')}),flush=True)

if __name__=='__main__':main()
