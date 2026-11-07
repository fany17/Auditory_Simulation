"""Aggregate complete, fixed final cells. Never tune or average failures."""
import csv
import json
import numpy as np
from pub_01_s03_ss07_baseline import ROOT,read,dump,sha
from pub_01_s03_ss07_final_manifest import require_ready
from pub_01_s03_ss07_eval_common import paired_arrays,descriptive_family_slices,summarize_absolute
from pub_01_s03_ss07_statistics import summarize,crossed_seed_person,crossed_family_sensitivity
from pub_01_s03_ss07_dtu_statistics import paired as dtu_paired


def csv_file(path,rows):
    assert rows
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    manifest,ready=require_ready();seeds=manifest['seeds'];main_ids=manifest['main_model_ids']
    for seed in seeds:
        cell=read(ROOT/f'final_results/seed{seed}/COMPLETED.json')
        assert cell['seed']==seed and cell['all_conditions_complete'],'failure/missing cells require review, never success-only averaging'
        assert cell['checkpoint_manifest_sha256']==sha(ROOT/'FINAL_CHECKPOINT_MANIFEST.json')
    destination=ROOT/'final_summary';destination.mkdir(exist_ok=False)
    results={};absolute=[];person_rows=[];source_rows=[];costs=[]
    for seed in seeds:
        ids=list(main_ids)
        if seed in manifest['screening_seeds']:ids+=sorted(set(manifest['exploratory_family_ids'])-set(ids))
        for condition in ids:
            item=read(ROOT/f'final_results/seed{seed}/INTERNAL_{condition}.json')
            assert item['seed']==seed and item['condition']==condition and item['complete']
            results[(condition,seed)]=item['segments'];summary=summarize_absolute(item['segments'])
            absolute.append({'seed':seed,'condition':condition,'weighted_r':summary['weighted_r'],
                'seen_r':summary['phase_macro_r']['test_seen'],'heldout_r':summary['phase_macro_r']['test_heldout'],'blocks':summary['blocks']})
            person_rows.extend(dict(seed=seed,condition=condition,**r) for r in summary['people'])
            groups={(r['phase'],r['subject'],r['record'],r['story'],r['source_family']) for r in item['segments']}
            for phase,subject,record,story,family in sorted(groups):
                rs=[r['r'] for r in item['segments'] if r['record']==record]
                source_rows.append({'seed':seed,'condition':condition,'phase':phase,'subject':subject,
                    'record':record,'story':story,'source_family':family,'mean_block_r':float(np.mean(rs)),'blocks':len(rs)})
        costs.append(read(ROOT/f'final_results/seed{seed}/COMPLETED.json'))
    values,blocks=paired_arrays(results,main_ids,seeds)
    primary=main_ids.index(manifest['primary']);comparisons=manifest['contrast_ids_and_comparators']
    differences=np.stack([values[primary]-values[main_ids.index(other)] for name,other in comparisons])
    report=summarize(differences,blocks,[name for name,_ in comparisons])
    report.update({'primary':manifest['primary'],'all_old_tests':'POST-HOC INTERNAL EXPLORATORY',
        'fixed_model_selection_not_repeated':True,'absolute_per_seed':absolute,
        'per_family_LOFO_descriptive':descriptive_family_slices(differences,blocks),
        'EMA_identity_alias':manifest['EMA_alias']['tau_seconds']==0,
        'duplicate_tau0_contrast_retained_in_prespecified_M':True})
    dump(destination/'INTERNAL_MAIN.json',report)
    exploratory_ids=['native_P']+manifest['exploratory_family_ids']
    values,metadata=paired_arrays(results,exploratory_ids,manifest['screening_seeds'])
    exploratory=summarize(values[1:]-values[:1],metadata,[id_+'-native_P' for id_ in exploratory_ids[1:]])
    exploratory.update({'exploratory_only':True,'separate_M5_family':True,
        'not_independent_replication_or_joint_FWER_with_main_family':True})
    dump(destination/'INTERNAL_FIVE_FAMILY_EXPLORATORY.json',exploratory)
    # Each history intervention gets its OWN frozen target keys and matched
    # true-history values. Do NOT apply main summarize to unequal subsets.
    pairs=read(ROOT/'INTERVENTION_PAIR_MASK.json')['pairs'];diagnostics=[]
    for kind in ['reset','content_blind','shuffled','older_outside_fused_support','same_person_wrong_family','same_story_wrong_person']:
        condition='history__'+kind;working={}
        for seed in seeds:
            item=read(ROOT/f'final_results/seed{seed}/INTERNAL_{condition}.json');assert item['complete']
            working[('true',seed)]=results[(manifest['primary'],seed)]
            working[('intervention',seed)]=item['segments']
        keys=[tuple(key) for key in pairs[kind]['eligible_keys']] if kind in pairs else None
        values,metadata=paired_arrays(working,['true','intervention'],seeds,keys);diff=values[:1]-values[1:]
        paired_keyset={(b['record'],b['start']) for b in metadata}
        coverage={phase:{'people':len({b['subject'] for b in metadata if b['phase']==phase}),
            'records':len({b['record'] for b in metadata if b['phase']==phase}),
            'blocks':sum(b['phase']==phase for b in metadata)} for phase in ['test_seen','test_heldout']}
        item={'intervention':kind,'coverage':coverage,'same_true_intervention_key_subset':True,
            'eligible_keys':[[r['record'],r['start']] for r in metadata],
            'interpretation':'OFFLINE distribution diagnostic, approximate exploratory intervals, no mechanistic causality/FWER claim'}
        if all(c['people']>0 for c in coverage.values()):
            point,samples=crossed_seed_person(diff,metadata);sensitivity=crossed_family_sensitivity(diff,metadata)
            item.update({'true_minus_intervention_seed_weighted_differences':point[0,:,0].tolist(),
                'weighted_mean_difference':float(point[0,:,0].mean()),
                'exploratory_95_approx_interval':np.quantile(samples[0,:,0],[.025,.975]).tolist(),
                'crossed_Exp_sensitivity_not_CI':np.quantile(sensitivity[0,:,0],[.025,.975]).tolist(),
                'matched_true_subset_absolute':[summarize_absolute([r for r in working[('true',s)] if (r['record'],r['start']) in paired_keyset]) for s in seeds]})
        else:item['weighted_endpoint_unavailable_reason']='subset lacks one original stratum; never renormalize away2/3+1/3 rule'
        diagnostics.append(item)
    dump(destination/'INTERNAL_HISTORY_DIAGNOSTICS.json',{'diagnostics':diagnostics,
        'not_main_contrast_family':True,'older_changes_recency_and_parity':True,'shuffle_may_use_later_observations':True})
    # External model rows are exact full18x60 keys, no hidden trial selection.
    external_values=[];external_absolute=[]
    for condition in main_ids:
        training_seeds=[]
        for seed in seeds:
            item=read(ROOT/f'final_results/seed{seed}/EXTERNAL_{condition}.json')
            assert item['seed']==seed and item['condition']==condition and item['complete']
            assert item['DTU_protocol_sha256']==sha(ROOT/'DTU_PROTOCOL.json')
            mapping={(r['subject'],r['trial']):r for r in item['trials']};assert len(mapping)==len(item['trials'])==1080
            rvalues=np.array([[mapping[(f'S{p}',t)]['rA'] for t in range(1,61)] for p in range(1,19)])
            accuracy=np.array([[mapping[(f'S{p}',t)]['aux_accuracy'] for t in range(1,61)] for p in range(1,19)])
            assert np.isfinite(rvalues).all() and np.isfinite(accuracy).all();training_seeds.append(rvalues)
            external_absolute.append({'condition':condition,'seed':seed,'attended_r':float(rvalues.mean()),
                'aux_accuracy_descriptive':float(accuracy.mean()),'subjects':18,'trials':1080,
                'per_person_attended_r':rvalues.mean(1).tolist(),'per_person_aux_accuracy':accuracy.mean(1).tolist()})
        external_values.append(np.stack(training_seeds))
    external_values=np.stack(external_values)
    difference=np.stack([external_values[primary]-external_values[main_ids.index(other)] for _,other in comparisons])
    external=dtu_paired(difference,[name for name,_ in comparisons]);external['absolute_per_seed']=external_absolute
    external.update({'primary':manifest['primary'],'fixed_input_adaptation_no_target_domain_model_fit':True,
        'stimulus_room_generalization_unproven':True,'DTU_protocol_sha256':sha(ROOT/'DTU_PROTOCOL.json')})
    dump(destination/'EXTERNAL_DTU_MAIN.json',external)
    csv_file(destination/'INTERNAL_PER_SEED.csv',absolute)
    csv_file(destination/'INTERNAL_PER_PERSON.csv',person_rows)
    csv_file(destination/'INTERNAL_PER_PERSON_RECORD_STORY_FAMILY.csv',source_rows)
    stage_totals={}
    for cell in manifest['training_costs']:
        stage=cell['stage'];c=cell['training'];total=stage_totals.setdefault(stage,
            {'cells':0,'epochs':0,'optimizer_updates':0,'actual_seconds_including_any_reported_cache':0.,
             'supervised_window_presentations':0,'aux_teacher_presentations':0,'aux_teacher48_scalar_targets':0})
        total['cells']+=1;total['epochs']+=c['epochs'];total['optimizer_updates']+=c['updates']
        total['actual_seconds_including_any_reported_cache']+=c.get('seconds',c.get('seconds_including_cache',0))
        total['supervised_window_presentations']+=c.get('supervised_window_presentations',c.get('target_window_presentations',0))
        total['aux_teacher_presentations']+=c.get('teacher_presentations',0)
        total['aux_teacher48_scalar_targets']+=48*c.get('teacher_presentations',0)
    assert stage_totals['screen']['cells']==54 and stage_totals['shared_pretrain']['cells']==5
    dump(destination/'COSTS.json',{'actual_training_costs':manifest['training_costs'],'stage_totals':stage_totals,
        'whole_method_search_includes_all54_screen_and_all_full_candidates_controls':True,
        'winner_formal_budget_separate_from_entire_search_cost':True,
        'five_shared_pretraining_runs_charged_once_not_per_condition':True,
        'engineering_probes_and_profiling_excluded_from_scientific_training_totals':True,
        'actual_evaluation_cost_components':costs,
        'shared_encoder_cost_not_hidden_in_head_only_timings':True,
        'timing_is_runtime_not_isolated_deployment_latency_or_FLOP_claim':True})
    dump(destination/'COMPLETED.json',{'all_fixed_cells_aggregated':True,
        'checkpoint_manifest_sha256':sha(ROOT/'FINAL_CHECKPOINT_MANIFEST.json'),
        'test_outcomes_not_used_to_change_any_choice':True,'scientific_goal_achieved_claimed':False,
        'next':'independent raw-input prediction/metric review, plots, scientific acceptance/remaining evidence'})
    print(json.dumps({'aggregated':True,'primary':manifest['primary'],'scientific_goal_complete':False}))


if __name__=='__main__':main()
