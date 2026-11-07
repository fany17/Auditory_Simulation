"""Independent D26 saved-evidence audit. Run on 2203; stdlib only, no model/data reads."""
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_contrastive_history_20261106')
REPORT = ROOT / 'report01'
FORMAL = ROOT / 'formal'
PROTOCOL = '8c249eabeb9edb9a5360006f1d3dabb9552e2a22ff649b9bae72801f75311b4d'
SEEDS = (1024, 2026, 31415)
ARMS = ('HH', 'CH', 'CC')
M = ('r', 'MSE')
HASHES = {}
CHECKS = Counter()


def need(ok, message):
    if not ok:
        raise AssertionError(message)
    CHECKS['assertions'] += 1


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    data = path.read_bytes()
    HASHES[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def equal(a, b, label):
    if isinstance(a, (float, int)) and not isinstance(a, bool):
        need(math.isfinite(float(b)) and abs(a-float(b)) <= 1e-11*max(1., abs(a)), label)
    else:
        need(a == b, label)


def fields(expected, observed, label):
    for key, value in expected.items():
        equal(value, observed[key], label + '/' + key)


def index(rows, keys):
    out = {tuple(str(row[k]) for k in keys): row for row in rows}
    need(len(out) == len(rows), 'Unique keys ' + str(keys))
    return out


def table(name, rows, keys):
    path = REPORT / (name + '.csv')
    HASHES[str(path)] = digest(path)
    with path.open(newline='') as stream:
        actual = list(csv.DictReader(stream))
    left, right = index(rows, keys), index(actual, keys)
    need(left.keys() == right.keys(), name + ' complete keys')
    for key, row in left.items():
        fields(row, right[key], name + str(key))
    CHECKS[name + '_rows'] = len(rows)


def grouped(blocks):
    people = defaultdict(list)
    for block in blocks:
        people[block['subject']].append(block)
    return {p: dict(subject=p, blocks=len(bs), **{m: mean(b[m] for b in bs) for m in M})
            for p, bs in people.items()}


def macro(blocks):
    persons = grouped(blocks)
    return dict(people=len(persons), blocks=len(blocks), records=len({b['record'] for b in blocks}),
                **{m: mean(p[m] for p in persons.values()) for m in M})


def main():
    wall, cpu = time.monotonic(), time.process_time()
    protocol = read(ROOT / 'PROTOCOL_V05.json')
    terminal = read(FORMAL / 'COMPLETED.json')
    inputs = read(ROOT / 'INPUTS.json')
    need(digest(ROOT / 'PROTOCOL_V05.json') == PROTOCOL and protocol['frozen'], 'Frozen protocol')
    need(digest(ROOT / 'INPUTS.json') == protocol['inventory_sha256'], 'Frozen inventory')
    need(terminal['passed'] and terminal['old_test_reads'] == 0 and terminal['protocol_sha256'] == PROTOCOL,
         'Original terminal with no test access')
    mask_path, = [Path(p) for p in protocol['sources'] if p.endswith('/block_eligibility/val.json')]
    mask = read(mask_path)
    need(digest(mask_path) == protocol['sources'][str(mask_path)] and mask['prediction_independent'], 'Frozen mask')
    manifest = read(REPORT / 'REPORT_ARTIFACTS.json')
    need(manifest['terminal_sha256'] == digest(FORMAL / 'COMPLETED.json'), 'Report terminal pin')
    for name, sha in manifest['files'].items():
        need(digest(REPORT / name) == sha, 'Report artifact ' + name)
    pinned_sources = {
        'pub_01_s03_ss07_contrastive_history_report.py': '550913f9c880c44687fbac280675db0285c42ee9bfeef18ee2b7a1619a149a08',
        'pub_01_s03_ss07_contrastive_history_report_cost.py': 'd07c8ab6bfb5fbae8292432da8ca4abac276846d8dadc780bf792c906709dbfd',
        'pub_01_s03_ss07_contrastive_history_report_figures.py': '9650a02253958d217ed0d6c5af8c9b6ec7a252f5050181e74459cac837f093f9',
        'HISTORICAL_REFERENCES.json': '5c95ce13af5eecf960a74d9c935e626a232ad048ccf31b1e4f7a6de5bf51b1bc',
        'D28_MATCHED_RAW_RIDGE_COMPLETED.json': '9be6b1dcc502318c4d6a1c7df734818af3deb5a64c76a2df739e2464599229f5',
    }
    need(manifest['sources'] == pinned_sources, 'Reviewed report sources')
    summary = read(REPORT / 'RESULT_SUMMARY.json')
    roster = {r['record']: r for r in inputs['phases']['val']}
    families = {r['source_family'] for r in inputs['phases']['train']}
    eligible = {(b['record'], b['start']): b['samples'] for b in mask['blocks'] if b['eligible']}
    need(len(inputs['phases']['train']) == 426 and len(roster) == 49 and len(eligible) == 752, 'Frozen cohort')
    need(mask['records'] == list(roster) and len({r['subject'] for r in roster.values()}) == 49, 'One VAL record per person')
    need(Counter(r['story'] for r in roster.values()) == {'audiobook_13_2': 6, 'audiobook_5_2': 23,
         'podcast_15': 4, 'podcast_21': 7, 'podcast_30': 9}, 'Five story counts')
    need(sum(r['source_family'] in families for r in roster.values()) == 29, '29/20 source-family split')
    need({r['subject'] for r in roster.values()} <= {r['subject'] for r in inputs['phases']['train']}, 'All VAL people TRAIN-seen')
    blocks_out, people_out, strata_out, primary, paired, paired_people = [], [], [], [], [], []
    values, source_scores, per_person, anchors = {}, {}, {}, {}

    def parse(saved, pure):
        bs = []
        seen = set()
        for row in saved:
            k = row['record'], row['start']
            need(k in eligible and k not in seen, 'Original block without duplicate')
            seen.add(k)
            meta = roster[k[0]]
            need(row['subject'] == meta['subject'] and row['source_family'] == meta['source_family'], 'Block metadata')
            support = eligible[k] - max(0, 960-max(k[1], 1)) if pure else eligible[k]
            need(row['retained_samples' if pure else 'samples'] == support, 'Original sample support')
            need(all(math.isfinite(row[m]) for m in M) and -1 <= row['r'] <= 1 and row['MSE'] >= 0, 'Finite block metrics')
            bs.append(dict(row, story=meta['story']))
        need(seen == eligible.keys(), 'All 752 block keys')
        return bs

    for seed in SEEDS:
        anchor = read(ROOT.parent / 's03_physical_stream_20261106' / 'fit' / f'seed{seed}_ANCHOR.json')['anchor']['val']
        anchor_blocks = parse(anchor['blocks'], False)
        fields({m: macro(anchor_blocks)[m] for m in M}, anchor, 'Anchor block macro')
        anchors[seed] = grouped(anchor_blocks)
        saved_people = {p['subject']: p for p in anchor['people']}
        for p, row in anchors[seed].items():
            fields(row, saved_people[p], 'Anchor per person')
        for arm in ARMS:
            folder = FORMAL / f'seed{seed}' / f'READOUT_{arm}'
            rec = read(folder / 'COMPLETED.json')
            need(rec in terminal['evaluations'] and rec['passed'] and rec['epoch'] == 4
                 and rec['lambda_fixed'] == .01 and rec['old_test_reads'] == 0, 'Final epoch 4 Ccap receipt')
            routes = ('near', 'far', 'copy') if arm == 'HH' else ('native',)
            for route in routes:
                score = read(folder / f'SCORE_{route}.json')
                source_scores[seed, arm, route] = score
                need(set(score) == ({'raw', 'affine', 'Ccap'} if route == routes[0] else {'Ccap'}), 'Exact readout routes')
                for readout, data in score.items():
                    for scope in (('full', 'pure_memory') if arm == 'HH' and readout == 'Ccap' else ('full',)):
                        pure = scope == 'pure_memory'
                        saved = data['pure_memory']['records'] if pure else data['segments']
                        bs = parse(saved, pure)
                        label = dict(seed=seed, arm=arm, route=route, readout=readout, scope=scope)
                        key = seed, arm, route, readout, scope
                        overall = macro(bs)
                        receipt = data['pure_memory'] if pure else data['phases']['val']
                        fields({m: overall[m] for m in M}, receipt, 'Saved macro')
                        if not pure:
                            need(receipt['windows'] == 30500, 'All VAL windows')
                        else:
                            need(receipt['first_physical_sample'] == 960, 'Pure memory starts at 960')
                        values[key] = overall
                        per_person[key] = grouped(bs)
                        if not pure:
                            archived = {p['subject']: p for p in data['people']}
                            need(archived.keys() == per_person[key].keys(), 'Saved people roster')
                            for p, row in per_person[key].items():
                                fields(row, archived[p], 'Saved person aggregation')
                        blocks_out.extend(dict(label, **b) for b in bs)
                        for p, row in per_person[key].items():
                            meta = next(m for m in roster.values() if m['subject'] == p)
                            people_out.append(dict(label, **row, story=meta['story'], source_family=meta['source_family']))
                        groups = defaultdict(list)
                        for b in bs:
                            for group in (('all', 'all'), ('story', b['story']), ('exact_source_family', b['source_family']),
                                          ('family', 'shared' if b['source_family'] in families else 'unseen')):
                                groups[group].append(b)
                        for (kind, name), subset in groups.items():
                            strata_out.append(dict(label, stratum=kind, label=name, **macro(subset)))
                        if not pure and route == routes[0]:
                            fields({'r': rec['VAL_r'][readout]}, overall, 'Terminal r')
                            primary.append(dict(seed=seed, arm=arm, readout=readout, **{m: overall[m] for m in M},
                                                delta_r_to_anchor=overall['r']-anchor['r']))
                if route == routes[0]:
                    raw = score['raw']['segments']; affine = score['affine']['segments']
                    need(len(raw) == len(affine) == 752 and all((a['record'], a['start']) == (b['record'], b['start'])
                         and abs(a['r']-b['r']) <= 1e-10 for a,b in zip(raw, affine)), 'Positive affine correlation invariant')
    need(len(values) == 42, 'All 42 conditions')
    for seed in SEEDS:
        chosen = {a: (seed,a,'near' if a == 'HH' else 'native','Ccap','full') for a in ARMS}
        pairs = [(a+'-anchor', per_person[chosen[a]], anchors[seed], 'full') for a in ARMS]
        pairs += [(a+'-'+b, per_person[chosen[a]], per_person[chosen[b]], 'full') for a,b in [('HH','CH'),('HH','CC'),('CH','CC')]]
        pairs += [('near-'+r, per_person[seed,'HH','near','Ccap','pure_memory'],
                   per_person[seed,'HH',r,'Ccap','pure_memory'], 'pure_memory') for r in ('far','copy')]
        for name, left, right, scope in pairs:
            need(left.keys() == right.keys(), 'Paired people')
            rows = []
            for p in left:
                meta = next(m for m in roster.values() if m['subject'] == p)
                rows.append(dict(seed=seed, subject=p, story=meta['story'], source_family=meta['source_family'], comparison=name,
                                 **{'delta_'+m: left[p][m]-right[p][m] for m in M}))
            paired_people.extend(rows)
            paired.append(dict(seed=seed, scope=scope, comparison=name,
                               **{'delta_'+m: mean(r['delta_'+m] for r in rows) for m in M}))
    labels = ('seed','arm','route','readout','scope')
    table('BLOCKS', blocks_out, labels+('record','start'))
    table('PEOPLE', people_out, labels+('subject',))
    table('STRATA', strata_out, labels+('stratum','label'))
    table('PRIMARY', primary, ('seed','arm','readout'))
    table('PAIRED', paired, ('seed','scope','comparison'))
    table('PAIRED_PEOPLE', paired_people, ('seed','subject','comparison'))
    for name, rows, keys in [('primary',primary,('seed','arm','readout')), ('paired',paired,('seed','scope','comparison')),
                              ('strata',strata_out,labels+('stratum','label'))]:
        actual = index(summary[name], keys)
        expected = index(rows, keys)
        need(actual.keys() == expected.keys(), 'Summary ' + name + ' keys')
        for k, row in expected.items():
            fields(row, actual[k], 'Summary ' + name)
    means = {a: mean(r['r'] for r in primary if r['arm'] == a and r['readout'] == 'Ccap') for a in ('CC','CH','HH')}
    winner = max(means, key=means.get)
    need(winner == summary['selected_full_arm'] == terminal['selected_full_arm'] == 'HH', 'Whole-arm winner')
    fields(means, terminal['means'], 'Terminal means')
    fields(means, summary['frozen_Ccap_means'], 'Summary means')

    costs = read(REPORT / 'STAGE_COSTS.json')
    need(costs == summary['actual_costs'], 'Same actual cost tables')
    epochs = read(REPORT / 'TRAINING_EPOCH_LEDGER.json')
    epoch_map = index(epochs, ('seed','branch','stage','epoch'))
    stage_map = index(costs['stages'], ('seed','branch','stage'))
    receipts, totals, expected_epochs = {}, Counter(), 0
    for seed in SEEDS:
        for stage, branches, updates, windows, cap, patience in [('CL',('H','C'),820,104960,100,15),
            ('HEAD',ARMS,3945,252454,250,10), ('FT',ARMS,3945,252454,5,None)]:
            rs = []
            for branch in branches:
                folder = FORMAL / f'seed{seed}' / f'{stage}_{branch}'
                rec = read(folder/'COMPLETED.json'); hist = rec['history']; rs.append(rec)
                receipts[seed,branch,stage] = rec
                need(rec['passed'] and rec['seed'] == seed and rec.get('arm',rec.get('role')) == branch, 'Stage identity')
                need(len(hist) == rec['actual_epochs'] <= cap, 'Stage epoch count')
                best = min(range(len(hist)), key=lambda i: hist[i]['val'])
                need(rec['selected_epoch'] == best and rec['updates'] == len(hist)*updates
                     and rec['selected_prefix_updates'] == (best+1)*updates, 'Actual selection and work')
                stale, bestval = 0, math.inf
                for i,h in enumerate(hist):
                    stale = 0 if h['val'] < bestval else stale+1; bestval = min(bestval,h['val'])
                    need(h['epoch'] == i and h['epoch_updates'] == updates and h['epoch_windows'] == windows
                         and h['total_updates'] == (i+1)*updates and h['stale'] == stale, 'Epoch order and updates')
                    need(read(folder/f'EPOCH_{i:03}.json') == h, 'Original immutable epoch')
                    need(epoch_map[str(seed),branch,stage,str(i)] == dict(seed=seed,branch=branch,stage=stage,**h), 'Reported epoch')
                    expected_epochs += 1
                if stage == 'FT':
                    need(rec['final_epoch'] == 4 and len(hist) == 5, 'Fixed final FT model')
                    minus = read(folder/'EPOCH_MINUS1.json')
                    need(minus['abs_difference'] <= 1e-4 and epoch_map[str(seed),branch,stage,'-1'] ==
                         dict(seed=seed,branch=branch,stage=stage,**minus), 'FT minus1 bridge is not model selection')
                    expected_epochs += 1
                actual = stage_map[str(seed),branch,stage]
                fields(dict(seed=seed,branch=branch,stage=stage,epochs=len(hist),committed_updates=rec['updates'],
                            selection_epoch=best,selected_prefix_updates=rec['selected_prefix_updates'],
                            used_model_epoch=4 if stage=='FT' else best,
                            used_prefix_updates=rec['updates'] if stage=='FT' else rec['selected_prefix_updates'],
                            elapsed_seconds=rec['elapsed_seconds'],CPU_seconds=rec['CPU_seconds'],
                            cuda_peak_reserved=rec['cuda_peak_reserved'],stopping=rec['stopping'],
                            parameters=rec.get('parameters',rec.get('head_parameters'))), actual, 'Stage ledger')
                totals[stage] += rec['updates']
            need(len({r['actual_epochs'] for r in rs}) == 1, 'Paired stage same epoch count')
            if patience:
                n = len(rs[0]['history'])
                need(n == cap or all(r['history'][-1]['stale'] >= patience for r in rs), 'Paired stopping')
                need(not any(all(r['history'][i]['stale'] >= patience for r in rs) for i in range(n-1)), 'First allowed paired stop')
    need(len(stage_map) == 24 and len(epochs) == expected_epochs, 'All stage and epoch records')
    need(dict(totals) == costs['committed_updates_by_stage'] == {'CL':126280,'HEAD':8876250,'FT':177525}, 'True total updates')
    heads = [r for (s,b,t),r in receipts.items() if t == 'HEAD']
    need(all(r['actual_epochs']==250 and r['selected_epoch']==249 and r['history'][-1]['stale']==0 for r in heads), 'All nine HEAD caps disclosed')
    for row in costs['branch_allocations']:
        s,a=row['seed'],row['arm']; cl=receipts[s,a[0],'CL']; head=receipts[s,a,'HEAD']; ft=receipts[s,a,'FT']
        share=1 if a=='HH' else .5
        fields(dict(pretraining_branch=a[0],pretraining_job_share=share,charged_CL_updates=share*cl['updates'],
                    full_standalone_CL_updates=cl['updates'],actual_selected_CL_prefix_updates=cl['selected_prefix_updates'],
                    selected_head_prefix_updates=head['selected_prefix_updates'],used_FT_updates=ft['updates'],
                    committed_head_updates=head['updates'],committed_FT_updates=ft['updates']), row, 'Shared CL allocation')
    need(len(costs['branch_allocations']) == 9 and sum(r['charged_CL_updates'] for r in costs['branch_allocations']) == totals['CL'], 'Shared CL charged once')
    for row in costs['CL_coverage']:
        hist=receipts[row['seed'],row['branch'],'CL']['history']; selected=receipts[row['seed'],row['branch'],'CL']['selected_epoch']
        for prefix, series in [('full',hist),('selected_prefix',hist[:selected+1])]:
            for phase in ('train','val'):
                coverage=[h[phase+'_coverage'] for h in series]
                current=sum(h['current_windows'] for h in coverage); slots=sum(h['history_slot_presentations'] for h in coverage)
                if phase=='train':
                    last=coverage[-1]['cumulative']; unique=last['unique_current_windows']; eeg=last['unique_EEG_samples']; audio=last['unique_audio_windows']; slot=[last[f'unique_slot{i}_samples'] for i in (1,2)]
                else:
                    need(all(h==coverage[0] for h in coverage), 'Fixed CL VAL traversal')
                    last=coverage[0];unique=last['unique_current_windows'];eeg=last['unique_all_EEG_sample_times'];audio=last['unique_audio_story_time_windows'];slot=[last[f'unique_slot{i}_EEG_sample_times'] for i in (1,2)]
                fields(dict(current_EEG_windows=current,memory_EEG_slots=slots,audio_target_windows=current,
                            unique_current_EEG_windows=unique,repeated_current_EEG_windows=current-unique,unique_EEG_sample_times=eeg,
                            unique_slot1_EEG_sample_times=slot[0],unique_slot2_EEG_sample_times=slot[1],
                            repeated_EEG_sample_time_presentations=192*(current+slots)-eeg,
                            unique_audio_story_time_windows=audio,repeated_audio_target_windows=current-audio), row[prefix+'_'+phase.upper()], 'CL coverage')
    need(len(costs['CL_coverage'])==6 and len(costs['supervised_coverage'])==18, 'All coverage jobs')
    for row in costs['supervised_coverage']:
        rec=receipts[row['seed'],row['branch'],row['stage']]; cached=row['stage']=='HEAD'
        for phase in ('train','val'):
            current=inputs['counts'][phase]['supervised_windows']; active=inputs['counts'][phase]['supervised_history_windows']
            physical=sum((r['windows']-1)*96+192 for r in inputs['phases'][phase])
            extra=int(not cached and phase=='val')
            for key,passes in [('full',rec['actual_epochs']+extra),('used_prefix',(rec['selected_epoch']+1 if cached else 5)+extra)]:
                slots=0 if cached else 2*active*passes
                fields(dict(current_encoder_windows=0 if cached else current*passes,memory_encoder_slots=slots,
                            target_or_head_windows=current*passes,unique_supervised_windows=current,repeated_supervised_windows=current*(passes-1),
                            unique_EEG_sample_times=None if cached else physical,
                            repeated_EEG_sample_time_presentations=None if cached else 192*(current*passes+slots)-physical), row[phase][key], 'Supervised coverage')
        if cached:
            folder=FORMAL/f"seed{row['seed']}"/f"HEAD_{row['branch']}"
            need([read(folder/name) for name in rec['cache_receipts']]==row['cache_attempts'], 'Original cache receipts')
            need(row['head_optimize']==rec['head_optimize'] and row['head_cached_VAL']==rec['head_cached_VAL'], 'Measured head costs')
    need(len(costs['readouts']) == 9, 'Nine readout solves')
    for row in costs['readouts']:
        rec=read(FORMAL/f"seed{row['seed']}"/f"READOUT_{row['arm']}"/'COMPLETED.json')
        need(row['train_fit']==rec['TRAIN_fit'] and row['ridge_fits']==1 and row['updates']==rec['new_neural_updates']==0, 'Actual TRAIN Ccap solve')
        fields(dict(current_EEG_windows=rec['encoder_current_windows'],memory_EEG_slots=rec['encoder_slots'],
                    CPU_seconds=rec['CPU_seconds'],elapsed_seconds=rec['elapsed_seconds']),row,'Readout costs')
        need(rec['TRAIN_fit']['TRAIN_fit_windows']==250750 and rec['TRAIN_fit']['TRAIN_physical_points']==24275628
             and rec['TRAIN_fit']['normal_equation_max_abs'] < 1e-12, 'Ccap full TRAIN fit receipt')

    ledger=costs['failure_and_worker_ledger']; workers=ledger['workers']
    need(len(workers)==30 and not ledger['failures'] and ledger['workers_without_timing']==0, 'All actual workers successful')
    roles={(s,stage,role) for s in SEEDS for stage in ('cl','head','ft','readout')
           for role in (('H','C') if stage in ('cl','head') else ARMS)}
    need({(w['seed'],w['stage'],w['role']) for w in workers}==roles, '30 original worker roles')
    registered=list(FORMAL.glob('ATTEMPT_*/*_PROCESS.json'))
    need(len(registered)==30, 'No extra registered workers')
    for w in workers:
        log=FORMAL/w['log']; stem=log.with_suffix('')
        need(read(Path(str(stem)+'_PROCESS.json'))==w['process'], 'Original process receipt')
        exit_receipt=read(Path(str(stem)+'_EXIT.json'))
        need(exit_receipt==w['exit'] and exit_receipt['pid']==w['process']['pid'] and exit_receipt['returncode']==0
             and exit_receipt['terminal_receipt'] and w['exact_extra_uncommitted_updates']==0, 'Worker successful exit')
        events=[json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
        timing=[e for e in events if e.get('event')=='worker_timing']
        need(timing==[w['timing']], 'Actual unique worker timing from log')
    equal(math.fsum(w['timing']['worker_CPU_seconds'] for w in workers),ledger['summed_worker_CPU_seconds'],'No duplicate worker CPU')
    equal(math.fsum(w['exit']['worker_wall_seconds_observed'] for w in workers),ledger['sum_observed_worker_wall_seconds'],'Worker wall sum')
    for attempt in ledger['attempts']:
        ws=[w for w in workers if w['attempt']==attempt['attempt']]
        equal(max(w['process']['launched_monotonic_seconds']+w['exit']['worker_wall_seconds_observed'] for w in ws)
              -min(w['process']['launched_monotonic_seconds'] for w in ws),attempt['observed_first_launch_to_last_exit_seconds'],'Actual matrix wall')
    engineering=read(ROOT/'ENGINEERING.json')
    need(engineering==summary['engineering_costs'] and sum(engineering['neural_v01']['exact_completed_updates'].values())
         +engineering['gradients_v02']['extra_CL_updates']==14 and engineering['Ccap_v02']['engineering_ridge_fits']==1,
         'Engineering work separate')
    need(engineering['neural_v01']['CPU_seconds'] is None and engineering['neural_v01']['wall_seconds'] is None,
         'Unknown engineering timings retained')

    replay_root=ROOT/'coordinator_final_replay_v02'; replay=read(replay_root/'COMPLETED.json')
    failures=[]; comparisons=[]; calls=Counter(); cpus=[]; walls=[]
    for seed in SEEDS:
        for arm in ARMS:
            file=replay_root/f'REPLAY_{seed}_{arm}.json'; case=read(file)
            need(digest(file)==replay['case_receipt_sha256'][f'{seed}_{arm}'] and case['execution_completed'], 'Final replay case pin')
            need(case['source_sha256']=='81734e974351a33a6c1d91c47379ce5efd5d1b75b847a0038a7cac47f9f53705'
                 and case['protocol_sha256']==PROTOCOL and case['old_test_reads']==case['new_fits']==case['optimizer_updates']==case['GPU']==0,
                 'Replay source and scope')
            need(case['EEG_samples_read']==3936 and case['target_samples_read']==3840 and case['record']==next(iter(roster)), 'Original fixed replay prefix')
            need(all(v<=2e-5 for v in case['memory_bridge_max_abs'].values()), 'Independent memory/library bridge')
            proc=read(replay_root/f'PROCESS_{seed}_{arm}.json'); ex=read(replay_root/f'EXIT_{seed}_{arm}.json')
            # Replay queue EXIT receipts contain the return code and observed time;
            # the paired PROCESS receipt carries the registered identity/argv.
            need(len(proc['process'])==3 and proc['process'][1]==1012 and ex['returncode']==0
                 and proc['argv'][-4:]==['--seed',str(seed),'--arm',arm]
                 and ex['observed_spawn_to_exit_seconds']>0, 'Replay registered case and successful exit')
            calls.update(case['actual_calls']);cpus.append(case['CPU_seconds']);walls.append(case['wall_seconds'])
            need(len(case['comparisons'])==(8 if arm=='HH' else 3), 'All intended replay comparisons')
            for c in case['comparisons']:
                score=source_scores[seed,arm,c['path']][c['output']]
                pure=c['support']=='physical_960_to_3839'
                expected=next(b for b in (score['pure_memory']['records'] if pure else score['segments'])
                              if b['record']==case['record'] and b['start']==0)
                fields({m:expected[m] for m in M},c['expected'],'Replay expected actual saved metric')
                need(c['observed']['samples']==(2880 if pure else 3839) and c['original_absolute_tolerance']==1e-4,'Unchanged replay threshold/support')
                errors={m:abs(c['observed'][m]-c['expected'][m]) for m in M}
                fields(errors,c['errors'],'Replay absolute error arithmetic')
                checks={m:errors[m]<=1e-4 for m in M}
                need(checks==c['original_checks'] and c['strict_passed']==all(checks.values()),'Unchanged strict judgment')
                comparisons.append(dict(seed=seed,arm=arm,**c))
                if not c['strict_passed']:
                    failures.append(dict(seed=seed,arm=arm,**c))
    need(failures==replay['original_threshold_failures'] and len(failures)==9
         and all(c['output']=='raw' and c['original_checks']=={'r':True,'MSE':False} for c in failures),'Nine raw MSE strict failures retained')
    need(len(comparisons)==42 and all(c['original_checks']['r'] for c in comparisons)
         and all(c['strict_passed'] for c in comparisons if c['output']!='raw'),'All r and calibrated outputs within original threshold')
    need(not replay['passed'] and not replay['strict_all_metrics_passed'] and replay['execution_completed'],'Strict failure not relabeled')
    need(dict(calls)==replay['actual_calls'],'All replay work counted once')
    equal(math.fsum(cpus),replay['CPU_seconds'],'Replay CPU');equal(math.fsum(walls),replay['worker_wall_seconds_sum'],'Replay wall')
    bridge=read(ROOT/'coordinator_cpu_bridge_v01'/'RESULT.json')
    need(digest(ROOT/'coordinator_cpu_bridge_v01'/'RESULT.json')=='5c040fc73b880d34e87bb9fc9dee1fc88f87daaaed2c27e13d584d1c0ed62169','Previously reviewed CPU bridge unchanged')
    combined={k:v+replay['prior_failed_actual_calls'][k]+bridge['actual_calls'][k] for k,v in calls.items()}
    pair_summary=[]
    for name in dict.fromkeys(p['comparison'] for p in paired):
        rows=[r for r in paired if r['comparison']==name]
        expected=dict(comparison=name,scope=rows[0]['scope'],**{'mean_delta_'+m:mean(r['delta_'+m] for r in rows) for m in M},
                      **{'n_positive_delta_'+m:sum(r['delta_'+m]>0 for r in rows) for m in M})
        actual=next(r for r in summary['paired_summary'] if r['comparison']==name)
        fields(expected,actual,'Paired three-seed summary');pair_summary.append(expected)
    result=dict(created_UTC=datetime.now(timezone.utc).isoformat(),saved_score_audit_passed=True,cost_audit_passed=True,
                numerical_strict_passed=False,numerical_execution_completed=True,all_r_checks_passed=True,
                calibrated_output_checks_passed=True,raw_MSE_failed_cases=9,selected_full_arm=winner,
                Ccap_means=means,paired_summary=pair_summary,checked_counts=dict(CHECKS),
                actual_stage_updates=dict(totals),HEAD_at_cap=9,HEAD_selected_epoch=249,HEAD_final_stale=0,
                formal_worker_CPU_seconds=ledger['summed_worker_CPU_seconds'],formal_worker_wall_sum=ledger['sum_observed_worker_wall_seconds'],
                formal_attempts=ledger['attempts'],engineering_updates=14,engineering_ridge_fits=1,
                replay_max_abs_r=max(c['errors']['r'] for c in comparisons),
                replay_max_calibrated_abs_MSE=max(c['errors']['MSE'] for c in comparisons if c['output']!='raw'),
                replay_raw_MSE_abs_range=[min(c['errors']['MSE'] for c in failures),max(c['errors']['MSE'] for c in failures)],
                diagnostic_calls_including_V01_V02_CPU_bridge=combined,
                source_sha256=digest(Path(__file__)),evidence_sha256=HASHES,
                new_signal_reads=0,new_neural_forwards=0,new_fits=0,new_optimizer_updates=0,
                scope='Independent stdlib aggregation of full saved scalar evidence, not another full-cohort raw/model replay.',
                limitations=['Repeated development VAL; all 49 people TRAIN-seen, only five stories.',
                             'Whole-arm HH marginal improvement; only two of three seeds beat anchor; below D22 B.',
                             'All nine HEAD jobs selected the still-improving budget cap; convergence is not established.',
                             'Raw first-block replay MSE strictly fails in all nine cases. Cause unresolved; no tolerance relaxation.',
                             'Raw/model replay covers one fixed first block, not all 752 blocks.',
                             'Historical reference protocols and budgets differ; no independent SOTA claim.'],
                audit_CPU_seconds=time.process_time()-cpu,audit_wall_seconds=time.monotonic()-wall)
    destination=ROOT/'COORDINATOR_FINAL_REVIEW.json'
    with destination.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))


if __name__ == '__main__':
    main()
