"""Freeze HDPS published split/valid indices against actual100Hz/M inputs.

Unavailable raw records are explicit failures; no64Hz replacement, no new
split, no invented candidates. Arrays stay on2203. Freeze precedes training.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from pub_01_s03_hdps_paper_model import normalize
from pub_01_s03_cohort_policy import select_subjects


def main(config):
    cfg = json.loads(config.read_text())
    base = Path(cfg['work'])
    if not (base / 'raw_batch01/completed.json').is_file() or not (base / 'mel_paper01/completed.json').is_file():
        raise RuntimeError('upstream processes still active')
    split = json.loads((base / 'hdps_split.json').read_text())
    idx = json.loads((base / 'valid_segments_metadata.json').read_text())
    mel = {r['story']: r for r in json.loads((base / 'mel_paper01/manifest.json').read_text())}
    out = Path(cfg.get('prepared_dir', str(base / 'prepared_paper_m01')))
    out.mkdir(exist_ok=False)
    (out / 'config_frozen.json').write_text(json.dumps(cfg, indent=2))
    subjects = list(split)
    stories = sorted({s for phases in split.values() for st in phases.values() for s in st})
    failures, records, windows = [], [], []
    feature_offsets, feature_indices, total_features = {}, {}, 0
    for story in stories:
        if story not in mel:
            failures.append({'story': story, 'type': 'feature_unavailable'})
            continue
        length = mel[story]['shape'][0]
        valid = [i for i in idx[story+'.wav'] if (i+1)*300 <= length]
        feature_offsets[story] = total_features
        feature_indices[story] = {segment: total_features+j for j, segment in enumerate(valid)}
        total_features += len(valid)
    feature = np.lib.format.open_memmap(out / 'feature_windows.npy', mode='w+', dtype='float32', shape=(total_features,300,80))
    for story, mapping in feature_indices.items():
        source = np.load(mel[story]['path'], mmap_mode='r')
        for segment, position in mapping.items():
            feature[position] = source[segment*300:(segment+1)*300]
    feature.flush()
    for subject in subjects:
        output_root = Path(cfg.get('raw_subject_outputs',{}).get(subject,str(base/('preprocessing100_author01_'+subject))))
        outputs = output_root / 'eeg' / subject
        for phase, expected_stories in split[subject].items():
            for story in expected_stories:
                paths = sorted(outputs.glob('*/'+'*_desc-preproc-audio-'+story+'_eeg.npy'))
                if not paths:
                    failures.append({'subject': subject, 'phase': phase, 'story': story, 'type': 'raw100Hz_unavailable'})
                    continue
                if story not in feature_indices:
                    continue
                for path in paths:
                    source = np.load(path, mmap_mode='r', allow_pickle=False)
                    if source.ndim != 2 or source.shape[0] != 64:
                        failures.append({'subject': subject, 'path': str(path), 'type': 'invalid_shape'})
                        continue
                    valid = list(feature_indices[story])
                    unsupported = [i for i in valid if (i+1)*300 > source.shape[1]]
                    if unsupported:
                        failures.append({'subject': subject, 'phase': phase, 'story': story,
                            'type': 'published_valid_indices_exceed_EEG_support', 'indices': unsupported, 'no_replacement': True})
                        continue
                    row = {'subject': subject, 'subject_id': subjects.index(subject), 'phase': phase,
                        'story': story, 'path': str(path), 'segment_count': len(valid), 'segments': valid}
                    records.append(row)
                    for segment in valid:
                        windows.append({'subject': subject, 'subject_id': subjects.index(subject), 'phase': phase,
                            'story': story, 'path': str(path), 'segment': segment,
                            'feature_id': feature_indices[story][segment], 'start_sample': segment*300})
    original_subjects = subjects
    subjects, excluded_subjects = select_subjects(split, records, failures, cfg.get('cohort_policy', {'mode':'strict_complete'}))
    selected_set = set(subjects)
    records = [r for r in records if r['subject'] in selected_set]
    windows = [w for w in windows if w['subject'] in selected_set]
    for row in records + windows:
        row['subject_id'] = subjects.index(row['subject'])
    # Exact source story identities cannot cross phases. Source-family parts
    # remain the author's definition, explicitly not a new family split.
    stories_by_phase = {p:{r['story'] for r in records if r['phase']==p} for p in ('train','val','test')}
    if any(stories_by_phase[a] & stories_by_phase[b] for a,b in (('train','val'),('train','test'),('val','test'))):
        raise RuntimeError('exact story identity crosses author phases')
    keys=[(w['subject'],w['phase'],w['story'],w['segment']) for w in windows]
    if len(set(keys)) != len(keys):
        raise RuntimeError('duplicate subject/phase/story/segment candidate identity')
    with (out / 'window_manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(windows[0]))
        writer.writeheader(); writer.writerows(windows)
    groups = {}
    for phase in ('train', 'val', 'test'):
        phase_windows = [w for w in windows if w['phase']==phase]
        array = np.lib.format.open_memmap(out / (phase+'_eeg.npy'), mode='w+', dtype='float32', shape=(len(phase_windows),300,64))
        pairs = np.zeros((len(phase_windows),2), dtype=np.int64)
        previous, raw = None, None
        for number, row in enumerate(phase_windows):
            if row['path'] != previous:
                raw = np.load(row['path'], mmap_mode='r').T
                previous = row['path']
            first = row['start_sample']
            chunk = normalize(np.asarray(raw[first:first+300], dtype=np.float32)[None])[0]
            if not np.isfinite(chunk).all():
                raise RuntimeError('nonfinite selected EEG window')
            array[number] = chunk
            pairs[number] = [row['subject_id'], row['feature_id']]
        array.flush()
        np.save(out / (phase+'_pairs.npy'), pairs, allow_pickle=False)
        groups[phase] = {subject: np.flatnonzero(pairs[:,0]==i).tolist() for i,subject in enumerate(subjects)}
    (out / 'candidate_groups.json').write_text(json.dumps(groups, indent=2))
    candidate_counts = {phase:{s:len(v) for s,v in rows.items()} for phase,rows in groups.items()}
    incomplete_subjects = sorted({row['subject'] for row in failures if 'subject' in row})
    result = {'config': cfg, 'subjects': subjects, 'record_count': len(records), 'window_count':len(windows),
        'phase_windows':{p:sum(candidate_counts[p].values()) for p in groups}, 'candidate_counts': candidate_counts,
        'failures':failures, 'incomplete_subjects':incomplete_subjects,
        'original_subjects':original_subjects,'excluded_subjects':excluded_subjects,
        'cohort_policy':cfg.get('cohort_policy', {'mode':'strict_complete'}),
        'selected_subjects_complete':True,'selected_input_failures':[],
        'records':records,'exact_story_phase_overlap':False,'duplicate_candidate_identity':False,
        'strict_published_cohort_match': not failures and len(subjects)==85,
        'actual_protocol': 'user-authorized available complete-subject subset; author split/valid candidates retained; original100Hz/M',
        'test_labels_used_for_training_or_selection': False, 'no_source_family_resplit': True}
    (out / 'manifest_frozen.json').write_text(json.dumps(result, indent=2))
    (out / 'completed.json').write_text(json.dumps({'engineering_preparation_finished':True,
        'ready_for_training':True,'windows':len(windows),'source_failures_excluded':len(failures)}, indent=2))
    print(json.dumps({'phase_windows':result['phase_windows'], 'failures':len(failures), 'incomplete_subjects':incomplete_subjects}), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    main(parser.parse_args().config)
