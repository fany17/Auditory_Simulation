"""Summarize existing100Hz/M preparation; never train or request payloads."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def main(base):
    split=json.loads((base/'hdps_split.json').read_text())
    mel={r['story']:r for r in json.loads((base/'mel_paper01/manifest.json').read_text())}
    idx=json.loads((base/'valid_segments_metadata.json').read_text())
    outcomes_path=base/'raw_batch01/outcomes.json'
    outcomes=json.loads(outcomes_path.read_text()) if outcomes_path.exists() else []
    repairs=[]
    repair=base/'preprocessing100_author02_sub-022/output_check.json'
    if repair.exists():
        repairs.append({'subject':'sub-022','output':str(repair.parent),'repair':'public events container, no APR/SNR in EEG numerical path'})
    raw_rows=[]
    for result in outcomes+repairs:
        summary=Path(result['output'])/'output_check.json'
        if not summary.exists():
            continue
        for record in json.loads(summary.read_text()):
            p=Path(record['path'])
            story=p.name.split('_desc-preproc-audio-')[1].split('_eeg.npy')[0]
            phases=[phase for phase,stories in split[result['subject']].items() if story in stories]
            eeg_frames=record['shape'][1]//300
            target_frames=mel[story]['shape'][0]//300 if story in mel else 0
            valid=idx.get(story+'.wav',[])
            # Public data.py/get_feat_SparrKUL explicitly removes indices beyond
            # feature frame count before EEG indexing. This is original tail
            # handling, not an extra cohort/window selection based on results.
            effective=[i for i in valid if i<target_frames]
            raw_rows.append(dict(subject=result['subject'],story=story,phase=';'.join(phases) if phases else 'NO_EXACT_STORY_MATCH',
                path=record['path'],channels=record['shape'][0],samples=record['shape'][1],sampling_rate=100,finite=record['finite'],
                author_valid_segment_n=len(valid),effective_valid_segment_n=len(effective),
                author_feature_tail_filter_excluded_indices=';'.join(str(i) for i in valid if i>=target_frames),
                unsupported_EEG_indices=sum(i>=eeg_frames for i in effective),
                unsupported_M_indices=0,no_64Hz_upsampling=True))
    with (base/'raw100_preparation_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(raw_rows[0]));writer.writeheader();writer.writerows(raw_rows)
    coverage=[]
    for subject,phases in split.items():
        for phase,stories in phases.items():
            for story in stories:
                matching=[r for r in raw_rows if r['subject']==subject and r['story']==story]
                coverage.append(dict(subject=subject,phase=phase,story=story,
                    status='ORIGINAL100Hz_M_ENGINEERING_READY' if matching and all(r['unsupported_EEG_indices']==r['unsupported_M_indices']==0 and r['finite'] for r in matching) else 'MISSING_OR_UNSUPPORTED',
                    processed_records=len(matching),valid_segment_n=matching[0]['effective_valid_segment_n'] if matching else 0))
    with (base/'author_split_input_coverage.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
    result={'raw_batch_completed':(base/'raw_batch01/completed.json').exists(),
        'subject_jobs_finished':len(outcomes),'subject_jobs_expected':len(split),
        'processed_100Hz_records':len(raw_rows),'subjects_with_some100Hz':len({r['subject'] for r in raw_rows}),
        'failed_subject_processes':[r['subject'] for r in outcomes if r['exit_code']!=0],
        'repairs':repairs,
        'record_processing_failures':[f for r in outcomes for f in r['failed_records'] if isinstance(f,dict)],
        'author_pairs':len(coverage),'author_pair_status':dict(Counter(r['status'] for r in coverage)),
        'missing_or_unsupported_by_phase':dict(Counter(r['phase'] for r in coverage if r['status']!='ORIGINAL100Hz_M_ENGINEERING_READY')),
        'missing_or_unsupported_pairs':[r for r in coverage if r['status']!='ORIGINAL100Hz_M_ENGINEERING_READY'],
        'mel_stories':len(mel),'mel_failures':json.loads((base/'mel_paper01/failures.json').read_text()),
        'scientific_training':'NOT_RUN','published_test_evaluation':'NOT_RUN',
        'partial_cohort_substitute':'NOT_RUN_USER_DOES_NOT_WANT_SUBSTITUTE',
        'performance_metrics':'NOT_ESTIMABLE','paper_consistency':'NOT_TESTED_SAME_CONDITIONS'}
    (base/'continuation_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('missing_or_unsupported_pairs','record_processing_failures')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    main(parser.parse_args().base)
