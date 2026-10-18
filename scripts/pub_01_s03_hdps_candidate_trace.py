"""Trace public-index and length-crop candidate counts; never change a test."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main(config):
    cfg=json.loads(config.read_text());base=Path(cfg['work']);prepared=Path(cfg['prepared_dir']);run=Path(cfg['run_dir'])
    split=json.loads((base/'hdps_split.json').read_text());index=json.loads((base/'valid_segments_metadata.json').read_text())
    manifest=json.loads((prepared/'manifest_frozen.json').read_text());selected=set(manifest['subjects'])
    rows=[]
    for subject,phases in split.items():
        for phase,stories in phases.items():
            public_count=sum(len(index[s+'.wav']) for s in stories)
            records=[r for r in manifest['records'] if r['subject']==subject and r['phase']==phase]
            actual=sum(r['segment_count'] for r in records) if subject in selected else None
            rows.append({'subject':subject,'phase':phase,'selected':subject in selected,'stories':';'.join(stories),
                'public_index_candidate_count_before_length_crop':public_count,
                'actual_after_M_length_crop':actual,'removed_by_original_length_rule':public_count-actual if actual is not None else None,
                'paper_described_range':'104-208','public_uncropped_outside_paper_range':not 104<=public_count<=208,
                'candidate_selection_based_on_model_output':False})
    def bounds(items):return {'min':min(items),'median':float(np.median(items)),'max':max(items)}
    summary={'public_author_commit':cfg['author_commit'],'index_file':'valid_seg_dict_6.pkl converted to pure metadata',
        'all81_public_index_test_uncropped':bounds([r['public_index_candidate_count_before_length_crop'] for r in rows if r['phase']=='test']),
        'selected76_public_index_test_uncropped':bounds([r['public_index_candidate_count_before_length_crop'] for r in rows if r['phase']=='test' and r['selected']]),
        'selected76_actual_test':bounds([r['actual_after_M_length_crop'] for r in rows if r['phase']=='test' and r['selected']]),
        'test_outliers_in_fixed_public_index':[r for r in rows if r['phase']=='test' and r['public_uncropped_outside_paper_range']],
        'interpretation':'public split/index already differ from paper described104-208; existing feature-length tail rule further reduces some counts; neither test scores nor candidate difficulty used to choose counts',
        'remaining_unknown':'paper exact valid-index revision and min length across M/W/G cannot be verified from current M-only outputs; do not guess a retrospective correction',
        'test_results_changed':False}
    with (run/'candidate_trace.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (run/'candidate_trace.json').open('x') as stream:json.dump(summary,stream,indent=2)
    print(json.dumps({k:v for k,v in summary.items() if k!='test_outliers_in_fixed_public_index'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);main(parser.parse_args().config)
