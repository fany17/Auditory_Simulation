"""Map existing public sidecar metadata to restricted BDF entries; no requests."""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path


def main(base):
    root = Path('/home/fanyu/auditory_simulation_m6a/pub_01/s06/sparrkulee_v3.1/raw')
    split = json.loads((base/'hdps_split.json').read_text())
    with (base/'source_public_inventory.csv').open(encoding='utf-8-sig',newline='') as stream:
        inventory = list(csv.DictReader(stream))
    rows=[]
    for source in inventory:
        rel=source['relative_path']
        subject=rel.split('/')[0]
        if subject not in split or source['restricted']!='True' or '_task-listeningActive_' not in rel or not rel.endswith('.bdf.gz'):
            continue
        events = root / rel.replace('_eeg.bdf.gz','_events.tsv')
        with events.open(newline='') as stream:
            events_rows=list(csv.DictReader(stream,delimiter='\t'))
        for event in events_rows:
            story=Path(event['stim_file']).name.split('.')[0]
            phases=[phase for phase,stories in split[subject].items() if story in stories]
            rows.append({'subject':subject,'file_id':source['file_id'],'relative_path':rel,'restricted':True,
                'bytes':source['bytes'],'source_url':source['source_url'],'source_inventory':'reports/pub_01/s01/sparrkulee_official_inventory.csv',
                'events_path':str(events),'story':story,'author_phase':';'.join(phases) if phases else 'NO_EXACT_STORY_MATCH',
                'access_requested':False,'payload_read':False,'raw100Hz_replacement':'NONE'})
    with (base/'restricted_raw_mapping.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'restricted_raw_files':len({r['relative_path'] for r in rows}),
        'subjects':dict(Counter(r['subject'] for r in rows)),
        'exact_author_phase_rows':dict(Counter(r['author_phase'] for r in rows)),
        'missing_author_pairs':{s:{phase:[story for story in stories if not any(r['subject']==s and r['story']==story for r in rows)]
            for phase,stories in split[s].items()} for s in sorted({r['subject'] for r in rows})},
        'all_affected_subjects_have_no_public_listening_BDF':all(not list((root/s).glob('*/eeg/*listeningActive*.bdf*')) for s in {r['subject'] for r in rows}),
        'no_data_download':True,'no_restricted_requests':True}
    (base/'restricted_raw_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    main(parser.parse_args().base)
