"""Correct a legacy metadata label without rewriting the frozen data split."""
from collections import Counter
import json
from pub_01_s03_ss05_data import ROOT,PUBLIC,dump
from pub_01_s03_ss05_train import digest
prior=json.loads((PUBLIC/'pub_01/s03_architecture_20261019/cache01/manifest.json').read_text())
current=json.loads((ROOT/'split_frozen.json').read_text());protocol=json.loads((ROOT/'FORMAL_PROTOCOL.json').read_text())
counts=dict(Counter(r['phase'] for r in prior['records']));used={r['record'] for r in current['records']}
assert counts=={'train':426,'val':49,'test_seen':69,'test_heldout':75}
assert used=={r['record'] for r in prior['records'] if r['phase']=='train'}
assert digest(ROOT/'split_frozen.json')==protocol['input_contract_sha256']['split']
dump(ROOT/'upstream_phase_audit.json',{'upstream_actual_phase_counts':counts,'SS05_used_oldtrain_records':len(used),
    'excluded_old_validation_records':49,'excluded_old_test_records':144,'excluded_all_nontrain_records':193,
    'legacy_split_field_note':'excluded_old_SS03_test_records=193 is a field-label misnomer; actual count is all nontrain, not test alone',
    'frozen_split_bytes_unchanged':True,'record_membership_unchanged':True,'scientific_metrics_changed':False})
print(json.dumps(counts))
