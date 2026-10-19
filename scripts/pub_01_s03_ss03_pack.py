"""Explicit lightweight copy allowlist and path/size manifest, no signal export."""
import csv
import json
from pathlib import Path
import shutil
root=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_architecture_20261019');out=root/'review01'
assert json.loads((out/'review.json').read_text())['technical']=='PASS'
evidence=out/'evidence';bootstrap=evidence/'bootstrap01';bootstrap.mkdir(parents=True,exist_ok=False)
for name in ('bootstrap_started.json','bootstrap_completed.json','smoke.json','pilot.json','formal_protocol.json','freeze_review.json',
    'constant_train_mean_control.json','all_training_review.json','formal_job_started.json','test_opened.json','formal_job_status.json'):
    shutil.copy2(root/name,bootstrap/name)
shutil.copy2(root/'cache01/manifest.json',bootstrap/'manifest.json')
protocol=json.loads((root/'formal_protocol.json').read_text())
for variant in protocol['variants']:
    for seed in protocol['seeds']:
        run=root/'formal01'/f'{variant}_seed{seed}';target=evidence/'formal_runs'/run.name;target.mkdir(parents=True,exist_ok=False)
        for name in ('history.json','train_completed.json'):shutil.copy2(run/name,target/name)
        shutil.copy2(run/'test01/result.json',target/'result.json')
rows=[]
for p in sorted(root.rglob('*')):
    if not p.is_file():continue
    kind='server_only_array' if p.suffix in ('.npy','.npz') else 'server_only_weights' if p.suffix=='.pt' else 'source_or_structured_evidence'
    rows.append({'host':'server2203','path':str(p),'bytes':p.stat().st_size,'kind':kind,'signal_payload_return':'NO'})
with (out/'SERVER_MANIFEST.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({'manifest_files':len(rows),'returned_signal_payloads':0,'lightweight_run_evidence':12}))
