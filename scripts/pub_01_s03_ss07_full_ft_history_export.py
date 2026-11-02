"""Export only whitelisted lightweight metadata/figures; never checkpoints/cache."""
import json,shutil
from pathlib import Path
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_full_ft_history_20261102')
assert (ROOT/'QUEUE_COMPLETED.json').exists() and (ROOT/'report01/PRESENTATION_COMPLETED.json').exists()
out=ROOT/'lightweight_delivery';assert not out.exists()
paths=[ROOT/n for n in ['PROTOCOL.json','ENGINEERING_CHECK.json','QUEUE_COMPLETED.json','QUEUE_LAUNCH.json',
    'WORKER_LAUNCHES.jsonl','COORDINATOR_TARGET_MAPPING.json','COORDINATOR_REVIEW_SOURCE_CHECK.json',
    'RESULTS_PRESENTATION_PLAN.json','PRESENTATION_METADATA_AMENDMENT.json','PLAN.md','COORDINATOR_DECISION.md'] if (ROOT/n).exists()]
for folder in (ROOT/'fits').iterdir():
    if folder.is_dir():
        paths.extend(folder/n for n in ['COMPLETED.json','INCUMBENT.json','SELECTED_METRICS.json','OLDER_METRICS.json','HISTORY.jsonl','STARTED.json'] if (folder/n).exists())
paths.extend(p for p in (ROOT/'report01').rglob('*') if p.is_file())
assert all(p.suffix.lower() in ['.json','.jsonl','.csv','.png','.md'] for p in paths)
size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())+sum(p.stat().st_size for p in paths)
assert size<128*1024**2,'Export duplication stays within new-run128MiB cap'
out.mkdir()
for p in paths:
    dest=out/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
with (out/'LIGHTWEIGHT_EXPORT.json').open('x') as f:json.dump(dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths),
    checkpoints_or_signal_arrays=False,paths=[str(p.relative_to(ROOT)) for p in paths]),f,indent=2)
print('WHITELISTED_LIGHTWEIGHT_DELIVERY_READY',flush=True)
