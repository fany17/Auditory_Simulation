"""D18 metadata-only reversible storage recovery. Never starts processes/models."""
import argparse,hashlib,json,os,shutil,time
from pathlib import Path

HOME=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_preprojection_memory_20261103')
SATA=Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_preprojection_memory_20261103')
PROOF=SATA.parent/'d18_storage_recovery_20261103'
BACKUP=HOME.with_name(HOME.name+'_home_backup_formal_attempt01')
QUEUE='pub_01_s03_ss07_preprojection_queue.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)

def verify_copy():
    assert HOME.resolve()==HOME and not HOME.is_symlink()
    assert SATA.resolve()==SATA and SATA.stat().st_uid==os.getuid()
    assert not BACKUP.exists() and not BACKUP.is_symlink()
    assert not any(p.is_symlink() for p in HOME.rglob('*'))
    before=read(PROOF/'BACKUP_MANIFEST.json')
    for relative,item in before.items():
        for root in (HOME,SATA):
            p=root/relative;s=p.stat()
            assert s.st_size==item['bytes'] and s.st_mtime_ns==item['mtime_ns'] and sha(p)==item['sha256'],str(p)
    for name in ('QUEUE_LAUNCH.json','WORKER_LAUNCHES.jsonl'):
        jobs=[read(HOME/name)] if name.endswith('.json') else [json.loads(x) for x in (HOME/name).read_text().splitlines()]
        for job in jobs:
            proc=Path(f"/proc/{job['pid']}")
            if proc.exists():
                line=(proc/'stat').read_text();rest=line[line.rfind(')')+2:].split()
                if rest[19]==job['start_ticks']:assert rest[0]=='Z','Original process must have exited; never kill here'
    assert shutil.disk_usage(SATA).free>=20*1024**3
    return before

def apply(approval_path):
    plan=read(PROOF/'RECOVERY_PLAN.json');approval=read(approval_path)
    assert approval['approved_for_migration_and_recovery_freeze']
    assert approval['decision_id']==plan['decision_id']
    assert approval['backup_manifest_sha256']==sha(PROOF/'BACKUP_MANIFEST.json')
    assert approval['absolute_deadline_unix']==plan['absolute_deadline_unix']
    assert approval['storage_upper_with_retained_attempt_bytes']==plan['conservative_storage_upper_with_retained_attempt_bytes']
    candidate=PROOF/(QUEUE+'.candidate')
    before=verify_copy();old_protocol=read(HOME/'PROTOCOL.json');old_sha=sha(HOME/'PROTOCOL.json')
    assert old_sha==plan['unmodified_original_protocol_sha256']==approval['old_protocol_sha256']
    original=(HOME/'input'/QUEUE).read_text()
    expected=original.replace("    start=time.time();p=read(ROOT/'PROTOCOL.json');verify(p);guard();checks()", "    attempt_start=time.time();amendment=ROOT/'RESOURCE_RECOVERY_AMENDMENT.json'\n    start=read(amendment)['original_queue_start_unix'] if amendment.exists() else attempt_start\n    p=read(ROOT/'PROTOCOL.json');verify(p);guard();checks()",1).replace("dict(time=start,pid=os.getpid(),uid=os.getuid(),wall_cap_seconds=86400", "dict(time=start,queue_attempt_start_unix=attempt_start,pid=os.getpid(),uid=os.getuid(),wall_cap_seconds=86400",1)
    assert candidate.read_text()==expected and expected!=original,'Only reviewed queue clock diff allowed'
    # Include any newly added coordinator metadata without overwriting evidence.
    additions=[]
    for source in HOME.rglob('*'):
        if not source.is_file():continue
        relative=str(source.relative_to(HOME));target=SATA/relative
        if relative not in before:
            assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            assert sha(source)==sha(target) and source.stat().st_mtime_ns==target.stat().st_mtime_ns
            additions.append(relative)
    HOME.rename(BACKUP)
    try:
        HOME.symlink_to(SATA,target_is_directory=True)
        assert HOME.resolve()==SATA and sha(HOME/'PROTOCOL.json')==old_sha
    except BaseException:
        # Preserve even a failed symlink, rather than unlink/delete it.
        if HOME.is_symlink():HOME.rename(HOME.with_name(HOME.name+'_failed_migration_link'))
        if not HOME.exists():BACKUP.rename(HOME)
        raise
    archive=HOME/'attempts/formal_attempt01';archive.mkdir(exist_ok=False)
    shutil.copytree(HOME/'input',archive/'input',copy_function=shutil.copy2)
    (HOME/'fits').rename(archive/'fits')
    names=['PROTOCOL.json','COORDINATOR_EXECUTION_REVIEW.json','QUEUE_LAUNCH.json','QUEUE_STARTED.json','QUEUE_STATUS.json','QUEUE_FAILURE.json','QUEUE.stdout.log','WORKER_LAUNCHES.jsonl','STARTUP_VERIFIED.json','H_history_seed1024.log','C_current_seed1024.log']
    for name in names:
        source=HOME/name
        assert source.exists(),name
        source.rename(archive/name)
    shutil.copy2(candidate,HOME/'input'/QUEUE)
    assert sha(archive/'input'/QUEUE)==old_protocol['source_pins'][str(HOME/'input'/QUEUE)]
    for source in (HOME/'input').glob('pub_01_s03_ss07_preprojection_*.py'):
        if source.name!=QUEUE:assert sha(source)==sha(archive/'input'/source.name),source.name
    costs=read(PROOF/'INTERRUPTED_COST_BOUNDS.json');write(HOME/'INTERRUPTED_FORMAL_COSTS.json',costs)
    amendment=dict(plan,stage='APPLIED_READY_PENDING_EXECUTION_REVIEW',plan_sha256=sha(PROOF/'RECOVERY_PLAN.json'),migration_approval_sha256=sha(approval_path),queue_candidate_sha256=sha(candidate),original_protocol_archived=str(archive/'PROTOCOL.json'),model_train_data_metrics_runtime_unchanged=True,engineering_weights_still_discarded=True,old_partial_checkpoints_never_loaded_for_restart=True)
    write(HOME/'RESOURCE_RECOVERY_AMENDMENT.json',amendment)
    p=read(archive/'PROTOCOL.json')
    for source in (HOME/'input'/QUEUE,HOME/'RESOURCE_RECOVERY_AMENDMENT.json',HOME/'INTERRUPTED_FORMAL_COSTS.json',archive/'PROTOCOL.json',Path(approval_path)):
        p['source_pins'][str(source)]=sha(source)
    p['stage']='STORAGE_RECOVERY_READY_PENDING_COORDINATOR_EXECUTION_REVIEW'
    p['resources']['physical_writable_ROOT']=str(SATA)
    p['resources']['home_role']='Read-only existing inputs/provenance; no new large outputs. The20GiB guard applies to physicalSATA writerROOT and SATA, not home.'
    p['resources']['original_queue_start_unix']=plan['original_queue_start_unix']
    p['resources']['original_absolute_deadline_unix']=plan['absolute_deadline_unix']
    p['storage_estimated_upper_bytes']=plan['conservative_storage_upper_with_retained_attempt_bytes']
    p['retained_interrupted_fit_attempts']=2;p['formal_fit_attempts_including_recovery']=14
    p['recovery_unchanged_completed_matrix_budget']=True
    write(HOME/'PROTOCOL.json',p)
    receipt=dict(passed=True,time_unix=time.time(),canonical_root=str(HOME),physical_root=str(HOME.resolve()),untouched_original_home_backup=str(BACKUP),preserved_files=len(before),preserved_bytes=sum(x['bytes'] for x in before.values()),all_original_content_and_mtime_verified_before_switch=True,new_coordinator_metadata_copied=additions,original_protocol_sha256=old_sha,new_protocol_sha256=sha(HOME/'PROTOCOL.json'),new_queue_sha256=sha(HOME/'input'/QUEUE),changed_source_files=[QUEUE],physical_writer_free_bytes=shutil.disk_usage(HOME).free,home_free_bytes_for_disclosure=shutil.disk_usage(BACKUP).free,original_deadline_unix=plan['absolute_deadline_unix'],new_NN=0,new_backward=0,process_launches=0,old_two_partial_fits_archived=True,ready_for_execution=False,requires_new_bound_execution_receipt=True)
    write(HOME/'STORAGE_MIGRATION_RECEIPT.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');parser.add_argument('--approval',type=Path);args=parser.parse_args()
    if args.apply:assert args.approval is not None;apply(args.approval)
    else:
        before=verify_copy();print(json.dumps(dict(passed=True,stage='DRY_RUN_ONLY',files=len(before),bytes=sum(x['bytes'] for x in before.values()),new_NN=0,process_launches=0)))
