"""D26 exact-scope multi-GPU scheduler. Failed attempts are never restarted automatically."""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pub_01_s03_ss07_contrastive_history_data as data

ROOT=data.ROOT
FORMAL=ROOT/"formal"
GATE=ROOT/"PROTOCOL_V05.json"
APPROVAL=ROOT/"COORDINATOR_EXECUTION_REVIEW.json"
GPU=("GPU-be77d5f7-abf2-5e41-2a1c-9510e4a832d9",
     "GPU-cd053687-ccd7-87d3-3f4c-a3beffadfcba")


def identity():
    fields=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()
    return dict(pid=os.getpid(),uid=os.getuid(),startticks=int(fields[19]))


def alive(process):
    path=Path(f"/proc/{process['pid']}")
    try:
        fields=(path/"stat").read_text().rsplit(")",1)[1].split()
        return (path.stat().st_uid==process["uid"] and int(fields[19])==process["startticks"]
                and fields[0] not in ("Z","X"))
    except FileNotFoundError:
        return False


def release(gpu):
    deadline=time.monotonic()+90
    while True:
        value=subprocess.check_output(
            ["nvidia-smi","-i",gpu,"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip()
        if value.isdecimal() and int(value)<=256:
            return
        data.check(time.monotonic()<deadline,f"GPU admission busy: {gpu}")
        time.sleep(2)


def launch(attempt,stage,seed,role,slot,resume):
    gpu=GPU[slot]
    release(gpu)
    worker=Path(__file__).with_name(f"pub_01_s03_ss07_contrastive_history_{'readout' if stage=='readout' else 'train'}.py")
    argv=[sys.executable,"-B",str(worker)]
    if stage=="readout":
        argv+=["--seed",str(seed),"--arm",role]
    else:
        argv+=[stage,"--seed",str(seed),"--role",role]
        if resume:
            argv+=["--resume"]
    name=f"{seed}_{stage}_{role}"
    path=attempt/f"{name}.log"
    env=os.environ|{"CUDA_VISIBLE_DEVICES":gpu,"PYTHONDONTWRITEBYTECODE":"1",
                   "OPENBLAS_NUM_THREADS":"8","OMP_NUM_THREADS":"4"}
    started=time.monotonic()
    with path.open("x") as stream:
        child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env)
    process=Path(f"/proc/{child.pid}")
    ticks=int((process/"stat").read_text().rsplit(")",1)[1].split()[19])
    data.save(attempt/f"{name}_PROCESS.json",dict(pid=child.pid,uid=process.stat().st_uid,startticks=ticks,
                                                 gpu=gpu,seed=seed,stage=stage,role=role,argv=argv,
                                                 launched_monotonic_seconds=started))
    return child,started


def stage_done(seed_root,stage,role):
    names={"cl":[f"CL_{role}"],"head":["HEAD_HH"] if role=="H" else ["HEAD_CH","HEAD_CC"],
           "ft":[f"FT_{role}"],"readout":[f"READOUT_{role}"]}[stage]
    return all((seed_root/name/"COMPLETED.json").is_file() for name in names)


def schedule(attempt,seed_root,stage,seed,roles,resume):
    remaining=[role for role in roles if not stage_done(seed_root,stage,role)]
    live={}
    failure=None
    while remaining or live:
        for slot in range(2):
            if remaining and slot not in live and failure is None:
                role=remaining.pop(0)
                try:
                    worker,started=launch(attempt,stage,seed,role,slot,resume)
                    live[slot]=(role,worker,started)
                except BaseException as error:
                    failure=f"Launch {seed}/{stage}/{role}: {error}"
                    if stage in ("cl","head") and not (seed_root/"ABORT.json").exists():
                        data.save(seed_root/"ABORT.json",dict(failure=failure,process=identity(),stage=stage))
        if failure is not None and not live:
            break
        for slot,(role,worker,started) in list(live.items()):
            code=worker.poll()
            if code is None:
                continue
            del live[slot]
            done=stage_done(seed_root,stage,role)
            data.save(attempt/f"{seed}_{stage}_{role}_EXIT.json",
                      dict(pid=worker.pid,returncode=code,terminal_receipt=done,
                           worker_wall_seconds_observed=time.monotonic()-started,
                           timing_scope="Spawn through observed exit; includes loader and init. Polling interval 2 s."))
            if code or not done:
                failure=f"Actual failure {seed}/{stage}/{role} ({code})"
                if stage in ("cl","head") and not (seed_root/"ABORT.json").exists():
                    data.save(seed_root/"ABORT.json",dict(failure=failure,process=identity(),stage=stage))
        if live:
            time.sleep(2)
    if failure is not None:
        raise RuntimeError(failure)


def main(resume):
    p=data.read(GATE)
    r=data.read(APPROVAL)
    data.check(r["passed"] and r["protocol_sha256"]==data.digest(GATE),"Independent actual source/engineering review")
    data.check(all(data.digest(path)==h for path,h in p["sources"].items()),"Frozen source drift")
    if resume:
        data.check(FORMAL.is_dir() and not (FORMAL/"COMPLETED.json").exists(),"Only an incomplete formal run")
        previous=sorted(FORMAL.glob("ATTEMPT_*"))
        data.check(previous and not alive(data.read(previous[-1]/"PROCESS.json")),"Previous exact queue still live")
        for receipt in previous[-1].glob("*_PROCESS.json"):
            data.check(not alive(data.read(receipt)),"Prior worker remains alive")
        index=len(previous)
    else:
        FORMAL.mkdir()
        index=0
    attempt=FORMAL/f"ATTEMPT_{index:03}"
    attempt.mkdir()
    data.save(attempt/"PROCESS.json",identity())
    try:
        for seed in data.SEEDS:
            seed_root=FORMAL/f"seed{seed}"
            seed_root.mkdir(exist_ok=resume)
            abort=seed_root/"ABORT.json"
            if resume and abort.exists():
                abort.rename(attempt/f"{seed}_PRIOR_ABORT.json")
            for stage,roles in (("cl",("H","C")),("head",("H","C")),("ft",data.ARMS),("readout",data.ARMS)):
                schedule(attempt,seed_root,stage,seed,roles,resume)
        rows=[]
        for seed in data.SEEDS:
            seed_root=FORMAL/f"seed{seed}"
            for arm in data.ARMS:
                rows.append(data.read(seed_root/f"READOUT_{arm}"/"COMPLETED.json"))
        means={arm:sum(r["VAL_r"]["Ccap"] for r in rows if r["arm"]==arm)/3 for arm in ("CC","CH","HH")}
        winner=max(means,key=means.get)
        data.save(FORMAL/"COMPLETED.json",dict(passed=True,means=means,selected_full_arm=winner,
                  evaluations=rows,process=identity(),protocol_sha256=data.digest(GATE),old_test_reads=0))
    except BaseException as error:
        data.save(attempt/"FAILURE.json",dict(message=str(error),process=identity(),traceback=traceback.format_exc()))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",action="store_true")
    main(parser.parse_args().resume)
