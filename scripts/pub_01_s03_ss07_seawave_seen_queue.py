"""D27 GPU queue, admitted only after the D26 training workers actually exit."""
import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pub_01_s03_ss07_seawave_seen_data as data

GATE=data.ROOT/"PROTOCOL.json"
REVIEW=data.ROOT/"COORDINATOR_EXECUTION_REVIEW.json"
FORMAL=data.ROOT/"formal"
GPU=("GPU-be77d5f7-abf2-5e41-2a1c-9510e4a832d9","GPU-cd053687-ccd7-87d3-3f4c-a3beffadfcba")


def identity(pid):
    path=Path(f"/proc/{pid}")
    fields=(path/"stat").read_text().rsplit(")",1)[1].split()
    return dict(pid=pid,uid=path.stat().st_uid,startticks=int(fields[19]),status=fields[0])


def live(receipt):
    try:
        actual=identity(receipt["pid"])
        return all(actual[k]==receipt[k] for k in ("pid","uid","startticks")) and actual["status"] not in ("Z","X")
    except FileNotFoundError:
        return False


def gpu_ready(uuid):
    expires=time.monotonic()+90
    while True:
        count=subprocess.check_output(
            ["nvidia-smi","-i",uuid,"--query-gpu=memory.used","--format=csv,noheader,nounits"],
            text=True).strip()
        if count.isdecimal() and int(count)<=256:
            return
        data.check(time.monotonic()<expires,f"GPU busy: {uuid}")
        time.sleep(2)


def admit_after_d26():
    terminal=data.D26/"formal/COMPLETED.json"
    value=data.read(terminal)
    data.check(value["passed"] and len(value["evaluations"])==9,"D26 must finish its complete matrix")
    identities=[]
    for attempt in sorted((data.D26/"formal").glob("ATTEMPT_*")):
        for path in (attempt/"PROCESS.json",*sorted(attempt.glob("*_PROCESS.json"))):
            record=data.read(path)
            data.check(not live(record),"D26 original process remains alive")
            identities.append(record["pid"])
    return dict(D26_terminal=str(terminal),D26_terminal_sha256=data.digest(terminal),
                D26_checked_exited_pids=identities)


def name(item):
    stage,seed,person=item
    return f"{stage}_{seed}"+(f"_{person}" if person else "")


def folder(item):
    stage,seed,person=item
    root=FORMAL/f"seed{seed}"
    return root/"initial" if stage=="initial" else root/"personal"/person


def launch(attempt,item,slot,resume):
    stage,seed,person=item
    gpu=GPU[slot]
    gpu_ready(gpu)
    path=Path(__file__).with_name("pub_01_s03_ss07_seawave_seen_train.py")
    command=[sys.executable,"-B",str(path),stage,"--seed",str(seed)]
    if person:
        command+=["--subject",person]
    if resume:
        command+=["--resume"]
    env=os.environ|{"CUDA_VISIBLE_DEVICES":gpu,"PYTHONDONTWRITEBYTECODE":"1",
                    "OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4"}
    started=time.monotonic()
    with (attempt/(name(item)+".log")).open("x") as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    process=identity(child.pid)
    data.write(attempt/(name(item)+"_PROCESS.json"),
               dict(**process,argv=command,gpu=gpu,started_monotonic=started))
    return child,started


def schedule(attempt,jobs,resume):
    jobs=[job for job in jobs if not (folder(job)/"COMPLETED.json").exists()]
    active={}
    error=None
    while jobs or active:
        for slot in range(2):
            if jobs and slot not in active and error is None:
                item=jobs.pop(0)
                try:
                    child,started=launch(attempt,item,slot,resume)
                    active[slot]=item,child,started
                except BaseException as ex:
                    error=f"Launch {name(item)}: {ex}"
        if error and not active:
            break
        for slot,(item,child,started) in list(active.items()):
            code=child.poll()
            if code is None:
                continue
            del active[slot]
            complete=(folder(item)/"COMPLETED.json").is_file()
            data.write(attempt/(name(item)+"_EXIT.json"),dict(pid=child.pid,returncode=code,
                       complete=complete,worker_wall_seconds_observed=time.monotonic()-started))
            if code or not complete:
                error=f"Failed {name(item)} returncode={code}"
        if active:
            time.sleep(2)
    if error:
        raise RuntimeError(error)


def finish():
    import pub_01_s03_ss07_seawave_seen_model as source
    inv=data.read(data.ROOT/"INVENTORY.json")
    rows=[]
    for seed in data.SEEDS:
        base=FORMAL/f"seed{seed}"
        group=data.read(base/"initial/SELECTED_VAL.json")
        initial=data.read(base/"initial/COMPLETED.json")
        check=[]
        blocks=[]
        for subject in inv["subjects"]:
            path=base/"personal"/subject
            receipt=data.read(path/"COMPLETED.json")
            data.check(receipt["initial_sha256"]==initial["selected_sha256"],"One true initial source per seed")
            check.append(receipt)
            blocks.extend(data.read(path/"SELECTED_VAL.json")["blocks"])
        adapted=source.aggregate(blocks)
        data.check(len(adapted["people"])==49 and len(adapted["blocks"])==752,"Complete shared protocol")
        families=set(inv["train_source_families"])
        adapted["families"]={key:source.aggregate([b for b in blocks if
                 (b["source_family"] in families)==(key=="shared")]) for key in ("shared","unseen")}
        published=base/"PERSONAL_SELECTED_VAL.json"
        if published.exists():
            data.check(data.read(published)==adapted,"Same previously published score")
        else:
            data.write(published,adapted)
        rows.append(dict(seed=seed,initial_no_globalFT_r=group["r"],personal_r=adapted["r"],
                         initial_updates=initial["actual_updates"],personal_updates=sum(r["actual_updates"] for r in check)))
    data.write(FORMAL/"COMPLETED.json",dict(passed=True,results=rows,initial_fits=3,personal_fits=147,
               protocol_sha256=data.digest(GATE),process=identity(os.getpid()),old_test_reads=0))


def run(resume):
    p=data.read(GATE)
    review=data.read(REVIEW)
    data.check(review["passed"] and review["protocol_sha256"]==data.digest(GATE),"Independent actual engineering")
    data.check(all(data.digest(path)==sha for path,sha in p["sources"].items()),"Frozen source")
    exited=admit_after_d26()
    if resume:
        attempts=sorted(FORMAL.glob("ATTEMPT_*"))
        data.check(attempts and not (FORMAL/"COMPLETED.json").exists(),"Incomplete formal run only")
        prev=attempts[-1]
        data.check(not live(data.read(prev/"PROCESS.json")) and
                   all(not live(data.read(path)) for path in prev.glob("*_PROCESS.json")),
                   "Do not duplicate an original live worker")
        next_attempt=len(attempts)
    else:
        FORMAL.mkdir()
        next_attempt=0
    attempt=FORMAL/f"ATTEMPT_{next_attempt:03}"
    attempt.mkdir()
    data.write(attempt/"PROCESS.json",dict(**identity(os.getpid()),**exited))
    subjects=data.read(data.ROOT/"INVENTORY.json")["subjects"]
    try:
        schedule(attempt,[("initial",seed,None) for seed in data.SEEDS],resume)
        schedule(attempt,[("personal",seed,person) for seed in data.SEEDS for person in subjects],resume)
        finish()
    except BaseException as error:
        data.write(attempt/"FAILURE.json",dict(message=str(error),process=identity(os.getpid()),
                                                 traceback=traceback.format_exc()))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",action="store_true")
    run(parser.parse_args().resume)
