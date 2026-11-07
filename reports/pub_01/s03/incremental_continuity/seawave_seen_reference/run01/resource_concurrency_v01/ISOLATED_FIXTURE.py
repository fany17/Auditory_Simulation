"""Only fixture-created processes are signaled. No signals, payloads, or model imports from D27."""
import errno
import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import d27_resource_queue as queue
from d27_resource_process import GPU, Process, identity, key, until

CHILD = """import os,signal,sys
os.read(int(sys.argv[1]),1)
if sys.argv[2]=='term':
    os.kill(os.getpid(),signal.SIGTERM)
else:
    os._exit(int(sys.argv[2]))
"""
PARENT = """import json,subprocess,sys
fd=int(sys.argv[1])
kids=[subprocess.Popen([sys.executable,'-B','-c',sys.argv[2],str(fd),mode],pass_fds=(fd,)) for mode in ('0','23','term')]
print(json.dumps([c.pid for c in kids]),flush=True)
print(json.dumps([c.wait() for c in kids]),flush=True)
"""
OBSERVER = """import errno,json,os,sys
from d27_resource_process import Process
p=Process(tuple(map(int,sys.argv[1:])))
try:
    os.waitpid(p.expected[0],os.WNOHANG)
except ChildProcessError as exc:
    assert exc.errno==errno.ECHILD
else:
    raise AssertionError('Observer cannot wait for non-child')
print(json.dumps(p.stat()))
p.close()
"""


def procfd_fixture():
    rd, wr = os.pipe()
    parent = subprocess.Popen([sys.executable,"-B","-c",PARENT,str(rd),CHILD], pass_fds=(rd,),
                              stdout=subprocess.PIPE, text=True)
    os.close(rd)
    handle = None
    stopped = False
    try:
        children = [identity(pid) for pid in json.loads(parent.stdout.readline())]
        assert all(r["ppid"] == parent.pid and r["readable_startcode"] for r in children)
        handle = Process(key(identity(parent.pid)))
        handle.signal(signal.SIGSTOP)
        stopped = True
        until(handle.stopped, 5, "Fixture parent did not stop")
        assert all(identity(r["pid"])["state"] != "T" for r in children)
        assert os.write(wr,b"xxx") == 3
        until(lambda: all(identity(r["pid"])["state"] == "Z" for r in children),5,"Fixture children did not terminate")
        observed = [json.loads(subprocess.check_output([sys.executable,"-B","-c",OBSERVER,*map(str,key(r))],
                    text=True)) for r in children]
        assert handle.stopped()
        for row in children:
            probe = Process(key(row))
            try:
                assert probe.quiescent_zombie()
            finally:
                probe.close()
        assert all(key(a)==key(b) and b["state"]=="Z" and b["ppid"]==parent.pid for a,b in zip(children,observed))
        raw = [r["wait_status_field52"] for r in observed]
        status = [os.waitstatus_to_exitcode(s) for s in raw]
        handle.signal(signal.SIGCONT)
        stopped = False
        data,_ = parent.communicate(timeout=5)
        truth = json.loads(data.strip())
        assert parent.returncode == 0 and truth == status == [0,23,-15]
        try:
            signal.pidfd_send_signal(handle.fd,0)
        except ProcessLookupError as exc:
            assert exc.errno == errno.ESRCH
        else:
            raise AssertionError("Stale proc directory descriptor still signalable")
        return dict(original_Popen=truth, nonancestor_decoded=status, proc_wait_status=raw, stale_fd_ESRCH=True)
    finally:
        os.close(wr)
        if stopped:
            handle.signal(signal.SIGCONT)
        if parent.poll() is None:
            parent.wait(timeout=5)
        if handle is not None:
            handle.close()


class MemoryPath:
    def __init__(self, registry, name):
        self.registry = registry
        self.name = name

    def __truediv__(self, name):
        return MemoryPath(self.registry, f"{self.name}/{name}")

    def exists(self):
        return self.name in self.registry

    def mkdir(self):
        assert not self.exists()
        self.registry[self.name] = "directory"


def error(message):
    raise RuntimeError(message)


def migration_logic(after_commit):
    files, calls = {}, []
    p = MemoryPath(files,"resource")
    f = MemoryPath(files,"formal")
    class OldQueue:
        def __init__(self, expected):
            self.expected = expected
            self.paused = False

        def stopped(self):
            return self.paused

        def signal(self, sig):
            assert self.expected == queue.ORIGINAL
            calls.append(sig)
            self.paused = sig == signal.SIGSTOP

        def close(self):
            pass

    def publish(path,value):
        assert not path.exists()
        files[path.name] = value

    original = {}
    def patch(**values):
        for k,v in values.items():
            original[k] = getattr(queue,k)
            setattr(queue,k,v)
    fixture_identity = identity(os.getpid())
    patch(RESOURCE=p, ATTEMPT=f/"ATTEMPT_001", SUPERVISORS=f/"ATTEMPT_001/supervisors", Process=OldQueue,
          confirmed_old=lambda parent,*a,stopped: parent.stopped()==stopped or error("wrong stop state"),
          verify_resource=lambda **kw:"fixture", child_environment=lambda _: {},
          inventory_jobs=lambda: [queue.PILOT], assignment_state=lambda *a:([],{}),
          observed_devices=lambda: ({g:(8034,46068) for g in GPU},{g:set() for g in GPU}), choose_gpu=lambda *a:GPU[0],
          identity=lambda pid: dict(fixture_identity,pid=pid), live=lambda _: True,
          SYSTEM_PY=sys.executable, publish=publish,
          prepare_worker=(lambda *a: (SimpleNamespace(pid=os.getpid()),123)) if after_commit else
                         (lambda *a:error("before release failure")),
          release_worker=lambda fd:calls.append(("release",fd)),
          start_reporter=lambda *a:error("after release failure"))
    try:
        try:
            queue.migration(False)
        except RuntimeError as exc:
            assert str(exc) == ("after release failure" if after_commit else "before release failure")
        else:
            raise AssertionError("Injected failure was hidden")
        assert files["resource/SCHEDULER_FAILURE.json"]["committed"] == after_commit
        assert calls == ([signal.SIGSTOP,("release",123)] if after_commit else [signal.SIGSTOP,signal.SIGCONT])
        assert ("resource/ROLLBACK.json" in files) != after_commit
    finally:
        for k,v in original.items():
            setattr(queue,k,v)


def capacity():
    g0,g1=GPU
    devices = ({g0:(8034,46068),g1:(8034,46068)},{g0:{1},g1:{2}})
    assert queue.choose_gpu([g0,g1],{g0:{1},g1:{2}},devices)==g0
    assert queue.choose_gpu([g0,g0,g1],{g0:{1},g1:{2}},devices)==g1
    assert queue.choose_gpu([g0,g0,g1,g1],{g0:{1},g1:{2}},devices) is None
    assert queue.choose_gpu([g0,g1],{g0:set(),g1:{2}},devices)==g1
    narrow = ({g0:(40000,46068),g1:(40000,46068)},devices[1])
    assert queue.choose_gpu([g0,g1],{g0:{1},g1:{2}},narrow) is None


if __name__ == "__main__":
    kernel = procfd_fixture()
    capacity()
    migration_logic(after_commit=False)
    migration_logic(after_commit=True)
    print(json.dumps(dict(passed=True, kernel=kernel, per_GPU_cap_and_foreign_process_rejection=True,
        clean_rollback_before_release=True, no_old_parent_resume_after_release=True,
        actual_formal_signals=0, actual_formal_workers=0, signal_payload_reads=0, model_forwards=0),indent=2))
