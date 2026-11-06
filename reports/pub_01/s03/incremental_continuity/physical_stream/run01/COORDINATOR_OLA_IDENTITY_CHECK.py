"""Synthetic OLA/frame identities only; run on server2203, no EEG/model/data files."""
import json, os, time, hashlib, pathlib
import numpy as np
start=time.time(); cpu=time.process_time()
L=192; hop=96; nw=5; T=(nw-1)*hop+L
w=np.hanning(L); weights=np.zeros(T)
for i in range(nw): weights[i*hop:i*hop+L]+=w
valid=weights>0
alpha=np.stack([np.divide(w,weights[i*hop:i*hop+L],out=np.zeros(L),where=weights[i*hop:i*hop+L]>0) for i in range(nw)])
def analysis(y):
    return np.stack([y[i*hop:i*hop+L] for i in range(nw)])
def analysis_adjoint(z):
    y=np.zeros(T)
    for i in range(nw): y[i*hop:i*hop+L]+=z[i]
    return y
def synthesis(z):
    return analysis_adjoint(alpha*z)
def synthesis_adjoint(y):
    return alpha*analysis(y)
rng=np.random.default_rng(20261106)
z=rng.standard_normal((nw,L)); target=rng.standard_normal(T)
project=lambda a:analysis(synthesis(a))
y=synthesis(z); p=project(z); null=z-p
g=synthesis_adjoint(2*(y-target)*valid/valid.sum())
gp=synthesis_adjoint(analysis_adjoint(synthesis_adjoint(2*(synthesis(p)-target)*valid/valid.sum())))
loss=lambda a:float(np.mean((synthesis(a)[valid]-target[valid])**2))
checks={
"AS_identity_valid":float(np.max(np.abs(synthesis(analysis(target))[valid]-target[valid]))),
"AP_equals_A":float(np.max(np.abs(synthesis(p)-y))),
"P_idempotent":float(np.max(np.abs(project(p)-p))),
"nonzero_removed_window_component_max":float(np.max(np.abs(null))),
"removed_component_physical_max":float(np.max(np.abs(synthesis(null)))),
"physical_loss_unchanged":abs(loss(z)-loss(p)),
"physical_loss_gradient_unchanged":float(np.max(np.abs(g-gp))),
"valid_physical_points":int(valid.sum()),
"window_coordinates":int(z.size)
}
assert checks["nonzero_removed_window_component_max"]>0.1
assert max(v for k,v in checks.items() if k not in ("nonzero_removed_window_component_max","valid_physical_points","window_coordinates"))<1e-12
out={"passed":True,"scope":"Synthetic independent window coordinates, n_windows=5, np.hanning(192), hop=96. No statement about rank/conditioning of learned shared-parameter D9 or D18 models.","checks":checks,"conclusion":"Postcomposing window outputs with frame(OLA(.)) and using the same final OLA physical loss leaves prediction, loss and gradient unchanged; hidden window redundancy remains. A genuinely unique-time output parameterization is a different model class; it does not prove better generalization or remove all possible parameter nonidentifiability.","real_eeg_loaded":False,"neural_forward":0,"backward":0,"optimizer_updates":0,"synthetic_gradient_analytic_check":True,"pid":os.getpid(),"source_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),"cpu_seconds":time.process_time()-cpu,"wall_seconds":time.time()-start}
pathlib.Path(__file__).with_suffix(".json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out))

