"""D9 geometry-only planning check, original record counts; no signal/target."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_physical_objective_20261101')
counts=json.loads((ROOT/'INPUT_INVENTORY.json').read_text())['counts']
N,R,T=counts['fit_windows'],counts['TRAIN_records'],counts['common_physical_target_points']
h=np.hanning(192);alpha=h[96:]/(h[96:]+h[:96]);beta=1-alpha;rows=[]
for j,(a,b) in enumerate(zip(alpha,beta)):
    # Previous active right + next active left, first active left/fixed neighbor,
    # and final record's uncovered-by-neighbor right half. No 60s truncation.
    q=np.array([b,a]);left=np.array([b,0.]);last=np.array([0.,1. if j<95 else 0.])
    F=((N-R)*np.outer(q,q)+R*np.outer(left,left)+R*np.outer(last,last))/T
    W=np.diag([N*b/T,((N-R)*a+(R if j<95 else 0.))/T])
    active=[int(k) for k in (0,1) if not (j==0 and k==0) and not (j==95 and k==1)]
    f=np.linalg.eigvalsh(F[np.ix_(active,active)]);w=np.linalg.eigvalsh(W[np.ix_(active,active)])
    assert f.min()>0 and w.min()>0
    rows.append(dict(output_pair=[j,j+96],free_indices=active,F_bias_block=F.tolist(),W_bias_block=W.tolist(),F_min_eigen=float(f.min()),F_condition=float(f.max()/f.min()),W_condition=float(w.max()/w.min())))
out=dict(scope='GEOMETRY_ONLY_NO_SIGNAL_TARGET_COEFFICIENT_MODEL_READ_OR_FIT',counts=counts,all_free_bias_blocks_positive=True,endpoint_output_columns_fixed=[0,191],
    minimum_F_bias_eigen=min(r['F_min_eigen'] for r in rows),maximum_F_bias_condition=max(r['F_condition'] for r in rows),maximum_W_bias_condition=max(r['W_condition'] for r in rows),blocks=rows)
with (ROOT/'GEOMETRY_PLAN_CHECK.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in out.items() if k!='blocks'},indent=2))
