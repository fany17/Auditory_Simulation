import pathlib,json,numpy as np,time,hashlib
r=pathlib.Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_physical_objective_20261101')
out={'scope':'EXISTING_SIX_COEFFICIENT_ALGEBRA_ONLY_NEWLY_OBSERVED_BIAS_RISK_NO_FIT_FORWARD_OR_NEW_EVALUATION','cases':[],'goal_complete':False}
h=np.hanning(192);a=h[:96]/(h[:96]+h[96:]);b=1-a;nfit=250750;nr=426;T=24112044
for seed in [1024,2026,31415]:
 for role in ['W','F']:
  path=r/f'seed{seed}'/f'{role}_COEFFICIENTS.npz';c=np.load(path);v=c['bias'];B=c['coef']
  interior=a*v[:96]+b*v[96:];left=a*v[:96];tail=v[96:191]
  D=((nfit-nr)*np.sum(a*b*(v[:96]-v[96:])**2)+nr*np.sum(a*b*v[:96]**2))/T
  independent_weighted_terms=np.maximum(abs(a*v[:96]),abs(b*v[96:]))
  rounded=a.astype(np.float32)*v[:96].astype(np.float32)+b.astype(np.float32)*v[96:].astype(np.float32)
  sol=json.loads((r/f'seed{seed}'/f'{role}_SOLVER.json').read_text())
  out['cases'].append(dict(seed=seed,role=role,coefficient_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bias_absolute_max=float(abs(v).max()),bias_l2=float(np.linalg.norm(v)),bias_max_column=int(abs(v).argmax()),bias_at_max=float(v[abs(v).argmax()]),B_absolute_max=float(abs(B).max()),B_frobenius=float(np.linalg.norm(B)),physical_interior_bias_absolute_max=float(abs(interior).max()),physical_interior_bias_rms=float(np.sqrt(np.mean(interior**2))),physical_first_learned_boundary_bias_absolute_max=float(abs(left).max()),physical_last_tail_bias_absolute_max=float(abs(tail).max()),weighted_bias_component_absolute_max=float(independent_weighted_terms.max()),bias_only_TRAIN_disagreement=float(D),total_TRAIN_disagreement=sol['train_objectives']['disagreement'],float32_coefficient_only_interior_fusion_max_abs=float(abs(rounded.astype(float)-interior).max())))
out['interpretation']=['Bias magnitude in virtual windows alone does not prove invalid fused predictions. First learned boundary and record tail must be examined in physical coordinates.','Bias-only disagreement may explain most observed disagreement but feature/cross terms remain; this algebra is not a fitted ablation.','Float32 coefficient-only cancellation check is a small algebraic diagnostic, not a full reduced-precision inference or robustness test.','No new candidates, bias penalty, lambda, inference data or test performance were introduced.']
with (r/'COORDINATOR_BIAS_DIAGNOSTIC.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))

