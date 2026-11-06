"""Metadata-only temporal permissions check; execute only on server2203.

No model, signal array, feature, label, or score is loaded. Integer support
formula uses the strictly positive indices 1..190 of the existing 192 Hann.
This is a permission/geometry result, not a test of predictive information.
"""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_recent_context_review_20261106')
META = Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_physical_stream_20261106/PREPARATION_METADATA.json')
META_SHA = '1068d1eede47893e074c7a67b7a57c1a967364cb974ffdf088a1d623121e24bd'

def support(t, n):
    lo = max(0, -((190-t)//96))  # ceil((t-190)/96)
    hi = min(n-1, (t-1)//96)
    assert lo <= hi
    return (96*lo, 96*hi+192)

def main():
    tic, cpu = time.perf_counter(), time.process_time()
    raw = META.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == META_SHA
    meta = json.loads(raw)
    checks, phases = 0, {}
    recent_overlaps, remote_overlaps = 0, 0
    recent_lead, remote_lead = [], []
    for phase in ('train', 'val'):
        records = meta['phases'][phase]['records']
        phases[phase] = {'records':len(records), 'points_sampled':0,
                         'covered_active_points':0}
        for rec in records:
            n, stop, start = rec['windows'], rec['covered_stop'], 1344
            assert stop == 96*(n-1)+191
            assert stop > start+384
            # All 96 interior phases plus actual prefix and tail/clamp boundaries.
            pts = sorted(set(range(start,start+288)) | set(range(stop-384,stop)))
            assert {t%96 for t in pts} == set(range(96))
            phases[phase]['points_sampled'] += len(pts)
            phases[phase]['covered_active_points'] += stop-start
            for t in pts:
                cur = support(t,n)
                # Verify current formula directly, with no numpy/Hann implementation.
                ids = [j for j in range(max(0,t//96-2), min(n,t//96+1))
                       if 0 < t-96*j < 191]
                assert cur == (96*min(ids),96*max(ids)+192)
                near_first, near_last = support(t-384,n), support(t-1,n)
                old_first, old_last = support(t-671,n), support(t-288,n)
                assert near_first[1] <= near_last[1] <= cur[1]
                assert old_first[1] <= old_last[1] <= cur[0]
                recent_overlaps += int(near_last[1] > cur[0])
                remote_overlaps += int(old_last[1] > cur[0])
                recent_lead.append(near_last[1]-t)
                remote_lead.append(old_last[1]-t)
                checks += 1
    result = {
        'passed':True, 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'metadata_sha256':META_SHA, 'phases':phases, 'sampled_coordinates':checks,
        'sample_rule':'All 96 interior stride phases plus actual first 288 active and last 384 covered points per record; not enumeration of every point.',
        'recent_prediction_lags_inclusive':[1,384],
        'D19_prediction_lags_inclusive':[288,671],
        'recent_latest_raw_end_minus_target_range':[min(recent_lead),max(recent_lead)],
        'D19_latest_raw_end_minus_target_range':[min(remote_lead),max(remote_lead)],
        'recent_latest_prediction_overlaps_current_raw_support_count':recent_overlaps,
        'D19_latest_prediction_overlaps_current_raw_support_count':remote_overlaps,
        'max_future_permission_extended':False,
        'general_reason':'For u<t, min(n-1,floor((u-1)/96)) <= min(n-1,floor((t-1)/96)); hence every prior frozen prediction ends no later in raw EEG than the current frozen prediction. This does not make the offline backbone causal or independent of current EEG.',
        'scope':'Integer geometry and existing TRAIN/VAL metadata only. No information-gain claim, model selection or performance computation.',
        'signal_feature_target_arrays_loaded':0,'model_forwards':0,'fits':0,'optimizer_updates':0,
        'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-tic,
        'pid':os.getpid(), 'goal_complete':False
    }
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'COORDINATOR_SUPPORT_CHECK.json').open('x') as f:
        json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
