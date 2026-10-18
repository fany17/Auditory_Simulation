"""Separate independent numerical review of actual HDPS engineering outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pub_01_s03_hdps_paper_model import similarity


def main(base):
    old = next((base/'preprocessing100_smoke05/eeg').rglob('*.npy'))
    current = next((base/'preprocessing100_smoke06/eeg').rglob('*.npy'))
    a,b = np.load(old,mmap_mode='r'),np.load(current,mmap_mode='r')
    no_change = a.shape==b.shape and np.array_equal(a,b)
    numeric_max_difference=float(np.max(np.abs(a-b)))
    relative_l2_difference=float(np.linalg.norm(a-b)/np.linalg.norm(a))
    mel=np.load(base/'mel_smoke01/audiobook_5_1.npy',mmap_mode='r')
    ids=json.loads((base/'valid_segments_metadata.json').read_text())['audiobook_5_1.wav'][:4]
    targets=np.stack([mel[i*300:(i+1)*300,:7] for i in ids]).astype(np.float32)
    calculated=similarity(torch.from_numpy(targets),torch.from_numpy(targets)).numpy()
    independently=np.zeros((len(ids),len(ids)))
    # NumPy scalar channels provide an independent implementation/reference.
    for i in range(len(ids)):
        for j in range(len(ids)):
            independently[i,j]=np.mean([np.corrcoef(targets[i,:,c],targets[j,:,c])[0,1] for c in range(7)])
    error=float(np.max(np.abs(independently-calculated)))
    coords=json.loads((base/'preprocessing100_smoke06/alignment_coordinates.json').read_text())[0]
    audio=json.loads((base/'mel_smoke01/manifest.json').read_text())[0]
    smoke=json.loads((base/'model_smoke01.json').read_text())
    checks={'observer_bit_exact_output':no_change,
        'observer_max_absolute_difference_microvolt':numeric_max_difference,
        'observer_relative_l2_difference':relative_l2_difference,
        'observer_numerical_equivalence_under_BLAS_thread_count_change':numeric_max_difference<1e-8 and relative_l2_difference<1e-10,
        'scalar_numpy_vs_torch_Pearson_max_error':error,
        'Pearson_agrees_with_independent_reference':error<1e-6,
        'actual_audio_duration_preserved_at16k':abs(audio['source_duration_seconds']-audio['resampled_duration_seconds'])<1/16000,
        'stimulus_first_trigger_zero':coords['stimulus_first_last_seconds'][0]==0,
        'brain_and_stimulus_trigger_count_equal':coords['brain_trigger_count']==coords['stimulus_trigger_count'],
        'all_published_first_story_valid_indices_supported':max(json.loads((base/'valid_segments_metadata.json').read_text())['audiobook_5_1.wav'])*300+300<=min(b.shape[1],len(mel)),
        'real_input_model256_backward_finite':all(r['finite_gradient'] for r in smoke['records']) and smoke['records'][-1]['batch']==256,
        'physical_hardware_synchronization_claim':'NOT_MADE',
        'model_performance_claim':'NOT_MADE','author_weight_numerical_parity':'NOT_CHECKED_AUTHOR_WEIGHTS_UNAVAILABLE'}
    technical=all(value for key,value in checks.items() if isinstance(value,bool) and key!='observer_bit_exact_output')
    result={'technical':'PASS_ENGINEERING_ONLY' if technical else 'REWORK','checks':checks,
        'acceptance':'HOLD_NO_SAME_CONDITION_TEST_RESULT','parent_consistency':'PASS_PUBLIC_ONLY',
        'independent_review':'separate post-smoke numerical pass; same agent, independent NumPy calculation'}
    (base/'engineering_review.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if not technical:
        raise RuntimeError('engineering validation failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    main(parser.parse_args().base)
