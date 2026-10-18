"""Run the GPL ExpORL raw pipeline with HDPS 100Hz settings on server2203.

Uses Challenge2024 GPL-3-or-later preprocessing + brain_pipe0.0.4. Does not
import the unlicensed HDPS model sources. Input is original published1024Hz BDF,
not the64Hz derivative. Originals remain readonly; every run gets a new directory.
"""
import importlib.util
import argparse
import json
import sys
from pathlib import Path
import scipy.signal


def main(base, mode, run, subject=None):
    original = Path("/home/fanyu/auditory_simulation_m6a/pub_01/s06/sparrkulee_v3.1/raw")
    view = base / ("raw_view_" + run)
    out = base / ("preprocessing100_" + run)
    view.mkdir(exist_ok=False); out.mkdir(exist_ok=False)
    (view / "stimuli").symlink_to(original / "stimuli", target_is_directory=True)
    if mode == "smoke":
        first = sorted((original / "sub-001").glob("*/eeg/*bdf*"))[0]
        dest = view / first.parent.relative_to(original)
        dest.mkdir(parents=True)
        prefix = first.name.split("_eeg")[0]
        for path in first.parent.glob(prefix + "*"):
            (dest / path.name).symlink_to(path)
    else:
        split = json.loads((base / "hdps_split.json").read_text())
        selected = [subject] if subject else list(split)
        if any(s not in split for s in selected):
            raise ValueError("subject outside frozen author split")
        for selected_subject in selected:
            (view / selected_subject).symlink_to(original / selected_subject, target_is_directory=True)
    spec = importlib.util.spec_from_file_location("licensed_sparrkulee", base / "challenge2024_sparrkulee_gpl.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    # APR only adds SNR. The selected EEG numerical steps never consume SNR;
    # trigger/crop coordinates come from the published trigger payload and BDF.
    # Missing APR is an input-container dependency, not missing EEG samples.
    adaptations = []
    original_extractor = mod.BIDSAPRStimulusInfoExtractor
    class PublicEventsExtractor(original_extractor):
        def __call__(self, data_dict):
            path = data_dict[self.brain_path_key]
            apr = Path('_'.join(path.split('_')[:-1]) + '_eeg.apr')
            if apr.is_file():
                return super().__call__(data_dict)
            events = mod.BIDSStimulusInfoExtractor.__call__(self, data_dict)
            if not events or any('snr' not in event or not __import__('math').isfinite(float(event['snr'])) for event in events):
                raise ValueError('missing APR and no explicit finite public SNR')
            adaptations.append({'record': str(path), 'missing_apr': str(apr), 'public_events_snr': [event['snr'] for event in events],
                'fields_APR_would_add': ['snr'], 'EEG_numeric_steps_consume_snr': False,
                'trigger_and_time_coordinates_changed': False, 'restricted_APR_requested_or_read': False})
            (out/'input_container_adaptations.json').write_text(json.dumps(adaptations, indent=2))
            return events
    mod.BIDSAPRStimulusInfoExtractor = PublicEventsExtractor
    coordinates = []
    original_aligner = mod.AlignPeriodicBlockTriggers
    def observed_aligner(*args, **kwargs):
        from brain_pipe.preprocessing.brain.trigger import default_drift_correction
        def observe(data, brain_indices, brain_fs, stimulus_indices, stimulus_fs):
            result = default_drift_correction(data, brain_indices, brain_fs, stimulus_indices, stimulus_fs)
            coordinates.append({"brain_fs": float(brain_fs), "stimulus_trigger_fs": float(stimulus_fs),
                "brain_trigger_count": len(brain_indices), "stimulus_trigger_count": len(stimulus_indices),
                "brain_first_last_seconds": [float(brain_indices[0] / brain_fs), float(brain_indices[-1] / brain_fs)],
                "stimulus_first_last_seconds": [float(stimulus_indices[0] / stimulus_fs), float(stimulus_indices[-1] / stimulus_fs)],
                "aligned_samples": result.shape[1], "aligned_duration_seconds": result.shape[1] / brain_fs,
                "coordinate_zero": "first EEG trigger after official drift correction; corresponding first stimulus trigger"})
            return result
        return original_aligner(*args, postprocessing_fn=observe, **kwargs)
    mod.AlignPeriodicBlockTriggers = observed_aligner
    resampler = mod.ResamplePoly
    mod.ResamplePoly = lambda rate, *a, **k: resampler(100 if rate == 64 else rate, *a, **k)
    pipeline = mod.DefaultPipeline

    class HDPSPipeline(pipeline):
        def __init__(self, steps, *args, **kwargs):
            if any(type(step).__name__ == "ArtifactRemovalMWF" for step in steps):
                steps = list(steps)
                steps.insert(-1, mod.SosFiltFilt(scipy.signal.butter(1, 40, "lowpass", fs=100, output="sos"), emulate_matlab=True, axis=1))
            else:
                # EEG linkage needs only raw stimulus loading. The unused
                # challenge envelope/Mel10/save steps are not HDPS Mel80 targets.
                steps = [steps[0]]
            kwargs["on_error"] = pipeline.RAISE
            super().__init__(steps=steps, *args, **kwargs)

    mod.DefaultPipeline = HDPSPipeline
    # DefaultRunner starts a process pool even at1 worker; the dynamically
    # constructed pipeline is not pickleable. Serial calls preserve the steps.
    class SerialRunner:
        def __init__(self, **kwargs):
            pass

        def run(self, pairs):
            for loader, processing in pairs:
                for item in loader:
                    path = str(item.get("data_path", "unknown"))
                    before = len(coordinates)
                    try:
                        processing(item)
                    except Exception as exc:
                        failures.append({"path": path, "error": repr(exc), "message": str(exc), "filename": getattr(exc, 'filename', None)})
                        if mode == "smoke":
                            raise
                    finally:
                        for entry in coordinates[before:]:
                            entry["record"] = path
                        (out / "alignment_coordinates.json").write_text(json.dumps(coordinates, indent=2))
                        (out / "failures.json").write_text(json.dumps(failures, indent=2))

    mod.DefaultRunner = SerialRunner
    failures = []
    (out / "input_protocol.json").write_text(json.dumps({"input_root": str(original), "view": str(view), "raw_fs": 1024,
       "target_fs": 100, "source": "GPL Challenge2024 bd6b018f + brain_pipe0.0.4",
       "steps": "0.5Hz highpass, interpolation, periodic trigger alignment, epochs, MWF, CAR,100Hz resample,40Hz lowpass",
       "mode": mode, "not64Hz_upsampling": True, "model_training": "NOT_RUN"}, indent=2))
    mod.run_preprocessing_pipeline(str(view), str(out / "stimuli"), str(out / "eeg"), nb_processes=1, overwrite=False, log_path=str(out / "pipeline.log"))
    import numpy as np
    outputs = []
    for path in (out / "eeg").rglob("*.npy"):
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        outputs.append({"path": str(path), "shape": list(array.shape), "dtype": str(array.dtype), "finite": bool(np.isfinite(array).all()), "sampling_rate": 100})
    if not outputs or not all(row["finite"] and 64 in row["shape"] for row in outputs):
        raise RuntimeError("raw100Hz pipeline did not produce valid EEG")
    (out / "output_check.json").write_text(json.dumps(outputs, indent=2))
    print(json.dumps(outputs), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--run", default="smoke01")
    parser.add_argument("--subject", default=None)
    args = parser.parse_args()
    main(args.base, args.mode, args.run, args.subject)
