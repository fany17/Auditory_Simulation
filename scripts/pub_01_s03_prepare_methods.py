"""Prepare frozen internal public subset and original XLSR19 feature on 2203."""
import argparse
import csv
import gzip
import json
import math
import platform
from pathlib import Path
import numpy as np
from scipy.signal import resample, resample_poly


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def freeze(cfg):
    root, work = Path(cfg["root"]), Path(cfg["work"])
    out = work / "prepared"
    out.mkdir(exist_ok=False)
    dump(out / "config_frozen.json", cfg)
    index = {r["source_recording"]: r for r in csv_rows(root / cfg["index"])}
    candidates = [r for r in csv_rows(root / cfg["upstream_split"]) if r["record_status"] == "READY"]
    subjects = sorted({r["subject"] for r in candidates})[:10]
    selected = [r for r in candidates if r["subject"] in subjects]
    families = sorted({r["source_family"] for r in selected})
    group = {family: ("test" if i % 5 == 0 else "val" if i % 5 == 1 else "train") for i, family in enumerate(families)}
    records, failures = [], []
    stats_n, stats_sum, stats_square = 0, np.zeros(64), np.zeros(64)
    for row in sorted(selected, key=lambda r: (r["subject"], r["source_recording"])):
        raw = index[row["source_recording"]]
        source_meta = root / "pub_01/s02_feature_cache_20250925" / row["raw_story"] / "metadata.json"
        try:
            if not source_meta.is_file():
                raise ValueError("no verified S02 audio source mapping")
            metadata = json.loads(source_meta.read_text())
            audio_path = Path(metadata["source_path"])
            if not audio_path.is_relative_to(root) or not audio_path.is_file():
                raise ValueError("audio path missing or outside project")
            eeg = np.load(raw["eeg_path"], mmap_mode="r", allow_pickle=False)
            if eeg.shape[0] == 64:
                eeg = eeg.T
            if eeg.ndim != 2 or eeg.shape[1] != 64:
                raise ValueError("bad EEG shape")
            n = min(len(eeg), int(raw["samples"]), math.floor(metadata["source_duration_seconds"] * 64))
            if n < 384 or not np.isfinite(eeg[:n]).all():
                raise ValueError("nonfinite or insufficient support")
            output = {"subject": row["subject"], "subject_id": subjects.index(row["subject"]),
                      "record": row["source_recording"], "story": row["raw_story"],
                      "family": row["source_family"], "phase": group[row["source_family"]],
                      "eeg_path": raw["eeg_path"], "audio_path": str(audio_path),
                      "common_samples": n, "segments": n // 192, "physical_sync": "UNKNOWN"}
            records.append(output)
            if output["phase"] == "train":
                stats_n += n
                stats_sum += np.asarray(eeg[:n], dtype=np.float64).sum(0)
                stats_square += np.square(np.asarray(eeg[:n], dtype=np.float64)).sum(0)
        except Exception as exc:
            failures.append({"record": row["source_recording"], "error": repr(exc), "no_replacement": True})
    if not records or not stats_n:
        raise RuntimeError("no training records")
    mean = stats_sum / stats_n
    std = np.sqrt(np.maximum(stats_square / stats_n - mean ** 2, 1e-12))
    dump(out / "scope_manifest.json", {"subjects": subjects, "records": records, "failures": failures,
                                     "family_split": group, "no_fit_yet": True})
    np.savez(out / "train_eeg_scaling.npz", mean=mean, std=std)
    # Evaluation candidate identities are fixed before any trained model output.
    windows = []
    for r in records:
        for segment in range(r["segments"]):
            windows.append({"subject": r["subject"], "subject_id": r["subject_id"], "story": r["story"],
                            "phase": r["phase"], "record": r["record"], "segment": segment,
                            "start_sample": segment * 192})
    with (out / "window_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(windows[0]))
        writer.writeheader(); writer.writerows(windows)
    dump(out / "evaluation_rules_frozen.json", {"large_pool": "all unique story/segment units per subject/phase",
         "2afc": "same-story segment+round(4*64/192) is NOT used: exact +256 samples window, if valid support; drop final4s only from 2AFC",
         "5afc": "four distinct same-story segment IDs; rng1024; candidate order seeded; no model-based selection",
         "EEG_shuffle": "cyclic query permutation by13 segment indices; candidate manifest unchanged"})
    print(json.dumps({"frozen_subjects": subjects, "records": len(records), "failures": failures, "family_split": group}), flush=True)


def load_audio(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        with np.load(stream, allow_pickle=False) as data:
            audio = np.asarray(data["audio"], dtype=np.float32)
            fs = int(np.asarray(data["fs"]).squeeze())
    if audio.ndim == 2:
        audio = audio.mean(axis=int(np.argmin(audio.shape)))
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("unsupported or nonfinite audio")
    divisor = math.gcd(fs, 16000)
    return resample_poly(audio, 16000 // divisor, fs // divisor).astype(np.float32)


def features(cfg, kind):
    import librosa
    work = Path(cfg["work"])
    scope = json.loads((work / "prepared/scope_manifest.json").read_text())
    stories = {r["story"]: r["audio_path"] for r in scope["records"]}
    cache = work / "features"
    cache.mkdir(exist_ok=True)
    if kind == "xlsr19":
        import torch
        from transformers import Wav2Vec2Model
        model_path = work / ("models/xlsr_dutch_" + cfg["xlsr_revision"])
        model = Wav2Vec2Model.from_pretrained(model_path, local_files_only=True).eval().to("cuda:0")
        torch.set_num_threads(8)
    for story, path in sorted(stories.items()):
        destination = cache / story
        destination.mkdir(exist_ok=True)
        target, meta = destination / (kind + ".npy"), destination / (kind + ".json")
        if target.exists() and meta.exists():
            continue
        audio = load_audio(path)
        if kind == "mel80":
            # Explicit internal adapter, not claimed identical to author compression.
            magnitude = np.abs(librosa.stft(audio, n_fft=1024, hop_length=160, win_length=1024, center=True))
            bands = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=80, fmin=0, fmax=8000, norm="slaney")
            native = np.log(np.maximum(bands @ magnitude, 1e-5)).T
            native_rate = 100
        else:
            import torch
            padded = np.pad(audio, (16000, 0))
            pieces = []
            for start in range(0, len(audio), 8 * 16000):
                chunk = padded[start:start + 10 * 16000]
                tensor = torch.from_numpy(chunk).unsqueeze(0).to("cuda:0")
                with torch.inference_mode():
                    hidden = model(tensor, attention_mask=torch.ones_like(tensor, dtype=torch.long), output_hidden_states=True).hidden_states[19][0].cpu().numpy()
                last = start + 10 * 16000 >= len(padded)
                cropped = hidden[50:] if last else hidden[50:-49]
                pieces.append(cropped)
                if len(pieces) % 20 == 0:
                    print(json.dumps({"story": story, "kind": kind, "chunks": len(pieces)}), flush=True)
            native = np.concatenate(pieces)
            native_rate = 50
        length = round(len(native) * 64 / native_rate)
        feature = resample(native, length, axis=0).astype(np.float32)
        if not np.isfinite(feature).all():
            raise ValueError("nonfinite feature")
        np.save(target, feature)
        dump(meta, {"story": story, "source": path, "kind": kind, "shape": list(feature.shape),
             "native_rate": native_rate, "native_frames": len(native), "fs": 64,
             "hidden_states_tuple_index": 19 if kind == "xlsr19" else None,
             "model_revision": cfg["xlsr_revision"] if kind == "xlsr19" else None,
             "all_finite": True, "time_axis": "source story origin; context crop then whole-story Fourier resample",
             "physical_sync": "UNKNOWN"})
        print(json.dumps({"feature_done": story, "kind": kind, "shape": list(feature.shape)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["freeze", "mel80", "xlsr19"], required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if platform.system() != "Linux" or cfg["patient_data"] != "FORBIDDEN":
        raise RuntimeError("2203 public execution only")
    freeze(cfg) if args.stage == "freeze" else features(cfg, args.stage)
