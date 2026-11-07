"""Frozen D26 TRAIN/VAL geometry and exact native contrastive batch sampling."""
import json
import math
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

HOME = Path("/home/fanyu/auditory_simulation_m6a/pub_01")
SATA = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01")
ROOT = SATA / "s03_contrastive_history_20261106"
OLD = HOME / "s03_strong_baselines_20261015/transformer_available01"
BASE = HOME / "s03_incremental_continuity_20261101"
D18 = SATA / "s03_preprojection_memory_20261103"
D19 = SATA / "s03_physical_stream_20261106"
D24 = SATA / "s03_source_family_crossfit_20261106"
NATIVE_CONFIG = OLD / "training_seed1024_attempt04/config_frozen.json"
ENVELOPE = OLD / "envelope_readout_seed1024"
PREPARED = OLD / "prepared"
FEATURES = OLD / "features/manifest.json"
FAMILY = BASE / "SOURCE_FAMILY_JOIN.json"
MASK = BASE / "block_eligibility/val.json"
PHASES = ("train", "val")
SEEDS = (1024, 2026, 31415)
ARMS = ("HH", "CH", "CC")
FRAME, HOP, READY = 192, 96, 864
NEAR = (-480, -288)
FAR = (-864, -672)
CL_BATCH = 128
BATCH = 64


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def stamp(path):
    data = Path(path).stat()
    return {"bytes": data.st_size, "mtime_ns": data.st_mtime_ns}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def plans(rows):
    indices = defaultdict(lambda: defaultdict(list))
    subjects = defaultdict(set)
    for i, row in enumerate(rows):
        story = row["story"]
        indices[story][row["start"]].append(i)
        subjects[story].add(row["subject"])
    result = []
    for story, times in sorted(indices.items()):
        starts = sorted(times)
        blocks = len(starts) // CL_BATCH
        if blocks:
            width = len(starts) // blocks
            for b in range(blocks):
                item = (story, starts[b*width:(b+1)*width], times)
                result.extend([item]*len(subjects[story]))
    return result


class Coordinates:
    def __init__(self, phase, manifest, audio=False):
        check(phase in PHASES, "TRAIN/VAL only")
        self.phase = phase
        self.rows = manifest["phases"][phase]
        self.windows = sum(r["windows"] for r in self.rows)
        self.starts = np.concatenate([np.arange(r["windows"], dtype=np.int32)*HOP for r in self.rows])
        self.record_ids = np.repeat(np.arange(len(self.rows), dtype=np.int32), [r["windows"] for r in self.rows])
        self.y = np.load(ENVELOPE/f"{phase}_envelope.npy", mmap_mode="r", allow_pickle=False)
        check(self.y.shape == (self.windows,1,FRAME) and self.y.dtype == np.float32, "Native target shape")
        norm = np.load(PREPARED/"train_only_normalization.npz", allow_pickle=False)
        self.mean, self.std = norm["mean"],norm["std"]
        norm.close()
        self.loaded = {}
        self.audio = audio
        if audio:
            self.cl_rows = read(PREPARED/f"{phase}_windows.json")
            lookup = {r["record_id"]:i for i,r in enumerate(self.rows)}
            self.cl_records = np.array([lookup[r["record_id"]] for r in self.cl_rows],dtype=np.int32)
            self.cl_starts = np.array([r["start"] for r in self.cl_rows],dtype=np.int32)
            self.cl_plans = plans(self.cl_rows)
            self.x = np.load(PREPARED/f"{phase}_eeg.npy", mmap_mode="r", allow_pickle=False)
            check(self.x.shape == (len(self.cl_rows),FRAME,64) and self.x.dtype == np.float32,
                  "Prepared native CL grid")
            stories = {r["story"] for r in self.rows}
            paths = {r["story"]:r["path"] for r in manifest["features"] if r["story"] in stories}
            self.features = {story:np.load(path,mmap_mode="r",allow_pickle=False) for story,path in paths.items()}

    def eeg(self, record):
        if record not in self.loaded:
            row = self.rows[record]
            path = np.load(row["eeg_path"],mmap_mode="r",allow_pickle=False)
            raw = path.T if path.shape[0] == 64 else path
            check(raw.ndim == 2 and raw.shape[1] == 64 and len(raw) >= row["common_samples"], "Raw EEG")
            samples = np.asarray(raw[:row["common_samples"]],np.float32)
            self.loaded[record] = ((samples-self.mean)/self.std).astype(np.float32)
        return self.loaded[record]

    def windows_at(self, ids, starts):
        return np.stack([self.eeg(int(rec))[int(s):int(s)+FRAME] for rec,s in zip(ids,starts)])

    def past(self, ids, starts, slots):
        chosen = starts >= READY
        if not chosen.any():
            return None
        return tuple(self.windows_at(ids[chosen],starts[chosen]+delay) for delay in slots)

    def grid(self, first, last, role, far=False):
        ids, starts = self.record_ids[first:last],self.starts[first:last]
        current = self.windows_at(ids,starts)
        past = self.past(ids,starts,FAR if far else NEAR) if role == "H" else None
        return current, starts >= READY, past, np.array(self.y[first:last],copy=True)

    def selections(self, seed, shuffle):
        rng = np.random.default_rng(seed)
        order = np.arange(len(self.cl_plans))
        if shuffle:
            rng.shuffle(order)
        for i in order:
            story, times, lookup = self.cl_plans[i]
            starts = rng.choice(times,size=CL_BATCH,replace=False)
            ids = np.array([int(rng.choice(lookup[int(s)])) for s in starts],dtype=np.int32)
            yield story,starts,ids

    def contrastive(self, selection, role):
        story, starts, ids = selection
        current = np.asarray(self.x[ids]).copy()
        past = self.past(self.cl_records[ids],starts,NEAR) if role == "H" else None
        target = np.stack([self.features[story][s:s+FRAME] for s in starts])
        return current, starts >= READY, past, target

    def cl_coverage(self, selections, role, union=None):
        ids = np.concatenate([r[2] for r in selections])
        starts = self.cl_starts[ids]
        rec = self.cl_records[ids]
        # Each visited interval spans exactly two 96-sample cells. This counts
        # physical (record, time) samples once even when windows overlap.
        stride = 1+max(r["common_samples"] for r in self.rows)//HOP
        position = rec.astype(np.int64)*stride + starts//HOP
        cells = lambda s: np.unique(np.r_[s,s+1])
        active = starts >= READY
        first = cells(position[active]+NEAR[0]//HOP) if role == "H" else cells(position[active])
        second = cells(position[active]+NEAR[1]//HOP) if role == "H" else first
        raw = cells(position)
        audio = {(story,int(s)) for story,ts,_ in selections for s in ts}
        report = dict(current_windows=len(ids), unique_current_windows=len(np.unique(ids)),
                    unique_current_EEG_sample_times=len(raw)*HOP,
                    memory_query_windows=int(active.sum()), history_slot_presentations=2*int(active.sum()),
                    unique_slot1_EEG_sample_times=len(first)*HOP,
                    unique_slot2_EEG_sample_times=len(second)*HOP,
                    unique_all_EEG_sample_times=len(np.unique(np.r_[raw,first,second]))*HOP,
                    unique_audio_story_time_windows=len(audio),
                    repeated_current_windows=len(ids)-len(np.unique(ids)))
        if union is not None:
            import torch
            names={s:i for i,s in enumerate(sorted(self.features))}
            audio_indices=np.array([names[story]*stride+s//HOP for story,s in audio],dtype=np.int64)
            for key,locations,size in (
                ("current_windows",ids,len(self.cl_rows)),
                ("EEG_samples",np.r_[raw,first,second],stride*len(self.rows)),
                ("slot1_samples",first,stride*len(self.rows)),
                ("slot2_samples",second,stride*len(self.rows)),
                ("audio_windows",audio_indices,stride*len(names))):
                if key not in union:
                    union[key]=torch.zeros(size,dtype=torch.bool)
                union[key].numpy()[locations]=True
            report["cumulative"]={
                f"unique_{key}":int(np.count_nonzero(value.numpy()))*(HOP if key.endswith("samples") else 1)
                for key,value in union.items()}
        return report


def inventory():
    prepared = read(PREPARED/"manifest_frozen.json")["records"]
    envelopes = read(ENVELOPE/"manifest_frozen.json")["records"]
    original = {r["record"]:(i,r) for i,r in enumerate(prepared) if r["phase"] in PHASES}
    family = {r["record"]:r for r in read(FAMILY)["records"] if r["phase"] in PHASES}
    result, budget = {}, {}
    files = set()
    val_mask = read(MASK)
    check(val_mask["prediction_independent"] and val_mask["eligible_blocks"] == 752,"Frozen score mask")
    for phase, number, count, b in (("train",426,252454,820),("val",49,30500,98)):
        rows = []
        offset = 0
        for row in envelopes:
            if row["phase"] != phase:
                continue
            key = row["record"]
            i, source = original[key]
            mapping = family[key]
            check(all(source[k] == row[k] for k in ("eeg_path","subject","story","common_samples","phase"))
                  and mapping["phase"] == phase, "Source record coordinate")
            windows = (row["common_samples"]-FRAME)//HOP+1
            check(row["phase_offset"] == offset and row["readout_windows"] == windows and windows>9, "Envelope grid")
            rows.append(dict(row,windows=windows,offset=offset,record_id=i,source_family=mapping["source_family"]))
            offset += windows
        check(len(rows)==number and offset==count, "Original cohort")
        cl = read(PREPARED/f"{phase}_windows.json")
        ids = {r["record_id"]:r for r in rows}
        for item in cl:
            record = ids[item["record_id"]]
            check(item["story"] == record["story"] and item["subject"] == record["subject"]
                  and item["phase"] == phase and item["start"]%HOP==0
                  and 0<=item["start"]<=record["common_samples"]-FRAME, "Exact native CL coordinate")
        batches = plans(cl)
        check(len(batches)==b, f"Native CL batch count: {phase} got {len(batches)}")
        result[phase] = rows
        budget[phase] = dict(records=number,supervised_windows=count,supervised_B64=math.ceil(count/BATCH),
                             supervised_B512=math.ceil(count/512),
                             supervised_history_windows=count-9*number,native_CL_index_rows=len(cl),
                             native_CL_batches=b,native_CL_presentations=b*CL_BATCH)
        files.update([PREPARED/f"{phase}_eeg.npy",ENVELOPE/f"{phase}_envelope.npy"])
        files.update(Path(r[k]) for r in rows for k in ("eeg_path","envelope_path"))
    check([r["record"] for r in result["val"]]==val_mask["records"],"Original VAL score order")
    stories = {r["story"] for rows in result.values() for r in rows}
    features = [r for r in read(FEATURES) if r["story"] in stories]
    check(len(features)==len(stories),"One XLSR source per legal story")
    files.update(Path(r["path"]) for r in features)
    files.add(PREPARED/"train_only_normalization.npz")
    train_families={r["source_family"] for r in result["train"]}
    check(sum(r["source_family"] in train_families for r in result["val"])==29,"Frozen family overlap")
    return dict(phases=result,features=features,counts=budget,inputs={str(p):stamp(p) for p in sorted(files)},
                allowed_phases=list(PHASES),VAL_people=49,VAL_blocks=752,VAL_shared_family_records=29)
