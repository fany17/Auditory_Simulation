"""Bounded 2203-only probes of HDPS source indices and raw preprocessing inputs."""
import argparse
import csv
import json
import pickle
import pickletools
from pathlib import Path


class MetadataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("No class loading allowed for public index")


def main(base):
    idxpath = base / "valid_seg_dict_6.pkl"
    opcodes = list(pickletools.genops(idxpath.read_bytes()))
    report = {"index_bytes": idxpath.stat().st_size,
              "pickle_global_opcodes": [op.name for op, _, _ in opcodes if op.name in {"GLOBAL", "STACK_GLOBAL", "REDUCE"}]}
    with idxpath.open("rb") as stream:
        indices = MetadataOnlyUnpickler(stream).load()
    if not isinstance(indices, dict) or not all(isinstance(k, str) and isinstance(v, list) and all(isinstance(i, int) for i in v) for k, v in indices.items()):
        raise ValueError("unsupported metadata schema")
    (base / "valid_segments_metadata.json").write_text(json.dumps(indices, indent=2))
    report["index_lengths"] = {k: {"n": len(v), "min": min(v) if v else None, "max": max(v) if v else None} for k, v in indices.items()}
    root = Path("/home/fanyu/auditory_simulation_m6a/pub_01/s06/sparrkulee_v3.1/raw")
    first = sorted((root / "sub-001").glob("*/eeg/*bdf*"))[0]
    # Header only, before any full raw load. No permanent decompression.
    import gzip
    opening = gzip.open if first.suffix == ".gz" else open
    with opening(first, "rb") as stream:
        fixed = stream.read(256)
        header_bytes = int(fixed[184:192])
        nchan = int(fixed[252:256])
        signal_header = stream.read(header_bytes - 256)
    offset = nchan * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    samples_per_record = [int(signal_header[offset + i * 8:offset + (i + 1) * 8]) for i in range(nchan)]
    duration = float(fixed[244:252])
    report["first_raw_header"] = {"path": str(first), "nchannels": nchan, "record_duration": duration,
                                  "channel0_fs": samples_per_record[0] / duration,
                                  "records": int(fixed[236:244])}
    report["raw_sidecars"] = [str(p) for p in first.parent.glob(first.name.split("_eeg")[0] + "*") if not p.name.endswith((".bdf", ".bdf.gz"))]
    (base / "hdps_input_probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"index_stories": len(indices), "header": report["first_raw_header"], "first_counts": dict(list(report["index_lengths"].items())[:8])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    main(parser.parse_args().base)
