"""D26 report from completed frozen stage metadata and saved VAL scores only."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from pub_01_s03_ss07_contrastive_history_report_cost import close, read, require, summarize_cost

ROOT = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_contrastive_history_20261106")
ANCHOR = ROOT.parent / "s03_physical_stream_20261106" / "fit"
PROTOCOL = "8c249eabeb9edb9a5360006f1d3dabb9552e2a22ff649b9bae72801f75311b4d"
SEEDS = (1024, 2026, 31415)
ARMS = ("HH", "CH", "CC")
READOUTS = ("raw", "affine", "Ccap")
STORIES = {"audiobook_13_2": 6, "audiobook_5_2": 23, "podcast_15": 4, "podcast_21": 7, "podcast_30": 9}
METRICS = ("r", "MSE")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def mean(values):
    return math.fsum(values) / len(values)


def save(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def admitted(root):
    gate_file = root / "PROTOCOL_V05.json"
    terminal_file = root / "formal" / "COMPLETED.json"
    require(terminal_file.is_file(), "Wait for full frozen matrix")
    gate = read(gate_file)
    terminal = read(terminal_file)
    review = read(root / "COORDINATOR_EXECUTION_REVIEW.json")
    require(digest(gate_file) == PROTOCOL and gate["frozen"]
            and review["passed"] and review["protocol_sha256"] == PROTOCOL
            and terminal["passed"] and terminal["protocol_sha256"] == PROTOCOL and terminal["old_test_reads"] == 0,
            "Original approved D26 V05 terminal")
    require(gate["seeds"] == list(SEEDS) and gate["arms"] == list(ARMS) and gate["FT_final_epoch"] == 4
            and gate["lambda_fixed"] == .01, "Frozen matrix and final model")
    rows = terminal["evaluations"]
    require(len(rows) == 9 and {(r["seed"], r["arm"]) for r in rows} == {(s, a) for s in SEEDS for a in ARMS},
            "Exactly one result for each seed and arm")
    require(digest(root / "INPUTS.json") == gate["inventory_sha256"], "Frozen input manifest")
    mask_paths = [Path(p) for p in gate["sources"] if p.endswith("/block_eligibility/val.json")]
    require(len(mask_paths) == 1 and digest(mask_paths[0]) == gate["sources"][str(mask_paths[0])], "Original scoring mask")
    return terminal, read(root / "INPUTS.json"), read(mask_paths[0]), digest(terminal_file)


def support(inventory, mask):
    train, val = (inventory["phases"][p] for p in ("train", "val"))
    require(len(train) == 426 and len(val) == 49
            and all(row["phase"] == "train" for row in train) and all(row["phase"] == "val" for row in val),
            "Exactly the frozen public TRAIN/VAL")
    families = {r["source_family"] for r in train}
    roster = {r["record"]: r for r in val}
    require(len(roster) == len({r["subject"] for r in val}) == 49
            and {r["subject"] for r in val} <= {r["subject"] for r in train}, "One VAL record per seen person")
    require(Counter(r["story"] for r in val) == STORIES
            and sum(r["source_family"] in families for r in val) == 29, "Five stories and original 29/20 family split")
    require(mask["prediction_independent"] and mask["records"] == list(roster), "Mask and record order")
    eligible = {(b["record"], b["start"]): b["samples"] for b in mask["blocks"] if b["eligible"]}
    require(len(eligible) == mask["eligible_blocks"] == 752 and {k[0] for k in eligible} == set(roster),
            "Original 752 block keys")
    return roster, families, eligible


def person_rows(blocks):
    groups = defaultdict(list)
    for block in blocks:
        groups[block["subject"]].append(block)
    return [dict(subject=subject, blocks=len(group),
                 **{k: mean([r[k] for r in group]) for k in METRICS}) for subject, group in sorted(groups.items())]


def aggregate(blocks):
    people = person_rows(blocks)
    return dict(people=len(people), blocks=len(blocks), records=len({r["record"] for r in blocks}),
                **{k: mean([r[k] for r in people]) for k in METRICS})


def assert_metrics(a, b, reason):
    require(all(close(a[k], b[k]) for k in METRICS), reason)


def checked_blocks(blocks, roster, eligible, pure):
    observed = {}
    for row in blocks:
        key = row["record"], row["start"]
        require(key not in observed and key in eligible, "Exact original block key")
        meta = roster[row["record"]]
        require(all(row[k] == meta[k] for k in ("subject", "source_family")), "Block subject/source identity")
        require(all(math.isfinite(row[k]) for k in METRICS) and -1.000000000001 <= row["r"] <= 1.000000000001
                and row["MSE"] >= 0, "Finite saved r and MSE")
        if pure:
            lost = max(0, 960-max(row["start"], 1))
            require(row["retained_samples"] == eligible[key]-lost, "Pure-memory common Hann support")
        else:
            require(row["samples"] == eligible[key] and row["phase"] == "val" and row["story"] == meta["story"],
                    "Original full Hann sample count")
        observed[key] = dict(row, story=meta["story"])
    require(observed.keys() == eligible.keys(), "No missing or duplicated eligible blocks")
    return observed


def stratify(rows, families):
    selectors = {
        "all": {"all": rows},
        "story": {key: [r for r in rows if r["story"] == key] for key in STORIES},
        "family": {"shared": [r for r in rows if r["source_family"] in families],
                   "unseen": [r for r in rows if r["source_family"] not in families]},
        "exact_source_family": {key: [r for r in rows if r["source_family"] == key]
                                for key in sorted({r["source_family"] for r in rows})},
    }
    return [dict(stratum=group, label=label, **aggregate(blocks))
            for group, strata in selectors.items() for label, blocks in strata.items()]


def parse_score(result, roster, families, eligible, scope, labels):
    pure = scope == "pure_memory"
    if pure:
        receipt = result["pure_memory"]
        require(receipt["first_physical_sample"] == 960 and receipt["people"] == 49 and receipt["blocks"] == 752,
                "Common pure-memory start and support")
        saved = receipt["records"]
    else:
        receipt = result["phases"]["val"]
        saved = result["segments"]
        require(set(result["phases"]) == {"val"} and receipt["windows"] == 30500, "Complete saved VAL phase")
    points = checked_blocks(saved, roster, eligible, pure)
    blocks = list(points.values())
    assert_metrics(aggregate(blocks), receipt, "Saved macro r/MSE matches block aggregation")
    per_person = person_rows(blocks)
    if not pure:
        archived = {r["subject"]: r for r in result["people"]}
        require(len(archived) == len(per_person) == 49, "Complete saved per-person rows")
        for row in per_person:
            assert_metrics(row, archived[row["subject"]], "Saved per-person r/MSE")
            require(row["blocks"] == archived[row["subject"]]["blocks"], "Per-person block count")
    people_meta = {row["subject"]: row for row in roster.values()}
    return dict(points=points, people=[dict(**labels, **r, story=people_meta[r["subject"]]["story"],
                                         source_family=people_meta[r["subject"]]["source_family"]) for r in per_person],
                strata=[dict(**labels, **r) for r in stratify(blocks, families)])


def comparison(kind, left, right, labels):
    return dict(**labels, comparison=kind,
                **{f"delta_{k}": left[k] - right[k] for k in METRICS})


def check_anchor(anchor, expected_r, roster, eligible):
    require(close(anchor["r"], expected_r) and len(anchor["people"]) == 49,
            "Frozen same-seed original anchor")
    original = [dict(b, phase="val", story=roster[b["record"]]["story"]) for b in anchor["blocks"]]
    points = list(checked_blocks(original, roster, eligible, pure=False).values())
    assert_metrics(aggregate(points), anchor, "Anchor full-support r and MSE")
    saved = {row["subject"]: row for row in anchor["people"]}
    for row in person_rows(points):
        assert_metrics(row, saved[row["subject"]], "Anchor per-person r and MSE")
        require(row["blocks"] == saved[row["subject"]]["blocks"], "Anchor per-person exact block count")


def scores(formal, terminal, inventory, mask):
    roster, families, eligible = support(inventory, mask)
    people, strata, blocks, primary_rows, paired_people = [], [], [], [], []
    parsed = {}
    anchor_scores = {}
    pure_support = set()
    for row in terminal["evaluations"]:
        seed, arm = row["seed"], row["arm"]
        folder = formal / f"seed{seed}" / f"READOUT_{arm}"
        require(read(folder / "COMPLETED.json") == row and row["passed"] and row["epoch"] == 4
                and row["lambda_fixed"] == .01 and row["old_test_reads"] == 0, "Fixed final readout receipt")
        native_path = "near" if arm == "HH" else "native"
        for route in (("near", "far", "copy") if arm == "HH" else ("native",)):
            record = read(folder / f"SCORE_{route}.json")
            require(set(record) == set(READOUTS if route == native_path else ("Ccap",)), "Original evaluated readout conditions")
            for readout, result in record.items():
                scopes = ("full", "pure_memory") if arm == "HH" and readout == "Ccap" else ("full",)
                for scope in scopes:
                    labels = dict(seed=seed, arm=arm, route=route, readout=readout, scope=scope)
                    table = parse_score(result, roster, families, eligible, scope, labels)
                    parsed[seed, arm, route, readout, scope] = table
                    people.extend(table["people"])
                    strata.extend(table["strata"])
                    blocks.extend(dict(**labels, **b) for b in table["points"].values())
                    if scope == "pure_memory":
                        pure_support.add(tuple(sorted((key, b["retained_samples"]) for key, b in table["points"].items())))
                        require(close(table["strata"][0]["r"], row["VAL_pure_memory"][route]), "Frozen HH intervention r")
                    if scope == "full" and route == native_path:
                        overall = table["strata"][0]
                        require(close(overall["r"], row["VAL_r"][readout]), "Terminal primary r")
                        primary_rows.append(dict(seed=seed, arm=arm, readout=readout, r=overall["r"], MSE=overall["MSE"],
                                                 delta_r_to_anchor=overall["r"]-row["anchor_r"]))
        require(row["affine"]["a"] > 0, "Positive TRAIN affine slope")
        raw, affine = (parsed[seed, arm, native_path, name, "full"]["points"] for name in ("raw", "affine"))
        require(max(abs(raw[k]["r"]-affine[k]["r"]) for k in raw) <= 1e-10, "Native to affine r invariance")
        anchor = read(ANCHOR / f"seed{seed}_ANCHOR.json")["anchor"]["val"]
        if seed not in anchor_scores:
            check_anchor(anchor, row["anchor_r"], roster, eligible)
            anchor_scores[seed] = anchor
        require(close(anchor_scores[seed]["r"], row["anchor_r"]), "Same anchor across all arms")
    require(len(pure_support) == 1, "Same pure-memory support across all nine HH counterfactual scores")
    chosen = {}
    paired = []
    for seed in SEEDS:
        for arm in ARMS:
            route = "near" if arm == "HH" else "native"
            chosen[seed, arm] = parsed[seed, arm, route, "Ccap", "full"]
            anchor = anchor_scores[seed]
            scores_by_person = {r["subject"]: r for r in anchor["people"]}
            require(len(scores_by_person) == 49, "Anchor per-person roster")
            for person in chosen[seed, arm]["people"]:
                old = scores_by_person[person["subject"]]
                require(old.get("blocks", person["blocks"]) == person["blocks"], "Anchor per-person support count")
                paired_people.append(comparison(f"{arm}-anchor", person, old, dict(seed=seed, subject=person["subject"],
                                               story=person["story"], source_family=person["source_family"])))
            paired.append(comparison(f"{arm}-anchor", chosen[seed, arm]["strata"][0], anchor, dict(seed=seed, scope="full")))
        for a, b in (("HH", "CH"), ("HH", "CC"), ("CH", "CC")):
            left, right = chosen[seed, a], chosen[seed, b]
            paired.append(comparison(f"{a}-{b}", left["strata"][0], right["strata"][0], dict(seed=seed, scope="full")))
            for x, y in zip(left["people"], right["people"]):
                require(x["subject"] == y["subject"], "Matched per-person arm comparisons")
                paired_people.append(comparison(f"{a}-{b}", x, y, dict(seed=seed, subject=x["subject"],
                                               story=x["story"], source_family=x["source_family"])))
        for route in ("far", "copy"):
            near = parsed[seed, "HH", "near", "Ccap", "pure_memory"]
            other = parsed[seed, "HH", route, "Ccap", "pure_memory"]
            paired.append(comparison(f"near-{route}", near["strata"][0], other["strata"][0], dict(seed=seed, scope="pure_memory")))
            for x, y in zip(near["people"], other["people"]):
                require(x["subject"] == y["subject"], "Matched per-person HH comparisons")
                paired_people.append(comparison(f"near-{route}", x, y, dict(seed=seed, subject=x["subject"],
                                               story=x["story"], source_family=x["source_family"])))
    frozen_means = {arm: sum(row["VAL_r"]["Ccap"] for row in terminal["evaluations"] if row["arm"] == arm)/3
                    for arm in ("CC", "CH", "HH")}
    winner = max(frozen_means, key=frozen_means.get)
    require(frozen_means == terminal["means"] and winner == terminal["selected_full_arm"], "Frozen whole-arm rank and tie order")
    anchors = [dict(seed=s, **{k: anchor_scores[s][k] for k in METRICS}) for s in SEEDS]
    arm_means = [dict(arm=a, readout=m, **{k: mean([r[k] for r in primary_rows if r["arm"] == a and r["readout"] == m])
                                         for k in METRICS}) for a in ARMS for m in READOUTS]
    paired_summary = []
    for name in dict.fromkeys(row["comparison"] for row in paired):
        group = [row for row in paired if row["comparison"] == name]
        require(len(group) == 3, "Complete three-seed pair")
        paired_summary.append(dict(comparison=name, scope=group[0]["scope"],
                                   **{f"mean_delta_{k}": mean([row[f"delta_{k}"] for row in group]) for k in METRICS},
                                   **{f"n_positive_delta_{k}": sum(row[f"delta_{k}"] > 0 for row in group)
                                      for k in METRICS}))
    support_geometry = dict(people=49, blocks=752, story_people=STORIES, shared_family=29, unseen_family=20,
                            full_samples=sum(eligible.values()),
                            pure_memory_samples=sum(n for _, n in next(iter(pure_support))))
    return dict(primary=primary_rows, strata=strata, people=people, blocks=blocks, paired=paired, paired_summary=paired_summary, paired_people=paired_people,
                anchor=anchors, arm_means=arm_means, selected_full_arm=winner, frozen_Ccap_means=frozen_means,
                geometry=support_geometry)


def generate(root, destination):
    terminal, inventory, mask, completion_sha = admitted(root)
    formal = root / "formal"
    data = scores(formal, terminal, inventory, mask)
    costs, epochs = summarize_cost(formal, inventory, terminal, SEEDS, ARMS)
    historical = read(Path(__file__).parent / "HISTORICAL_REFERENCES.json")
    ridge = read(Path(__file__).parent / "D28_MATCHED_RAW_RIDGE_COMPLETED.json")
    engineering = read(root / "ENGINEERING.json")
    require(engineering["passed"] and ridge["passed"] and ridge["VAL_eligible_blocks"] == 752,
            "Original engineering evidence and D28 completed comparator")
    summary = {k: v for k, v in data.items() if k not in ("blocks", "people", "paired_people")}
    summary.update(terminal_sha256=completion_sha, protocol_sha256=PROTOCOL, historical_references=historical,
                   D28_development_reference=ridge["scores"], actual_costs=costs, engineering_costs=engineering,
                   scope="Saved scalar evidence only. No new fitting, neural forward, or raw EEG/envelope read.")
    destination.mkdir()
    save(destination / "RESULT_SUMMARY.json", summary)
    save(destination / "TRAINING_EPOCH_LEDGER.json", epochs)
    save(destination / "STAGE_COSTS.json", costs)
    for key in ("primary", "strata", "people", "blocks", "paired", "paired_people"):
        save_csv(destination / (key.upper() + ".csv"), data[key])
    from pub_01_s03_ss07_contrastive_history_report_figures import draw
    draw(destination, summary, epochs)
    save(destination / "REPORT_ARTIFACTS.json",
         dict(passed=True, terminal_sha256=completion_sha, protocol_sha256=PROTOCOL,
              files={p.name: digest(p) for p in sorted(destination.iterdir())},
              sources={p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*")) if p.is_file()},
              new_signal_reads=0, new_fits=0, new_neural_forwards=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(ROOT, args.output)
