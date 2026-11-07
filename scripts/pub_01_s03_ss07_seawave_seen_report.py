"""Produce D27's same-support result package without loading any signals or weights."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

from pub_01_s03_ss07_seawave_seen_report_cost import budget, fit, need, read, same, worker_accounting

ROOT = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_seawave_seen_reference_20261106")
D26 = ROOT.parent/"s03_contrastive_history_20261106"
D28 = ROOT.parent/"s03_matched_raw_ridge_20261107"
D26_HELPERS = D26/"report_source_v03"
GATE = "8442767e6fb1a220bab129d2c2163321e760610102fdefb986f43a06079f7476"
REVIEW = "e5cfcb770962a322e8f25304621478db63a0fdd1835445c0907064400cef033d"
SEEDS, METRICS = (1024, 2026, 31415), ("r","MSE")

# Reuse the already reviewed metadata-only common-support checker and summarizer.
sys.path.insert(0, str(D26_HELPERS))
from pub_01_s03_ss07_contrastive_history_report import (
    ANCHOR, PROTOCOL as D26_GATE, aggregate, assert_metrics, check_anchor, checked_blocks,
    digest, mean, person_rows, save, save_csv, stratify, support,
)

HELPER_HASHES = {
    "pub_01_s03_ss07_contrastive_history_report.py": "550913f9c880c44687fbac280675db0285c42ee9bfeef18ee2b7a1619a149a08",
    "pub_01_s03_ss07_contrastive_history_report_cost.py": "d07c8ab6bfb5fbae8292432da8ca4abac276846d8dadc780bf792c906709dbfd",
}


def admission(root):
    formal = root/"formal"
    end = formal/"COMPLETED.json"
    need(end.is_file(), "Wait for all 150 formal D27 fits")
    gate_path, review_path = root/"PROTOCOL_V04.json", root/"COORDINATOR_EXECUTION_REVIEW.json"
    gate, review, end_receipt = (read(p) for p in (gate_path, review_path, end))
    need(digest(gate_path) == GATE and digest(review_path) == REVIEW and gate["frozen"]
         and review["passed"] and review["protocol_sha256"] == GATE, "Approved exact D27 V04")
    need(all(digest(D26_HELPERS/name) == h for name,h in HELPER_HASHES.items()), "Reviewed common-support report helpers")
    need(end_receipt["passed"] and end_receipt["protocol_sha256"] == GATE and end_receipt["old_test_reads"] == 0
         and end_receipt["initial_fits"] == 3 and end_receipt["personal_fits"] == 147, "Complete D27 V04 terminal")
    need(len(end_receipt["results"]) == 3 and {r["seed"] for r in end_receipt["results"]} == set(SEEDS),
         "Three original seeds")
    manifest = root/"INVENTORY.json"
    mask_paths = [Path(p) for p in gate["sources"] if p.endswith("/block_eligibility/val.json")]
    need(digest(manifest) == gate["inventory_sha256"] and len(mask_paths) == 1
         and digest(mask_paths[0]) == gate["sources"][str(mask_paths[0])], "Frozen cohort and exact original mask")
    return formal, {r["seed"]:r for r in end_receipt["results"]}, read(manifest), read(mask_paths[0]), gate, digest(end)


def checked_score(score, roster, mask):
    original = [dict(row, story=roster[row["record"]]["story"], phase="val") for row in score["blocks"]]
    points = list(checked_blocks(original, roster, mask, pure=False).values())
    people = person_rows(points)
    saved = {r["subject"]:r for r in score["people"]}
    need(len(saved) == len(score["people"]) == len(people), "Complete saved participant rows")
    for row in people:
        assert_metrics(row, saved[row["subject"]], "Saved participant r/MSE")
        need(row["blocks"] == saved[row["subject"]]["blocks"], "Saved participant block count")
    assert_metrics(aggregate(points), score, "Recomputed macro r/MSE on exact support")
    return points


def summarize(formal, terminal, inventory, mask, settings):
    train, val = (inventory["phase_records"][k] for k in ("train","val"))
    roster, families, eligible = support(dict(phases=inventory["phase_records"]), mask)
    train_by_person = defaultdict(list)
    for row in train:
        train_by_person[row["subject"]].append(row)
    val_by_person = {row["subject"]:row for row in val}
    need(set(inventory["subjects"]) == set(val_by_person) and len(train_by_person) == 70,
         "All 49 original VAL participants seen in TRAIN")
    need(sorted(families) == inventory["train_source_families"], "Exact shared source families")
    mask_by_person = defaultdict(dict)
    for key,n in eligible.items():
        mask_by_person[roster[key[0]]["subject"]][key] = n
    fit_costs, trajectories, paired, block_table, people_table, strata_table, seed_scores = ([] for _ in range(7))
    for seed in SEEDS:
        base = formal/f"seed{seed}"
        group, initial_cost, epochs = fit(base/"initial",seed,"initial",None,train,val,settings["initial"],inventory["initial"],None)
        fit_costs.append(initial_cost)
        trajectories.extend(epochs)
        initial = checked_score(group, roster, eligible)
        group_people = {r["subject"]:r for r in person_rows(initial)}
        personal = []
        for subject in inventory["subjects"]:
            v = val_by_person[subject]
            selected, cost, history = fit(base/"personal"/subject,seed,"personal",subject,train_by_person[subject],
                                          [v],settings["personal"],inventory["personal"][subject],initial_cost["selected_sha256"])
            fit_costs.append(cost)
            trajectories.extend(history)
            points = checked_score(selected, roster, mask_by_person[subject])
            personal.extend(points)
            initial_person = group_people[subject]
            incumbent = history[0]
            if cost["selected_epoch"] == -1:
                assert_metrics(selected, incumbent, "Preserved epoch -1 selection")
            pair = dict(seed=seed, subject=subject, story=v["story"], source_family=v["source_family"],
                        blocks=initial_person["blocks"], personal_epoch=cost["selected_epoch"],
                        personal_epochs_run=cost["actual"]["training_epochs"],
                        personal_updates=cost["actual"]["optimizer_updates"],
                        selected_personal_prefix_updates=cost["used_prefix"]["optimizer_updates"])
            for k in METRICS:
                pair.update({f"initial_{k}":initial_person[k], f"epoch_minus1_{k}":incumbent[k], f"personal_{k}":selected[k],
                             f"delta_{k}":selected[k]-initial_person[k],
                             f"delta_to_personal_incumbent_{k}":selected[k]-incumbent[k]})
            paired.append(pair)
        combined = read(base/"PERSONAL_SELECTED_VAL.json")
        saved_personal = checked_score(combined, roster, eligible)
        need({(r["record"],r["start"]):r for r in saved_personal} ==
             {(r["record"],r["start"]):r for r in personal}, "Exactly the 49 selected personal score sets")
        need(same(aggregate(initial)["r"], terminal[seed]["initial_no_globalFT_r"])
             and same(aggregate(personal)["r"], terminal[seed]["personal_r"]), "Formal seed terminal r")
        for stage, points in (("initial",initial),("personal",personal)):
            labels = dict(seed=seed, stage=stage)
            seed_scores.append(dict(**labels, **aggregate(points)))
            block_table.extend(dict(**labels, **r) for r in points)
            people_table.extend(dict(**labels, **r, story=val_by_person[r["subject"]]["story"],
                                     source_family=val_by_person[r["subject"]]["source_family"]) for r in person_rows(points))
            strata_table.extend(dict(**labels, **r) for r in stratify(points, families))
    need(len(fit_costs)==150 and len(block_table)==4512 and len(paired)==147, "Complete selected score matrix")
    overall = [dict(stage=stage, **{k:mean([r[k] for r in seed_scores if r["stage"]==stage]) for k in METRICS})
               for stage in ("initial","personal")]
    seed_differences = []
    for seed in SEEDS:
        people = [r for r in paired if r["seed"]==seed]
        seed_differences.append(dict(seed=seed, improved_person_r=sum(r["delta_r"]>1e-11 for r in people),
                                     reduced_person_MSE=sum(r["delta_MSE"]< -1e-11 for r in people),
                                     **{f"mean_delta_{k}":mean([r[f"delta_{k}"] for r in people]) for k in METRICS}))
    tables = dict(seed_scores=seed_scores, blocks=block_table, people=people_table, strata=strata_table, pairs=paired)
    return (dict(overall=overall,seed_differences=seed_differences,
                 epoch_histogram=dict(sorted(Counter(r["personal_epoch"] for r in paired).items())),
                 shared_to_personal_incumbent_max_abs={
                     k:max(abs(r[f"epoch_minus1_{k}"]-r[f"initial_{k}"]) for r in paired) for k in METRICS}),
            tables, fit_costs, trajectories, roster, eligible)


def comparators(roster, eligible):
    rows = []
    for seed in SEEDS:
        anchor_file = ANCHOR/f"seed{seed}_ANCHOR.json"
        anchor = read(anchor_file)["anchor"]["val"]
        check_anchor(anchor, anchor["r"], roster, eligible)
        rows.append(dict(method="Transformer FT+Ccap", seed=seed, personal=False, source_sha256=digest(anchor_file),
                         **{k:anchor[k] for k in METRICS}))
    terminal_file = D26/"formal/COMPLETED.json"
    d26 = read(terminal_file)
    need(d26["passed"] and d26["protocol_sha256"] == D26_GATE and len(d26["evaluations"]) == 9
         and {(r["seed"],r["arm"]) for r in d26["evaluations"]} == {(s,a) for s in SEEDS for a in ("HH","CH","CC")},
         "Frozen original D26 nine results")
    for r in d26["evaluations"]:
        seed, arm = r["seed"], r["arm"]
        route = "near" if arm=="HH" else "native"
        score_file = D26/f"formal/seed{seed}/READOUT_{arm}/SCORE_{route}.json"
        score = read(score_file)["Ccap"]
        points = list(checked_blocks(score["segments"],roster,eligible,pure=False).values())
        agg = aggregate(points)
        assert_metrics(agg, score["phases"]["val"], "D26 full support comparator")
        need(same(agg["r"],r["VAL_r"]["Ccap"]), "D26 final Ccap comparator")
        rows.append(dict(method=f"D26 {arm}",seed=seed,personal=False,source_sha256=digest(score_file),**{k:agg[k] for k in METRICS}))
    d28file = D28/"formal/COMPLETED.json"
    d28review = read(D28/"COORDINATOR_RESULT_REVIEW.json")
    d28 = read(d28file)
    need(d28["passed"] and d28review["passed"] and d28review["completed_sha256"] == digest(d28file)
         and d28review["exact_original_support"] and d28["VAL_eligible_blocks"] == 752, "Independently reviewed D28 same support")
    for name in ("G","I"):
        rows.append(dict(method=f"D28 {name}",seed=None,personal=name=="I",source_sha256=digest(d28file),**d28["scores"][name]))
    historical = read(D26_HELPERS/"HISTORICAL_REFERENCES.json")
    means = [dict(method=method, **{k:mean([r[k] for r in rows if r["method"]==method]) for k in METRICS})
             for method in dict.fromkeys(r["method"] for r in rows)]
    return dict(seed_rows=rows,means=means,d26_frozen_selected_arm=d26["selected_full_arm"],historical_r_only=historical)


def generate(root, out):
    formal, terminal, inventory, mask, gate, terminal_sha = admission(root)
    summary, tables, costs, epochs, roster, eligible = summarize(formal,terminal,inventory,mask,gate["settings"])
    expected = {(seed,"initial",None) for seed in SEEDS} | {(seed,"personal",s) for seed in SEEDS for s in inventory["subjects"]}
    summary.update(protocol_sha256=GATE,formal_completed_sha256=terminal_sha,VAL_people=49,VAL_blocks=752,
                   same_support_comparators=comparators(roster,eligible),**budget(costs,inventory,terminal),
                   process_ledger=worker_accounting(root,formal,expected,costs),engineering=read(root/"ENGINEERING.json"),
                   scope="Only frozen scalar receipts, file sizes and saved block scores; no weights, raw signals, fitting or forward.")
    out.mkdir()
    save(out/"RESULT_SUMMARY.json",summary)
    save(out/"FIT_COSTS.json",costs)
    save_csv(out/"TRAINING_EPOCHS.csv",epochs)
    for name,rows in tables.items():
        save_csv(out/(name.upper()+".csv"),rows)
    from pub_01_s03_ss07_seawave_seen_report_figures import draw
    draw(out,summary,tables,epochs,costs)
    save(out/"REPORT_ARTIFACTS.json",dict(passed=True,protocol_sha256=GATE,formal_completed_sha256=terminal_sha,
             artifacts={p.name:digest(p) for p in sorted(out.iterdir())},D26_helper_hashes=HELPER_HASHES,
             sources={p.name:digest(p) for p in Path(__file__).parent.glob("*.py")},
             signal_reads=0,weight_reads=0,model_forwards=0,new_fits=0))


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",required=True,type=Path)
    generate(ROOT,cli.parse_args().output)
