"""Render D21 scalar comparisons and server-resident coefficient arrays."""

import csv

from pub_01_s03_ss07_output_ridge import D20, FIRST, LAMBDAS, ROOT, SEEDS, read, sha, write


def table(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gather(fit, np):
    grid, people, selected = [], [], []
    d20 = read(D20 / "fit/COMPLETED.json")["seeds"]
    for seed in SEEDS:
        x = read(fit / f"seed{seed}_METRICS.json")
        anchor = x["anchor"]
        with np.load(fit / f"seed{seed}_COEFFICIENTS_AND_GRAM.npz", allow_pickle=False) as weights:
            for name, model in x["models"].items():
                v = model["val"]
                grid.append(dict(seed=seed, model=name, role=model["role"], penalty=model["penalty"],
                                 r=v["r"], delta_r=v["r"] - anchor["r"], anchor_r=anchor["r"],
                                 D20_r=d20[str(seed)]["selected_r"], full_record_r=v["full_record_r"],
                                 train_active_MSE=model["train_active_MSE"],
                                 train_anchor_active_MSE=x["train_anchor_active_MSE"],
                                 val_active_MSE=v["active_MSE"], val_anchor_active_MSE=anchor["active_MSE"],
                                 val_block_MSE=v["MSE"], val_anchor_block_MSE=anchor["MSE"],
                                 intercept=model["intercept"], physical_intercept=float(weights[name + "_physical_intercept"]),
                                 current_gain=model["current_gain"], past_gain_sum=model["past_gain_sum"],
                                 past_gain_L1=model["past_gain_L1"], beta_l2=model["beta_l2"],
                                 normal_abs=model["normal_abs"], objective=model["objective"]))
        for name, metric in [("anchor", anchor)] + [(k, m["val"]) for k, m in x["models"].items()]:
            for row, base in zip(metric["people"], anchor["people"], strict=True):
                assert row["subject"] == base["subject"]
                people.append(dict(seed=seed, model=name, **row, anchor_r=base["r"],
                                   delta_r=row["r"] - base["r"]))
        for role, s in x["selection"].items():
            chosen = s["selected"]
            metric = anchor if chosen is None else x["models"][chosen]["val"]
            selected.append(dict(seed=seed, role=role, selected=chosen or "anchor", r=s["r"],
                                 delta_r=s["delta_r"], delta_D20=s["r"] - d20[str(seed)]["selected_r"],
                                 val_active_MSE=metric["active_MSE"], full_record_r=metric["full_record_r"]))
    return grid, people, selected


def charts(fit, out, grid, np, plt):
    colors = ("#627882", "#a85421", "#005a8c")
    labels = ("1", "0.01", "0.0001")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 3.8), layout="constrained")
    for ax, seed in zip(axes, SEEDS):
        rows = [row for row in grid if row["seed"] == seed]
        for role, color, symbol in (("A", "#a85421", "o"), ("F", "#005a8c", "s")):
            values = [1000 * next(x["delta_r"] for x in rows if x["role"] == role and x["penalty"] == l)
                      for l in LAMBDAS]
            ax.plot(range(3), values, marker=symbol, color=color, label=role, linewidth=1.5)
        reference = 1000 * (rows[0]["D20_r"] - rows[0]["anchor_r"])
        ax.axhline(reference, color="#657a29", linestyle="--", lw=1.2, label="D20 selected")
        ax.axhline(0, color="#728394", lw=.9, label="Anchor")
        ax.set_xticks(range(3), labels)
        ax.set_title(f"Seed {seed}", loc="left", fontweight="bold")
        ax.set_xlabel("Ridge penalty")
        ax.set_ylabel("Development VAL Δr × 1,000")
        ax.grid(axis="y", color="#e7ecf0")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out / "VAL_GRID.png", dpi=180, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, sharey="row", figsize=(12.6, 6.9), layout="constrained")
    tap = np.arange(-384, 0) / 64
    for i, seed in enumerate(SEEDS):
        with np.load(fit / f"seed{seed}_COEFFICIENTS_AND_GRAM.npz", allow_pickle=False) as weights:
            for l, color, label in zip(LAMBDAS, colors, labels):
                name = f"F_{l:g}"
                axes[0, i].step(tap[3::4], weights[name + "_beta"][1:], where="mid",
                                color=color, label=f"λ {label}", lw=1.3)
                axes[1, i].step(tap, weights[name + "_physical_kernel"][:-1] * 1000,
                                where="mid", color=color, lw=1.3)
        for j, label in enumerate(("Standardized history β", "Physical gain per sample × 1,000")):
            ax = axes[j, i]
            ax.axhline(0, color="#728394", lw=.75)
            ax.set_title(f"Seed {seed}", loc="left", fontweight="bold")
            ax.set_xlabel("Seconds relative to target")
            ax.set_ylabel(label)
            ax.grid(axis="y", color="#e7ecf0")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(out / "FIR_COEFFICIENTS_AND_KERNEL.png", dpi=180, facecolor="white")
    plt.close(fig)


def archive_index(fit, out, np):
    rows = []
    for seed in SEEDS:
        metric = read(fit / f"seed{seed}_METRICS.json")
        anchor = {(r["record"], r["start"]): r["r"] for r in metric["anchor"]["blocks"]}
        mixed, active = [], []
        for block in metric["models"]["A_0.0001"]["val"]["blocks"]:
            group = mixed if block["start"] < FIRST else active
            group.append(block["r"] - anchor[block["record"], block["start"]])
        path = fit / f"seed{seed}_COEFFICIENTS_AND_GRAM.npz"
        with np.load(path, allow_pickle=False) as weights:
            history = weights["F_0.0001_physical_kernel"][:-1]
            peak = int(np.argmax(np.abs(history)))
            tap = float(history[peak])
        rows.append(dict(seed=seed, remote_array_file=str(path), bytes=path.stat().st_size, sha256=sha(path),
                         mixed_A_blocks=len(mixed), mixed_A_delta_r_mean=float(np.mean(mixed)),
                         active_only_A_blocks=len(active), active_only_A_delta_r_max_abs=float(np.max(np.abs(active))),
                         nominal_F_peak_tap_lag=peak - len(history), nominal_F_peak_tap=tap))
    write(out / "ARRAYS_AND_PREFIX.json", rows)


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = ROOT / "fit"
    done = read(fit / "COMPLETED.json")
    assert done["passed"] and done["counters"]["exact_ridge_solves"] == 18
    assert not (fit / "FAILURE.json").exists()
    out = ROOT / "presentation"
    out.mkdir()
    grid, people, selected = gather(fit, np)
    table(out / "GRID.csv", grid)
    table(out / "PEOPLE.csv", people)
    table(out / "SELECTED.csv", selected)
    charts(fit, out, grid, np, plt)
    archive_index(fit, out, np)


if __name__ == "__main__":
    main()
