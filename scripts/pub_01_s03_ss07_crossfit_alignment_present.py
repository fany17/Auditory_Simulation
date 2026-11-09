"""Summarize D25 saved scores, without loading any signal or model."""
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT=Path("/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_crossfit_source_alignment_20261106")
SEEDS=(1024,2026,31415)
EPOCHS=(0,19)
VARIANTS=("B","S")


def csv_rows(name):
    with (ROOT/name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    completion=json.loads((ROOT/"COMPLETED.json").read_text())
    assert completion["passed"] and completion["new_optimizer_updates"]==0
    data=csv_rows("RESULTS.csv")
    per_record=csv_rows("RECORD_SCORES.csv")
    assert len(data)==36 and len(per_record)==1335
    item=lambda group,seed,variant,epoch: next(row for row in data
        if (row["group"],int(row["seed"]),row["variant"],int(row["epoch"]))==(group,seed,variant,epoch))
    out={}
    for group in ("all","shared","unseen"):
        out[group]={}
        for variant in VARIANTS:
            for epoch in EPOCHS:
                rows=[item(group,s,variant,epoch) for s in SEEDS]
                anchor_keys=("fold_anchor_r","fold_model_r","fold_delta_r","fold_anchor_active_MSE",
                             "fold_model_active_MSE","residual_RMS")
                full_keys=("full_r","full_anchor_r","full_delta_r","delta_of_deltas")
                cols=anchor_keys+full_keys if group=="all" else anchor_keys
                average={k:mean(float(row[k]) for row in rows) for k in cols}
                average["above_fold_anchor"]=sum(float(row["fold_delta_r"])>0 for row in rows)
                average["fold_active_MSE_difference"]=mean(float(row["fold_model_active_MSE"])-
                                                         float(row["fold_anchor_active_MSE"]) for row in rows)
                if group=="all":
                    average["relative_loss_smaller_than_full"]=sum(float(row["delta_of_deltas"])>0 for row in rows)
                if variant=="B":
                    average["B_minus_S"]=mean(float(row["fold_model_r"])-
                                             float(item(group,int(row["seed"]),"S",epoch)["fold_model_r"])
                                             for row in rows)
                out[group][f"{variant}_{epoch}"]=average
    scales={}
    for variant in VARIANTS:
        scale_rows=[r for r in per_record if r["epoch"]=="0" and r["variant"]==variant]
        count=Counter((r["seed"],r["record"]) for r in scale_rows)
        points=lambda r:int(r["active_points"])/count[r["seed"],r["record"]]
        n=sum(points(r) for r in scale_rows)
        scales[variant]={name:math.sqrt(sum(float(r[f"standardized_{name}_RMS"])**2*points(r)
                                           for r in scale_rows)/n) for name in ("current","context")}
    with (ROOT/"SUMMARY.json").open("x") as stream:
        json.dump(dict(metrics=out, standardized_RMS_matched_epoch0=scales),stream,indent=2)
        stream.write("\n")
    os.environ["MPLCONFIGDIR"]=str(ROOT/"matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(11.8,3.9),sharey=True,layout="constrained")
    colors={"B":"#a95520","S":"#005d86"}
    for seed,ax in zip(SEEDS,axes):
        ax.axhline(0,color="#7b8996",lw=.8)
        for variant in VARIANTS:
            rows=[item("all",seed,variant,epoch) for epoch in EPOCHS]
            for k,style,marker in (("fold_delta_r","-","o"),("full_delta_r","--","s")):
                ax.plot(EPOCHS,[1000*float(row[k]) for row in rows],marker=marker,
                        linestyle=style,lw=1.7,color=colors[variant])
        ax.set_xticks(EPOCHS,("0","19"))
        ax.set_xlabel("Fixed training epoch")
        ax.set_title(f"Seed {seed}",loc="left",fontweight="bold")
        ax.grid(axis="y",color="#e3e8ee",lw=.6)
    axes[0].set_ylabel("Development VAL delta to own anchor (x1000)")
    handles=[Line2D([0],[0],color=colors[x],label=x,lw=2) for x in VARIANTS]
    handles.extend((Line2D([0],[0],color="#455c6e",marker="o",label="Fold-matched input"),
                    Line2D([0],[0],color="#455c6e",marker="s",linestyle="--",label="Full-TRAIN input")))
    axes[0].legend(handles=handles,frameon=False,fontsize=8,loc="lower left")
    fig.savefig(ROOT/"FIXED_SOURCE_ALIGNMENT.png",dpi=180,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    main()
