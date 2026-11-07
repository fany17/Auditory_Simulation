"""Figures from saved D27 scores and fitted-stage metadata."""
from collections import Counter, defaultdict
from math import fsum

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = (1024,2026,31415)
STAGES = ("initial","personal")
COLORS = {"initial":"#23577b","personal":"#a56720"}


def average(values):
    return fsum(values)/len(values)


def main_scores(out, summary, tables):
    references = summary["same_support_comparators"]["seed_rows"]
    series = {}
    for label in ("Transformer FT+Ccap","D26 HH","D26 CH","D26 CC","D28 G","D28 I"):
        rows = [r for r in references if r["method"]==label]
        series[label] = rows, rows[0]["personal"]
    for stage in STAGES:
        series["Sea-Wave "+stage] = [r for r in tables["seed_scores"] if r["stage"]==stage], stage=="personal"
    fig, axes = plt.subplots(1,2,figsize=(11,5),layout="constrained")
    labels = list(series)
    for axis, key, title in zip(axes,("r","MSE"),("49-person macro Pearson r","49-person macro MSE")):
        for i,(rows,personal) in enumerate(series.values()):
            color=COLORS["personal" if personal else "initial"]
            values=[r[key] for r in rows]
            if len(values)==3:
                axis.scatter(values,[i-.08,i,i+.08],color=color,s=21,alpha=.5)
            axis.scatter(average(values),i,color=color,s=42,marker="D")
        axis.set(xlabel=title,yticks=range(len(labels)),yticklabels=labels)
        axis.invert_yaxis()
        axis.grid(axis="x",alpha=.2)
    fig.savefig(out/"val_same_support.png",dpi=175)
    plt.close(fig)


def adaptation(out, pairs):
    stories=sorted({r["story"] for r in pairs})
    fig, axes=plt.subplots(2,2,figsize=(12,7),layout="constrained")
    for axis,key,title in zip(axes[0],("delta_r","delta_MSE"),("Personal minus shared Pearson r","Personal minus shared MSE")):
        for j,story in enumerate(stories):
            rows=[r for r in pairs if r["story"]==story]
            by_person=defaultdict(list)
            for r in rows:
                by_person[r["subject"]].append(r[key])
            points=[average(values) for values in by_person.values()]
            axis.scatter(points,[j]*len(points),color=COLORS["initial"],s=21,alpha=.65)
            axis.scatter(average(points),j,color=COLORS["personal"],s=45,marker="D")
        axis.axvline(0,color="#46515b",lw=1)
        axis.set(yticks=range(len(stories)),yticklabels=stories,xlabel=title)
        axis.grid(axis="x",alpha=.18)
    axis=axes[1,0]
    for i,seed in enumerate(SEEDS):
        counts=Counter(r["personal_epoch"] for r in pairs if r["seed"]==seed)
        axis.bar([j+(i-1)*.23 for j in range(-1,21)],[counts[j] for j in range(-1,21)],.2,label=str(seed))
    axis.set(xlabel="Selected personal epoch (-1 = initial)",ylabel="Participants")
    axis.legend(fontsize=8)
    axis=axes[1,1]
    for i,seed in enumerate(SEEDS):
        rows=[r for r in pairs if r["seed"]==seed]
        axis.scatter([r["personal_epochs_run"] for r in rows],[r["personal_epoch"] for r in rows],
                     label=str(seed),s=17,alpha=.5)
    axis.set(xlabel="Actual personal epochs run",ylabel="Selected personal epoch")
    axis.grid(alpha=.18)
    fig.savefig(out/"personal_adaptation.png",dpi=170)
    plt.close(fig)


def training(out, epochs, fits):
    indexed=defaultdict(list)
    for row in epochs:
        indexed[row["seed"],row["stage"],row["subject"]].append(row)
    by_fit={(r["seed"],r["stage"],r["subject"]):r for r in fits}
    fig, axes=plt.subplots(3,2,figsize=(10.5,8.4),layout="constrained")
    for i,seed in enumerate(SEEDS):
        initial=indexed[seed,"initial",None]
        axis=axes[i,0]
        axis.plot([r["epoch"] for r in initial],[r["r"] for r in initial],color=COLORS["initial"])
        chosen=by_fit[seed,"initial",None]
        axis.scatter(chosen["selected_epoch"],chosen["selected_r"],color=COLORS["personal"],s=32,zorder=3)
        axis.set(title=f"{seed} / shared initial",ylabel="VAL Pearson r")
        axis.grid(alpha=.18)
        axis=axes[i,1]
        for (s,stage,subject),rows in indexed.items():
            if s!=seed or stage!="personal":
                continue
            y0=rows[0]["r"]
            axis.plot([r["epoch"] for r in rows],[r["r"]-y0 for r in rows],color=COLORS["initial"],alpha=.18,lw=.8)
            chosen=by_fit[s,stage,subject]
            axis.scatter(chosen["selected_epoch"],chosen["selected_r"]-y0,color=COLORS["personal"],s=9,alpha=.5,zorder=3)
        axis.axhline(0,color="#46515b",lw=.7)
        axis.set(title=f"{seed} / 49 personal fits",ylabel="VAL r minus epoch -1")
        axis.grid(alpha=.18)
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    fig.savefig(out/"training_selection.png",dpi=170)
    plt.close(fig)


def draw(out,summary,tables,epochs,costs):
    main_scores(out,summary,tables)
    adaptation(out,tables["pairs"])
    training(out,epochs,costs)
