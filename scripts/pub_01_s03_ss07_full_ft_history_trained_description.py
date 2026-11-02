"""Describe existing trained curves separately from selected zero incumbents; no NN."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_full_ft_history_20261102');out=ROOT/'report01'
read=lambda p:json.loads(p.read_text())
rows=list(csv.DictReader((out/'FIT_RESULTS.csv').open()));assert len(rows)==9
for row in rows:
    folder=ROOT/f"fits/{row['arm']}_seed{row['seed']}";h=[json.loads(x) for x in (folder/'HISTORY.jsonl').read_text().splitlines()];assert len(h)==20
    anchor=read(folder/'INCUMBENT.json')['phases']['val']['r'];best=max(h,key=lambda x:x['validation']['r'])
    row.update(best_trained_epoch=best['epoch'],best_trained_r=best['validation']['r'],best_trained_delta_to_anchor=best['validation']['r']-anchor,
        best_trained_validation_MSE=best['validation']['MSE'],last_trained_r=h[-1]['validation']['r'],last_validation_MSE=h[-1]['validation']['MSE'],
        online_TRAIN_first_MSE=h[0]['train_window_MSE'],online_TRAIN_last_MSE=h[-1]['train_window_MSE'])
with (out/'FIT_RESULTS_WITH_TRAINED.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
description=dict(best_trained_deltas=[dict(arm=r['arm'],seed=int(r['seed']),epoch=r['best_trained_epoch'],r=r['best_trained_r'],delta=r['best_trained_delta_to_anchor']) for r in rows],
    no_new_checkpoint_selection_or_replay=True,NN=0,original_FIT_RESULTS_retained=True)
with (out/'TRAINED_DESCRIPTION.json').open('x') as f:json.dump(description,f,indent=2)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');labels=[f"{r['arm']}\n{r['seed']}" for r in rows]
axes[0].bar(range(9),[float(r['wall_seconds'])/60 for r in rows],color=['#457b9d','#e9a23b','#b23a48']*3)
axes[0].set(xticks=range(9),xticklabels=labels,ylabel='Fit elapsed minutes',title='Each fit: 20 epochs, 78,900 updates; parallel elapsed not summed')
values=[int(r['selected_epoch'])+1 for r in rows];axes[1].bar(range(9),values,color=['#457b9d','#e9a23b','#b23a48']*3)
axes[1].scatter(range(9),values,color='black',s=35,zorder=5,clip_on=False)
axes[1].set(xticks=range(9),xticklabels=labels,ylim=(0,21),ylabel='Selected added epoch (0 = incumbent)',title='Dots at zero explicitly mark selected incumbents')
path=out/'figures/04_COSTS_AND_SELECTED_EPOCHS_v2.png';assert not path.exists();fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
with (out/'TRAINED_DESCRIPTION_PROVENANCE.json').open('x') as f:json.dump(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    numeric_change='Add descriptive best-trained/last rows from all existing epochs; no new model choice',
    display_change='Explicit markers for observed zero-height selected-epoch bars; original image retained',NN=0),f,indent=2)
print(json.dumps(description),flush=True)
