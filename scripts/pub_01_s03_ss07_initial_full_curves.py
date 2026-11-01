"""Render completed initial15 train/VAL metadata on2203, never fit or score.

Checkpoint criterion is window-level3s; family selection is60s/person VAL.
These selected-on-VAL results are descriptive, not independent confirmation.
"""
import csv
import json
import math
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pub_01_s03_ss07_baseline import ROOT, read, dump, sha


def main():
    protocol = read(ROOT/'INCREMENTAL_PROTOCOL.json')
    widths = read(ROOT/'FULL_SELECTION.json')
    primary = read(ROOT/'FULL_PRIMARY_SELECTION.json')
    ema = read(ROOT/'EMA_SELECTION.json')
    seeds = protocol['screening_seeds']; assert seeds == [1024, 2026, 31415]
    assert ema['screening_seeds'] == seeds and ema['tau_seconds'] == 0
    ids = widths['selected_family_condition_ids']
    assert ids == [f'm{i}_h16' for i in range(1, 6)] and primary['primary'] == 'm4_h16'
    assert primary['five_seed_primary_not_reselected_after_extra_seeds']
    colors = ['#0072B2', '#D55E00', '#009E73']
    rows = []; histories = {}; sources = {}
    for name in ['INCREMENTAL_PROTOCOL.json', 'FULL_SELECTION.json', 'FULL_PRIMARY_SELECTION.json', 'EMA_SELECTION.json']:
        sources[name] = sha(ROOT/name)
    for id_ in ids:
        for seed in seeds:
            folder = ROOT/f'full/{id_}_seed{seed}'
            completed = read(folder/'completed.json'); history = read(folder/'history.json')
            assert completed['seed'] == seed and completed['stage'] == 'full' and not completed['test_read']
            assert completed['condition']['id'] == id_ and completed['protocol_sha256'] == sha(ROOT/'INCREMENTAL_PROTOCOL.json')
            assert len(history) == completed['epochs'] < 250 and history[-1]['stale'] == 10
            assert completed['updates'] == 3945*len(history)
            losses = [h['validation_negative_Pearson'] for h in history]
            winner = min(range(len(losses)), key=lambda i: (losses[i], i))
            assert winner == completed['selected_epoch'] and losses[winner] == completed['best_validation_negative_Pearson']
            assert all(h['epoch'] == i and h['updates'] == (i+1)*3945 for i, h in enumerate(history))
            assert all(math.isfinite(h[k]) for h in history for k in ['train_negative_Pearson', 'validation_negative_Pearson'])
            score = completed['validation_macro']['macro_r']; native = ema['paired_scores']['0'][seeds.index(seed)]
            rows.append({'condition': id_, 'seed': seed, 'epochs': len(history), 'updates': completed['updates'],
                'selected_epoch_0based': winner, 'stopping': 'PATIENCE10',
                'selected_validation_mean3s_r': -losses[winner], 'selected_validation60s_person_r': score,
                'native_P_validation60s_person_r': native, 'validation60s_difference_vs_native_P': score-native})
            histories[(id_, seed)] = history
            for p in [folder/'completed.json', folder/'history.json']:
                sources[str(p.relative_to(ROOT))] = sha(p)
    assert len(rows) == 15
    means = {id_: math.fsum(r['selected_validation60s_person_r'] for r in rows if r['condition'] == id_)/3 for id_ in ids}
    assert means == primary['full_family_validation_means']
    destination = ROOT/'initial_full_curves02'; destination.mkdir(exist_ok=False)
    with (destination/'FULL_INITIAL15_CURVE_SUMMARY.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (destination/'FULL_INITIAL15_HISTORY.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['condition', 'seed', 'epoch_1based', 'train_mean3s_r',
            'validation_mean3s_r', 'updates', 'stale', 'selected_checkpoint'])
        w.writeheader()
        for row in rows:
            for h in histories[(row['condition'], row['seed'])]:
                w.writerow({'condition': row['condition'], 'seed': row['seed'], 'epoch_1based': h['epoch']+1,
                    'train_mean3s_r': -h['train_negative_Pearson'], 'validation_mean3s_r': -h['validation_negative_Pearson'],
                    'updates': h['updates'], 'stale': h['stale'], 'selected_checkpoint': h['epoch'] == row['selected_epoch_0based']})
    fig, axes = plt.subplots(5, 2, figsize=(10.4, 12.2), layout='constrained')
    for i, id_ in enumerate(ids):
        for seed, color in zip(seeds, colors):
            history = histories[(id_, seed)]; epochs = [h['epoch']+1 for h in history]
            row = next(r for r in rows if r['condition'] == id_ and r['seed'] == seed)
            for j, metric in enumerate(['train_negative_Pearson', 'validation_negative_Pearson']):
                axes[i, j].plot(epochs, [-h[metric] for h in history], color=color, lw=1.25, label=str(seed))
            axes[i, 1].scatter(row['selected_epoch_0based']+1, row['selected_validation_mean3s_r'],
                marker='s', s=40, facecolor='none', edgecolor=color, linewidth=1.3, zorder=4)
        for j, label in enumerate(['Train', 'Validation']):
            axes[i, j].set(title=f'{id_} | {label}', xlabel='Epoch (1-based)', ylabel='Mean 3 s correlation')
            axes[i, j].spines[['top', 'right']].set_visible(False)
            axes[i, j].grid(axis='y', alpha=.18, lw=.5)
            axes[i, j].tick_params(labelsize=8); axes[i, j].set_axisbelow(True)
            axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 0].legend(title='Paired seed', frameon=False, fontsize=8, title_fontsize=8)
    fig.suptitle('Initial15 full-budget candidates | saved train/VAL logs\n'
        'All stop by patience10; open squares select the earliest best3s VAL checkpoint\n'
        'Train: online batch means. Validation: fixed checkpoint pass.', fontsize=11)
    fig.savefig(destination/'FULL_INITIAL15_CURVES.png', dpi=180, facecolor='white'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), layout='constrained')
    differences = []
    for i, id_ in enumerate(ids):
        ds = [next(r for r in rows if r['condition'] == id_ and r['seed'] == seed)['validation60s_difference_vs_native_P'] for seed in seeds]
        for shift, seed, color, delta in zip([-.10, 0, .10], seeds, colors, ds):
            ax.scatter(i+shift, delta, color=color, s=38, label=str(seed) if i == 0 else None, zorder=4)
        mean = math.fsum(ds)/3; differences.append(mean)
        ax.plot([i-.18, i+.18], [mean, mean], color='black', lw=2.0, zorder=3)
        ax.annotate(f'mean {mean:+.6f}', (i, min(ds)), xytext=(0, -16), textcoords='offset points', ha='center', fontsize=8)
    ax.axhline(0, color='#555555', lw=1, linestyle='--')
    ax.set_xticks(range(5), ['M1', 'M2', 'M3', 'M4\n(frozen primary)', 'M5'])
    ax.set(xlabel='H16 full-budget family (fixed order)', ylabel='VAL60s/person r difference vs native P',
        title='All five family means below native P on first3seed VAL\n'
              'Dots: paired seeds; black bars: means, no uncertainty intervals')
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.18, lw=.5)
    ax.set_axisbelow(True); ax.margins(y=.22)
    ax.legend(title='Paired seed', frameon=False, ncol=3, fontsize=8, title_fontsize=8, loc='lower left')
    fig.savefig(destination/'FULL_INITIAL15_VAL60S.png', dpi=200, facecolor='white'); plt.close(fig)
    dump(destination/'FULL_INITIAL15_CURVE_SNAPSHOT.json', {'time': time.time(), 'sources': sources,
        'source_sha256': sha(ROOT/'input/pub_01_s03_ss07_initial_full_curves.py'), 'rows': rows,
        'initial_full_cells': 15, 'histories_only_no_signal_checkpoint_prediction_or_test_reads': True,
        'train_curves_are_online_batch_means_not_frozen_checkpoint_training_evaluations': True,
        'train_window_pool_and_VAL_window_pool_are_different': True,
        'VAL_used_for_checkpoint_and_family_selection_not_independent_confirmation': True,
        'mean_VAL60s_differences_vs_native_P': dict(zip(ids, differences)),
        'P_reference_is_original_prediction_exact_EMA_tau0_alias': True,
        'no_training_selection_budget_split_or_frozen_source_changes': True,
        'no_global_convergence_causal_or_test_superiority_claim': True})
    print(json.dumps({'rendered_completed_initial_full15': True, 'destination': str(destination)}))


if __name__ == '__main__': main()
