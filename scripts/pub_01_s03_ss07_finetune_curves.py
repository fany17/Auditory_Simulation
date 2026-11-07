"""Completed fixed FT validation/budget snapshot; metadata only on2203."""
import csv
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pub_01_s03_ss07_baseline import ROOT, read, dump, sha


def main():
    protocol = read(ROOT/'FINETUNE_PROTOCOL.json')
    assert protocol['frozen'] and protocol['budget']['max_epochs'] == 5
    rows = []; curves = []; sources = {}
    for seed in protocol['seeds']:
        folder = ROOT/f'finetune/native_P_seed{seed}'
        completed = read(folder/'completed.json'); history = read(folder/'history.json')
        initial = read(folder/'INCUMBENT_EQUIVALENCE.json')
        head = read(ROOT/f'baseline/head_seed{seed}/completed.json')
        assert completed['seed'] == seed and not completed['test_read']
        assert completed['protocol_sha256'] == sha(ROOT/'FINETUNE_PROTOCOL.json')
        assert len(history) == completed['epochs'] == 5
        assert completed['updates'] == 5*3945 == history[-1]['updates']
        assert completed['target_window_presentations'] == 5*252454
        assert completed['validation_passes_including_incumbent'] == 6
        assert abs(completed['initial_validation_negative_Pearson']-head['best_validation']) <= 1e-4
        losses = [completed['initial_validation_negative_Pearson']] + [h['validation_negative_Pearson'] for h in history]
        selected = min(range(len(losses)), key=lambda i: (losses[i], i))-1
        assert completed['selected_epoch'] == selected
        assert completed['best_validation_negative_Pearson'] == min(losses)
        start = -losses[0]; finish = -min(losses)
        rows.append({'seed': seed, 'epochs': completed['epochs'], 'optimizer_updates': completed['updates'],
            'selected_epoch_0based_minus1_incumbent': selected,
            'initial_validation_3s_r_raw_global64': start, 'selected_validation_3s_r': finish,
            'validation_3s_r_gain_descriptive': finish-start,
            'extra_target_window_presentations': completed['target_window_presentations'],
            'validation_passes_including_incumbent': completed['validation_passes_including_incumbent'],
            'FT_trainable_parameters': completed['registered_trainable_parameters'],
            'extra_phase_runtime_seconds_not_isolated': completed['seconds'],
            'initial_vs_original_cached_validation_absolute_difference': abs(losses[0]-head['best_validation']),
            'stopping': 'FIXED_EPOCH_CEILING5'})
        curves.append([-value for value in losses])
        for p in [folder/'completed.json', folder/'history.json', folder/'INCUMBENT_EQUIVALENCE.json']:
            sources[str(p)] = sha(p)
    destination = ROOT/'finetune_curve_snapshot01'; destination.mkdir(exist_ok=False)
    fig, ax = plt.subplots(figsize=(7.4, 4.0), layout='constrained')
    for row, curve, color in zip(rows, curves, ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']):
        ax.plot(range(6), curve, marker='o', ms=3.5, color=color, lw=1.5, label=str(row['seed']))
        ax.scatter(row['selected_epoch_0based_minus1_incumbent']+1, row['selected_validation_3s_r'],
                   s=55, marker='s', facecolor='none', edgecolor=color, zorder=4)
    ax.set(xlabel='Additional fine-tuning epochs (0 = native P incumbent)',
           ylabel='Mean 3 s validation correlation', title='Fixed native encoder + head fine-tuning')
    ax.set_xticks(range(6)); ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    ax.legend(title='Paired seed', ncol=2, frameon=False, fontsize=9, title_fontsize=9)
    fig.savefig(destination/'FINETUNE_CURVES.png', dpi=220, facecolor='white'); plt.close(fig)
    with (destination/'FINETUNE_BUDGET.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dump(destination/'FINETUNE_SNAPSHOT.json', {'time': time.time(), 'all_five_FT_completed': True,
        'protocol_sha256': sha(ROOT/'FINETUNE_PROTOCOL.json'), 'sources': sources, 'rows': rows,
        'only_history_and_completed_metadata_no_signals_or_test_scores': True,
        'native_pretraining_and_head_costs_are_additional_separate_upstream_costs': True,
        'validation_mean3s_not_60s_person_macro_test_endpoint': True,
        'no_convergence_or_test_superiority_claim': True,
        'all_selected_at_final_budget_epoch': all(r['selected_epoch_0based_minus1_incumbent'] == 4 for r in rows),
        'source_sha256': sha(ROOT/'input/pub_01_s03_ss07_finetune_curves.py')})
    print(json.dumps({'five_FT_completed': True, 'epochs_each': 5, 'optimizer_updates_each': 19725,
                      'selected_validation_3s_r': [r['selected_validation_3s_r'] for r in rows]}))


if __name__ == '__main__': main()
