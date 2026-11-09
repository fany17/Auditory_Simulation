"""Present complete, independently reviewed SS07 summaries on2203 only.

No signal/checkpoint/prediction-array reader, fitting, model selection, new
intervals or goal completion. Fixed manifest order; missing endpoints stay
missing. The output is a numerical reading aid, not scientific acceptance.
"""
import json
import math
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pub_01_s03_ss07_baseline import ROOT, read, dump, sha
from pub_01_s03_ss07_review_metrics import gated

FILES = ['INTERNAL_MAIN.json', 'EXTERNAL_DTU_MAIN.json',
         'INTERNAL_FIVE_FAMILY_EXPLORATORY.json', 'INTERNAL_HISTORY_DIAGNOSTICS.json', 'COSTS.json']
STAGES = ['shared_pretrain', 'native_head', 'screen', 'full', 'finetune']
KINDS = ['reset', 'content_blind', 'shuffled', 'older_outside_fused_support',
         'same_person_wrong_family', 'same_story_wrong_person']
FIGURES = ['01_MAIN_CONTRASTS.png', '02_FIVE_FAMILY_EXPLORATORY.png',
           '03_HISTORY_DIAGNOSTICS.png', '04_ABSOLUTE_PER_SEED.png', '05_SEARCH_COSTS.png']


def reviewed_gate():
    done = read(ROOT/'INDEPENDENT_REVIEW_COMPLETED.json')
    assert done['technical_review_passed'] and not done['scientific_goal_complete']
    manifest, ready = gated()
    assert done['protocol_sha256'] == sha(ROOT/'INDEPENDENT_REVIEW_PROTOCOL.json')
    for path in ['independent_raw_review/RAW_INPUT_REVIEW.json', 'independent_review/METRICS_REVIEW.json']:
        proof = read(ROOT/path)
        assert proof['passed'] and proof['checkpoint_manifest_sha256'] == sha(ROOT/'FINAL_CHECKPOINT_MANIFEST.json')
    return manifest


def finite(value):
    assert math.isfinite(float(value)); return float(value)


def interval(value):
    assert len(value) == 2
    a, b = map(finite, value); assert a <= b
    return a, b


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=.15, linewidth=.5); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def forest(ax, rows, labels, point_key, interval_key, color, seed_key=None):
    assert len(rows) == len(labels)
    ax.axvline(0, color='#777777', lw=.8, zorder=0)
    for i, row in enumerate(rows):
        point = finite(row[point_key]); low, high = interval(row[interval_key])
        # Do not assume an approximate bootstrap interval contains its mean.
        ax.hlines(i, low, high, color=color, lw=2)
        ax.vlines([low, high], i-.09, i+.09, color=color, lw=1)
        ax.scatter(point, i, color=color, s=28, zorder=4)
        if seed_key:
            values = row[seed_key]; assert len(values) in (3, 5)
            ax.scatter([finite(v) for v in values], i+np.linspace(-.16, .16, len(values)),
                       color='#333333', alpha=.45, s=10, zorder=3)
    ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis(); style(ax)
    ax.set_xlabel('Correlation difference', fontsize=9)


def formatted(value):
    return f'{finite(value):.6f}'


def formatted_interval(value):
    low, high = interval(value); return f'[{low:.6f}, {high:.6f}]'


def render(destination, manifest, documents, fixture_label=None):
    """Metadata-only renderer, also usable with explicit synthetic fixtures."""
    internal, external, exploratory, history, costs = [documents[name] for name in FILES]
    comparisons = manifest['contrast_ids_and_comparators']; names = [r[0] for r in comparisons]
    labels = [r[1] for r in comparisons]; seeds = manifest['seeds']; ids = manifest['main_model_ids']
    assert len(names) in (6, 7) and len(set(names)) == len(names)
    assert seeds == [1024, 2026, 31415, 42, 12345]
    assert internal['primary'] == external['primary'] == manifest['primary']
    assert [r['contrast'] for r in internal['contrasts']] == [r['contrast'] for r in external['contrasts']] == names
    assert internal['M'] == external['M'] == len(names)
    assert internal['draws'] == external['draws'] == 20000
    if manifest['EMA_alias']['tau_seconds'] == 0:
        a, b = labels.index('native_P'), labels.index('P_EMA')
        for document, fields in [(internal, ['weighted_mean_difference',
                'weighted_Bonferroni_M_approx_bootstrap_interval', 'seed_paired_weighted_differences']),
                (external, ['mean_r_difference', 'Bonferroni_M_approx_interval', 'paired_seed_r_differences'])]:
            assert all(document['contrasts'][a][key] == document['contrasts'][b][key] for key in fields), 'tau0 contrast alias changed'
    title_prefix = (fixture_label+' | ') if fixture_label else ''
    destination.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.9+.3*(len(names)-6)))
    fig.subplots_adjust(left=.21, right=.98, bottom=.24, top=.82, wspace=.82)
    forest(axes[0], internal['contrasts'], labels, 'weighted_mean_difference',
           'weighted_Bonferroni_M_approx_bootstrap_interval', '#0072B2', 'seed_paired_weighted_differences')
    forest(axes[1], external['contrasts'], labels, 'mean_r_difference',
           'Bonferroni_M_approx_interval', '#D55E00', 'paired_seed_r_differences')
    axes[0].set_title('Sparr: internal exposed tests', fontsize=10)
    axes[1].set_title('DTU: fixed adaptation, no domain fit', fontsize=10)
    fig.suptitle(title_prefix+f'{manifest["primary"]} minus each fixed comparator', fontsize=11)
    fig.text(.02, .045, f'Intervals: Bonferroni M={len(names)} approximate crossed seed/person bootstrap (20,000 draws).\n'
        'Colored points: five-seed means; gray dots: individual paired seeds.\n'
        'Internal: 2/3 seen + 1/3 heldout person macro r, POST-HOC EXPLORATORY. DTU: trial r -> person -> seed.\n'
        'Separate comparison families; no joint internal/external FWER or finite-sample guarantee. DTU stimulus/room independence unproven.', fontsize=8)
    fig.savefig(destination/FIGURES[0], dpi=220, facecolor='white'); plt.close(fig)
    family_names = [id_+'-native_P' for id_ in manifest['exploratory_family_ids']]
    assert exploratory['M'] == 5 and [r['contrast'] for r in exploratory['contrasts']] == family_names
    fig, ax = plt.subplots(figsize=(8.1, 4.3)); fig.subplots_adjust(left=.26, right=.97, bottom=.25, top=.82)
    forest(ax, exploratory['contrasts'], manifest['exploratory_family_ids'], 'weighted_mean_difference',
           'weighted_Bonferroni_M_approx_bootstrap_interval', '#009E73', 'seed_paired_weighted_differences')
    fig.suptitle(title_prefix+'Five-family exploration minus native P (three seeds)', fontsize=11)
    fig.text(.03, .045, 'Bonferroni M=5 approximate intervals; green points: means; gray dots: three paired seeds.\n'
        'Separate prespecified exploration, not independent replication.\n'
        'Old internal tests exposed; not a reselection of the five-seed primary and not joint FWER with main comparisons.', fontsize=8)
    fig.savefig(destination/FIGURES[1], dpi=220, facecolor='white'); plt.close(fig)
    diagnostics = history['diagnostics']; assert [r['intervention'] for r in diagnostics] == KINDS
    history_labels = ['Reset', 'Content-blind', 'Shuffled', 'Older: recency + parity', 'Same person / wrong family', 'Same story / wrong person']
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.8), gridspec_kw={'width_ratios': [1.25, 1.]})
    fig.subplots_adjust(left=.23, right=.98, bottom=.23, top=.81, wspace=.42)
    axes[0].axvline(0, color='#777777', lw=.8); coverage = []; missing = []
    for i, row in enumerate(diagnostics):
        a = row['coverage']['test_seen']; b = row['coverage']['test_heldout']
        coverage.append([a['people'], b['people'], a['records']+b['records'], a['blocks']+b['blocks']])
        if 'weighted_mean_difference' in row:
            assert a['people'] > 0 and b['people'] > 0
            point = finite(row['weighted_mean_difference']); low, high = interval(row['exploratory_95_approx_interval'])
            axes[0].hlines(i-.06, low, high, color='#0072B2', lw=2)
            lo, hi = interval(row['crossed_Exp_sensitivity_not_CI'])
            axes[0].hlines(i+.09, lo, hi, color='#CC79A7', lw=1.5, linestyle='dotted')
            axes[0].scatter(point, i-.06, s=25, color='#0072B2')
        else:
            assert 'weighted_endpoint_unavailable_reason' in row and not (a['people'] > 0 and b['people'] > 0)
            axes[0].text(.03, i, 'Endpoint unavailable', transform=axes[0].get_yaxis_transform(), va='center', fontsize=8)
            missing.append(row['intervention'])
    axes[0].set_yticks(range(6), history_labels); axes[0].invert_yaxis(); style(axes[0])
    axes[0].set_xlabel('True minus intervention, own matched subset', fontsize=9)
    axes[1].axis('off'); table = axes[1].table(cellText=coverage, colLabels=['Seen n', 'Held n', 'Records', 'Blocks'],
        cellLoc='center', loc='center'); table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.6)
    axes[1].set_title('Coverage in the same row order', fontsize=9)
    fig.suptitle(title_prefix+'History diagnostics: each contrast owns its paired keys', fontsize=11)
    fig.text(.02, .04, 'Blue: exploratory 95% crossed-bootstrap interval. Purple dotted: Exp sensitivity, NOT calibrated CI.\n'
        'Do not compare mismatched true populations. Missing strata are not renormalized.\n'
        'Offline diagnostics only: shuffled history may include later observations; older changes recency AND parity. No mechanistic causality claim.', fontsize=8)
    fig.savefig(destination/FIGURES[2], dpi=220, facecolor='white'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.7)); fig.subplots_adjust(left=.07, right=.98, bottom=.37, top=.8, wspace=.27)
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
    for ax, document, key, title in [(axes[0], internal, 'weighted_r', 'Internal weighted person r'),
                                    (axes[1], external, 'attended_r', 'External DTU attended r')]:
        mapping = {(r['condition'], r['seed']): r for r in document['absolute_per_seed']}
        assert len(mapping) == len(document['absolute_per_seed'])
        if manifest['EMA_alias']['tau_seconds'] == 0:
            assert all(mapping[('native_P', seed)][key] == mapping[('P_EMA', seed)][key] for seed in seeds), 'tau0 absolute alias changed'
        for i, condition in enumerate(ids):
            points = [finite(mapping[(condition, seed)][key]) for seed in seeds]
            for j, (point, color) in enumerate(zip(points, colors)):
                ax.scatter(i+(j-2)*.09, point, color=color, s=20, label=str(seeds[j]) if i == 0 else None)
            ax.plot([i-.26, i+.26], [sum(points)/5]*2, color='#222222', lw=1.5)
        ax.set_xticks(range(len(ids)), ids, rotation=60, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylabel('Correlation'); ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    axes[1].legend(title='Training seed', frameon=False, ncol=3, fontsize=7, title_fontsize=8)
    fig.suptitle(title_prefix+'Absolute performance: all fixed main models and five seeds', fontsize=11)
    fig.text(.02, .025, 'Colored dots: trained seeds; black bars: arithmetic five-seed means. No confidence intervals in this panel.\n'
        'Each internal seed already uses the original person/stratum weights; DTU uses 18 people x 60 trials, not 1,080 independent subjects.', fontsize=8)
    fig.savefig(destination/FIGURES[3], dpi=220, facecolor='white'); plt.close(fig)
    totals = costs['stage_totals']; assert set(totals) == set(STAGES)
    assert totals['screen']['cells'] == 54 and totals['shared_pretrain']['cells'] == 5
    updates = [int(totals[s]['optimizer_updates']) for s in STAGES]
    hours = [finite(totals[s]['actual_seconds_including_any_reported_cache'])/3600 for s in STAGES]
    assert all(v > 0 for v in updates+hours)
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.3)); fig.subplots_adjust(left=.1, right=.97, bottom=.29, top=.81, wspace=.35)
    short = ['Shared pretrain', 'Native head', 'All screening', 'All full cells', 'Fine-tuning']
    for ax, values, title in [(axes[0], updates, 'Optimizer updates (different losses)'),
                              (axes[1], hours, 'Sum of per-run elapsed hours')]:
        ax.scatter(range(5), values, color='#0072B2', s=38); ax.set_yscale('log'); ax.margins(y=.27)
        for i, value in enumerate(values):
            label = f'{value:,.0f}' if values is updates else f'{value:.2f}'
            ax.annotate(label, (i, value), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
        ax.set_xticks(range(5), short, rotation=25, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10); ax.spines[['top','right']].set_visible(False)
    fig.suptitle(title_prefix+'Whole-method search cost: all candidates and controls', fontsize=11)
    fig.text(.02, .035, 'Includes all 54 screening cells and every unique full cell; five shared pretrains charged once.\n'
        'Per-run elapsed time includes shared-GPU contention and reported caches; it is not server wall time, FLOPs or isolated deployment latency.', fontsize=8)
    fig.savefig(destination/FIGURES[4], dpi=220, facecolor='white'); plt.close(fig)
    text = [f'# {fixture_label or "SS07 已复核数值汇总（科学解释待完成）"}', '',
        f'冻结主模型：`{manifest["primary"]}`；五seed：{seeds}。以下仅转述已复核数据，不自动认定goal完成或投稿充分。', '',
        '## 主比较', '', '| 比较 | 内部原weighted差值 | 内部Bonferroni近似区间 | 外部DTU差值 | 外部Bonferroni近似区间 |',
        '|---|---:|---|---:|---|']
    for a, b in zip(internal['contrasts'], external['contrasts']):
        text.append(f'| {a["contrast"]} | {formatted(a["weighted_mean_difference"])} | {formatted_interval(a["weighted_Bonferroni_M_approx_bootstrap_interval"])} | {formatted(b["mean_r_difference"])} | {formatted_interval(b["Bonferroni_M_approx_interval"])} |')
    text += ['', '内部为已暴露测试的POST-HOC INTERNAL EXPLORATORY；外部固定适配、无目标域拟合，但刺激/房间独立性未证明。两套比较各自M6/7，不主张联合FWER或有限样本覆盖。', '',
        '![主比较](01_MAIN_CONTRASTS.png)', '', '## 五family与历史诊断', '',
        '五family只使用预定前三seed、单列探索M5；不是新主模型选择或独立复现。历史差值都使用其自身同key true子集，缺分层不重新赋权。', '',
        '![五family探索](02_FIVE_FAMILY_EXPLORATORY.png)', '', '![历史诊断及覆盖](03_HISTORY_DIAGNOSTICS.png)', '',
        '## 逐seed与成本', '', '![全部main逐seed](04_ABSOLUTE_PER_SEED.png)', '', '![全搜索成本](05_SEARCH_COSTS.png)', '',
        '| 阶段 | Cell数 | 更新数 | 监督窗口呈现 | 辅助teacher呈现 | 逐run elapsed秒之和 |', '|---|---:|---:|---:|---:|---:|']
    for stage in STAGES:
        row = totals[stage]
        text.append(f'| {stage} | {row["cells"]} | {row["optimizer_updates"]} | {row["supervised_window_presentations"]} | {row["aux_teacher_presentations"]} | {finite(row["actual_seconds_including_any_reported_cache"]):.1f} |')
    text += ['', '预训练的负样本/样本呈现预算另见COSTS与原BASELINE_PROTOCOL，不能把本表监督窗口列中的0理解为没有预训练数据成本。各loss一次更新计算量不同，逐run elapsed相加不是设备隔离计时或总服务器wall time。', '',
        '原始输入重放范围、实际metric复核、暴露/泛化限制及相关工作/新意缺口须与本表一起解释。技术通过、单seed阳性或图上区间不跨0均不自动构成论文级科学接受。']
    (destination/'NUMERICAL_SUMMARY.md').write_text('\n'.join(text)+'\n')
    return {'main_contrasts': names, 'main_M': len(names), 'fixed_main_model_order': ids,
        'exploratory_family_count': 5, 'history_missing_endpoints_preserved': missing,
        'main_seed_dots': 5*len(names)*2, 'absolute_seed_dots': 5*len(ids)*2,
        'separate_internal_external_comparison_families': True,
        'no_new_fit_selection_interval_or_scientific_acceptance': True, 'figures': FIGURES}


def main():
    manifest = reviewed_gate()  # before reading ANY actual outcome summaries
    documents = {name: read(ROOT/'final_summary'/name) for name in FILES}
    destination = ROOT/'final_presentation01'
    result = render(destination, manifest, documents)
    dump(destination/'PRESENTATION_MANIFEST.json', dict(result, time=time.time(),
        sources={name: sha(ROOT/'final_summary'/name) for name in FILES},
        source_sha256=sha(ROOT/'input/pub_01_s03_ss07_result_figures.py'),
        independent_review_complete_sha256=sha(ROOT/'INDEPENDENT_REVIEW_COMPLETED.json'),
        scientific_goal_complete=False, original_data_and_summaries_unchanged=True))
    print(json.dumps({'presentation_generated': True, 'scientific_goal_complete': False, 'figures': FIGURES}))


if __name__ == '__main__': main()
