#!/usr/bin/env python3
"""Finalize S6 reports after controller-side browser and figure QA."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "TB001-DEMO001"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def main() -> int:
    pointer = read_json(ROOT / "outputs" / TASK_ID / "current_run_group.json")
    group = ROOT / pointer["relative_path"]
    runs = read_json(group / "runs.json")
    vis = read_json(group / "visualization.json")
    interventions = [r for r in runs if r["layer_id"] is not None and r["status"] == "SUCCESS"]
    hidden_rank = sorted(interventions, key=lambda r: r["final_hidden_cosine_distance_from_baseline"], reverse=True)
    js_rank = sorted(interventions, key=lambda r: r["ctc_logit_divergence_from_baseline"], reverse=True)
    edited = [r for r in interventions if r["transcript_edit_distance_from_baseline"] > 0]
    adjacent = vis["baseline"]["adjacent_layer_cosine_distance"]
    norms = vis["baseline"]["mean_hidden_norm"]
    top_adjacent_index = max(range(len(adjacent)), key=adjacent.__getitem__)
    top_norm_index = max(range(len(norms)), key=norms.__getitem__)
    l12_a0 = next(r for r in interventions if r["layer_id"] == 12 and r["alpha"] == 0.0)
    l12_a05 = next(r for r in interventions if r["layer_id"] == 12 and r["alpha"] == 0.5)
    report = f"""# {TASK_ID} 科研结果解读

## 跨图综合

在固定模型 revision、固定 5.12 s 英语语音和规定残差调节下，24/24 个非基线 run 成功，且全部 greedy transcript 与 baseline 完全一致。阴性文本结果并不意味着干预无效：最终 hidden cosine distance 的最大值为 L12/α=0 的 `{l12_a0['final_hidden_cosine_distance_from_baseline']:.6f}`，而 CTC JS divergence 的最大值出现在 L{js_rank[0]['layer_id']}/α={js_rank[0]['alpha']}（`{js_rank[0]['ctc_logit_divergence_from_baseline']:.6f}`）。因此，这条样本上“表示改变”“CTC 分布改变”和“greedy 文本改变”明显不是同一层级的判据。

α=0 的效应总体大于同层 α=0.5，但不是可直接解释为线性功能剂量：后续 11 个 block、layer norm 与 CTC head 会对输入分布偏移做非线性传播。结果可重复区分部分层与强度，满足第一版 Go 的敏感性条件；Go 只表示可以考虑第二条语音，不表示已有跨语音结论。

## Figure 1：最终 hidden state 距离

- 文件：`reports/figures/panel_01_final_hidden_distance.png`（另有独立 SVG）。
- 纵轴是实际一基编号 L1–L12，横轴是 α=0 与 α=0.5；颜色及单元格数字是调整后最终 hidden state 相对 baseline 的逐位置 cosine distance 均值。
- 主模式：L12 明显高于其余层，α=0 为 `{l12_a0['final_hidden_cosine_distance_from_baseline']:.6f}`，α=0.5 为 `{l12_a05['final_hidden_cosine_distance_from_baseline']:.6f}`；其后为 L{hidden_rank[2]['layer_id']}/α={hidden_rank[2]['alpha']}（`{hidden_rank[2]['final_hidden_cosine_distance_from_baseline']:.6f}`）。L1–L6 的绝对偏移总体较小。
- 解释：因为 L12 后面几乎直接接近最终归一化/CTC 读出，干预造成的表示差异较少被后续 Transformer block 重新整合；这只是当前架构和样本下的传播结果，不证明 L12“更重要”或具有某种专属功能。

## Figure 2：CTC Jensen–Shannon divergence

- 文件：`reports/figures/panel_02_ctc_js_divergence.png`（另有独立 SVG）。
- 坐标与 Figure 1 相同；指标是 baseline 与 adjusted 的逐 frame softmax JS divergence（自然对数、概率下限 `1e-12`）再对时间取均值。
- 主模式：α=0 的峰值集中在 L{js_rank[0]['layer_id']}（`{js_rank[0]['ctc_logit_divergence_from_baseline']:.6f}`）、L{js_rank[1]['layer_id']}（`{js_rank[1]['ctc_logit_divergence_from_baseline']:.6f}`）和 L{js_rank[2]['layer_id']}（`{js_rank[2]['ctc_logit_divergence_from_baseline']:.6f}`）；α=0.5 的对应差异整体更小。
- 跨图关系：L12 的最终 hidden distance 最大，但其 α=0 CTC JS divergence 仅 `{l12_a0['ctc_logit_divergence_from_baseline']:.6f}`，低于 L9/L10 峰值，说明大幅表示旋转或尺度变化并不必然产生最大 CTC 分布变化。
- 限制：JS divergence 是模型输出分布差异，不是校准置信度、识别正确率或统计显著性。

## Figure 3：转写字符编辑距离

- 文件：`reports/figures/panel_03_transcript_edit_distance.png`（另有独立 SVG）。
- 12×2 单元格全部为 0，即 {len(interventions) - len(edited)}/24 个成功干预的 greedy transcript 都保持 `AS FOR ETCHINGS THEY ARE OF TWO KINDS BRITISH AND FOREIGN`。
- 科研意义：当前样本的最终离散文本对单层旁路/减弱表现出稳健性，但 Figure 1–2 已证明中间表示和 CTC 分布并非不变。不能把全零图解释成“这些层没有信息”或“识别不受干预”。
- 限制：这里只比较单条干净语音；没有跨说话人、噪声、语速或重复试次的统计分布。

## Figure 4：baseline 层级动力学

- 文件：`reports/figures/panel_04_baseline_layer_dynamics.png`（另有独立 SVG）。
- 上 panel 的横轴为 H0→H1 至 H11→H12，相邻表示 cosine distance 在后两次转换明显增大；最大是 H{top_adjacent_index}→H{top_adjacent_index + 1}（`{adjacent[top_adjacent_index]:.6f}`）。
- 下 panel 的横轴为 H0–H12，平均向量范数在 H{top_norm_index} 达到峰值 `{norms[top_norm_index]:.6f}`，H12 随后回落到 `{norms[12]:.6f}`。这一幅图显示后段表示的几何和幅值变化较大，也为 Figure 1 中晚层干预的高敏感性提供数值背景。
- 限制：cosine distance 与向量范数受 layer norm、残差和输出组织共同影响，不能据此为某一层命名或建立脑区对应。

## 阴性结果、失败与证据边界

- 当前成功结果中没有任何 transcript 改变；该阴性结果完整保留。
- 首次 S3 identity test 失败也保留：直接计算 `h_in + 1×(h_out-h_in)` 的 float32 舍入误差经后续层放大到 logits 最大误差 `0.0023346`。修复为数学等价的 α=1→`h_out`、α=0→`h_in` 数值稳定端点后，identity error 为 `0.0`，token IDs、transcript 与 hidden shapes 均严格一致。
- 本实验不能支持层的声音/节律/语义功能命名、脑区映射、跨语音重要性排序、训练机制解释或人类听觉等价。

## 下一步

由人类先核对 HTML、25 个结果、identity report、失败记录和四幅图；接受后只增加第二条获许可英语语音，检查 Figure 1–2 的层级模式是否跨输入复现，不同时加入噪声、变速或第二模型。
"""
    write_text(ROOT / "reports" / f"{TASK_ID}_SCIENTIFIC_INTERPRETATION.md", report)

    browser_qa = {
        "status": "PASS",
        "checked_at_utc": utc_now(),
        "method": "controller-side in-app browser against a localhost copy of the SSH-downloaded lightweight package",
        "checks": {
            "page_loaded": True,
            "audio_ready_state": 4,
            "audio_duration_seconds": 5.12,
            "canvas_count": 6,
            "initial_state": {"layer": 1, "mode": "observe", "alpha": 1.0, "run_id": "baseline-final-001"},
            "layer12_alpha0": {"run_id": "layer-12-alpha-0p0", "hidden_distance": l12_a0["final_hidden_cosine_distance_from_baseline"], "bypass_highlight": True},
            "layer12_alpha05": {"run_id": "layer-12-alpha-0p5", "hidden_distance": l12_a05["final_hidden_cosine_distance_from_baseline"]},
            "reset_restored_baseline": True,
            "metric_help_toggle": True,
            "console_errors_or_warnings": 0,
            "local_script_sources": ["data/data.js", "app.js"],
        },
        "note": "Browser QA is a controller-side self-check, not an independent scientific review.",
    }
    write_json(ROOT / "reports" / f"{TASK_ID}_BROWSER_QA.json", browser_qa)

    status = {
        "status": "COMPLETE",
        "go_no_go": "GO_FOR_HUMAN_REVIEW_AND_OPTIONAL_SECOND_UTTERANCE",
        "completed_at_utc": utc_now(),
        "execution_method": "LOCAL_CONTROLLER_OVER_SSH",
        "remote_agent_used_for_completion": False,
        "unique_inference_results": len(runs),
        "nonbaseline_success": len(interventions),
        "nonbaseline_failed": 24 - len(interventions),
        "transcript_changed_runs": len(edited),
        "identity_test": read_json(ROOT / "reports" / f"{TASK_ID}_IDENTITY_TEST.json")["status"],
        "root_verify": read_json(ROOT / "reports" / f"{TASK_ID}_FINAL_VERIFY.json")["status"],
        "browser_qa": browser_qa["status"],
        "scientific_review_boundary": "Agent self-check complete; final acceptance remains human.",
    }
    write_json(ROOT / "reports" / f"{TASK_ID}_FINAL_STATUS.json", status)

    execution_path = ROOT / "reports" / f"{TASK_ID}_EXECUTION_REPORT.md"
    execution = execution_path.read_text(encoding="utf-8")
    marker = "## 最终可视与科研复核"
    if marker not in execution:
        execution += f"""

## 最终可视与科研复核

- controller-side localhost 浏览器 QA：`PASS`；音频 5.12 s 可加载，6 个 canvas 有效，L12/α=0、L12/α=0.5、指标解释开关与恢复 baseline 均实际交互通过，控制台 0 warning/error。
- 四幅 PNG/SVG 已逐张人工可视检查；详细 Results-level 解读见 `reports/{TASK_ID}_SCIENTIFIC_INTERPRETATION.md`。
- 最终状态：`COMPLETE`；第一版达到 `GO_FOR_HUMAN_REVIEW_AND_OPTIONAL_SECOND_UTTERANCE`。这不是自动授权进入第二条语音。
"""
        write_text(execution_path, execution)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
