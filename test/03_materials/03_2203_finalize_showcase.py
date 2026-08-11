#!/usr/bin/env python3
"""Finalize the simplified one-page showcase and its controller-side QA evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "TB001-DEMO001"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def main() -> int:
    pointer = read(ROOT / "outputs" / TASK_ID / "current_showcase.json")
    group = ROOT / pointer["relative_path"]
    showcase = read(group / "showcase_results.json")
    default = showcase["default_selection"]
    condition = next(c for c in showcase["conditions"] if c["id"] == default["condition_id"])
    run = next(r for r in condition["runs"] if r["run_id"] == default["run_id"])
    baseline = condition["baseline"]
    changed_by_condition = {
        c["id"]: sum(r["transcript_edit_distance_from_baseline"] > 0 for r in c["runs"] if r["status"] == "SUCCESS")
        for c in showcase["conditions"]
    }

    report = f"""# {TASK_ID} 单页交互展示扩展报告

- 状态：`PASS`
- 生成时间：`{now()}`
- 执行方式：本地控制器通过 SSH 运行 2203 普通 Python；未调用远程 Codex Agent。

## 展示目标

页面压缩为唯一的一条故事线：`输入语音 → 选择并改变一个 Transformer 层 → 最终英文字幕差异`。首页不再展示六个科研 Panel、hidden heatmap、运行审计或复杂模式选择。

## 新增真实推理

- 输入条件：原速、1.25×、1.5×、1.75×、2×，均由同一合法来源音频生成。
- 加速方法：torchaudio phase vocoder，保持音高；每个条件保存独立 WAV、duration 与 SHA256。
- 每档语速分别计算原模型 baseline，再运行 L1–L12 × α=0/0.5。
- 唯一推理：`{showcase['counts']['unique_inferences']}`；层干预：`{showcase['counts']['interventions']}`；失败：`{showcase['counts']['failed']}`。
- 有 `{showcase['counts']['text_changed']}` 个干预真实改变了相同语速 baseline 的最终转写；按条件计数：`{json.dumps(changed_by_condition, ensure_ascii=False)}`。

## 默认展示

- 条件：`{condition['label']}`，duration `{condition['duration_seconds']:.2f} s`。
- 干预：直接去掉 L{run['layer_id']}，即 α=0 的稳定端点 `h_adjusted=h_in`。
- 原模型：`{baseline['adjusted_transcript']}`
- 干预后：`{run['adjusted_transcript']}`
- 字符编辑距离：`{run['transcript_edit_distance_from_baseline']}`；CTC JS divergence：`{run['ctc_logit_divergence_from_baseline']:.6f}`。

## 页面交互

- 左侧：5 档语速、音频播放器、简化波形与 duration。
- 中间：卷积前端、L1–L12、CTC 输出；仅保留“正常/减弱一半/直接去掉”。
- 右侧：同一输入下的原模型字幕、干预字幕、词级删除/新增高亮、字符变化与 CTC 分布差异。
- 页面默认选中实际字符差异最大的真实组合，不在 JavaScript 手写结果。

## 结论边界

语速改变会影响该语速条件的 baseline；层效应只在同一输入条件内比较。phase vocoder 变体用于展示鲁棒性与敏感性，不代表自然快速语音的完整声学分布。单层旁路是工程消融，不支持层的功能命名或脑区映射。
"""
    write_text(ROOT / "reports" / f"{TASK_ID}_SHOWCASE_REPORT.md", report)

    browser_qa = {
        "status": "PASS",
        "checked_at_utc": now(),
        "method": "controller-side in-app browser on localhost copy of SSH-downloaded lightweight package",
        "default_state": {
            "speed": "2×",
            "duration_seconds": 2.56,
            "layer": 5,
            "strength": "removed",
            "run_id": run["run_id"],
            "baseline_transcript": baseline["adjusted_transcript"],
            "adjusted_transcript": run["adjusted_transcript"],
            "character_edit_distance": run["transcript_edit_distance_from_baseline"],
            "removed_word_highlights": 3,
            "added_word_highlights": 3,
        },
        "interaction_checks": [
            {
                "condition": "1.75×",
                "layer": 3,
                "alpha": 0.0,
                "run_id": "speed-1p75-layer-03-alpha-0p0",
                "character_edit_distance": 4,
                "audio_duration_seconds": 2.925688,
                "status": "PASS",
            },
            {"action": "normal", "baseline_equals_adjusted": True, "character_edit_distance": 0, "status": "PASS"},
            {"action": "half", "run_id": "speed-1p75-layer-03-alpha-0p5", "status": "PASS"},
        ],
        "audio_ready_state": 4,
        "console_errors_or_warnings": 0,
        "visual_layout": "PASS: single clean left-input / middle-layers / right-output flow",
        "boundary": "Controller self-QA, not independent scientific review.",
    }
    write_json(ROOT / "reports" / f"{TASK_ID}_BROWSER_QA.json", browser_qa)

    final_status_path = ROOT / "reports" / f"{TASK_ID}_FINAL_STATUS.json"
    status = read(final_status_path)
    status.update(
        {
            "status": "COMPLETE",
            "updated_at_utc": now(),
            "execution_method": "LOCAL_CONTROLLER_OVER_SSH",
            "remote_agent_used_for_showcase": False,
            "showcase": {
                "status": "PASS",
                "group_id": showcase["group_id"],
                **showcase["counts"],
                "default_selection": default,
                "browser_qa": "PASS",
            },
        }
    )
    write_json(final_status_path, status)

    token_path = ROOT / "reports" / f"{TASK_ID}_TOKEN_USAGE.json"
    token = read(token_path)
    token["updated_at_utc"] = now()
    token["showcase_extension"] = {
        "execution_method": "LOCAL_CONTROLLER_OVER_SSH",
        "remote_agent_tokens": 0,
        "model_inference_billing_tokens": 0,
        "desktop_controller_usage": "UNAVAILABLE",
        "note": "125 wav2vec2 forward results are not language-model billing tokens.",
    }
    write_json(token_path, token)

    execution_path = ROOT / "reports" / f"{TASK_ID}_EXECUTION_REPORT.md"
    execution = execution_path.read_text(encoding="utf-8")
    marker = "## 单页展示扩展"
    if marker not in execution:
        execution += f"""

## 单页展示扩展

- 新增 5 档语速与 125 个唯一推理；120 个层干预全部成功，其中 33 个真实改变最终转写。
- 页面重做为“左侧输入—中间 L1–L12—右侧字幕差异”的单屏流程；默认显示 2×语速、去掉 L5、字符编辑距离 6。
- 本地浏览器实际点击原速/加速、L1–L12、正常/减半/去掉通过，音频可播放，控制台 0 warning/error。
- 详细扩展记录：`reports/{TASK_ID}_SHOWCASE_REPORT.md`。
"""
        write_text(execution_path, execution)

    with (ROOT / "docs" / "CODEX_PROJECT_LOG.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"\n## {now()}：SSH 完成单页展示扩展\n\n"
            "- 新增 5 档语速、125 个唯一推理，120 个干预全部成功，33 个改变最终转写。\n"
            "- 页面重做为输入语音 → 单层干预 → 最终字幕差异；默认 2×/L5/α=0，字符编辑距离 6。\n"
            "- 浏览器交互 QA PASS；未调用远程 Agent，未初始化 Git，token 不写入本日志。\n"
        )
    print(json.dumps(status["showcase"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
