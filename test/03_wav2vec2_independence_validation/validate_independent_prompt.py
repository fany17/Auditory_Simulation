#!/usr/bin/env python3
"""Validate that the external task document is location-neutral and self-contained."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCUMENT = (ROOT.parent / "03_wav2vec2_independent" / "03_最小可计算任务与Demo_v2.md").resolve()

EXPECTED_SECTIONS = [
    "1. 任务规划 Prompt",
    "2. 启动 Prompt",
    "3. 其他 Prompt",
    "4. 任务说明",
]

BANNED_PATTERNS = {
    "specific_machine": r"2203|server2203",
    "completed_results": r"已完成|已经完成|当前结果|此前|历史两次",
    "external_context": r"README|AGENTS|其他上下文|01_|02_|\x60(?:01|02)\x60",
    "hash_requirement": r"\bSHA256\b|哈希|\bhash\b",
    "absolute_windows_path": r"[A-Za-z]:[\\/]",
    "absolute_home_path": r"/home/",
    "remote_execution_method": r"\bSSH\b|远程 Agent",
}

REQUIRED_TERMS = {
    "model": "facebook/wav2vec2-base-960h",
    "input_rate": "16 kHz",
    "hidden_states": "H0–H12",
    "intervention_formula": "h_adjusted = h_in + α × (h_out - h_in)",
    "identity_gate": "identity test",
    "speed_matrix": "125 次",
    "frontend_matrix": "15 次",
    "total_matrix": "140 次真实模型推理",
    "stride_50hz": "[5,2,2,2,2,2,2]",
    "stride_100hz": "[5,2,2,2,2,2,1]",
    "stride_200hz": "[5,2,2,2,2,1,1]",
    "single_page": "单屏三段式",
    "offline": "断网浏览",
    "figure_outputs": "PNG/SVG",
    "source_table": "源 CSV",
    "fresh_review": "全新环境交付复核",
}

START_REQUIRED = [
    "预检与准备",
    "基线与逐层表示",
    "单层干预门禁",
    "语速与层实验",
    "卷积前端实验",
    "结果与页面",
    "测试与交付",
]

PLAN_REQUIRED = [
    "环境与资源判断",
    "目录和文件规划",
    "分阶段实施步骤",
    "预计推理数量与资源",
    "风险、阻塞项和需要人类决定的问题",
]


def section(text: str, start: str, end: str | None) -> str:
    begin = text.index(f"## {start}")
    finish = text.index(f"## {end}", begin) if end else len(text)
    return text[begin:finish]


def main() -> int:
    text = DOCUMENT.read_text(encoding="utf-8")
    findings: list[dict[str, str]] = []

    sections = re.findall(r"(?m)^## (.+)$", text)
    if sections != EXPECTED_SECTIONS:
        findings.append({
            "check": "top_level_sections",
            "detail": f"expected={EXPECTED_SECTIONS!r}, actual={sections!r}",
        })

    for name, pattern in BANNED_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            findings.append({
                "check": f"banned:{name}",
                "detail": f"matched={match.group(0)!r}",
            })

    for name, term in REQUIRED_TERMS.items():
        if term not in text:
            findings.append({
                "check": f"required:{name}",
                "detail": f"missing={term!r}",
            })

    fence_count = text.count("```")
    if fence_count % 2:
        findings.append({
            "check": "markdown_fences",
            "detail": f"unbalanced marker count={fence_count}",
        })

    plan = section(text, EXPECTED_SECTIONS[0], EXPECTED_SECTIONS[1])
    for term in PLAN_REQUIRED:
        if term not in plan:
            findings.append({
                "check": "planning_prompt",
                "detail": f"missing={term!r}",
            })

    startup = section(text, EXPECTED_SECTIONS[1], EXPECTED_SECTIONS[2])
    for term in START_REQUIRED:
        if term not in startup:
            findings.append({
                "check": "startup_prompt",
                "detail": f"missing={term!r}",
            })

    other_prompt_count = len(re.findall(r"(?m)^### 3\.\d+ ", text))
    if other_prompt_count < 10:
        findings.append({
            "check": "other_prompt_coverage",
            "detail": f"expected at least 10, actual={other_prompt_count}",
        })

    result = {
        "status": "PASS" if not findings else "FAIL",
        "document": DOCUMENT.name,
        "top_level_sections": sections,
        "other_prompt_count": other_prompt_count,
        "markdown_fence_markers": fence_count,
        "required_term_checks": len(REQUIRED_TERMS),
        "banned_pattern_checks": len(BANNED_PATTERNS),
        "findings": findings,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
