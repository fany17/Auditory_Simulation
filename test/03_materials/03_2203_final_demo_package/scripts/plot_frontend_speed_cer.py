#!/usr/bin/env python3
"""Plot real speech-speed × convolutional-frontend CER results."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "TB001-DEMO001"
DATA_PATH = ROOT / "demo" / TASK_ID / "data" / "frontend_results.json"
FIGURE_DIR = ROOT / "reports" / "figures"
DEMO_FIGURE_DIR = ROOT / "demo" / TASK_ID / "figures"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    expected_variants = ["frontend-50hz", "frontend-100hz", "frontend-200hz"]
    variant_labels = {
        "frontend-50hz": "标准时间采样 · 50 Hz (20 ms)",
        "frontend-100hz": "加密时间采样 · 100 Hz (10 ms)",
        "frontend-200hz": "过密实验条件 · 200 Hz (5 ms)",
    }
    colors = {
        "frontend-50hz": "#176B87",
        "frontend-100hz": "#2E8B67",
        "frontend-200hz": "#C05A47",
    }
    markers = {"frontend-50hz": "o", "frontend-100hz": "s", "frontend-200hz": "^"}
    speeds = [float(condition["speed_factor"]) for condition in data["conditions"]]
    series: dict[str, list[float]] = {key: [] for key in expected_variants}
    rows: list[dict[str, object]] = []

    for condition in data["conditions"]:
        variants = {item["frontend_variant_id"]: item for item in condition["variants"]}
        if set(variants) != set(expected_variants):
            raise RuntimeError(f"Incomplete frontend matrix for {condition['id']}")
        for key in expected_variants:
            cer_percent = 100.0 * float(variants[key]["character_error_rate_vs_reference"])
            series[key].append(cer_percent)
            rows.append(
                {
                    "speed_factor": condition["speed_factor"],
                    "frontend_variant": key,
                    "frontend_label": variant_labels[key],
                    "ctc_frame_rate_hz": variants[key]["actual_ctc_frame_rate_hz"],
                    "cer_percent": cer_percent,
                    "transcript": variants[key]["adjusted_transcript"],
                }
            )

    # Demonstration policy selected from the same utterance, not held-out validation.
    adaptive_variant = ["frontend-50hz" if speed <= 1.5 else "frontend-100hz" for speed in speeds]
    adaptive_cer = [series[key][index] for index, key in enumerate(adaptive_variant)]
    for index, key in enumerate(adaptive_variant):
        rows.append(
            {
                "speed_factor": speeds[index],
                "frontend_variant": "same-sample-adaptive-demo",
                "frontend_label": "同样本补偿示范策略",
                "ctc_frame_rate_hz": 50.0 if key == "frontend-50hz" else 100.0,
                "cer_percent": adaptive_cer[index],
                "transcript": "derived selection from existing run; no new inference",
            }
        )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIGURE_DIR / "frontend_speed_cer_source.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    font_candidates = [
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    ]
    font_path = next((Path(path) for path in font_candidates if Path(path).is_file()), None)
    if font_path is not None:
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10.5})

    fig, ax = plt.subplots(figsize=(9.4, 5.8), constrained_layout=False)
    fig.patch.set_facecolor("#F8F6F0")
    ax.set_facecolor("#FFFDF8")
    ax.axvspan(1.625, 2.025, color="#F3C9A8", alpha=0.23, linewidth=0)
    ax.text(1.82, 34.0, "快速语音区", color="#8B5637", ha="center", va="center", fontsize=9)

    for key in expected_variants:
        ax.plot(
            speeds,
            series[key],
            color=colors[key],
            marker=markers[key],
            markersize=6.5,
            linewidth=2.1,
            label=variant_labels[key],
            zorder=3,
        )
    ax.plot(
        speeds,
        adaptive_cer,
        color="#1E2A32",
        marker="D",
        markersize=5.5,
        linewidth=2.2,
        linestyle="--",
        label="同样本补偿示范*",
        zorder=4,
    )

    ax.annotate(
        "2×：31.6%",
        xy=(2.0, series["frontend-50hz"][-1]),
        xytext=(-64, -4),
        textcoords="offset points",
        color=colors["frontend-50hz"],
        fontsize=9,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": colors["frontend-50hz"], "lw": 1},
    )
    ax.annotate(
        "2×：1.8%",
        xy=(2.0, series["frontend-100hz"][-1]),
        xytext=(-58, 22),
        textcoords="offset points",
        color=colors["frontend-100hz"],
        fontsize=9,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": colors["frontend-100hz"], "lw": 1},
    )
    ax.text(
        0.03,
        0.93,
        "当前样本补偿规则\n≤1.5× → 50 Hz\n≥1.75× → 100 Hz",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#263640",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EEF3F2", "edgecolor": "#B8C8C4", "linewidth": 0.8},
    )

    ax.set_title("语音加速与卷积前端时间采样共同决定识别误差", loc="left", fontsize=16, fontweight="bold", pad=17)
    ax.text(0.0, 1.015, "同一条真实语音的 15 次 wav2vec2 推理；纵轴越低越好", transform=ax.transAxes, ha="left", va="bottom", color="#5C6970", fontsize=9.5)
    ax.set_xlabel("语音速度倍数")
    ax.set_ylabel("相对数据集标注的字符错误率 CER (%)")
    ax.set_xticks(speeds, ["原速", "1.25×", "1.5×", "1.75×", "2×"])
    ax.set_ylim(-1.5, 36.5)
    ax.set_yticks(range(0, 36, 5))
    ax.grid(axis="y", color="#D8D5CD", linewidth=0.8, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9EA7AA")
    ax.legend(loc="upper center", bbox_to_anchor=(0.53, -0.16), ncol=2, frameon=False, fontsize=9)
    fig.text(0.065, 0.015, "* 规则由同一条语音选择，仅用于展示补偿思路；不是 held-out 验证，也不代表通用最优前端。", ha="left", va="bottom", color="#6A7479", fontsize=8.5)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.27)

    png_path = FIGURE_DIR / "frontend_speed_cer.png"
    svg_path = FIGURE_DIR / "frontend_speed_cer.svg"
    fig.savefig(png_path, dpi=220, facecolor=fig.get_facecolor())
    fig.savefig(svg_path, format="svg", facecolor=fig.get_facecolor())
    plt.close(fig)
    shutil.copy2(png_path, DEMO_FIGURE_DIR / png_path.name)
    shutil.copy2(svg_path, DEMO_FIGURE_DIR / svg_path.name)

    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "figure_id": "frontend_speed_cer",
        "title": "语音加速与卷积前端时间采样共同决定识别误差",
        "source_json": DATA_PATH.relative_to(ROOT).as_posix(),
        "source_csv": csv_path.relative_to(ROOT).as_posix(),
        "outputs": [png_path.relative_to(ROOT).as_posix(), svg_path.relative_to(ROOT).as_posix()],
        "real_inferences": 15,
        "adaptive_curve": "derived from existing runs; no new inference",
        "adaptive_demo_policy": "speed <= 1.5x: frontend-50hz; speed >= 1.75x: frontend-100hz",
        "adaptive_demo_cer_percent": adaptive_cer,
        "boundary": "same-utterance demonstration, not held-out validation or a general optimum",
    }
    (FIGURE_DIR / "frontend_speed_cer_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "png": str(png_path), "svg": str(svg_path), "csv": str(csv_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
