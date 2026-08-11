# TB001-DEMO001：Wav2Vec 2.0 Layer Lab

本目录实现 `docs/STANDALONE_EXECUTION_SPEC.md` 的第一版完整 Demo。计算仅由普通 Python 完成；控制端可通过 SSH 调用服务器命令，服务器上不需要也不得启动 Codex Agent。

## 稳定入口

从本目录任意位置均可使用显式脚本路径；脚本自行解析根目录：

```text
预检：python scripts/preflight.py
运行：python scripts/run_demo.py --config configs/TB001-DEMO001.yaml
验收：python scripts/verify_delivery.py
浏览：python scripts/serve_demo.py --port 8000
```

运行阶段强制使用本地模型 snapshot（`local_files_only` 和 offline 环境变量），不会下载模型或数据。权重准备属于显式 prepare/S0–S2 阶段，不在 `run_demo.py` 中发生。

## 入口与范围

- HTML：`demo/TB001-DEMO001/index.html`
- 结果：`outputs/TB001-DEMO001/current_run_group.json`
- 执行报告：`reports/TB001-DEMO001_EXECUTION_REPORT.md`
- 科研解读：`reports/TB001-DEMO001_SCIENTIFIC_INTERPRETATION.md`
- token：`reports/TB001-DEMO001_TOKEN_USAGE.json`
- 轻量交付：`delivery/TB001-DEMO001_lightweight/`

单页展示包含同一英语语音的原速、1.25×、1.5×、1.75× 和 2× 保持音高语速条件。每档输入分别计算原模型 baseline，再与 L1–L12 的 α=0/0.5 单层干预比较；页面默认选择真实字符差异最大的组合，不手写演示结果。改变的是层输出贡献，不是重新训练权重。

展示数据共 140 个实际推理：原有 125 个语速×Transformer 层干预结果，另加 15 个语速×卷积前端时间分辨率结果。卷积前端比较标准约 50 Hz 与未重训练的约 100/200 Hz 步幅变体；Transformer 的“减弱一半”是只保留该层新增量 `Δh=h_out-h_in` 的 50%，不是把权重或整层激活除以二。结果只能说明固定模型和这些输入变体下的工程敏感性，不支持为层命名、脑区映射或跨语音泛化。

右侧的“真实文本”来自输入条目 `hf-internal-testing/librispeech_asr_demo:clean:validation:8` 的数据集标注。标准模型和当前干预输出均直接与该标注计算字符错误率（CER）；红色词表示未能与标注词序列对齐的识别结果。

独立图 `reports/figures/frontend_speed_cer.{png,svg}` 汇总 5 档语速 × 3 种卷积前端的 15 个真实 CER 结果；对应源数据为 `reports/figures/frontend_speed_cer_source.csv`。图中的补偿曲线只从同一条语音的已有结果选择 `≤1.5×→50 Hz、≥1.75×→100 Hz`，没有新增推理，也不是 held-out 验证。
