# TB001-DEMO001 完整执行报告

- 状态：`S0-S6_EXECUTED`
- 生成时间：`2026-08-06T11:27:00Z`
- 执行链路：本地控制器通过 SSH 调用 2203 上的普通 Python；未启动远程 Codex Agent。
- 远端 Git：未初始化、未提交、未推送。
- run group：`outputs/TB001-DEMO001/run_groups/run-20260806T112651Z`

## 环境与固定来源

- 模型：`facebook/wav2vec2-base-960h`，revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`。
- Processor revision：`22aad52d435eb6dbaf354bdad9b0da84ce7d6156`。
- 输入：公开 LibriSpeech demo 单条 16 kHz 单声道英语语音；来源、许可、item 与 SHA256 见 `stimuli/TB001-DEMO001/manifest.json`。
- 模型关键文件与 SHA256 见 `models/model_manifest.json`；加载键审计见 `reports/TB001-DEMO001_MODEL_LOADING_AUDIT.json`。
- 唯一缺失键 `wav2vec2.masked_spec_embed` 为训练期 SpecAugment 参数；本次 `eval()` 且显式 `apply_spec_augment=false`。任何其他 missing/unexpected/mismatched key 都会令运行失败。

## 实际运行与状态

- baseline：1/1 SUCCESS；13 组 hidden-state shape 已记录。
- identity test：`PASS`；α=1 logits 最大绝对误差 `0.0`，阈值 `1e-05`；token IDs / transcript / hidden shapes 严格一致。
- 非基线干预：24/24 SUCCESS，0 FAILED；每次只 hook 一个一基编号层，forward 后 hook 残留为 0。
- greedy transcript 改变：0/24；未改变：24/24。阴性结果未过滤。
- 唯一推理结果：25（1 baseline + 12×2 intervention）。HTML 的 12 个 α=1 选择共同引用 baseline。

## 稳定命令与退出码

```text
python scripts/preflight.py                                      # exit 0
python scripts/run_demo.py --config configs/TB001-DEMO001.yaml       # exit 0
python -m unittest discover -s tests -v                          # exit 0
python scripts/verify_delivery.py                                # 由最终验收日志给出
python scripts/serve_demo.py --port 8000                         # 人工浏览入口
```

运行阶段设置 Hugging Face offline/local-files-only，只读取已固定的本地 snapshot，不访问网络。日志只追加写入 `logs/`。

## HTML 与图

- 离线入口：`demo/TB001-DEMO001/index.html`；无 CDN，结果由 `outputs/` 导出的 `data/*.json` 和派生 `data.js` 提供，`app.js` 不包含手填实验数值。
- 独立图件：4 张 PNG 和对应 SVG，位于 `reports/figures/`；逐图科研解读见 `reports/TB001-DEMO001_SCIENTIFIC_INTERPRETATION.md`。
- 控件包含音频、L1–L12、观察/残差调节、α=0/0.5/1、恢复 baseline、解释开关、复制配置和导出当前 JSON。

## token 口径

- 本轮 S3–S6 使用 SSH + Python/模型推理，远端 Agent token 为 0。
- 此处的 CTC frame、token id 和字符数不是模型计费 token。
- 当前桌面控制会话的精确 usage 未由服务器暴露，标记为 `UNAVAILABLE`，不估算。
- 历史上两次误用远端 Codex CLI 的 314,074 tokens 单列为历史沉没消耗，不计入本轮 SSH 完成链路。

## 结论与边界

本 Demo 最多支持：在固定模型 revision、固定单条语音和规定工程干预下，不同层与强度对中间表示、CTC 分布及最终转写呈现可复现的不同敏感性。它不能支持层的语义/节律命名、脑区映射、跨语音普遍排序或人类听觉等价结论。

## Go / No-go

如果最终 verifier、独立轻量包验收和人工 HTML 检查均通过，则第一版 Demo 达到 Go；Go 仅表示可以考虑第二条语音，不授权自动扩展。


## 最终可视与科研复核

- controller-side localhost 浏览器 QA：`PASS`；音频 5.12 s 可加载，6 个 canvas 有效，L12/α=0、L12/α=0.5、指标解释开关与恢复 baseline 均实际交互通过，控制台 0 warning/error。
- 四幅 PNG/SVG 已逐张人工可视检查；详细 Results-level 解读见 `reports/TB001-DEMO001_SCIENTIFIC_INTERPRETATION.md`。
- 最终状态：`COMPLETE`；第一版达到 `GO_FOR_HUMAN_REVIEW_AND_OPTIONAL_SECOND_UTTERANCE`。这不是自动授权进入第二条语音。


## 单页展示扩展

- 新增 5 档语速与 125 个唯一推理；120 个层干预全部成功，其中 33 个真实改变最终转写。
- 页面重做为“左侧输入—中间 L1–L12—右侧字幕差异”的单屏流程；默认显示 2×语速、去掉 L5、字符编辑距离 6。
- 本地浏览器实际点击原速/加速、L1–L12、正常/减半/去掉通过，音频可播放，控制台 0 warning/error。
- 详细扩展记录：`reports/TB001-DEMO001_SHOWCASE_REPORT.md`。

## Convolutional frontend extension (2026-08-07)

See `reports/TB001-DEMO001_FRONTEND_REPORT.md`. Added 15 real frontend-stride inferences and the interactive convolutional-frontend explanation. Remote Agent tokens: 0.

## Frontend CER figure (2026-08-07)

Generated standalone SVG/PNG and an auditable CSV from the existing 15 frontend inferences. The same-sample policy CER (%) is `[0.0, 0.0, 0.0, 0.0, 1.7543859649122806]`; it is explicitly not held-out. No additional inference or remote Agent token was used.
