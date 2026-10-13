# SUBTASK — PUB-01-S03_SS02 — Strong EEG Auditory Decoding Baseline Reproduction

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S03_SS02` |
| Parent Node | `PUB-01-S03` |
| Parent Task | `PUB-01` |
| Title | Strong EEG Auditory Decoding Baseline Reproduction |
| Status | `READY`（仅文档/执行门批准；尚未运行） |
| Blocking for Parent | YES |
| Patient Data | `FORBIDDEN` |
| Upstream | `S01` / `S02` COMPLETED, `S03_SS01` COMPLETED / PASS_WITH_LIMITATION |
| Inputs | server2203 已授权公开 SparrKULee、官方论文与开源代码（先独立核对）、冻结 protocol/data manifests |
| Deliverables | 文献/代码对标矩阵、复现环境与运行日志、完整指标/逐被试表、误差分析、可复现审阅报告 |
| Acceptance | 目标：HDPS + Transformer-CLIP-EEG 至少两套独立强基线完成实际结果与可复现审计；若任一缺资源/许可/官方测试定义只能标 HOLD/PASS_WITH_LIMITATION，不能伪称两套成功；challenge 先审查协议；不伪造接近论文数值 |
| Validation | Technical → Acceptance（PASS/PASS_WITH_LIMITATION/REWORK/HOLD/NO_GO）→ Parent consistency；独立审阅 |
| Go/Hold/Fail | 作者数据划分/候选构造无法获得→该 protocol HOLD；可复现部分单列；绝不调整测试难度追成绩 |
| Files Allowed to Change | `doc/tasks/PUB-01/**`、`doc/CODEX_START_HERE.md`、`configs/pub_01_s03*`、`scripts/pub_01_s03*`、`src/**`（限新复现适配）、`tests/test_pub_01_s03*`、`reports/pub_01/s03/**`；禁止改写历史产物 |
| Last Updated | 2026-10-13 |

## 1. Objective

在研发任何新架构前建立可靠的 EEG 听觉解码强基线：分别复现**直接重建**与**候选识别**已有方法，说明它们输入什么、输出什么、在哪种测试条件下取得成绩。目标是实验对标，不是马上获得 SOTA。

## 2. Preflight：先审论文协议，后开计算

先核对 `doc/tasks/PUB-01/BENCHMARK_REGISTRY.md` 里的 4 个官方项目入口（已验证 GitHub repo 存在，**不是声明运行可复现**）。记录各协议：原论文/代码版本、许可证、数据集版本、signal sampling、preprocessing、3 s/其他窗口、subject 划分、story/stimulus 漏泄防护、candidate pool size 与构成、负例来源、训练/验证/测试隔离、声音特征（Mel/wav2vec2/GPT-2 等）、是否调用语言模型、评估宏/微平均、随机种子、硬件和运行资源。

特别注意：S02/S03_SS01 使用的是发布方 64 Hz derivative、被试内跨 source-family split；HDPS 等论文可能使用 100 Hz、不同记录划分。**不得因目标都是 SparrKULee 就混用原有 EEG preprocessing/split 并声称复现原论文。**

任何官方隐藏测试集或原代码不可获得，明确写 `UNREPRODUCIBLE_AS_PUBLISHED`；可另立冻结的内部共同协议，在该协议重跑所有比较模型，不把内部结果说成官方 SOTA。

## 3. Priority execution

A. `HDPS` — https://github.com/bobwangPKU/HDPS ：核对论文的多候选定义、代码复现说明、模型使用的声音/语言先验。优先复现其**公开可重跑的 EEG Top-1 / Top-5 / Top-10**。先跑首被试 smoke，再按事先锁定的复现范围扩展。

B. `Transformer-CLIP-EEG` — https://github.com/exporl/transformer-clip-eeg ：核对原 paper/task 使用的 match-mismatch（二选一）与连续目标协议；按真实可执行条件复现。不要把二选一准确率直接和 HDPS 百余候选 Top-1 比较。

C. `Auditory EEG Challenge 2023` — https://github.com/exporl/auditory-eeg-challenge-2023-code 及 `2024` — https://github.com/exporl/auditory-eeg-challenge-2024-code ：审协议、评估脚本/可得 split、二选一/多选一和复现边界。可作为同一复现包的独立协议审计，**不得擅自下载/训练新的数据集**；需要额外许可或大量算力时记录独立请求，不自动加任务。

主线最少要求 A+B 两个强 baseline 的可核验结果；C 优先完成 protocol audit，实际复现视授权数据与原协议可用性。SS02 不负责训练新架构、GPT 重排序、sEEG transfer 或患者结果。

## 4. Fairness and failure controls

- 官方论文 protocol 与我们统一内部 protocol 分开报告，附 immutable split / candidate manifests；候选池由合法测试素材构建，不包含标签泄露或重复音频捷径。
- 训练模型、归一化、声音表征选择、正负例生成规则只使用 train/validation；不以 test performance 反复调超参数或筛除被试。
- 清楚区分 EEG-only 信息、固定声音预训练表示、语言/上下文先验；提供错误 EEG/错配输入、audio-only、prior-only 等可执行对照或记录 why not estimable。
- 分别给 per-subject 与 aggregate 成绩，以及精度随候选数量变化、可重复 seed 的不确定性。
- 完整保留失败尝试、不可用数据、模型源码差异、实际环境，不把合成 fixture 当科研结果。
- 不读取患者/STN 数据；服务器 2203 执行数据、模型与计算；Windows/Git 仅回传代码、配置、轻量 CSV/JSON/PNG/报告；不在服务器启动第二个 Git 仓库。
- 复现代码不能覆盖 `S01/S02/S03_SS01` 历史结果，新增独立运行输出目录。

## 5. Mandatory deliverables

1. `reports/pub_01/s03/strong_baselines/PROTOCOL_MATRIX.md`（论文→原任务→数据/split/候选/分数口径→代码/许可→实际复现状态）。
2. `reports/pub_01/s03/strong_baselines/REPRO_REPORT.md`（环境、命令、真实运行、差异、错误和阻塞）。
3. `reports/pub_01/s03/strong_baselines/RESULTS.csv`（dataset, protocol, method, split, subjects, candidate_n, seed, Top1/5/10 or 2AFC/5AFC, direct metrics, status, provenance）。
4. `reports/pub_01/s03/strong_baselines/REVIEW.md`（独立技术/验收/父任务一致性三级审阅）。
5. 代码/配置/tests 的准确入口和 2203-only 产物 manifest；未运行项必须标 `NOT_RUN`，不得编造指标。

## 6. Completion and stop rule

`NOT_STARTED`。SS02 完成后，只更新 `SUBTASKS.md` 和 completion record，然后 **STOP**；用户/研究者审阅决定 SS03 的模型创新，不自动训练 SaShiMi、TimeMixer++、GPT 或任何未经批准的新模型。
