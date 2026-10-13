# SUBTASK — PUB-01-S03_SS04 — Unified Direct Reconstruction and Retrieval Evaluation

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S03_SS04` |
| Parent Node | `PUB-01-S03` |
| Parent Task | `PUB-01` |
| Title | Unified Direct Reconstruction and Retrieval Evaluation |
| Status | `BACKLOG` |
| Blocking for Parent | YES |
| Patient Data | `FORBIDDEN` |
| Upstream | `PUB-01-S03_SS02` baseline + `PUB-01-S03_SS03` frozen new model |
| Inputs | frozen model outputs / held-out splits / test candidate manifests |
| Deliverables | 分协议直接重建、2AFC、5AFC、百余候选检索；逐被试结果/强基线/消融/数据效率/negative controls |
| Acceptance | 一套主 decoder 的 direct outputs 能被独立测评；各 protocol 无数据划分/候选泄漏，结果不合并成假统一 SOTA |
| Validation | 运行重现、候选 pool leakage checks、模型不依赖候选的推理 canary、独立复核 |
| Go/Hold/Fail | 官方协议无法重现时单列内部协议、不得宣称官方 SOTA |
| Files Allowed to Change | 公共新评估代码、配置、报告与当前 subtask 文档；历史报告只读 |
| Last Updated | 2026-10-13 |

## Objective and protocol

固定一份 EEG→声音时间特征预测输出（不能因候选集变化重训练主模型），然后按各任务独立打分。**直接重建**为第一主指标；候选检索为同一解码器的衍生评价，既要有 2AFC/5AFC，也要有 100+ pool，但数据来自不同 protocol 时必须分别统计。

主要核验：输出对齐/分辨率、feature reconstruction score、检索 metric、跨故事、跨未见被试、参数/算力/校准量、公平正负对照和至少多个预定种子。为便于解释，保留 S03_SS01 线性 TRF、S02 特征和 SS02 强基线。不得因模型结果更好就变更 test negative pool。

`60%/80%` 是竞争愿景，不是本 SS04 的完成判据。GPT 重排序不包含在本节点中；若后续正式考虑，先独立决议允许的文本/上下文和 LM-only 控制，再新建符合父任务的 SS##。

## Completion

`NOT_STARTED`。
