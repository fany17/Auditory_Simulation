# PUB-01 Subtask Registry

唯一子任务状态登记表（2026-11-01）；路线参见 `RESEARCH_BRANCHES.md`。父状态以 `doc/TASKS.md` 为准。当前唯一执行SS07，旧SS06 STOP为历史。

| Subtask ID | Title | Status | Blocking | Upstream | Task Spec | Validation |
|---|---|---|---|---|---|---|
| `PUB-01-S01` | Public data preparation and exploration | `COMPLETED` | YES | NONE | `subtasks/PUB-01-S01_DATASET_LICENSE_TIMING_AUDIT.md` | PASS_WITH_LIMITATION；许可/定向时序限制依任务保留 |
| `PUB-01-S02` | Acoustic & frozen-audio feature → EEG encoding baseline | `COMPLETED` | YES | S01 | `subtasks/PUB-01-S02_FEATURE_BANK_BASELINE.md` | 83人/659录音；acoustic/log-mel/wav2vec2，HuBERT HOLD；PASS_WITH_LIMITATION |
| `PUB-01-S03` | **A线：SparrKULee EEG direct auditory decoding** | `ACTIVE` | YES | S02 | `subtasks/PUB-01-S03_PUBLIC_EEG_BENCHMARK.md` | SS01/02/03/05/06冻结；新授权SS07全量原强基线+连续残差，SS04 BACKLOG |
| `PUB-01-S04` | **B线：ds004703 independent intracranial baseline + transfer** | `BACKLOG` | YES | S01数据准入；仅 transfer 需 S03 | `subtasks/PUB-01-S04_INTRACRANIAL_TRANSFER.md` | SS01/02 未执行；两线不串行阻塞数据基线 |
| `PUB-01-S05` | Public artifact candidate + parent integration review | `BACKLOG` | YES | S01–S04 | `subtasks/PUB-01-S05_ARTIFACT_INTEGRATION.md` | 阶段结果与失败/许可统一审核；未产生 ART-AUDREP-v1 |

## S01：已完成的数据准备 / 初探（保持历史）

| Child ID | Parent Node | Title | Status | Upstream | Task Spec | Acceptance |
|---|---|---|---|---|---|---|
| `PUB-01-S01_SS01` | S01 | Server Dataset Download / Initial Cleaning | `COMPLETED` | S01许可/访问门 | `subtasks/PUB-01-S01/PUB-01-S01_SS01_DOWNLOAD_INITIAL_CLEANING.md` | 2026-09-20 PASS_WITH_LIMITATION；历史 S06 别名 |
| `PUB-01-S01_SS02` | S01 | Data exploration and problem localization | `COMPLETED` | SS01 | `subtasks/PUB-01-S01/PUB-01-S01_SS02_BENCHMARK_ADMISSION.md` | 数据波形/频谱/时轴示例，用户阶段验收 |

## S03：优先执行的 SparrKULee EEG decoding 主线

| Child ID | Parent Node | Title | Status | Upstream | Task Spec | Acceptance |
|---|---|---|---|---|---|---|
| `PUB-01-S03_SS01` | S03 | EEG→envelope backward TRF | `COMPLETED` | S02 | `subtasks/PUB-01-S03/PUB-01-S03_SS01_ENVELOPE_DECODING.md` | 2026-10-11 PASS_WITH_LIMITATION；10 人 93 条，r≈0.1819，R²≈0.0178；只作为线性低层特征对照 |
| `PUB-01-S03_SS02` | S03 | **Strong EEG auditory decoding baseline reproduction（Codex）** | **`COMPLETED`** | S01/S02+SS01 | `subtasks/PUB-01-S03/PUB-01-S03_SS02_STRONG_BASELINE_REPRODUCTION.md` | 2026-10-15 PASS_WITH_LIMITATION：HDPS76人Top1/5/10=12.03/30.45/42.10%；Transformer70训练、69seen/13heldout，2AFC加权82.38%、直接包络加权r=.1593；三运行独立review及交付检查；原完整论文条件未复现，不启下一节点 |
| `PUB-01-S03_SS03` | S03 | Architecture hypothesis and direct decoder | **`COMPLETED`** | SS02 PASS_WITH_LIMITATION + researcher authorization 2026-10-19 | `subtasks/PUB-01-S03/PUB-01-S03_SS03_ARCHITECTURE_HYPOTHESIS.md` | 2026-10-19 PASS_WITH_LIMITATION：4×3真实同预算训练/test；multiscale seen/heldout r=.10305/.11380，高于long/middle，四预设contrast family95%区间>0；same-agent separate review，非生理机制/强基线超越声明；SS04不启动 |
| `PUB-01-S03_SS04` | S03 | Direct reconstruction + 2AFC/5AFC/large-pool evaluation | `BACKLOG` | SS02+SS03 | `subtasks/PUB-01-S03/PUB-01-S03_SS04_DIRECT_RECONSTRUCTION_AND_RETRIEVAL.md` | 分协议公平评价 / 强对照 / 跨被试 / 消融 |
| `PUB-01-S03_SS05` | S03 | History-aware predictive/associative memory (Codex execution) | **`COMPLETED`** | SS03 frozen + 2026-10-19 researcher authorizes Codex | `subtasks/PUB-01-S03/PUB-01-S03_SS05_PREDICTIVE_MEMORY.md` | 2026-10-19 PASS_WITH_LIMITATION：新train norm、15tests/pilot/15run×7条件完成；21人/47record/7family五组r约.091–.092，六primary区间跨0，真实阴性不扩架构；同agent separate review，历史STOP；#4已合并60c3404 |
| `PUB-01-S03_SS06` | S03 | Five continuity frameworks and matched strong-baseline comparison | **`COMPLETED`** | run01 frozen + 2026-10-26 matched-scale authorization executed | `subtasks/PUB-01-S03/PUB-01-S03_SS06_CONTINUITY_FRAMEWORKS.md` | PASS_WITH_LIMITATION：run02 C/O各3×3262/P完整两阶段3×639pre+2035head，r=.123014/.151762/.141466，六项C配对校正区间全跨0；18原始CPU r差3.36e-8、协调SUMMARY2.78e-17；已看audit/5family/单pass及权限/额外预算限制，报告3图/轻量证据齐备；run01不改，#6 Draft，STOP |

| `PUB-01-S03_SS07` | S03 | Full-scale native strong baseline + incremental continuity research | **`ACTIVE`** | SS02/SS06 frozen + explicit2026-11-01 full-scale goal authorization | `subtasks/PUB-01-S03/PUB-01-S03_SS07_FULL_SCALE_INCREMENTAL_CONTINUITY.md` | 五native两阶段及五FT完整完成、head均250上限、FT各5epoch/19725更新，screen40/54快照；EMA原规则选tau0、DTU固定适配冻结。FT曲线/成本与结果前EVIDENCE_MATRIX已交；评估88448和复核113154经协调审阅按门排队，尚无实际test/外部NN结果，不宣称超过基线 |

## S04：独立保留的 ds004703 公共颅内支线

| Child ID | Parent Node | Title | Status | Upstream | Task Spec | Acceptance |
|---|---|---|---|---|---|---|
| `PUB-01-S04_SS01` | S04 | ds004703 independent direct baseline | `BACKLOG` | S01公开数据准入，**不依赖 S03** | `subtasks/PUB-01-S04/PUB-01-S04_SS01_INTRACRANIAL_INDEPENDENT_BASELINE.md` | 独立 sEEG→声音目标基线或明确 HOLD |
| `PUB-01-S04_SS02` | S04 | Public EEG→sEEG fixed-budget transfer | `BACKLOG` | SS01 + S03 frozen public representation | `subtasks/PUB-01-S04/PUB-01-S04_SS02_PUBLIC_CROSS_RECORDING_TRANSFER.md` | from-scratch/frozen/small adapter，正/无/负迁移全部保留 |

## Governance, branch dependencies and historical mapping

- S03 和 S04 是两条并列数据研究支线；**S04_SS01 可独立审计/跑自己的 baseline，不依赖 S03/SOTA；S04_SS02 的 cross-recording transfer 才依赖 S03**。
- SS02/SS03/SS05冻结；#3/#4已按后续授权合并。2026-10-23 SS06先原型、后用户“启动吧”真实执行，现39监督+3SSL/内部审计/后置review完成有限制并STOP。#6仍Draft；SS04/S04/父S05保持BACKLOG，不自动启动。早先“仅合成/不训练”属于prototype01历史，不能覆盖真实完成记录。
- `S03_SS01` 与 `S02` 已有科学结果不改写；之前的 PUB-01-S06 错号只作为历史 `S01_SS01` 别名，其历史报告、脚本、server路径均不重命名。
- 阴性结果合法；模型名/性能目标不能触发额外模型、更改 split 或候选难度。
- SS## 只在已有 S## 内，不建立更深层级；同一执行链一次一个节点；Technical/Acceptance/Parent consistency 三级验收，S05 还需父整合审核。
