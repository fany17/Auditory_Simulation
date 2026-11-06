# PUB-01 Subtask Registry

唯一子任务状态登记表（2026-11-01）；路线参见 `RESEARCH_BRANCHES.md`。父状态以 `doc/TASKS.md` 为准。当前唯一执行SS07，旧SS06 STOP为历史。

2026-11-06 SS07 最新验收记录。D19 九组正式训练和独立复核已完成，180 个训练轮次均低于锚点，九组正式选点回退，预设收益门未过。该候选为 PASS_WITH_LIMITATION 的开发阴性证据，SS07 仍为 ACTIVE。[完整结果与四图](../../../reports/pub_01/s03/incremental_continuity/physical_stream/run01/REPORT.md)

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

| `PUB-01-S03_SS07` | S03 | Full-scale native strong baseline + incremental continuity research | **`ACTIVE`** | SS02/SS06 frozen + explicit2026-11-01 full-scale goal authorization | `subtasks/PUB-01-S03/PUB-01-S03_SS07_FULL_SCALE_INCREMENTAL_CONTINUITY.md` | D4-D11冻结；ds004408三seed完整380记录/19人/988块/45429窗及独立4560记录/11856块/2280FP32复算通过。Ccap−PFT +.008040823/Holm5.34e−5、3seed/18人正；F−W未达阈值/单窗退化和融合MSE更差保留。阶段PASS_WITH_LIMITATION，全文/图实际审阅通过、13图完整中文HTML与交付核验完成（同run ARCHIVE.md）；全record未来EEG gain/SI等限制保留，当前读出迁移不证明历史机制，goal未达到；D15九fit目标对照及独立数值/图文/4PNG HTML已封存；H全回退anchor、历史收益门FALSE，梯度退化限制保留，D16九fit/20轮和631239窗独立复算通过，首批通路活动改善、TRAIN拟合增强，但H仅2026微增且低于P、其余回锚；三臂last VAL均值低于D15，历史收益门FALSE。五图中文全文/成本经协调实际审阅，HTML170产物/5图/10限制/链接及最新source_bundle交付检查通过，D16封存；D17固定小批384NN/8输入反传/0fit已完成，selected响应很小、三last仍有明显响应，不能统称塌缩；1325个0NN标量核验/全文科学审阅及34产物/完整中文3表HTML交付检查通过，D17封存。有限只读实现审计纠正warm joint既有覆盖，下一协调已选projection前64token memory四臂H/C/T/P、共同强FT+Ccap/strict5-3历史，D18最终12行工程/累计28临时更新全部丢弃及独立源/输入审阅已过，正式首轮因home磁盘阈值终止，2partial及准确/有界成本已完整保留；已可恢复迁至SATA并0NN核验，新协议实际独立审阅通过后同配置干净重开12fit，队列2050834运行中；原24h截止不变，恢复证据见preprojection_memory/run01，不重开外部。D14九fit顺序对照已独立验收及4PNG HTML归档，修正TRAIN退化但历史收益门FALSE；下一D15仅目标对照源/工程准备，goal ACTIVE。D13固定last误差/三梯度独立复核及四图HTML已交付，TRAIN物理MSE升高/r下降、VAL偏置改善、梯度cos正；D14仅顺序干预方案准备，goal ACTIVE；完整FT单结构9×20轮/710100更新已完成且全部selected−1；best-trained H低于强锚点.0063–.0079，收益门FALSE；独立手写/非零last19/目标/图文表审阅通过、5PNG HTML已归档，本候选封存；旧外部冻结，无自动扩数据/其他节点/患者/PR合并。2026-11-06 D18 候选封存：12 个完整拟合、60 个固定 TRAIN、24 个 best/last 独立核验，H−C 两正一负且 H−P 三组均负，预设收益门未过。完整中文 8 图 HTML 与交付验证见 preprojection_memory/run01/ARCHIVE.md，SS07 总目标未达。 |

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
