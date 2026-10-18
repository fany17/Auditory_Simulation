# TASK — PUB-01 — Public Auditory–Neural Representation

| Field | Value |
|---|---|
| Task ID | `PUB-01` |
| Title | `Public Auditory–Neural Representation` |
| Track | `PUBLIC_METHOD` |
| Class | `MAINLINE_EXECUTION` |
| Status | `ACTIVE` |
| Owner Repo | `Auditory_Simulation` |
| Blocking | `NO` for direct STN READ |
| Patient Data | `FORBIDDEN` |
| Upstream | historical public baselines/model bank |
| Inputs | SparrKULee, public intracranial dataset candidates, frozen audio models |
| Deliverables | leak-safe public benchmark + `ART-AUDREP-v1` candidate |
| Acceptance | public EEG and public intracranial audit/benchmark/transfer evidence completed regardless of transfer sign |
| Go/No-go | positive/no/negative transfer all valid; never blocks direct STN science |
| Downstream | `INT-01` optional comparator |
| Subtask Registry | `doc/tasks/PUB-01/SUBTASKS.md` |
| Last Updated | `2026-10-15` |

## 1. Scientific question

> 在完全不读取患者/STN 数据的前提下，能否从公开 EEG 与公开 intracranial 数据建立一个可审计、可测试迁移、可用于少样本比较的 auditory-neural representation？

本任务提供 public prior/comparator，不承担 STN 生物学 claim，也不要求 transfer 必须阳性。

## 2. Scope

Primary public EEG：SparrKULee。  
Primary public intracranial candidate：OpenNeuro `ds004703`，其独立 baseline 和 transfer 需要分别完成对应 license/use-condition / pairing/timing audit。

父任务覆盖：

- dataset/license/timing/pairing audit；
- interpretable acoustic/timing baseline；
- frozen wav2vec2/HuBERT layer bank；
- channel-agnostic neural encoder；
- encoding + contrastive/retrieval benchmark；
- held-out subject/session/story；
- EEG→public intracranial transfer；
- few-shot / negative-transfer characterization；
- `ART-AUDREP-v1` candidate packaging。

## 3. Non-goals / forbidden actions

- 不读取患者/STN 数据或患者派生 embedding；
- 不把 EEG→intracranial→STN 写成生物学层级；
- 不继续无限扩 pretrained model 名单；
- 不用 test set 选 layer/hyperparameter；
- 不因阴性反复改 split/window/negative sampling；
- 不把 shared geometry 解释为功能同源；
- 不把 public result 写成 stimulation/write-in evidence。

## 4. Inputs and Preconditions

任何正式训练/benchmark 前必须冻结：dataset source/version/license、audio-neural pairing、subject/session/story keys、sampling/channel/reference、timing/event metadata、bad/missing recordings 和 split policy。

患者数据始终 `FORBIDDEN`。

## 5. Execution Plan

父任务按 `SUBTASKS.md` 分解：

1. `PUB-01-S01` Dataset / License / Timing Audit；
2. `PUB-01-S02` Feature Bank and Leak-safe Baseline（已完成有限制的 acoustic/log-mel/frozen wav2vec2→EEG encoding）；
3. **A 支线**：`PUB-01-S03` SparrKULee EEG 听觉直接解码（先复现强基线，再经研究者审阅确定新架构，最后做无候选直接重建与二选一/五选一/百余候选检索）；
4. **B 支线**：`PUB-01-S04` ds004703 独立公开颅内基线与后续 EEG→sEEG 迁移（独立 sEEG 基线不依赖 S03，迁移阶段才依赖 S03 frozen output）；
5. `PUB-01-S05` Artifact Candidate and Parent Integration Review。

两条支线严格按 `RESEARCH_BRANCHES.md`、`SUBTASKS.md` 执行；当前唯一 READY 节点是 `PUB-01-S03_SS02`，详细 Codex 入口在 `doc/CODEX_START_HERE.md`。

2026-09-19 用户要求持续推进至数据下载和初步清洗完成，新增 `PUB-01-S01_SS01 — Server Dataset Download and Initial Cleaning` 作为 S01 审计后的独立数据准备包。2026-09-24纠正原S06编号并嵌入S01；执行依赖为 S01许可/访问准入 → S01_SS01 → S01配对/时序与实验准入 → S02 → S03 → S04 → S05。S01_SS01 对已经完成许可/身份核验的数据可分支先行下载和非破坏性 QC，不绕过 S01 的 benchmark 准入验收。

已有S##节点内的追加工作若需独立交付/验收，登记为该节点的SS##；小修改写节点完成记录。仅新的独立主阶段使用S##。

## 6. Controls / leakage / confounds

至少阻断：相邻时间泄漏、同 story/clip 重复、test-subject normalization leakage、test-driven layer selection、recording identity shortcut、trivial negative sampling。

所有 normalization/PCA/feature/layer selection 只在 train fold 内。

## 7. Deliverables

父任务最终必须具备：

- auditable dataset/license manifests；
- frozen split manifest；
- acoustic/timing baseline；
- SparrKULee EEG 无候选直接声音特征解码 + 2AFC/5AFC/100+ 候选分协议 benchmark、强基线与消融；
- 严格记录是否只是在候选中匹配，不能将候选识别当成可听语音/语义重建；
- ds004703 独立公开颅内基线及 public intracranial transfer/few-shot evidence；
- negative-transfer/known-failures report；
- `ART-AUDREP-v1` candidate manifest/schema/runtime/canary/provenance。

## 8. Acceptance Criteria

1. license/timing/pairing 可审计；
2. split leak-safe；
3. interpretable baseline 可复现；
4. held-out subject/session/story 结果完整报告；明确 direct reconstruction 与候选检索任务的不同 protocol、分组和指标，不以不同数据集的分数声称同一 SOTA；
5. ds004703 独立 baseline 及 transfer 按预定义条件执行；只有迁移依赖 S03 模型冻结；
6. positive/no/negative transfer 全保留；
7. candidate artifact 记录 provenance、license 与 known failures；
8. patient/STN data boundary 未被突破。

## 9. Go / Hold / Fail

### GO

形成可由 `INT-01` 消费的 `ART-AUDREP-v1` candidate。

### HOLD

数据许可、audio-neural pairing 或 timing 不可审计时，相应分支停止等待。

### NO-GO

某 dataset/model 不支持可靠 benchmark时记录 failure，停止该分支；不自动新增替代模型追阳性。

`PUB-01` 的无迁移结果不构成 direct STN READ 的 no-go。

## 10. Known risks

held-out subject weak、EEG→intracranial negative transfer、layer instability、representation 只复现 envelope/onset、license 限制 artifact distribution。

## 11. Artifact / Handoff Contract

`ART-AUDREP-v1` 至少包含：semantic version、source task/subtasks/commit/config/seed、model/feature spec、preprocessing/timing、neural encoder（若许可）、runtime/transform、canary、benchmark/few-shot evidence、known failures、schema/validator、license/provenance manifest。

不得包含患者/STN 数据。

## 12. Amendment Record

2026-10-15：用户明确“缺几个就缺几个，没关系”，经协调会话转达，授权SS02在现有合法可用数据上正式训练/验证/测试，不再要求凑齐原cohort。采用只依据输入可用性的完整被试子集：纳入者保留作者全部train/val/test story及有效候选，缺任一必要输入则整名排除；冻结在拟合前，禁止test成绩筛人。沿用原100Hz/M、seed1024、预算/validation early stopping，M只对照M；此为cohort偏差的显式父amendment，不把子集成绩等同完整论文结果，不放宽权限/患者禁令或A+B两套结果验收，不启SS03/S04。严格完整模式保留，可用子集模式明确记录此次授权。

2026-10-13：研究者决定保持 `PUB-01` 同一父任务，明确两条并行公开数据研究支线（不新增患者项目或新 PUB 父任务）：
- A / `PUB-01-S03`：SparrKULee EEG 听觉直接解码，近期 ICML 方向；主 decoder 在推理时不接受候选音频/候选答案，直接输出连续声音表示，候选任务为冻结输出的后验评估。
- B / `PUB-01-S04`：ds004703 独立 sEEG 声音重建基线（可平行推进）+ 未来 EEG→sEEG 的受控迁移（依赖 S03 冻结 public model）。
- 在 S03 下新增 SS02 强基线复现（READY）、SS03 架构（BACKLOG）、SS04 统一评测（BACKLOG）；在 S04 下新增 SS01 独立基线、SS02 transfer（均 BACKLOG）。原 S01/S02/S03_SS01 的全部历史产物与验收不变。
- “60% 大候选 Top-1 / GPT 后 80%”仅为挑战愿景，不改变 baseline 任务难度、primary acceptance 或允许 test-driven selection；GPT 目前不是执行单元。
- 此项属于父任务 scientific emphasis/acceptance 的正式 amendment；保留最终公共表征、迁移及 `ART-AUDREP-v1` 的原定含义，不将 EEG 结果直接外推至 STN。

2026-10-11：统筹转达用户要求本周继续PUB并交实际初步解码结果。S02在a063/2203的已有交付核验后接续S03_SS01：同一公开SparrKULee cohort、既有split、ridge/TRF方法下增加EEG→envelope线性重建，冻结初步10名范围和null。该方向补充不改变父任务最终Acceptance或artifact语义，不替代S03 neural/retrieval/held-out-subject要求；不宣称语义重建/患者泛化。SS01完成后停止，父任务ACTIVE。

2026-09-19：用户明确要求设置持续 goal，至少完成 PUB-01 数据集下载及初步清洗，并跟进执行而非仅派发后停止。此处将“初步清晰”按上下文落实为初步清洗/质量核验。新增 S01_SS01 数据准备交付；范围仍为 SparrKULee 与 ds004703，全部数据工作在 server2203，原始数据只读保留，清洗产物另存。许可不明/访问受限必须记录真实阻塞，不能绕过；不授权模型训练、患者资料或改变科学 endpoint。协调者在 S01及其SS01 之间核验后推进，无需重复询问已有启动授权。

## 13. Completion Record

2026-10-15：同SS02按事先授权的可用公开条件完成两套方法及直接包络endpoint，Technical PASS / Acceptance PASS_WITH_LIMITATION / Parent consistency PASS；SS02 COMPLETED。HDPS复用100Hz/M、作者split/valid，输入完整76人（排除019/020/021/022/026），Top1/5/10=12.03/30.45/42.10%。Transformer按原fold4/READY记录冻结70train、69seen/13heldout，XLSR19；2AFC加权82.38%、冻结EEG编码器原Conv32包络加权r=.1593；所有选择validation-only、一次test，独立review含逐条评分/CPU重建。两者条件不同，未形成统一协议优劣判定；原完整cohort/隐藏packet/ensemble/M论文数字等限制保留。报告`reports/pub_01/s03/strong_baselines/REPRO_REPORT.md`，7行真实结果与10行原条件NOT_RUN分开。代码/config/命令和2203路径见该包；main起点ec3adef、新增代码未提交，不冒充source commit。未改变S01/S02/SS01、患者边界或artifact语义；S03/PUB仍ACTIVE，无ART-AUDREP-v1。停止SS02，后续研究需研究者指定。

2026-10-13追加：S03_SS02交付固定官方协议矩阵、2203首被试10条数据/GPU preflight、4项技术tests与18项单独review检查；A+B模型smoke/训练/评估NOT_RUN，实验Acceptance HOLD，SS02 WAITING_EXTERNAL。报告`reports/pub_01/s03/strong_baselines/REPRO_REPORT.md`列许可/原协议资源缺口。此记录不修改S02/SS01历史结论、不完成S03或PUB-01；父任务ACTIVE，无frozen decoder/ART-AUDREP-v1，不启动SS03/S04。

2026-10-13同节点续行（以上初次preflight保持历史）：按用户同论文对照目标，原raw100Hz602条、M70story、独立论文功能模型真实batch256工程smoke与独立数值review已完成；作者647对中600对工程就绪，必要47条raw受限。sub022public events容器修复有代码消费字段依据、无APR请求。Transformer冻结EEGcheckpoint/official test未齐；原XLSR声音权重已取得。没有同条件正式训练/测试，不启动76人/64Hz替代结果；作业全部结束，SS02 WAITING_EXTERNAL / HOLD，最小恢复资源见strong_baselines/RESOURCE_RECOVERY。原固定GPT2特征可以属于HDPS方法，仍不执行GPT新模型/rerank。父范围、历史科学结论与artifact验收不变。

`IN_PROGRESS`：2026-10-11已交S03_SS01初步公开EEG→包络解码（10名93条、test/null/静态图/周报、技术及独立review PASS_WITH_LIMITATION），报告见reports/pub_01/s03/envelope_decode_20261011/REPORT.md。S03完整neural/retrieval/held-out-subject、S04和S05尚未完成，父任务保持ACTIVE；未生成ART-AUDREP-v1，未进入患者侧。

## 2026-09-24 用户修订：先探索数据，再决定正式实验规则

用户明确要求下载完成即开展探索性分析、画图并报告实际数据内容。此前将正式benchmark准入作为所有分析前提的解释作废。SS02追加图形化数据概览与问题定位，优先交付实际图和中文说明；旧准入报告作为历史判断保留，不能作为阻止探索的门槛。

- 已下载且已获本项目内部研究使用授权的公开数据，可直接做描述性统计、代表性波形/频谱、事件叠图和对齐诊断。已知损坏/不能读取的对象跳过并注明，受限载荷不请求；不为未知物理同步、历史预处理revision或尚未分配fold停止其余探索。
- 基础可读性、维度和坐标单位检查与分析同时完成；问题标注到图表和已存在的记录索引中，不再先全量重跑QC或循环完善审计文字。
- 正式训练、性能比较、因果/生物学结论仍依赖适合相应问题的预处理与数据划分。探索不宣称这些已通过，也不以相关峰或好看的图证明硬件同步。
- 数据/权重/信号读取与计算继续仅在2203。允许回传本轮必要的静态汇总/示例图及轻量HTML/Markdown报告供用户阅读；不回传原始音频/神经信号数组或嵌入它们的交互式载荷。
- 不读取本项目私有患者/STN资料、不扩大下载范围、不自动训练；只有实际访问/读取失败或明确使用禁止才阻断受影响对象，其余可进行的分析继续。

## 用户阶段验收与后续安排

用户明确接受S01作为“公开数据准备、内容理解及初步探索”阶段完成。SS01已完成公开范围下载/读取核验，SS02已提交真实信号、PSD、音频/事件和尾差图。现将S01及SS02标记COMPLETED / PASS_WITH_LIMITATION，转入S02。

这是阶段范围与验收安排的显式修订，不是声称全部QC或模型准入已经解决。原准入报告中的640/26划分、ds004703 HOLD保留为历史判断，不再作为整体停止条件。仍有科学价值的问题随具体分析处理；3秒尾差提示不作为整条记录自动排除阈值。不能读取的对象跳过，不能配对的区间单独排除。受限载荷不请求，不读取私有患者/STN资料。

S02先完成SparrKULee声学基线：复用发布方64Hz EEG/envelope，按实际共同区间分析；最小必要清洗、对齐和分组检查与实现同时完成。拟合前固定首次实验范围、时间滞后、正则化候选、分组和指标；选择只使用训练/验证数据，测试集不参与。分别说明新故事和新被试的评估目标，不能笼统要求所有分组维度同时不交叉。

首个交付是可复现的ridge/TRF预测结果、时间错配对照、逐被试性能表和实际图文解释。先小规模跑通，再执行预定首轮范围；阴性或失败照实交付，不反复改方案追阳性。ds004703保留已有探索，不阻塞EEG首轮；其信号预处理在颅内分析时按实际需要处理。

完成第一版声学基线后汇报；这不等于S02全部完成。深度特征与S03以后不自动启动。所有数据计算继续在2203，报告可回传静态图与轻量结果。无需补修上一版图形排版或重做全量审计作为本轮前置条件。
