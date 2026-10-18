# SUBTASK — PUB-01-S03_SS02 — Strong EEG Auditory Decoding Baseline Reproduction

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S03_SS02` |
| Parent Node | `PUB-01-S03` |
| Parent Task | `PUB-01` |
| Title | Strong EEG Auditory Decoding Baseline Reproduction |
| Status | `COMPLETED` / `PASS_WITH_LIMITATION`（2026-10-15可用公开条件两基线与直接包络endpoint；原完整论文条件未复现） |
| Blocking for Parent | YES |
| Patient Data | `FORBIDDEN` |
| Upstream | `S01` / `S02` COMPLETED, `S03_SS01` COMPLETED / PASS_WITH_LIMITATION |
| Inputs | server2203 已授权公开 SparrKULee、官方论文与开源代码（先独立核对）、冻结 protocol/data manifests |
| Deliverables | 文献/代码对标矩阵、复现环境与运行日志、完整指标/逐被试表、误差分析、可复现审阅报告 |
| Acceptance | 目标：HDPS + Transformer-CLIP-EEG 至少两套独立强基线完成实际结果与可复现审计；若任一缺资源/许可/官方测试定义只能标 HOLD/PASS_WITH_LIMITATION，不能伪称两套成功；challenge 先审查协议；不伪造接近论文数值 |
| Validation | Technical → Acceptance（PASS/PASS_WITH_LIMITATION/REWORK/HOLD/NO_GO）→ Parent consistency；独立审阅 |
| Go/Hold/Fail | 作者数据划分/候选构造无法获得→该 protocol HOLD；可复现部分单列；绝不调整测试难度追成绩 |
| Files Allowed to Change | `doc/tasks/PUB-01/**`、`doc/CODEX_START_HERE.md`、`configs/pub_01_s03*`、`scripts/pub_01_s03*`、`src/**`（限新复现适配）、`tests/test_pub_01_s03*`、`reports/pub_01/s03/**`；禁止改写历史产物 |
| Last Updated | 2026-10-15 |

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

2026-10-15最终Completion Record：

- 输入/版本：复用2203既有100Hz/M与作者HDPS commit42a5e1ec split/valid，76完整输入被试/599对；不完整019/020/021/022/026在拟合前排除。Transformer作者commit8be963ab/fold4，可用READY64Hz长记录70train、69seen/13heldout，原Apache2 XLSR Dutch19；cohort/query在拟合前固定，原official packet不可用。
- 实际结果：HDPS Top1/5/10=12.0327/30.4454/42.1004%，100–235候选，shuffle Top1=.5291%；Transformer2AFC seen83.5295/heldout80.0786%，加权82.3792%，shuffle约50%；冻结EEG encoder原Conv32读出直接包络r=.168810/.140314，加权.159311，shuffle近0。三项均仅validation选checkpoint后一次test；2AFC与大候选不直接排名。
- 预算/偏差：HDPS9轮、best3；Transformer24轮、best8/patience15；readout250轮、best249，达到事先上限时val仍改善。原完整paper cohort/ensemble/official packet未复现；M论文数值未核实，未用MWG20%作差；源family/物理同步/单seed限制保留。没有test-driven调参/重选人或扩展模型。
- 输出/命令：`reports/pub_01/s03/strong_baselines/`四mandatory报告、17行RESULTS（10原条件NOT_RUN+7实际分组行）、逐被试CSV/静态图、COMMANDS/SERVER_MANIFEST。2203运行根`pub_01/s03_strong_baselines_20261015/{hdps_m_available01,transformer_available01}`；信号/特征/权重/逐query评分留服务器。本地轻量证据在`evidence/available_20261015/`。
- 代码/config改动：`configs/pub_01_s03_{hdps_m_available,transformer_available,transformer_envelope}_20261015.json`，`scripts/pub_01_s03_*`中cohort guard、HDPS prepare/train/review、Transformer inputs/train/readout/review/finalize/package/plot；维护本spec、parent登记/入口和报告。历史TRF/CNN未覆盖。branch main，起点ec3adefc5c7d3dd37e1b043fb1956a7dda11cda1；新代码未提交，不把起点伪作运行代码commit。
- 失败/测试：保留torchaudio CUDA依赖、尾clip分组、DataParallel/LSTM InferenceMode失败日志；后者无test，修复no_grad后相同seed从头跑，双GPU train→val→train smoke通过。严格/可用模式5项tests通过。HDPS独立rank重算误差4.77e-8，Transformer2AFC重算0，readout独立CPU全test段重建maxr误差8.18e-6、汇总0、0无效段；包schema/原未跑项/轻量回传检查见available_package_review.json。
- 三级验收：Technical PASS；Acceptance PASS_WITH_LIMITATION（用户预先授权可用条件方法复现，不声明完整论文数字一致）；Parent consistency PASS。审阅为同一agent另行post-execution pass，不是外部签署。SS02 COMPLETED不等于S03/PUB完成；无ART-AUDREP-v1，SS03/SS04/S04仍BACKLOG，当前停止。

2026-10-15执行记录（进行中）：用户经协调会话明确“缺几个就缺几个，没关系”，授权在现有合法输入上正式训练/验证/测试。父TASK Amendment已记录。复用既有600/647工程就绪对及70story M，不重做预处理；按预定输入规则筛完整被试（每名保留全部作者train/val/test story，任何必要输入缺失则整名排除），拟合前冻结，不按test表现挑选。新运行路径`2203:pub_01/s03_strong_baselines_20261015/hdps_m_available01`，新配置`configs/pub_01_s03_hdps_m_available_20261015.json`。原严格模式仍拒绝缺项；明确授权的available_complete_subjects允许源数据缺项，但选定cohort必须完整且无selected input failure。原论文M数字未核实不阻止实际指标交付，M不冒充MWG20.0%，Transformer后续在可实现合法条件下推进。此记录撤销10-13“不得子集训练”的当前约束，历史记录保留。

2026-10-13追加执行记录（不代表SS02实验完成）：

- 用户经协调会话明确只要同论文条件的复现实测。原64Hz内部适配配置/prepare留NOT_RUN；不启动76人替代训练，不以M对照MWG20.0%。原方法固定GPT2特征可以归属HDPS复现，条款未明确时局部等待，不执行新GPT/rerank。
- 在2203隔离env01执行GPL Challenge2024/brain_pipe的原1024Hz→100Hz处理，参数逐项对应作者版本；取得固定public valid_seg并安全读取70story纯metadata。81个原raw作业结束，602条100Hz产物覆盖77人部分输入，作者600/647对工程就绪。原raw/张量/权重仍只存2203。
- 原Mel参数70story全部生成且finite；实际NPZ fs→16k时长保真，避免错误WAV头。按作者get_feat的feature长度尾截断处理原索引，不test选窗；初次summary错误归入10个尾项不支持，复查原contract后保留快照并纠正。
- 独立论文功能模型（非clean-room，不import无LICENSE作者模型）用实际100Hz/M batch256前向/反向/Adam工程smoke通过，有限梯度、两L40各约10.3GiB；重复window只作资源检查。正式训练/原论文测试NOT_RUN，无性能数字/checkpoint/prediction。
- 作者split必要47条raw=37train/5val/5test仍restricted，分属sub019/020/021/022/026；49受限BDF中2条无精确作者story映射不计必需。逐对象sidecar→story→phase在2203完成，未请求受限payload。sub022另1条public raw初因APR缺失失败；GPL代码证明APR仅加snr，选定grouper和EEG数值步骤不消费snr，公开events给相同语义/原路径；显式v7容器修复恢复该记录，无APR访问。失败与修复目录均保留。
- 论文/code差异预先固定在configs/pub_01_s03_hdps_paper_m_20261013.json（同一Implementation段lr/subject/FFL，代码补未写细节，不test选配置）。原M TableI/II图像数值仍未核验，留空。Transformer两个checkpoint入口真实失效，有界迁移查验结束；原XLSR Dutch权重明确Apache2并仅2203取得，不能代替EEG模型；GroNLP权重条款不从关联repo Apache推定。
- 三级验证：Engineering PASS限定（独立NumPy Pearsonmax误差1.19e-7、真实模型backward、raw/speech时间支持）；Acceptance HOLD（无同条件结果，不能说一致或不一致）；Parent consistency PASS（public-only/2203-only/不改历史结论）。raw两次不同BLAS线程结果数值差1.69e-10μV，非bit-exact；第一次严格检查REWORK保留说明。
- 当前交付追加到原4报告及COMMANDS/SERVER_MANIFEST，轻量evidence/continuation01与最小RESOURCE_RECOVERY；训练入口拒绝inputfailures。未执行prepare聚合partial tensors/正式train，未提交Git、未移除旧资产、未启动SS03/S04，无后台作业。协调会话集中向用户询问合法资源位置，执行会话不重复过程清单。
- 同节点恢复条件为47条必要raw/无法由公开信息等价恢复的metadata或同源100Hz包；原三特征/权重条款与Transformer对应冻结模型+官方测试位置。仍归SS02，不改变父验收、不产ART-AUDREP-v1。本轮已有可完成工程作业全部结束，等待同条件资源。

以下首次执行记录保留历史原文，其无索引/无依赖/无100Hz、GPT一律不计算等解释已由追加记录纠正，不再作为当前整体停止理由。

2026-10-13 execution record（不代表SS02实验完成）：

- 输入/版本：main `ec3adefc5c7d3dd37e1b043fb1956a7dda11cda1`，2203既有SparrKULee V3.1、S01 cleaning index、S02 split；HDPS/Transformer/Challenge2023/2024固定源码commit见`reports/pub_01/s03/strong_baselines/PROTOCOL_MATRIX.md`。保护原stash、两处tracked改动、旧CNN资产与全部frozen证据；未切换/重置Git或初始化远端Git。
- 预执行门：ID/parent/upstream/边界/允许文件/验收一致；训练准入未通过：HDPS与Transformer独立代码/权重许可未明确、原协议数据/特征条件未满足。因此没有进入模型ACTIVE，也未冻结替代内部训练协议。静态源码与已授权数据可读性检查在许可门之外完成。
- 实际执行：2203 `pub_01/s03_strong_baselines_20261013/run01`保存37个源码HTTPS读取失败；`run02`经轻量源码staging读取37/37成功、Python AST通过。sub-001全部10条indexed EEG/envelope读取/shape/finite/common-support通过；CUDA基础smoke通过；**作者模型smoke、训练、评估全部NOT_RUN**。HDPS作者81人647 record-phase对有现有64Hz路径，但没有100Hz/相同目标就绪证据；Transformer manifest仅71/91 train subjects存在，官方test payload缺失。
- 交付：`reports/pub_01/s03/strong_baselines/{PROTOCOL_MATRIX.md,REPRO_REPORT.md,RESULTS.csv,REVIEW.md,COMMANDS.md,SERVER_MANIFEST.csv,review_checks.json,evidence/}`；全部未运行性能指标留空，无checkpoint/预测/最终candidate manifest。源码许可、候选定义、source-family与采样率差异、官方测试包和语言先验分列。
- 新代码/config/test：`scripts/pub_01_s03_strong_preflight.py`、`scripts/pub_01_s03_stage_source.ps1`、`scripts/pub_01_s03_review_strong.py`、`configs/pub_01_s03_strong_baselines_20261013.json`、`tests/test_pub_01_s03_strong_preflight.py`。实际命令和服务器仅存资产见COMMANDS/manifest。以上是审计入口，未伪装为完整训练adapter。
- 文档修改范围：本任务书、parent SUBTASKS/TASK、S03主节点、BENCHMARK_REGISTRY、RESEARCH_BRANCHES及必要的CODEX_START_HERE/CURRENT_TASK/TASKS路由状态同步；未修改项目科学目标/验收或原历史记录。新增工作尚未提交，source commit字段为起点main，不冒充新增代码的commit。
- 三级验收：Technical **PASS（preflight/审查包限定）**；Acceptance **HOLD（A+B实际复现未达到）**；Parent consistency **PASS（public-only，父ACTIVE，未交artifact）**。单独post-execution review为同一agent独立步骤；2203四项unit tests和18项metadata检查全部通过，不等于模型验收。
- 限制/继续条件：作者可核验代码/权重许可；HDPS原100Hz处理/特征和有效片段/candidate定义；Transformer合法官方test包与权重/训练manifest。完整paper训练含非公开数据，不请求它；公开子集复现须另列部分/内部协议，不能声称原paper数字。challenge新增资源不自动下载，GPT不计算。详见REPRO_REPORT逐项条件。
- 父任务影响/停止：SS02保持WAITING_EXTERNAL / HOLD；S03与PUB-01仍ACTIVE，SS03/SS04/S04 BACKLOG，未形成ART-AUDREP-v1。下一步仍是SS02条件满足后的模型smoke与实际复现，**本轮到此STOP**。未将HOLD伪称完成或自动启动新架构。

SS02实际复现完成后，研究者审阅决定SS03的模型创新；不自动训练SaShiMi、TimeMixer++、GPT或任何未经批准的新模型。
