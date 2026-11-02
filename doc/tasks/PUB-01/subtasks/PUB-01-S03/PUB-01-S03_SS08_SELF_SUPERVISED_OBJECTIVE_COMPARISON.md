# SUBTASK — PUB-01-S03_SS08 — 固定主干的EEG自监督训练目标对照

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S03_SS08` |
| Parent Node | `PUB-01-S03` |
| Parent Task | `PUB-01` |
| Title | Fixed-backbone masked context versus causal future prediction for auditory EEG decoding |
| Status | **`BACKLOG` / DESIGN_ONLY — NOT_RUNNING**；本文件为研究者拟议任务书，**未授权抢占正在执行的SS07** |
| Blocking for Parent | NO；独立方法学对照，阴性结果合法 |
| Patient Data | **`FORBIDDEN`** |
| Upstream Subtasks | SS06 `COMPLETED / PASS_WITH_LIMITATION`；SS07 `ACTIVE`、当前外部确认仍在运行；开始训练前须再次核对当前冻结状态 |
| Inputs | 既有获准的公开EEG与声音包络、SS06 M3及SS07强基线配置和合法train数据；**SS07当前ds004408运行数据/评估保留，不自动纳入SS08** |
| Deliverables | 结构统一/训练目标比较的冻结协议、Codex代码和运行日志、训练/验证与（具资格时）新保留集结果、消融、阴性记录、三级审阅 |
| Acceptance | 明确遮挡表征与未来预测各自是否帮助声音解码及历史依赖；相同数据/主干/下游任务、规范预算与暴露控制；不以SSL loss下降替代声音解码改善 |
| Validation | Technical → Acceptance → Parent consistency；Codex负责实现与运行，协调者只审阅科学设计与证据 |
| Go/Hold/Fail | 若SS07仍占用资源或无有效新heldout，则 `HOLD_RUNTIME`/`HOLD_CONFIRMATION`；train/val探索可单列，不能虚称外部确认 |
| Files Allowed to Change | 新的 `SS08` 源码/config/test及 `reports/pub_01/s03/ssl_objectives/`；经审阅方可更新父登记。**禁止改SS02/03/05/06结果、SS07当前执行入口、已有冻结protocol/queue** |
| Last Updated | `2026-11-02` |
| Source review | PR#6 `codex/pub-01-s03-ss06-continuity`，基准commit `95f0013fff01d59efef1ec0eb68a97278206108f` |

## 1. Objective / 科学问题

**固定EEG编码主干、下游声音解码读出及输入数据，比较不同的自监督时序学习目标是否能学到更有用的听觉相关EEG信息。**

核心区分：
1. **上下文补全**：遮挡一段连续EEG，允许看前后两侧，判断是否能从上下文恢复原信号或其神经表征。这是离线/双向信息任务。
2. **向前预测**：仅用过去的EEG预测未来EEG表征，判断是否能建立因果方向的时间记忆；不得读取未来作为输入。
3. **下游迁移**：预训练能否改善相同连续 EEG→声音包络任务，是否提升对真实历史的内容特异利用，而非只学低频平滑/记录身份。

不要把“预训练重建误差更低”“模型会用未来上下文”“模型可预测EEG”直接写成“听觉信息表示变好”。更不能把EEG→包络写成可听语音重建或STN编码机制。

## 2. Existing evidence / 先消化M3，禁止无视阴性结果

SS06 `eeg_run01` 的 **M3已执行未来神经特征预测**：64Hz、causal LocalEncoder+GRU context、同输入无标签EEG的EMA teacher、horizons `(4,16,64)` samples（62.5ms/250ms/1s）；future teacher latent仅供训练损失，推理不使用。3个seed各3511个完整60s训练片段做SSL（fresh state，非跨段teacher持续记忆），监督下游统一3262步/3seed；frozen和finetune使用同seed预训练起点；GRU为相同构架no-pretrain对照。

前次 **prospective INTERNAL / EXPLORATORY audit**：GRU `r=0.125221`，M3 frozen `0.061478`，M3 finetune `0.123747`；finetune−GRU `Δr=−0.001474`，subject/source-family Bonferroni区间 `[−0.005002,0.001951]`，未检出相对scratch的额外预训练收益；finetune明显胜frozen。这不是自监督原理被否定：训练步数有限，模型/目标/有效时间尺度/数据量可导致阴性，不应根据旧audit反复追参。

SS07后续大规模强Transformer实验及D11内部分析也**未稳定证实额外过去信息收益**；其原 `P` / `Ccap` / `Past` 等结果来自不同目标与观察权限，不能作为本新实验相同协议的数值基线，也不能在新模型设计时把已打开的测试值当验证集。SS07截至2026-11-02仍在公开ds004408外部确认数据预处理/最终冻结阶段，**不得修改/并发干扰其服务器作业或结果解释**。

证据出处（只读）：
- `reports/pub_01/s03/continuity/eeg_run01/REPORT.md`、`FORMAL_PROTOCOL.json`；
- `src/m6a_public/continuity/predictive.py`；
- `reports/pub_01/s03/incremental_continuity/observation_state_probe/run01/REPORT.md`；
- `doc/CURRENT_TASK.md` 最新SS07运行状态。

## 3. Fixed structure / “固定网络结构”究竟意味着什么

正式比较前，Codex必须冻结一个统一的 `EEG backbone E_θ` 与统一的下游 `envelope readout G_φ`，包括**层数、隐藏维度、参数量、卷积前端、位置编码、有效输入采样率/长度、归一化、输出头、初始化和训练时钟**。所有主比较臂使用相同编码器与下游readout，实现同seed配对初始化及相同监督训练。

**特别注意：SS06 M3是单向GRU，本身无法在被遮挡的中段同时利用右侧未来输入。** 因此不能声称在原M3结构完全不变的情况下自然完成双向cloze。Codex可在预实验审计后确定一个支持固定参数下切换**双向/因果attention mask**的统一模型（如固定层数的Transformer），并将它用于所有臂。若保留GRU，双向补全需要额外反向计算/共享权重机制，必须明确计算和可用上下文的差异。**正式训练前需做一次技术选择并冻结；不得因test结果换backbone。**

当双向遮挡与因果预测采用不同attention可见性时，**两者同时改变了训练目标与预训练输入权限**，直接B−C差值只能解释为训练策略差异，不能纯归因于loss；增加一个相同因果可见性的“未来连续块遮挡”对照以分离权限因素。下游全部臂采用**同一事先固定的信息权限**（优先与所选现有强baseline一致，单列offline/causal）；不得让某一模型额外看未来EEG而另一个不看。

所有SSL专用predictor/reconstruction heads只参与预训练，微调时移除；头部参数和SSL计算成本单独计账。若已有SS07 native强基线checkpoint初始化被选择为共同起点，则所有臂同样使用该**相同合规train-only起点**，不能混用旧见过新heldout的checkpoint。否则统一随机初始化，不能把两种起点的分数混在一起。

## 4. Prespecified arms / 先三组，组合暂缓

| 组别 | 自监督阶段 | 统一下游阶段 | 首轮性质 |
|---|---|---|---|
| **A — scratch** | 无 | `EEG→envelope` 直接监督训练 | 必做基线 |
| **A+ — compute-matched** | 不使用新SSL；在相同合法train数据上做等额额外监督训练/训练更新（明确目标访问量） | 同一读出与最终选择规则 | **必做成本对照**，不能把额外SSL计算隐去 |
| **B — contiguous mask** | 遮挡整个连续EEG片段，恢复**被遮挡位置的teacher latent**，可单列原始波形预测 | 与A完全同一微调协议 | 必做 |
| **C — causal future latent** | 仅以过去预测未来4/16/64采样点的stop-gradient teacher latent，保留M3复现对照 | 与A完全同一微调协议 | 必做 |
| **B_causal** | 与C同一因果可见性，预测后续完整遮挡时间块的latent | 同一微调协议 | 机制控制；若算力不足，明确HOLD机制归因 |
| **D — joint B+C** | 同一预训练预算内进行加权双目标，权重仅在train/dev定 | 同一微调协议 | **第二阶段可选，不在首轮自动启动** |

第一轮先执行 A/A+/B/C 和必要的B_causal权限检查；**不得一上来加入D、通道遮挡、多个新foundation模型或无限多网格**。D、通道遮挡属于下一个预注册消融，需看第一轮train/val证据后由研究者决定。

对B：在train/dev只探索连续mask **32/64/128 samples**（0.5/1/2s）及预先限定的mask覆盖率；不得使用随机独立采样点遮挡伪装长上下文。主目标优先teacher表征，原始EEG的标准化波形重建另作同目标简单插值基线。可补多尺度带通/谱域目标，但不得在第一轮三种任务和多个loss上做无界组合。

对C：以M3已有 **4/16/64 samples** 为必备复核，128 samples（2s）最多作为train/dev上预登记的额外诊断；必须记录不同horizon的loss、有效监督配对数量、teacher方差和collapse监测。不得用高horizon低loss直接选最好的声音解码模型。

**只遮通道**可以检验空间信息冗余和坏通道鲁棒性，但与时间历史利用的主假设不同；仅作为完成B/C后独立登记的次要任务，避免本次结果难以归因。

## 5. Data and leakage contract / 数据、时间与前处理

1. 所有用于**自监督预训练、teacher训练、train-only统计**的EEG必须来自本次合法 `train`，不能把validation/audit/test无标签EEG偷偷加入预训练后再宣称严格inductive generalization。保留subject、record、story、source_family、session与完整支持坐标。
2. 预训练数据划分先按来源/故事/被试关系审计：**不能将相邻窗口、同一录音的另一些窗、同一故事的不同录音随机分进train与heldout**。如无法形成新的独立证据，允许train/val探索，但只记 `HOLD_CONFIRMATION`。
3. SS02/03/05/06以及SS07的原test/audit/DTU相关指标已经在开发中查看；不得把它们重新命名为“全新独立测试”，也不得再用于本任务架构/预算/lag/窗口选择。正在处理的ds004408仅在SS07当前已授权scope中使用，SS08没有自动取得其训练或测试许可。
4. 遮挡必须发生在最前面的模型可见输入之前：**不能先用完整EEG做卷积/时频变换再把latent盖住**，从而让缺失片段经卷积邻域、overlap、时间域归一化、padding或标签重新泄漏。teacher可访问原始完整train目标，但teacher输出必须detach；不能作为student输入。仅对真正被遮挡的中心区计B损失，并按有效支持排除掩码边界泄漏。
5. 未来预测的每个输入计算图必须满足 `output(t)` 不依赖 `EEG[t+1:]`；teacher的未来特征只是loss target。若teacher本身有双向感受野，则应显式限定目标特征的支持，禁止未来目标泄漏进student或跨record。
6. 任何lag、带宽、mask长度、loss权重、归一化与encoder状态reset规则只在train/dev确定。已知发布方 `physical_sync=UNKNOWN` 不能因本实验推断精确生理延迟。下游推理不输入候选音频/ID/真实声音包络。
7. **独立project边界**：本任务只在 `Auditory_Simulation` (M6A)；`STN_Decoding_Encoding` (M6B) 的患者、STN、SEEG及派生embedding永不进入本项目。若未来交付，只有冻结、版本化公共artifact单向传入患者项目。

## 6. Fairness / 训练成本与评估设计

第一轮每种SSL方法以相同**预训练可见train EEG token数、有效target token数（若不可等，报告）、优化更新上限、验证频率**训练。A与B/C不可能自动满足“相同总算力”，所以A+要匹配**预训练+微调总消耗的优化/计算成本**，并额外报告GPU-hours、FLOPs估计、有效参数与实际训练样本重复遍历。每个A/B/C在声音微调阶段的train、batch顺序、seed、更新上限、早停与readout完全一致；不得只挑预训练后的最佳seed。

初步固定 `seeds=(1024,2026,31415)` 至少三个；训练/验证工程pilot可先1seed，**pilot不读正式新heldout**。正式训练前冻结统一下游声音目标、完整60s Pearson/subject-macro、MSE/音频时延支持、pairing与统计方法。学习率等细节由Codex在train/dev有限网格选后写 `FORMAL_PROTOCOL.json` 冻结，不能沿用一组方案的最优后对另一组不调公平性却声称无偏比较。

必须显式区分两种推断：
- **pretrain效果**：同backbone同finetune，B/C减A，在相同supervised budget下；B/C减A+可讨论总计算成本效率，不能无视额外预训练。
- **音频表征收益**：`EEG→envelope` 的person/record平均60s Pearson r（primary）及train-only标准化MSE（secondary）；将每seed点、paired per-person差与分组bootstrap区间呈现。不得跨SS06/SS07不同split/loss/window直接排名。

简单对照不可省略：B原始波形重建要对照**线性/样条插值和局部频段均值**；C波形/latent预测对照**持值/线性自回归（AR）/简单谱预测**；复杂模型不超越简单模型时不得称其学会高级时序关系。mask/forecast的自身训练loss只是辅助诊断，不用于确认下游改善。

历史信息机制检查必须**保持当前同一段EEG输入及参数不变**，分别用真实历史、清空状态、同一record错时历史、跨record错配历史；阻断record identity shortcut及声学目标泄漏，并明确不同方法是否实际有可传递状态。计算 `Δr_history=r(true_history)-r(mismatched_history)`，按person/record而非重叠窗进行推断。mask重建的双向结果**不能**拿来证明因果向前预测。

## 7. Codex ordered execution / 不打断SS07

### Gate 0 — 纯只读预审（当前唯一可推进动作）

- 读取 `AGENTS.md` 与父任务/本任务；核对SS06 M3实际训练结果及EMA teacher/source、SS07当前checkpoint与外部处理状态，核对是否存在可独立使用的新的train-only数据及新heldout。
- 确定“固定网络”的唯一配置与对B/C可见性支持，说明为何双向mask与因果future不会变成不一致的模型结构。准备 `DESIGN_REVIEW.md`、成本预算和冻结数据分组方案供研究者审阅。
- **当前SS07独立确认仍在进行**：不得中断/复用其PID/任务队列、改预处理、借用其新source样本训练或抢占GPU。当前阶段不运行训练、不生成新performance分数。

### Gate 1 — 仅在SS07明确释放/研究者授权后

- Codex自己实现统一backbone/损失模块和元数据严格loader；进行CPU/GPU forward/gradient和mask/causality/no-leak单元测试。现有SS06/SS07实现可只读参考，不能修改旧canonical源来适配新结果。
- 用既有合法train/validation小样本跑有界1-seed pilot，验证B连续遮挡真实难度、C各horizon可学性、teacher EMA、内存与速度。阴性、collapse、NaN均如实记录。
- 确认train/val规模可行后**一次冻结**四主臂、共同init、结构/时间权限、目标、训练预算、seed、统计、多重比较族与停止规则，禁止test驱动试错。

### Gate 2 — 有条件正式实验

- 仅在新 `FORMAL_PROTOCOL.json` 完整且有资格的、新冻结holdout可用时，Codex执行3seed主对照、一次新heldout评估和独立数值审阅。若数据只能提供反复查看的开发集，交train/val结论，标 `HOLD_CONFIRMATION`，不得虚称独立证据。
- 第一轮只在A/A+/B/C及因果mask检查上得出结论；**D联合预训练、通道遮挡、新foundation模型、S03_SS04或S04**都不自动执行。

## 8. Required deliverables / 文件和验收

Codex在独立新目录 `reports/pub_01/s03/ssl_objectives/run01/` 交：
- `DESIGN_REVIEW.md`：模型固定与B/C可见性、M3差异、训练目标与可证伪假设，已知论文对比；
- `DATA_CONTRACT.md` 与 `SPLIT_PROVENANCE.md`：公共合法数据、train-only统计、source-family/subject/story隔离、mask/lag坐标、已暴露集合；
- `COMPUTE_MATCHING.md`：A+和B/C的实际总算力/样本/目标访问公平性；
- `FORMAL_PROTOCOL.json`：只有确实冻结才创建，含source/config/split/checkpoint identity及seed/停止规则；
- `RESULTS.csv`、逐人表与 `REPORT.md`：区分SSL任务效果、train/dev效果、合格新heldout及其统计；无数据不得伪写；
- `REVIEW.md`、`EXECUTION_LOG.md`：Codex实际执行命令、CPU/GPU/来源复算、负例/失败、三级验收；无正式运行状态仅 `NOT_RUN`。

三级验收：
1. **Technical**：shape/causal/mask boundary/teacher detach/数据独立/预算/结果可复算。
2. **Acceptance**：A/B/C是否有额外下游收益、历史内容特异效应与简单基线差距；允许 `PASS_WITH_LIMITATION` 的完整阴性结论；**不能因loss下降或低r差点估计提前宣布方法创新**。
3. **Parent consistency**：不改变SS07现有外部确认及旧报告，不将开发集称external confirmation，不触及患者项目；全量公开项目目标仍ACTIVE。

状态规则：`BACKLOG / DESIGN_ONLY` → （设计审核/资源门全部允许后）`READY` → `ACTIVE` → `REVIEW` → `COMPLETED / PASS[_WITH_LIMITATION]` 或 `HOLD/REWORK`。协调者只做任务编排和结果审查；**实际代码/实验必须由Codex执行**。

## 9. Relevant literature / 只用于原理与方法区分

- Kostas et al., BENDR (Frontiers in Human Neuroscience, 2021): https://doi.org/10.3389/fnhum.2021.653659 。连续mask表征/对比预训练，并非简单波形缺口插值。
- Jiang et al., LaBraM (ICLR 2024): https://arxiv.org/abs/2405.18765 。EEG channel patch的神经code/token遮挡预测。
- Wang et al., EEGPT (NeurIPS 2024): https://papers.nips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html 。mask-based spatial-temporal representation alignment。
- SS06 M3不是完整CPC复现，SS08也不得使用BENDR/LaBraM/EEGPT名称来暗示逐项复现原论文。

## 10. Completion Record

**NOT_STARTED — 仅协调者编制任务书，无Codex新任务执行、无新代码、无新数据/模型运行。**
