# PUB-01 — 两条公共数据研究支线（2026-10-13 决策）

此文件是父任务 `TASK.md` 的执行路线图，**不构成第二份状态登记**；状态一律以 `SUBTASKS.md` 和 `doc/TASKS.md` 为准。

## 科学与论文目标

近期 ICML 方法论文的优先目标：研究一个**不依赖候选声音作为模型输入**的 EEG→连续听觉表征解码架构，并在同一模型冻结输出下评估直接重建、二选一、五选一、百余候选声音片段检索。追求公开可比协议上的强基线/SOTA；60%/80% 是研究者提出的冲刺愿景，**不是已实现效果、验收硬阈值或允许更改测试集的理由**。

“直接解码”最低限度是：模型输入只有 EEG（以及推理时允许的固定元数据），输出时间对齐的连续声音特征（先 envelope / Mel 或获准的冻结语音特征）；**不能在编码器前给它候选语音 ID、答案文本或候选音频**。输出音频特征不等于恢复可听语音、逐词转写或语义。候选检索只能是对固定预测输出的后验评分/检索，不得依测试候选集重训 EEG 主模型。

## A 支线 — SparrKULee：优先完成 EEG 听觉解码论文

归属：`PUB-01-S03`；前置 `S01` 数据准备与 `S02` 声音→EEG 编码/特征基线均为 COMPLETED / PASS_WITH_LIMITATION。

- `PUB-01-S03_SS01`：已完成 10 人 / 93 条 EEG→包络 backward TRF（历史证据完整保留，**不代表语音内容解码或 SOTA**）。
- `PUB-01-S03_SS02`：2026-10-15 **COMPLETED / PASS_WITH_LIMITATION**。HDPS76人原100Hz/M，Top1/5/10=12.03/30.45/42.10%；Transformer70训练/69seen/13heldout，2AFC加权82.38%，EEG-only包络加权r=.1593；三运行独立review完成。完整论文/official test/ensemble等价性未通过；不同候选任务分开报告。当前停止，后续需研究者指定。
- `PUB-01-S03_SS03`：2026-10-19 **COMPLETED / PASS_WITH_LIMITATION**。多尺度对三单尺度、4×3真实同预算训练/test；seen/heldout r=.10305/.11380，高于long .06411/.06979及middle .09297/.10413。限四原型/当前fold预算，非生理机制或超过SS02；报告与same-agent separate review在`reports/pub_01/s03/architecture/`。
- `PUB-01-S03_SS05`：**ACTIVE / SYNTHETIC_PROBE_COMPLETED**。历史预测误差、多速率状态和关联记忆已实现并通过本地合成任务；公开EEG/新heldout均未执行。保留阴性事实：未证明预测误差更新额外有效。旧SS03 test不能重复用于SS05选结构；下一步按 `PUB-01-S03_SS05_PREDICTIVE_MEMORY.md`。SS05在逻辑上为SS04前的独立探索，编号保留以保护既有任务文件。
- `PUB-01-S03_SS04`：锁定单一 decoder，比较直接特征预测、二选一、五选一、百余候选检索；分别保留原论文 protocol 和统一的内部 benchmark，不混用准确率。
- S03 parent 完成需要 S03 范围内的跨 story/subject、强对照、负例和消融证据；未达到则保持 ACTIVE/REVIEW。

必须保留 `S02` 已完成的 acoustic/log-mel/frozen wav2vec2→EEG encoding；这是**正向编码对照**，不能被写成已经得到 EEG→wav2vec2 decoder。

## B 支线 — ds004703：公开颅内独立基线与后续迁移

归属：`PUB-01-S04`。数据准备已在 `S01` 阶段完成有限制的 inventory/virtual QC；**当前仍 BACKLOG，未训练模型**。

- `PUB-01-S04_SS01`（BACKLOG）：独立审计公开 sEEG 的刺激配对、时间轴、记录/电极/被试及可用任务，建立它自身的直接重建基线。**不依赖 S03 模型成功**；仅需 S01 可用资料和本支线科学准入。
- `PUB-01-S04_SS02`（BACKLOG）：只有 S03 冻结可复现的 EEG decoder/表征、S04_SS01 独立 baseline 后，才比较从头训练、冻结公共表示与小适配；无收益/负迁移必须保留。

两个数据集不能被默认为同任务、同脑区、同信号分布；EEG→sEEG 不是默认生物学层级，跨模态只在共同可定义的声音目标上比较。

## 最终父任务与患者侧边界

`PUB-01-S05`：整合 A/B 两支线的真实 benchmark、失败、许可证和版本，决定能否形成 `ART-AUDREP-v1` candidate。S04 transfer 阴性不导致重追网络阳性。所有公共载荷/权重/训练留在 server2203，Windows Git 只存轻量代码/报告。患者数据和 STN embeddings 禁止进入公开项目；只有冻结的公共 artifact 可单向进入患者 `INT-01`。STN direct READ 完全独立推进。

## 当前执行路由：SS05交接Codex（2026-10-19）

- **科研问题**：历史信息如何持续参与当前 EEG→连续声音特征预测；真实历史是否优于错配历史；关联读写是否优于GRU/状态更新；预测误差项是否提供进一步增益。维持通用时序模型和公开方法学定位，不把合成任务阳性等同于听觉生物学证据。
- **唯一执行节点**：`PUB-01-S03_SS05` = `READY_FOR_CODEX`，研究者已授权Codex从代码审查、合成复核、公开 EEG 数据准入、训练/验证推进；仅在新的前瞻内部holdout隔离成立时一次测试。协调助手不代跑。
- **历史**：SS03四尺度卷积实验完成有限制，SS02强基线完成有限制。SS05此前合成模型/结果由协调助手起草，**Codex尚未独立复核，不得自动升为可用真实EEG模型**。
- **顺序**：SS05（机制审计和新公共EEG实验）→ 用户审阅 → 按需要另行激活SS04统一直接重建及候选检索。S04 ds004703独立路线依旧BACKLOG，不以EEG架构突破为前置；S05公共artifact打包依赖规范验收。
- **禁令**：SS05不用SS03已打开的test来挑选结构、lag或分组；患者/STN数据或embedding不得进入公开仓库。模型/信号与运行在server2203，Windows/Git仅代码与轻量报告；PR#3/PR#4暂Draft，不合并main。

## 当前SS03执行顺序（2026-10-19）

1. `SS02` 已经 `COMPLETED / PASS_WITH_LIMITATION`，冻结为历史基线。原论文完整cohort和官方评价尚未复现，不能补称同等。
2. 用户批准的SS03多时间尺度单假设已完成：真实window audit/smoke/100步pilot→冻结25000步/2500步验证/3seed→12run实际test→separate review。当前COMPLETED/PASS_WITH_LIMITATION，详见任务书与REPORT。
3. 研究者2026-10-19已明确授权新的SS05历史记忆探索：当前只有合成CPU结果；PR#3仍Draft，不合并main；SS05在独立分支开展，不修改历史SS03。公开EEG真实训练/SS04尚未开始。
4. `SS04` 仍BACKLOG；不得根据test挑选多尺度/单尺度、窗口、seed或负例，也不得把2AFC与HDPS百余候选任务的性能直接比较。
5. ds004703公开颅内独立基线（S04_SS01）可以作为另一支线单独规划；本轮SS03不启动它、不读任何患者/STN数据。冻结且版本化artifact之前两项目不交换运行数据。
