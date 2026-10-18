# PUB-01 — 两条公共数据研究支线（2026-10-13 决策）

此文件是父任务 `TASK.md` 的执行路线图，**不构成第二份状态登记**；状态一律以 `SUBTASKS.md` 和 `doc/TASKS.md` 为准。

## 科学与论文目标

近期 ICML 方法论文的优先目标：研究一个**不依赖候选声音作为模型输入**的 EEG→连续听觉表征解码架构，并在同一模型冻结输出下评估直接重建、二选一、五选一、百余候选声音片段检索。追求公开可比协议上的强基线/SOTA；60%/80% 是研究者提出的冲刺愿景，**不是已实现效果、验收硬阈值或允许更改测试集的理由**。

“直接解码”最低限度是：模型输入只有 EEG（以及推理时允许的固定元数据），输出时间对齐的连续声音特征（先 envelope / Mel 或获准的冻结语音特征）；**不能在编码器前给它候选语音 ID、答案文本或候选音频**。输出音频特征不等于恢复可听语音、逐词转写或语义。候选检索只能是对固定预测输出的后验评分/检索，不得依测试候选集重训 EEG 主模型。

## A 支线 — SparrKULee：优先完成 EEG 听觉解码论文

归属：`PUB-01-S03`；前置 `S01` 数据准备与 `S02` 声音→EEG 编码/特征基线均为 COMPLETED / PASS_WITH_LIMITATION。

- `PUB-01-S03_SS01`：已完成 10 人 / 93 条 EEG→包络 backward TRF（历史证据完整保留，**不代表语音内容解码或 SOTA**）。
- `PUB-01-S03_SS02`：2026-10-15 **COMPLETED / PASS_WITH_LIMITATION**。HDPS76人原100Hz/M，Top1/5/10=12.03/30.45/42.10%；Transformer70训练/69seen/13heldout，2AFC加权82.38%，EEG-only包络加权r=.1593；三运行独立review完成。完整论文/official test/ensemble等价性未通过；不同候选任务分开报告。当前停止，后续需研究者指定。
- `PUB-01-S03_SS03`：2026-10-19获研究者授权，**ACTIVE / PROTOTYPE**。选择多时间尺度卷积（dilation 1/4/16）对比参数量匹配单尺度（1/1/1，4/4/4，16/16/16）；只有协议与源码，没有实际server训练成绩。
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

## 当前SS03执行顺序（2026-10-19）

1. `SS02` 已经 `COMPLETED / PASS_WITH_LIMITATION`，冻结为历史基线。原论文完整cohort和官方评价尚未复现，不能补称同等。
2. 用户已批准 `SS03` 的第一项多时间尺度假设。当前 `ACTIVE / PROTOTYPE`；详见 `PUB-01-S03_SS03_ARCHITECTURE_HYPOTHESIS.md`。**尚无2203模型训练或测试结果**。
3. Codex应先在独立worktree运行代码与数据准入检查，做100步以内train/val工程pilot，冻结正式预算和测试前协议，然后运行参数匹配4种配置、多种子对照。
4. `SS04` 仍BACKLOG；不得根据test挑选多尺度/单尺度、窗口、seed或负例，也不得把2AFC与HDPS百余候选任务的性能直接比较。
5. ds004703公开颅内独立基线（S04_SS01）可以作为另一支线单独规划；本轮SS03不启动它、不读任何患者/STN数据。冻结且版本化artifact之前两项目不交换运行数据。
