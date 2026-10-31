# PUB-01 — 两条公共数据研究支线（2026-10-13 决策）

此文件是父任务 `TASK.md` 的执行路线图，**不构成第二份状态登记**；状态一律以 `SUBTASKS.md` 和 `doc/TASKS.md` 为准。

## 当前交付与停止（2026-10-23）

SS06真实连续EEG比较COMPLETED/PASS_WITH_LIMITATION：39监督+3SSL、243真实结果、四图和后置原型CPU/独立汇总复核已交。八primary三项为正（M1multi−matched long、M3ft−frozen、M4adaptive−uniformfixed），M2最高点估计但未稳健胜GRU，SSL额外收益/M5额外收益/内容特异历史收益未建立。报告continuity/eeg_run01/REPORT.md及REVIEW.md；INTERNAL/5family/历史暴露限制保留。当前STOP，SS04/S04/父S05 BACKLOG，无artifact。PR#3/#4已按后续授权合并，#6仍Draft，不自行合并。下方10-19执行路由及#4 Draft仅为历史，不重新启动。

## 科学与论文目标

近期 ICML 方法论文的优先目标：研究一个**不依赖候选声音作为模型输入**的 EEG→连续听觉表征解码架构，并在同一模型冻结输出下评估直接重建、二选一、五选一、百余候选声音片段检索。追求公开可比协议上的强基线/SOTA；60%/80% 是研究者提出的冲刺愿景，**不是已实现效果、验收硬阈值或允许更改测试集的理由**。

“直接解码”最低限度是：模型输入只有 EEG（以及推理时允许的固定元数据），输出时间对齐的连续声音特征（先 envelope / Mel 或获准的冻结语音特征）；**不能在编码器前给它候选语音 ID、答案文本或候选音频**。输出音频特征不等于恢复可听语音、逐词转写或语义。候选检索只能是对固定预测输出的后验评分/检索，不得依测试候选集重训 EEG 主模型。

## A 支线 — SparrKULee：优先完成 EEG 听觉解码论文

归属：`PUB-01-S03`；前置 `S01` 数据准备与 `S02` 声音→EEG 编码/特征基线均为 COMPLETED / PASS_WITH_LIMITATION。

- `PUB-01-S03_SS01`：已完成 10 人 / 93 条 EEG→包络 backward TRF（历史证据完整保留，**不代表语音内容解码或 SOTA**）。
- `PUB-01-S03_SS02`：2026-10-15 **COMPLETED / PASS_WITH_LIMITATION**。HDPS76人原100Hz/M，Top1/5/10=12.03/30.45/42.10%；Transformer70训练/69seen/13heldout，2AFC加权82.38%，EEG-only包络加权r=.1593；三运行独立review完成。完整论文/official test/ensemble等价性未通过；不同候选任务分开报告。当前停止，后续需研究者指定。
- `PUB-01-S03_SS03`：2026-10-19 **COMPLETED / PASS_WITH_LIMITATION**。多尺度对三单尺度、4×3真实同预算训练/test；seen/heldout r=.10305/.11380，高于long .06411/.06979及middle .09297/.10413。限四原型/当前fold预算，非生理机制或超过SS02；报告与same-agent separate review在`reports/pub_01/s03/architecture/`。
- `PUB-01-S03_SS05`：2026-10-19 **COMPLETED / PASS_WITH_LIMITATION**。Codex在2203完成源码/合成复核、15tests、train/val pilot、新train统计、15run及新内部holdout一次测试。五组r约.091–.092，六预设配对区间均跨0，未检出复杂历史记忆额外收益。有效阴性限定本轮数据/包络目标/原型/2000步；报告与same-agent separate review在`reports/pub_01/s03/predictive_memory/codex_run01/`。当前停止，#4仍Draft。
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

## 历史执行路由：SS05交付后停止（2026-10-19）

- **科研问题**：历史信息如何持续参与当前 EEG→连续声音特征预测；真实历史是否优于错配历史；关联读写是否优于GRU/状态更新；预测误差项是否提供进一步增益。维持通用时序模型和公开方法学定位，不把合成任务阳性等同于听觉生物学证据。
- **已交付节点**：`PUB-01-S03_SS05` = `COMPLETED / PASS_WITH_LIMITATION`。353/26/47条、23/7/7family的新内部划分；全部训练完成才开放holdout，新train-only归一化/预算/一次测试核验通过。人与训练重叠，来源曾用于SS03，不是外部独立复制。
- **审阅范围**：同agent另作post-execution pass，90个原始EEG CPU相关max差8.13e-6；协调仅核对指定轻量文件一致性，不称第三方科学签署。SS02/SS03历史结论冻结不变。
- **停止**：无自动下一READY节点。SS04、S04、父S05依旧BACKLOG；阴性不触发换架构/扩预算或新artifact。
- **边界**：SS05未用旧SS03 test选模型/lag。患者/STN禁止；数据/模型/计算仅2203，Windows/Git只轻量代码/报告/必要静态图。PR#3按用户新授权已合并8d21d0a；PR#4仍Draft，不自行合并。

## 历史SS03执行顺序（2026-10-19）

1. `SS02` 已经 `COMPLETED / PASS_WITH_LIMITATION`，冻结为历史基线。原论文完整cohort和官方评价尚未复现，不能补称同等。
2. 用户批准的SS03多时间尺度单假设已完成：真实window audit/smoke/100步pilot→冻结25000步/2500步验证/3seed→12run实际test→separate review。当前COMPLETED/PASS_WITH_LIMITATION，详见任务书与REPORT。
3. 研究者2026-10-19授权的SS05已完成真实EEG内部holdout阴性研究及review。独立分支/工作树保留，主目录CNN/TRF/stash及历史SS03不改；PR#3已按后续授权合并，#4保持Draft，SS04未启动。
4. `SS04` 仍BACKLOG；不得根据test挑选多尺度/单尺度、窗口、seed或负例，也不得把2AFC与HDPS百余候选任务的性能直接比较。
5. ds004703公开颅内独立基线（S04_SS01）可以作为另一支线单独规划；本轮SS03不启动它、不读任何患者/STN数据。冻结且版本化artifact之前两项目不交换运行数据。
