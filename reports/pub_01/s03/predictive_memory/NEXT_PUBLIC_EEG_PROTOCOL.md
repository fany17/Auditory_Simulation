> **执行者与证据纠偏（2026-10-19）**：本协议交由 **Codex** 在真实环境执行；协调助手不替代运行。此前 PI-MRAM 合成实验及10项本地测试为协调助手编写的历史报告，需 Codex 自行审查/重跑/记录来源；**公开 EEG 的新训练/评估尚未发生**。
>
> **最高优先级**：`doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS05_PREDICTIVE_MEMORY.md`；本文件为配套准入/统计契约。如两个文档冲突，以该SS05任务书为准。所有原SS03 `test_seen/test_heldout` 统计已在既往阶段查看，禁止用来选择新结构。

# SS05 — 下一轮公开 EEG 实验协议（**尚未执行**）

本文件是给 Codex 的**下一步执行约束**，不是运行记录。仅允许 server2203 上的公开数据；Windows Git 只回传自有代码、轻量摘要。保留全部 SS02/SS03 已冻结 checkpoint、输出、旧CNN/TRF本地改动。患者/STN 数据或embedding禁止进入公开仓库。

## 1. 新数据划分先行

1. 先只读审计 SS03 `cache01/manifest.json`、SS02 train-only mean/std、`source_family` 与 `story` 的完整身份映射，列出可形成多少互不重叠的录音家族；需要来自真实文件，**不能凭空推测数量**。
2. 从原SS03 **train** 子集设计新的内部 train/dev/heldout；以 `source_family` 或确实独立的 `story` 为分组单元，必须先冻结组别再加载正式指标。之前的SS03 `test_seen` / `test_heldout` **整个SS05期间禁用作模型选择**，不得把新的架构再用旧测试集追分。
3. 如果划分样本/独立source-family不足，记录 `HOLD_NEW_TEST`，允许 train/validation 的探索分析，但不能报告新的独立验证。因为新划分来自此前已进入SS03训练的信息，最多称“SS05前瞻内部holdout”，不冒称真正外部独立复现。

## 2. 计算及对齐契约

- 64Hz EEG，每3秒192点，**非重叠**；将同一录音真实连续20个窗口组成60秒序列。输入 `[B,20,192,64]`；包络输出 `[B,20,192,1]`；60秒Pearson→被试平均→总体。此协议与SS03之前50%重叠Hann结果不同，因此不直接比较旧数值。
- train-only标准化；原时间轴绝不通过test最优相关峰微调。
- 声音先于诱发的 EEG 反应。必须在train/validation阶段冻结 EEG 与包络目标的lag（初始可考虑 0/8/16/32 采样点，作为待验证的工程网格）；严格指出物理同步仍标 `UNKNOWN`，不能声称精确生理延迟。
- `WindowPredictiveDecoder` 跨窗口递归，但当前窗口的卷积可看到整个3秒，是**非实时窗口级因果**；与严格sample-causal的对照需分开报告。
- 任意状态仅可沿相同record的连续窗传播。录音/被试/会话边界必须reset；使用TBPTT时明确detach位置及评估warmup。绝不把随机打乱的训练窗口错误地串成一条历史。

## 3. 分组、对照、目标

在相同时间窗、声音目标、划分与优化配置下对照：

- `local`：无持续历史；
- `state`：三速率递归状态；
- `assoc_no_pred`：关联记忆＋递归状态；
- `full`：再增加历史预测残差；
- `WindowGRUDecoder`：标准GRU，参数与当前完整原型相差约2.78%，但需报告实际活跃参数、FLOPs与运行成本。
- `predictive_state` 可做单独消融，不得因为test结果临时新增或删除。

两个核心科学问题：

`H1: assoc_no_pred > state`（关联读写是否有独立贡献）；

`H2: full > assoc_no_pred`（预测误差更新是否还有独立贡献）。

本地两项合成试验中 **H2 未获支持**。若真实数据仍不支持，就停止宣传预测误差新机制。

历史反事实：保持同一段**当前3秒EEG不变**，只替换真实过去：历史打乱、其他录音的历史、历史强制重置，以及不提供历史；所有变体应使用相同当前音频目标且独立记录识别标签。结果应按 person/record 分组而非把同一录音的窗口当独立样本。用 true-history 减错配-history 的相关增量量化内容特异历史收益。

## 4. 任务关口

- **A. Read-only preflight**：公开数据可用性、group/source-family与样本身份、时长、lag/记录重置、完整性；不运行正式test。
- **B. Bounded train/validation pilot**：用可读合法公开数据跑窗口模型及GRU，真实CPU/GPU shape、梯度、峰值内存、训练速度和验证 r；失败保留。
- **C. Freeze**：在新内部测试读取前锁定所有变体、时间窗、lag选择、主体指标、训练步数、seed及模型选择规则，再做≥3个seed。
- **D. One-time test**：只有新保留集来源充分且确已隔离，才允许一次正式test与same-agent后置review；涉及多个source-family共享故事的统计不确定性需单独说明。
- **E. STOP**：阴性照实交付，不自动增加SaShiMi、TimeMixer++、Mamba、GPT rerank；SS04依旧BACKLOG。

最终产出 `DATA_CONTRACT.md` / `FORMAL_PROTOCOL.json` / `RESULTS.csv` / `REPORT.md` / `REVIEW.md`，原始公开信号/张量/权重留2203，且仅当真正完成时可标 COMPLETED。
