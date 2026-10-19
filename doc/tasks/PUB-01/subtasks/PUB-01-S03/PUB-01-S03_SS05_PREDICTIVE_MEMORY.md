> **Codex实际接续（2026-10-19）**：Issue#5已启动，实测审计/pilot完成，正式15run训练中，见codex_run01/EXECUTION_LOG。最新用户授权下PR#3已由协调者合并main@8d21d0a；PR#4仍Draft、base main，旧“PR#3保持Draft”仅为历史限制。

# SUBTASK — PUB-01-S03_SS05 — 跨窗口历史记忆与选择性信息更新

| 字段 | 值 |
|---|---|
| 子任务 | `PUB-01-S03_SS05`，从属于 `PUB-01-S03` / `PUB-01` |
| 当前状态 | **`TRAIN_VAL_FROZEN`**：真实元数据/15tests/合成审查/五组四lag pilot已完成；新train归一化修正，15run正式训练中，新holdout尚未评分 |
| 唯一执行者 | **Codex** 负责代码审查、实现、真实运行、数据统计、结果记录；协调助手仅更新任务书、复核提交和把关结论 |
| 上游 | `SS02` 强基线与 `SS03` 多尺度实验已完成有限制；只读、不得覆盖 |
| 当前 Git | `work/pub-01-s03-ss05-predictive-memory`，独立managed worktree/PR#4 Draft，base main；PR#3按新授权已合并8d21d0a，主目录CNN/TRF/stash不动 |
| 现有原型 | `scripts/pub_01_s03_ss05_predictive_memory.py`、`scripts/pub_01_s03_ss05_window_decoder.py` 等 |
| 患者数据 | **FORBIDDEN**：本任务仅 `Auditory_Simulation` 的公开 EEG/公开模型，不读 STN/患者数据或 embedding |
| 本轮验收 | 可验证的历史依赖、真实 EEG train/validation 与预先冻结的内部 heldout（若符合准入）；明确阴性、运行失败和证据边界 |
| 与 SS04 关系 | SS05 为新的方法探索；SS04 仍 `BACKLOG`，不自动启动其2AFC/5AFC/大候选评价 |

## 0. 角色纠偏与已有材料的证据地位

2026-10-19 研究者明确要求：**研究者决定问题，任务协调者制定/检查任务书，Codex 执行代码与实验。** 此前协调助手自行加入了 PI-MRAM 原型、合成任务代码和 `reports/pub_01/s03/predictive_memory/SYNTHETIC_RESULTS.json`。保留这些作为**尚待 Codex 独立审查的历史参考**；它们不是 Codex 亲自复现过的结果，更不是对新结构有效性的验收。

Codex 的第一步是只读检查上述代码和数据来源，独立重跑必要的单元测试/合成 probe，核对所有报告数字、随机种子、输出与真实执行日志。如果无法独立验证，明确写 `UNVERIFIED`、`REWORK`，不能引用已写的数值来宣称新算法有效。可重构现有代码，原型绝非不可修改的最终架构。

本任务本轮更新**仅修改 Markdown 任务书/状态路由**，不应由协调助手额外执行 Python、训练、合成实验或远程命令。

## 1. 核心研究问题与可证伪假设

研究目标：在直接 EEG→声音时间特征的解码中，建立跨时间段的历史联系，使网络保留过去、按内容调用记忆，并判断预测误差是否真的提升选择性更新。聚焦计算机制而非堆叠模型。

- **H1 历史信息是否有效**：固定同一当前 EEG 窗口时，使用真实连续历史，相比同录音错时、异录音历史或重置历史，能否提高目标声音特征的 heldout 重建质量？
- **H2 关联记忆是否有额外贡献**：在公平输入/参数或计算预算下，`assoc_no_pred` 是否优于 `state` 和标准 GRU？比较不仅看最终 Pearson r，还看训练效率、历史间隔、跨录音泛化。
- **H3 预测误差是否必要**：`full` 是否优于 `assoc_no_pred`？此前合成任务草稿没有支持这个差异，**必须允许再次得到阴性**。
- **H4 记忆真的按内容检索吗**：打乱历史顺序、仅替换相关历史事件、替换无关事件、延长事件—查询间隔后，输出改变是否符合目标信息？历史影响本身不等于有用的记忆。

首轮严格对照：`local`（无持续记忆）、`state`（多速率状态）、`assoc_no_pred`（状态＋关联读写）、`full`（再加预测残差）、**标准 GRU**。可将 `predictive_state` 作为预登记次要消融。需要说明参数**实例化数量**、实际激活数量、FLOPs/显存和迭代预算不是同一个概念。与已知 GRU/SSM/Delta Rule/Gated Delta Networks/Mamba/Titans 的区别必须实事求是，不得仅凭拼接现有模块就宣称原创算子。

## 2. Codex 首轮执行顺序（无需再征求相同授权）

### A. 只读核验 / smoke：第一个实际动作

1. 在 Codex 具备访问的本地环境检查现有 Git worktree、`origin`、PR#3/#4、未提交 CNN/TRF 与 stash；只用**独立 worktree**操作 SS05 分支，禁止强制 checkout/reset/清理原主目录。
2. 只读核验现有原型、测试和报告；分别核实代码能否导入、shape、有限梯度、跨块与整段计算一致、无未来跨窗口泄漏、跨录音 state reset、反事实历史影响。保留失败记录；不得把已有报告中的合成成功直接当作新执行证据。
3. 如果可访问 `server2203`，完成真实公开数据的**可用性/配对/分组元数据审计**。若不可用，完成代码审查并标记 `HOLD_RUNTIME`，不要制造远端执行或训练成功。
4. 可用 Python/CUDA 仅由 Codex 在实际执行环境调用；失败保留命令、错误和环境信息。

### B. 构造真正的连续 EEG 历史

1. 仅复用 `server2203:/home/fanyu/auditory_simulation_m6a` 已获准的公开 SparrKULee 数据及上游 frozen `train_only_normalization`、时间轴/record 清单；患者/STN路径与派生 embedding 一律禁止。
2. 首轮将连续一个 record 的 64Hz EEG 按 **192 samples = 3s、无重叠** 切窗，同一连续 20窗约60秒，输入 `[B,20,192,64]`、输出 `[B,20,192,1]`；需核验缺失、尾段、时轴与输出对应。窗口内可访问整段3s，因此仅声称**跨窗口因果的离线模型**。
3. 同一 subject/record 内按时间顺序传递历史状态，不得跨独立录音、被试、story/session 边界传递；清楚定义 warmup、TBPTT 长度与梯度截断。随机分 batch 不能把不相邻窗口当作连续记忆。
4. **重要**：EEG 相对声音有神经响应时延，必须在 train/val 中先定义目标方向和 lag；若比较 0/8/16/32 sample，仅能在训练/验证选择并冻结，绝不依据 test 找最优 lag；物理同步依旧 `UNKNOWN`。

### C. 新的验证协议：杜绝 SS03 test 重用

1. 旧 `SS03 test_seen/test_heldout` **已被打开**，不得再用于 SS05 模型选择、架构试错或确认性结果。
2. 仅在尚可用的 **SS03 train 来源**中，依据真实 `subject/story/source_family/record` 清单，先划分 SS05 train / validation / 新的内部 holdout（按组，而非把相邻3秒窗随机打散）。先写 frozen split manifest；确认测试被试、故事的隔离定义，保留无法同时满足的限制。
3. 这是 **SS05 prospective internal holdout**，因为来源在上一项目阶段曾被使用，不能冒充原论文官方或完全独立外部复现。若已没有可接受的非交叉数据组，使用 train/val 做探索，结论仅限 validation，状态记 `HOLD_NEW_TEST`，不可造“独立测试”。
4. 确认录音源身份、正确标签和时间窗，不得通过音频ID、候选音频、故事文本泄漏目标；推理输入只允许 EEG 与预先授权的固定元数据。

### D. 训练/验证工程 pilot，然后冻结正式实验

1. 先有限步数运行 local、state、assoc_no_pred、full、GRU 的真实公开 train/val 流程，完成至少一次全长度连续序列预测和历史干预测试；报告真实GPU资源、运行时间与是否出现损失/梯度异常。
2. 根据 pilot 的 **train/val 而非 test** 冻结模型版本、数据集/split、lag、预处理、窗口长度、历史长度、训练步数、checkpoint选择规则、计算预算以及至少3个随机种子；在 `FORMAL_PROTOCOL.json` 中写入源代码 SHA256/环境/负对照定义。
3. 公平比较严格要求同训练数据、输出目标、划分、候选或负例定义、计算预算；不同参数量和算力时显式报告，并考虑加一个匹配参数量的 GRU 或固定预算对照。
4. 不用旧SS03模型的 heldout表现设置新模型超参，不通过 test 后再增加架构/轮数。

### E. 一次正式 holdout 测试（仅前述准入成立时）

1. 分别报告同当前3秒 EEG 的真实历史与历史错配/重置时的 Pearson r、逐 subject 的 `Delta r_history`，以及模型间配对差值；负值和失败必须保留。
2. primary：60秒segment Pearson r → recording/subject 汇总 → subject-macro；另报告每seed、每subject及合理的分组bootstrap区间，避免把相关窗口/三个seed当独立被试。
3. 次要：记忆间隔、干扰密度、训练收敛、实际推理延迟、参数/FLOPs，分别分析 `assoc_no_pred - state`、`full - assoc_no_pred`、`full - GRU`。
4. 每个冻结模型测试一次；新 test 不能指导追加模型、参数或改变分组。若H3阴性，应考虑取消无效的预测误差模块，而不是额外制造复杂性。
5. 在有保留集证据以前不得宣布相对 SS02 的 `r=.1593` 或旧 SS03 `r=.10663` 有提升；两者原协议不同，严禁直接排名。

## 3. 文件与运行边界

- 服务器仅保留公开 EEG/声音、缓存、权重、测试逐时间窗预测：在 `/home/fanyu/auditory_simulation_m6a/pub_01/s03_history_memory_20261019/<new-run>`，新目录不可覆盖旧SS02/SS03。
- 本地 Git 仅收录**自有代码、config、轻量 JSON/CSV、静态图、Markdown**，不收原始数据、权重、派生个人神经表示、大量数组、外部未授权源码。
- 本轮提交到 `work/pub-01-s03-ss05-predictive-memory` / PR#4 Draft；PR#3已按用户新授权合并main@8d21d0a，不回退此事实。当前Codex不主动merge#4/修改占用中的main或force-push。
- 没有自动开启 SS04 大候选检索、S04公开颅内迁移、S05父任务 artifact，也不接触 `STN_Decoding_Encoding`。
- 若结构需要独立的声音/模拟神经元输入任务，可在 SS05 的**公共方法探索**下明示，不能凭此声称真实 EEG 已验证。

## 4. Codex 必交付成果与状态规则

必须新建 `reports/pub_01/s03/predictive_memory/codex_run01/`，按真正完成程度交付：

- `EXECUTION_LOG.md`：逐阶段具体命令、commit、环境、真实运行目录、失败与恢复、当前状态。
- `CODE_AUDIT.md`：复核历史助手原型及合成报告的结果，哪些可复现、哪些需 REWORK，说明原创性边界。
- `DATA_CONTRACT.md`：公开记录/被试/声音/时轴/split、历史连续规则、目标lag、train-only标准化和所有阻塞。
- `FORMAL_PROTOCOL.json`：仅在预注册方案完成时冻结，不造假。
- `RESULTS.csv`、`REPORT.md`、`REVIEW.md`：实际模型跑过并有结果才填数；未跑项目保留 `NOT_RUN`，不得补合成指标充当 EEG。
- 若没有合格的新 holdout：只能交 train/val 的真实探索结果和 `HOLD_NEW_TEST`，不能标 `COMPLETED`。

**状态顺序**：`READY_FOR_CODEX` → `CODEX_RUNNING`（有真实执行记录）→ `PILOT_COMPLETE` → `TRAIN_VAL_FROZEN` → `TEST_REVIEW` → `COMPLETED/PASS_WITH_LIMITATION` 或 `HOLD/REWORK`。只有 Codex 的真实运行证据满足节点验收，才能提升状态；协调者不能代跑或替代 Codex 的结果。

## 5. Codex执行记录（2026-10-19）

独立managed worktree为C:/Users/fanyu/.codex/worktrees/pub-01-s03-ss05/Auditory_Simulation，起点5724a4ac；新2203根pub_01/s03_history_memory_20261019/codex_run01。真实source-family元数据一次冻结23/7/7组、353/26/47记录、70/17/21人，仅来自旧SS03train；为prospective INTERNAL而非外部复现。旧norm含新holdout，已在353新train单独重拟合，旧norm/SS02/SS03不改。

原10tests/合成A/B/GRU600重跑及5新增tests、实际CPU/GPU smoke通过；原local隐含两past窗在新wrapper修正。每query前20连续窗、当前窗20query构成60s段，有限context reset，全部21token BPTT；共用120s元数据warmup以支持错时历史，不跨record/state。预测头无显式forecast目标，full仅按learned history-residual分支解释。

五组×四lag×100步train/val pilot选择lag8；在新holdout payload前冻结2000步/200步val/三seed/同批序/七历史条件、六primary contrasts/source环境于FORMAL_PROTOCOL。全部15训练完核对预算后才构建holdout并一次评分。容量审计同时给实例化、tensor-reachable、实际标量梯度活动，不声称全部消融等容量或等墙钟成本。当前继续真实结果/review，不能停在准备阶段；详见codex_run01。
