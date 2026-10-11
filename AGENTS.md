# AGENTS.md — Auditory_Simulation Local Agent Rules

| Field | Value |
|---|---|
| Status | `ACTIVE` |
| Effective date | `2026-09-15` |
| Applies to | `Auditory_Simulation/` |
| Program governance | `AuditoryReading/AGENTS.md` |

本文件只补充 `Auditory_Simulation` 的本地执行规则。全项目 task/subtask 编号、目录、创建、校验、完成、amendment 和 handoff 规则以 `AuditoryReading/AGENTS.md` 为权威。

若当前 workspace 无法读取 sibling `AuditoryReading`，必须至少遵守本文件、`doc/TASK_EXECUTION_STANDARD.md`、`doc/TASK_TEMPLATE.md`、`doc/SUBTASK_TEMPLATE.md`；不得凭旧 M6A 文档恢复历史执行逻辑。

## 1. Public-project hard boundary

本仓库只负责：

- `PUB-##` 公共数据/公共方法任务；
- 公开声音模型；
- 公开 EEG/intracranial neural data；
- 通用 auditory-neural representation；
- transfer/few-shot benchmark；
- 冻结 public artifact。

严格禁止患者/STN raw/intermediate/derivative、患者派生 embedding、私有临床 metadata、使用患者结果反向选择 public model/layer/hyperparameter，以及患者 PINS/DBS/SEEG 刺激分析。

所有数据集下载（含续传）、解压、读取核验、预处理和派生数据生成必须在 `server2203` 完成；模型权重下载、特征提取、训练/评估及缓存也统一在 2203。不得在 Windows 本地下载数据后再上传，不得因服务器不可用而回退到本地执行。

现有服务器项目根为 `/home/fanyu/auditory_simulation_m6a`。先盘点并复用可核验的既有资产，不覆盖历史数据或 frozen evidence。本地只保留轻量代码、配置、来源/许可/路径清单、结构化汇总结果、task/subtask 记录和报告；可查阅官方文档与许可网页。数据集载荷、音频/神经信号、权重、特征张量、cache 与大型训练输出不得回传本地或进入 Git。代码按需复制到服务器，不在远端初始化第二个 Git 工作区。

## 2. Fixed read order

每次执行按顺序读取：

1. 用户当前明确指令；
2. Program `AuditoryReading/AGENTS.md`（可见时）；
3. 本 `AGENTS.md`；
4. `doc/PROJECT_BOUNDARY.md`；
5. `doc/DOCUMENT_INDEX.md`；
6. `doc/TASK_EXECUTION_STANDARD.md`；
7. `doc/TASKS.md`；
8. `doc/CURRENT_TASK.md`；
9. 父任务 `doc/tasks/<TASK-ID>/TASK.md`；
10. `doc/tasks/<TASK-ID>/SUBTASKS.md`；
11. 指定 subtask；
12. spec 列出的数据/config/artifact/report。

## 3. Current/future task layout

```text
doc/tasks/<TASK-ID>/
  TASK.md
  SUBTASKS.md
  subtasks/
    <TASK-ID>-S01_<TITLE>.md
```

历史 `M6A-PUBLIC-001/002/003` 与旧总纲/charter 已按用户 2026-09-19 指令归档到 `doc/archive/pre_pub_20260919/`；原文保留 provenance，旧路径见归档映射，不自动执行。

## 4. Subtask rule

- 父任务定义 question/scope/acceptance；subtask 不得改变。
- `PUB-01-S##` 只表示 `PUB-01` 内部执行单元。
- 一次 execution 只执行一个 primary subtask。
- 追加工作先定位所属S##；需要独立交付/验收时新建其SS##，只有新主阶段才新建S##。
- 普通小修改直接记录在当前 subtask Completion Record。
- 已有节点内追加工作使用 S##_SS##。
- 新科学问题必须申请新 `PUB-##`，不能藏在 S## 中。

## 5. Subtask pre-execution gate

开始前必须确认：ID/filename/`SUBTASKS.md` 一致；parent 正确；patient data 仍 `FORBIDDEN`；upstream 满足；license/data access 合规；deliverable/acceptance/validation 已写；split/leakage 规则冻结；允许修改文件范围明确。

任何一项失败，不进入 `ACTIVE`。

## 6. Validation

subtask 完成必须经过：

1. Technical validation：输出/schema/test/reproduction；
2. Acceptance validation：`PASS / PASS_WITH_LIMITATION / REWORK / HOLD / NO_GO`；
3. Parent consistency：确认不改变 PUB parent scope、claim 和 data boundary。

Executor 不能只凭“跑通”自行完成；至少做一次独立 review pass。

父 `PUB-##` 完成还必须独立做 Integration Review。

## 7. Negative evidence

positive / no transfer / negative transfer 均是合法结果。

禁止因为阴性自动改 split、换窗口、加新 backbone、改 negative sampling，或新建“再试一个”的 S##。新增分支必须回到 parent scope 与 acceptance 判断。

## 8. Cross-repo handoff

公共→患者只通过冻结、版本化 artifact，例如 `ART-AUDREP-v1`。

必须记录 source task/subtask/commit/config、schema、runtime/transform、canary、benchmark、known failures、license/provenance。

患者 consumer 不静默修改 producer artifact；失败返回 rework decision。

## 9. Git / destructive operations

- 不擅自切换/重置/强制推送/改写历史；
- 不覆盖用户修改或 frozen evidence；
- 删除 current 临时文件前确认新 canonical 路径已建立且可读；
- 历史 M6A provenance 保留原文；2026-09-19 用户授权归档只改变目录，映射见 `doc/archive/pre_pub_20260919/ARCHIVE_MAP.csv`；
- 若仓库既有规则禁止 checksum/hash 审计，继续遵守，不主动新增此类完整性流程。

## 10. Stop rule

完成一个指定 subtask 后：更新其 Completion Record、parent `SUBTASKS.md`，必要时更新 parent `TASK.md` / owner `TASKS.md` / `CURRENT_TASK.md`，然后停止。

不得自动启动 sibling subtask、下一个 parent task 或患者侧任务。

## 追加工作与节点层级（2026-09-24 用户规则）

- S## 表示父任务下的主节点；归属于已有节点的追加工作必须登记为该节点的 SS##，例如 PUB-01-S01_SS01，不得仅因追加时间较晚占用新的主节点编号。
- SS## 在所属 S## 内从 01 独立编号；先查已有 owner，再决定新增。小修改留在现有完成记录；重复冻结流程仍使用 run，不创建 SS##。
- 主任务登记仍集中在父任务 SUBTASKS.md：主节点表登记 S##，其下按 S## 分区登记 SS##，包含 Parent Node、依赖、状态、任务书和验收。不得创建第二份竞争登记表。
- 子节点任务书路径：subtasks/<TASK-ID>-S##/<TASK-ID>-S##_SS##_<TITLE>.md；主节点任务书仍在 subtasks/<TASK-ID>-S##_<TITLE>.md。复用 SUBTASK_TEMPLATE，补 Parent Node 字段。
- SS## 继承所属节点的范围、权限、验收边界；完成 SS## 不等于完成 S##。依赖可表达为“节点前置准入 → SS## → 节点最终验收”，不得写成互相等待的循环。
- 只有超出已有节点职责的独立主阶段才分配新的 S##；不得因整理文档自动新增研究任务。更深层工作先写步骤，不自动递归建任务。
- 编号迁移必须记录旧 ID、新 ID、任务书路径和原有产物位置。已发布旧编号不复用为另一任务；当前登记移除旧节点，历史报告/脚本/服务器路径保留别名说明，不批量搬动数据。

## 2026-09-24 用户修订：先探索数据，再决定正式实验规则

用户明确要求下载完成即开展探索性分析、画图并报告实际数据内容。此前将正式benchmark准入作为所有分析前提的解释作废。SS02追加图形化数据概览与问题定位，优先交付实际图和中文说明；旧准入报告作为历史判断保留，不能作为阻止探索的门槛。

- 已下载且已获本项目内部研究使用授权的公开数据，可直接做描述性统计、代表性波形/频谱、事件叠图和对齐诊断。已知损坏/不能读取的对象跳过并注明，受限载荷不请求；不为未知物理同步、历史预处理revision或尚未分配fold停止其余探索。
- 基础可读性、维度和坐标单位检查与分析同时完成；问题标注到图表和已存在的记录索引中，不再先全量重跑QC或循环完善审计文字。
- 正式训练、性能比较、因果/生物学结论仍依赖适合相应问题的预处理与数据划分。探索不宣称这些已通过，也不以相关峰或好看的图证明硬件同步。
- 数据/权重/信号读取与计算继续仅在2203。允许回传本轮必要的静态汇总/示例图及轻量HTML/Markdown报告供用户阅读；不回传原始音频/神经信号数组或嵌入它们的交互式载荷。
- 不读取本项目私有患者/STN资料、不扩大下载范围、不自动训练；只有实际访问/读取失败或明确使用禁止才阻断受影响对象，其余可进行的分析继续。
