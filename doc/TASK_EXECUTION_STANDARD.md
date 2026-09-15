# Auditory_Simulation Task Execution Standard

版本：`semantic-pipeline-v2.0 / 2026-09-15`  
全局规范：`AuditoryReading/TASK_EXECUTION_STANDARD.md`。

本文件是公共执行仓库的本地镜像与边界补充；数据边界以本仓库 `PROJECT_BOUNDARY.md` 为最高约束。

## 1. Task identity and location

当前/未来 task 统一使用语义型 ID，并放在：

```text
doc/tasks/<SEMANTIC-TASK-ID>.md
```

当前 canonical task：`PUBLIC-REPRESENTATION`。

禁止 current/future ID 使用：

- `M<number>` milestone；
- `R0/P0/W0/S0` 等顺序号；
- 日期；
- `NEXT/HIGH/FINAL` 状态词；
- task version number。

状态、日期、版本属于 metadata，不属于 Task ID。

历史 `M6A-PUBLIC-001..003` 保留 provenance，不批量改名；不再创建新的 `M6A-*` task。

状态 source of truth：`doc/TASKS.md`。  
当前入口：`doc/CURRENT_TASK.md`。

## 2. Required metadata

新任务必须包含：

- Task ID；
- Pipeline Node（默认与 Task ID 同名）；
- Track；
- Class；
- Status；
- Owner Repo；
- Blocking；
- Patient Data；
- Upstream；
- Inputs；
- Deliverables；
- Acceptance；
- Go/No-go；
- Downstream；
- Last Updated。

模板：`doc/TASK_TEMPLATE.md`。

## 3. Allowed status

`BACKLOG / READY / ACTIVE / WAITING_EXTERNAL / BLOCKED / REVIEW / FROZEN / COMPLETED / CANCELLED / HISTORICAL_REFERENCE`

旧自定义状态只保留历史报告；Registry 必须映射到统一状态。

## 4. Public-project hard boundaries

- 患者/STN 数据：`FORBIDDEN`。
- 不读取患者派生 embedding、私有临床 metadata 或患者模型权重。
- 不修改患者项目科学结论。
- public artifact 只能依据 public data 选择模型/layer/hyperparameter。
- STN 结果不得反向用于挑选 `ART-AUDREP-v1` 内容。

## 5. Current scientific role

`PUBLIC-REPRESENTATION` 是 optional public prior / comparator。

合法结论包括：

- positive transfer；
- no transfer；
- negative transfer。

它不构成 direct `STN-READ-COMPUTATION` 的硬门。

## 6. Execution discipline

一次 agent execution 只能指定一个 primary Task ID。

禁止：

- 自动扩 pretrained model 名单；
- 因阴性更换 split/负样本/窗口追阳性；
- 相邻 story/time leakage；
- test subject 参与 normalization/layer selection；
- 把高 representation correlation 写成脑区同源；
- 自动启动患者侧 downstream。

任务结束时：更新 Completion Record、`TASKS.md`、`CURRENT_TASK.md`，然后停止。

## 7. Cross-repo artifact

公共→患者只通过版本化 artifact，例如 `ART-AUDREP-v1`。

必须包含：version、source task/commit/config/seed、schema、runtime/transform、canary、benchmark、known failures、license/provenance。

Consumer validation 不通过时，由患者侧输出 rework；consumer 不静默修补 producer artifact。

## 8. Documentation rule

- long-form charter/总纲：background，不是 current registry；
- historical milestone task/interface drafts：provenance，不能自动执行；
- `reports/`：结果证据，不反向改变 task 状态；
- current/future task 只进入 `doc/tasks/` 且使用 semantic ID。