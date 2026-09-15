# Auditory_Simulation Task Execution Standard

版本：`numbered-pipeline-v2.1 / 2026-09-15`  
全局规范：`AuditoryReading/TASK_EXECUTION_STANDARD.md`。

## 1. Task identity and location

Current/future public tasks use one canonical namespace: `PUB-##`, independently starting at `01`.

Task spec path:

```text
doc/tasks/<TASK-ID>_<SEMANTIC-SLUG>.md
```

Current task：`PUB-01 — Public Auditory–Neural Representation`。

旧 `M6A-PUBLIC-001..003` 只保留 provenance，不继续编号；不再使用第二套纯语义 ID。

状态 source of truth：`doc/TASKS.md`。当前入口：`doc/CURRENT_TASK.md`。

## 2. Required metadata

新任务必须包含 Task ID、Title、Track、Class、Status、Owner Repo、Blocking、Patient Data、Upstream、Inputs、Deliverables、Acceptance、Go/No-go、Downstream、Last Updated。

模板：`doc/TASK_TEMPLATE.md`。

## 3. Allowed status

`BACKLOG / READY / ACTIVE / WAITING_EXTERNAL / BLOCKED / REVIEW / FROZEN / COMPLETED / CANCELLED / HISTORICAL_REFERENCE`

## 4. Hard boundaries

- 患者/STN 数据：`FORBIDDEN`；
- 不读取患者派生 embedding、私有临床 metadata 或患者模型权重；
- public artifact 只能依据 public data 选择模型/layer/hyperparameter；
- STN 结果不得反向用于挑选 `ART-AUDREP-v1` 内容。

## 5. Current scientific role

PUB-01 是 optional public prior / comparator。positive/no/negative transfer 均可正常验收，不构成 READ-03 的硬门。

## 6. Execution discipline

一次 execution 只指定一个 primary Task ID。

禁止自动扩 pretrained model 名单、因阴性换 split/window 追阳性、时间/story 泄漏、test subject 参与 normalization/layer selection、把 representation correlation 写成脑区同源、自动启动患者侧 downstream。

## 7. Cross-repo artifact

公共→患者只通过版本化 artifact，例如 `ART-AUDREP-v1`。必须包含 version、source task/commit/config/seed、schema、runtime/transform、canary、benchmark、known failures、license/provenance。

## 8. Documentation rule

- long-form charter/总纲：background；
- historical M6A tasks/interface drafts：provenance；
- `reports/`：结果证据；
- current/future task 只进入 `doc/tasks/`，使用 `PUB-##` canonical ID。
