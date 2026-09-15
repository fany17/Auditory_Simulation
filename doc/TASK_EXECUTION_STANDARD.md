# Auditory_Simulation Task Execution Standard

版本：`program-standard-v1.0 / 2026-09-15`  
Program authoritative standard：`AuditoryReading/TASK_EXECUTION_STANDARD.md`。

本文件是公共执行仓库的本地镜像与边界补充；与 Program 标准冲突时，患者/公共数据边界以本仓库 `PROJECT_BOUNDARY.md` 为最高约束。

## 1. Local task location

今后新任务统一放在：

```text
doc/tasks/<TASK-ID>.md
```

状态 source of truth：`doc/TASKS.md`。  
当前入口：`doc/CURRENT_TASK.md`。

历史 `doc/tasks/M6A-PUBLIC-001..003.md` 保留 provenance，不为了格式统一重写全文；其当前状态由 `doc/TASKS.md` 映射。

## 2. Required metadata

新任务必须包含：

- Task ID；
- Program Node；
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

不再新建 `NEXT-HIGH`、`READY_TO_EXECUTE`、`COMPLETED_EXECUTION_READY_FOR_REVIEW` 等自定义状态；旧文字可保留在历史报告，但 Registry 必须映射到统一状态。

## 4. Public-project hard boundaries

- 患者/STN 数据：`FORBIDDEN`。
- 不读取患者派生 embedding、私有临床 metadata 或患者模型权重。
- 不修改 `STN_Decoding_Encoding` 的患者科学结论。
- 公共 artifact 只能依据公共数据选择模型/layer/hyperparameter。
- STN 结果不得反向用于挑选 `ART-AUDREP` 内容。

## 5. Current scientific role

`M6A-PUBLIC-004` 对应 Program `R1`：**optional public prior / comparator**。

它的成功与失败都必须可报告：

- positive transfer；
- no transfer；
- negative transfer。

它不再是 STN READ `R2–R4` 的硬门。不能因为公共 transfer 未完成而要求患者侧停止 direct/simple model science。

## 6. Execution discipline

一次 agent execution 只能指定一个 primary Task ID。

禁止：

- 自动扩 pretrained model 名单；
- 因结果阴性更换 split/负样本/窗口追阳性；
- 把同一 story 相邻片段泄漏到 train/test；
- test subject 参与 normalization/layer selection；
- 把高 representation correlation 写成脑区同源；
- 自动启动 downstream M6B。

任务结束时：更新 task Completion Record、`TASKS.md`、`CURRENT_TASK.md`，然后停止。

## 7. Cross-repo artifact

M6A→M6B 只通过版本化 artifact，例如 `ART-AUDREP-v1`。

必须包含：version、source commit/config/seed、schema、runtime/transform、canary、benchmark、known failures、provenance。

Consumer cross-test 不通过时，由 M6B 输出 rework；M6B 不静默修补 producer artifact。

## 8. Documentation rule

- `PROJECT_CHARTER.md`、`01_听觉时变信息处理项目总纲.md`：long-form background，不是当前 task registry。
- `M6A-PUBLIC-001_*`、exchange-contract draft：historical/interface evidence，除非 `TASKS.md` 明确激活，否则不可自动执行。
- `reports/`：结果证据，不反向改变任务状态。
- 新任务只进入 `doc/tasks/`。