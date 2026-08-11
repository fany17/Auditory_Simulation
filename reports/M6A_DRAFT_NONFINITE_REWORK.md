# M6A revised DRAFT 非有限数值返工

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-11 |
| 输入问题 | `NONFINITE_NUMERIC_VALUES_ACCEPTED` |
| 处置 | `IMPLEMENTED_BY_M6A_AWAITING_STN_REVIEW` |
| Contract 状态 | `REVISED_DRAFT_ACCEPTED_FOR_CANDIDATE_PREPARATION` |
| Candidate/frozen | `NO / NO` |

## 门禁修正

- CLI 严格拒绝非标准 JSON parse constants：`NaN`、`Infinity`、`-Infinity`；
- `validate_exchange_manifest` 对直接 dict 递归执行有限数检查，因此 canary tolerance、frame/time/metric 相关浮点值、benchmark numeric value 与 null value 均 fail closed；
- JSON Schema 保留类型、非负和范围约束，但 Python 的非标准浮点值可能绕过单独的 schema 比较，因此 schema 不能替代语义 validator；
- 非有限值错误是 hard fail，不降级为 warning。

## 测试证据

2203 项目专用环境 `auditory_m6a_public_001` 真实运行：

```text
44 passed, 18 subtests passed in 0.19s
All checks passed!
Success: no issues found in 14 source files
```

运行范围分别为 pytest `tests/`、Ruff `src tests scripts`、mypy `src tests scripts`。CLI 对 `NaN`、`Infinity`、`-Infinity` 均返回非零；直接 dict 在 tolerance、benchmark value 与 null value 中的三类非有限值均 fail closed。

## 发布边界

本轮只修订 producer DRAFT 规范、validator、测试与报告。没有真实 exchange manifest、method package 或 canary；不得声称 candidate、consumer cross-test、accepted 或 frozen。

后续 STN consumer 二审与协调独立复跑已确认 non-finite 与 layer-order 反例 fail closed，`REVISED_DRAFT_REVIEW=ACCEPT`。consumer 当前为 `READY_WAITING_M6A_CANDIDATE`；真实 candidate 仍不存在，因此 `CONSUMER_CROSS_TEST=NOT_RUN`，release 未 accepted/frozen。
