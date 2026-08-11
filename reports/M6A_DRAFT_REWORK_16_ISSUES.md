# M6A exchange DRAFT 16 项返工处置

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-11 |
| 上游审核 | `STN CROSSWALK_REVIEW=PASS_WITH_REWORK` |
| 本轮范围 | Auditory 自有 schema、contract、validator 与测试 |
| 处置 | `16/16 IMPLEMENTED_BY_M6A` |
| 当前状态 | `REVISED_DRAFT_ACCEPTED_FOR_CANDIDATE_PREPARATION` |
| Consumer cross-test | `NOT_RUN` |
| Candidate/frozen artifact | `NO / NO` |

## 逐项 disposition

| # | issue code | Auditory 处置 | 验证证据 |
|---:|---|---|---|
| 1 | `accepted_validation_evidence_can_be_empty` | leakage/null/metrics/benchmark 均 `minItems=1`；未运行必须显式 `NOT_ESTIMABLE` | empty/profile fail-closed schema tests |
| 2 | `benchmark_summary_has_no_minimum_structure` | 定义 scope/dataset/split/target/model/metric/value/null/status/limitations 最小结构 | `test_benchmark_item_requires_minimum_structure` |
| 3 | `canary_expected_output_contract_incomplete` | 拆分 input/output，增加 layer order、shape、dtype、frame time 和三类 tolerance | canary schema + cross-check tests |
| 4 | `canary_inventory_link_missing` | input/output 必须映射 inventory 的受控角色；candidate 必须本地包含 | `test_canary_files_must_be_in_inventory` |
| 5 | `acceptance_states_are_free_strings` | 三字段统一枚举 `PENDING/PASS/FAIL/NOT_APPLICABLE` | `test_acceptance_state_is_controlled` |
| 6 | `draft_schema_allows_premature_release_states` | draft 仅允许 DRAFT/REVISED/CANDIDATE，不允许 FROZEN/ACCEPTED/RETIRED | `test_frozen_status_is_not_allowed_by_draft_schema` |
| 7 | `illegal_state_transitions_not_prevented` | 引入 ordered status history 与显式 transition table | `test_illegal_transition_is_rejected` |
| 8 | `inventory_path_uniqueness_and_safety_missing` | 路径唯一；拒绝绝对、drive、反斜杠、`.`/`..` 与 root escape | `test_unsafe_inventory_path_is_rejected` |
| 9 | `inventory_required_roles_not_defined` | 枚举角色；candidate 对六个 cross-test 单例角色要求恰好一次 | candidate complete/missing-role tests |
| 10 | `duplicate_layer_entries_not_rejected` | schema `uniqueItems=true` | schema validation tests |
| 11 | `layer_identity_uniqueness_not_defined` | layer key/ordinal 唯一，ordinal 从 0 有序连续 | duplicate key/ordinal tests |
| 12 | `extraction_spec_files_not_linked` | entrypoint/runtime/config/output schema 均显式关联 inventory 角色 | linked path tests |
| 13 | `method_package_inventory_link_missing` | method files 非空且全部属于 inventory；entrypoint/runtime 双向一致 | `test_method_file_must_be_in_inventory` |
| 14 | `revision_limitation_not_required` | `revision_immutable=false` 条件触发必填 limitation | `test_nonimmutable_revision_requires_limitation` |
| 15 | `runtime_versions_have_no_minimum_profile` | 必填 python/torch/transformers/numpy 且值非空 | `test_runtime_profile_is_nonempty_and_minimum` |
| 16 | `transform_inventory_link_missing` | transform 文件必须在 inventory，candidate 必须本地包含 | `test_transform_must_link_to_inventory` |

## 运行证据

- 2203 项目专用环境：`33 passed in 0.11s`；
- 本地 stdlib JSON 解析与 Python compileall：通过；
- ruff/mypy：项目专用环境未安装，本轮未运行，不写作通过；
- 外部 STN 文件：只读接收，未修改；
- 完整性边界：只使用文件名、字节数、时间戳、数量、schema 与可读性，不进行哈希或校验和验证。

## 科研与发布边界

本轮可声称 producer 已对 16 项 schema/validator 返工逐项实现并通过本项目测试。不能声称 STN 已复核修订稿、consumer cross-test 已运行、真实 method/canary 已交付、contract 已接受或冻结，也不能声称任何公开脑数据 baseline 结果。

## STN 二审补充 disposition

原 16 项已由 STN 独立复核为全部 resolved。随后新增问题 `NONFINITE_NUMERIC_VALUES_ACCEPTED`，处置状态为 `IMPLEMENTED_BY_M6A_AWAITING_STN_REVIEW`：

1. exchange CLI 改用严格 JSON loader，`NaN`、`Infinity`、`-Infinity` 解析即失败并返回非零；
2. 直接传入 dict 时，语义 validator 递归检查所有浮点值的 `math.isfinite`，覆盖 canary tolerance、frame-time/shape 相关数值、benchmark value 与 null value；
3. JSON Schema 继续执行类型、非负与范围约束，但不单独承担非有限值门禁；
4. 新增三类非有限值在 tolerance、benchmark value/null value 与 CLI 文本入口的 fail-closed 反例测试；
5. producer 返工时 contract 状态保持 `REVISED_DRAFT_AWAITING_CONSUMER_REVIEW`，未升级 candidate。后续 STN 二审与协调复跑已接受该 revised DRAFT，当前为 `REVISED_DRAFT_ACCEPTED_FOR_CANDIDATE_PREPARATION`，仍无真实 candidate。
