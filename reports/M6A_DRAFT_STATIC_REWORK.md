# M6A revised DRAFT 静态审核返工

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-11 |
| 输入审核 | `STATIC_REVIEW=REWORK` |
| 当前 contract 状态 | `REVISED_DRAFT_AWAITING_CONSUMER_REVIEW` |
| Candidate/frozen | `NO / NO` |
| 执行环境 | 2203 专用 `auditory_m6a_public_001` |

## 最小修正

1. 移除 `tests/test_exchange_validator.py` 未使用的 `copy` 导入；
2. 将 `split_guard.py` 两个循环映射分别命名为 `required_groups_by_value` 与 `optional_groups_by_value`，消除 mypy `no-redef`；
3. 在项目专用环境安装与 `jsonschema 4.25.1` 匹配的 `types-jsonschema 4.25.1.20251009`，未设置 jsonschema ignore；
4. 增加项目可审计的 Ruff `E4/E7/E9/F` 基线；
5. 对没有可直接采用类型声明的 MNE 与 SoundFile，仅在各自延迟导入行使用精确 `import-untyped` 抑制；
6. 为 S3 轻量 inventory 增加 `TypedDict`，使全体 `src/tests/scripts` 通过 mypy。

## 开发工具版本

- Ruff `0.16.2`；
- mypy `2.3.0`；
- types-jsonschema `4.25.1.20251009`。

安装只发生在 `/home/fanyu/.conda/envs/auditory_m6a_public_001`，没有修改其他科研环境。

## 真实运行结果

```text
33 passed in 0.10s
All checks passed!
Success: no issues found in 11 source files
```

运行范围：

- pytest：仓库 `tests/`；
- Ruff：`src tests scripts`；
- mypy：`src tests scripts`。

本轮不读取、生成或比较任何哈希或校验和。G2 下载继续运行，未因静态返工中断。

## 声称边界

可声称：协调指出的三类静态问题已在 Auditory 项目专用环境收口，pytest、Ruff、mypy 当前通过。

不可声称：consumer 已接受 revised DRAFT、真实 candidate bundle 已存在、consumer cross-test 已运行或 contract 已冻结。
