# Auditory_Simulation 当前任务状态

| 字段 | 内容 |
|---|---|
| 日期 | `2026-09-07` |
| 状态 | `M6A-PUBLIC-004 READY_TO_EXECUTE` |
| 当前方向 | 公开声音模型、公开神经数据与通用 auditory-neural representation |
| 当前优先任务 | `M6A-PUBLIC-004` |
| 正式任务书 | `doc/tasks/M6A-PUBLIC-004.md` |
| 患者/STN 数据 | `FORBIDDEN` |

## 当前任务

`M6A-PUBLIC-004：Auditory–Neural Representation Alignment and Transfer`

第一主数据使用 SparrKULee EEG，随后使用 ds004703 做公开 intracranial transfer。目标是建立：

```text
audio representation
      ↕
public EEG neural representation
      ↓ transfer
public sEEG neural representation
      ↓ freeze
ART-AUDREP-v1 candidate
      ↓
STN_Decoding_Encoding consumer
```

当前不再增加新的 pretrained architecture，不读取患者/STN 数据，不微调大型 audio backbone。

执行顺序：

1. dataset/timing/leakage audit；
2. acoustic + wav2vec2 + HuBERT layerwise baseline；
3. channel-agnostic neural encoder；
4. encoding + contrastive/retrieval；
5. held-out subject；
6. EEG→public sEEG transfer；
7. few-shot learning curve；
8. `ART-AUDREP-v1` candidate package。

## M6A-PUBLIC-003 状态

`M6A-PUBLIC-003` 已完成执行，状态为 `COMPLETED_EXECUTION_READY_FOR_REVIEW`。

45/45 formal runs 与 supplementary 10/20/50 ms probes 已完成。当前结果不支持继续通过 early/late downsampling、multiscale RF、RF-growth 或 explicit change branch 追加结构探索。阴性结果保留，003 不再阻塞 004。

## Frozen baseline

`M6A-PUBLIC-002`：9/9 pretrained inference、8/9 unified temporal representation probe，作为 004 的 frozen model bank/reference。

`M6A-PUBLIC-001`：ds004703/wav2vec2 单被试 preliminary checkpoint，只作历史参考；004 将重新以 leak-safe、held-out 和 transfer 设计建立正式公共证据。

## 跨项目边界

- `Auditory_Simulation`：公开数据、声音模型、公共 EEG/sEEG、通用 representation、M6A。
- `STN_Decoding_Encoding`：患者实验、STN 数据、STN-specific adaptation、PINS/SEEG 写入验证、M6B。
- 仅通过冻结、版本化 artifact 连接。
- `ART-AUDREP-v1` 未经 STN consumer cross-test 前，只能称 `candidate`，不得称 accepted/frozen contract。
