# M6A-PUBLIC-003 Summary

状态：`READY_FOR_REVIEW`。这是 Agent B 的执行交付，不是 `ACCEPT` 或 `HUMAN_ACCEPTED`。

- Runs recorded: `45`; PASS: `45`; failed/diverged/other: `0`.
- Scope: exactly the three approved groups; no pretrained model, model download, patient/STN data, cochlear frontend, fast/slow branch or Mamba gating.
- Primary evidence: `m6a_public_003_metrics_by_seed.csv`, `m6a_public_003_metrics_summary.csv`, `m6a_public_003_model_parameters.csv`, `receptive_field_by_layer.csv` and generated figures.

## Engineering PASS criteria

- Parameter/RF smoke completed before formal training.
- Every formal variant was attempted under the same optimizer, epoch count, batch size, split and three seeds.
- RF recurrence and output frame step are computed from the exact model specification.
- Negative, failed and parameter-confounded states are retained in structured outputs.

## Evidence boundary

A performance difference in this synthetic benchmark is engineering evidence about these small model graphs and task distributions. It is not evidence of brain-region correspondence, auditory-system homology, patient/STN applicability, causality in neural tissue or clinical utility. Theoretical RF is not effective RF, and three seeds do not estimate biological variability.

## Open scientific questions

- Whether the observed task effects persist across real audio perturbations and public neural recordings remains unresolved.
- Whether effective rather than theoretical RF tracks the performance differences remains unresolved.
- Whether explicit change information adds stable value after a broader family of matched controls remains unresolved.

下一步必须经过 Agent A/人工审核，不由本 worker 自动启动。
