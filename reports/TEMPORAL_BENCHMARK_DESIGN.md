# M6A-PUBLIC-003 Temporal Benchmark Design

## 状态与图形契约

- 状态：`READY_FOR_REVIEW`（Agent B 产物，未写 ACCEPT/HUMAN_ACCEPTED）。
- 核心工程命题：在同一合成时序任务、统一优化和分组匹配下，结构变量改变快速事件的可定位性、区分性或未见参数泛化时，才构成结构 benchmark 证据。
- 图形类型：quantitative grid；主证据为任务性能与 RF/temporal-resolution 曲线，控制证据为参数匹配、representation distance、event-triggered response 和 autocorrelation。
- 后端：Python/Matplotlib；导出 PNG、SVG、PDF，所有图由同一 Python 脚本生成。
- Reviewer risks：3 seeds 不是生物学重复；theoretical RF 不等价于 effective RF；synthetic signal 不等价于声音、神经记录或听觉生理；参数超出 ±10% 的组标为 `CONFOUNDED`。

## Data and split

- Sequence length: `512 samples`, `1.0 ms/sample`; input is a synthetic 1D temporal signal, not patient or STN data.
- Train/validation/test-seen/test-unseen sizes: `1024/256/512/512`.
- Train and validation use the seen parameter regime. Test-seen uses an independent generation seed with the same ranges. Test-unseen uses held-out rates, jitter magnitudes and phase magnitudes.
- No adjacent-window, clip, story, subject or patient records are read; each sample is generated from an independent seed stream and split arrays are reused by every architecture.

## Tasks and metrics

- Rate discrimination: 4/6/8/10 Hz seen classes; balanced accuracy and rate MAE.
- Regular vs jitter: binary balanced accuracy and jitter-magnitude MAE.
- Omission detection: binary balanced accuracy.
- Phase-shift detection: binary balanced accuracy and phase-magnitude MAE.
- Onset/event timing: absolute error in milliseconds.
- Unseen generalization: held-out rate, jitter and phase magnitude MAE.
- Representation checks without additional fitting: no-fit latent distance after a 10-ms shift, event-triggered response, temporal lag-1 autocorrelation.

## Matched training contract

- Optimizer: AdamW, learning rate `0.002`, weight decay `0.0001`; epochs `8`; batch `64`; gradient clip `5.0`.
- Seeds: `[11, 22, 33]`. No per-architecture hyperparameter search or early stopping; the final epoch is evaluated for every run.
- Failure policy: retain non-finite loss, divergence, timeout and failed seed rows; never replace a failed run with a best run.

## RF convention

For a layer with kernel `k`, dilation `d`, stride `s`, previous jump `j` and RF `r`: `r_new = r + (k-1)d j`, `j_new = j s`. RF is reported as the number of inclusive input samples and `RF_ms = RF_samples × 1 ms`; output temporal resolution is the jump, not the RF width. Theoretical RF describes graph connectivity and is not an effective influence estimate learned from data.

## Reproducibility artifacts

- Code: `scripts/m6a_public_003_temporal_benchmark.py`.
- Configuration: `configs/m6a_public_003.json`.
- Raw model checkpoints and large training tensors are not saved; the report bundle contains only lightweight structured results, logs and figures.
