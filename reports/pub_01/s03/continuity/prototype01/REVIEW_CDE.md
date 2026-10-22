# SS06 independent review — M3/M4/M5 and integration contracts

- TASK_ID: `PUB-01-S03_SS06`
- REVIEWER: `prototype01_AB`, independently reviewing other workers/root files, not own M1/M2/common implementation.
- STATUS: `PASS_WITH_LIMITATION` — engineering review closed on 2026-10-23 (Asia/Shanghai).
- Runtime evidence: initial findings below were source-only; the closure section records subsequent independent inspection of the actual lightweight runtime logs and final source. This reviewer did not execute numerical code locally.
- Scope: synthetic prototype correctness and fidelity to the declared research controls. Scientific efficacy remains `UNVERIFIED`; real training and data access remain `NOT_RUN`.

## Reviewed paths

- `src/m6a_public/continuity/predictive.py`
- `src/m6a_public/continuity/alignment_innovation.py`
- `src/m6a_public/continuity/streaming.py`
- `src/m6a_public/continuity/objectives.py`
- `src/m6a_public/continuity/registry.py`
- `tests/test_ss06_predictive.py`
- `tests/test_ss06_alignment_innovation.py`
- `tests/test_ss06_integration.py`
- `scripts/pub_01_s03_ss06_smoke.py`
- `configs/pub_01_s03_ss06_prototype_20261023.json`
- `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS06_CONTINUITY_FRAMEWORKS.md`

## Findings requiring resolution before final report

1. **P2 — Misleading nonzero-gradient scalar count.** At initial review, smoke used `sum(p.numel() for p in active.values() if torch.count_nonzero(p.grad))` for `nonzero_gradient_parameters`. This counts every scalar in any parameter tensor with at least one nonzero gradient. It does not count scalar entries with nonzero gradient; large-dilation convolution taps without supporting history demonstrate why these can differ. Replace with summed actual nonzero counts, or rename the field explicitly. Retain the separate graph-participating parameter count. This affects accounting, not model output.

2. **P2 — Taskbook overstates absence of parameter changes during unit checks.** Section 6 initially said smoke does not update parameters, while `test_teacher_is_static_until_explicit_ema` deliberately perturbs online parameters and tests explicit EMA with decay 0/1. This is a legitimate deterministic unit check, not optimization, and the main smoke itself verifies unchanged parameters. Describe the distinction: no data-dependent fitting or optimizer step; fixed perturbation/EMA boundary unit tests are allowed. Do not claim all unit checks leave each temporary model untouched.

3. **P3 — M5 equation needs precision.** Section 8 initially describes gain projecting the innovation. The implemented minimal model has a same-width diagonal gain, `prior=tanh(A*z+b)`, `innovation=f-H(prior)`, `gain=sigmoid(G([prior,innovation]))`, `posterior=tanh(prior+gain*innovation)`, without a separate innovation projection matrix. State this explicitly in the taskbook rather than implying a full matrix observation correction.

These findings were sent to root before report creation. The reviewer did not change code. Root should record the final disposition and runtime evidence in its completion review. Known M1/M2 equation differences were already assigned to root and are outside this independent review.

## Verified by source inspection

- **M3 target separation:** downstream/predictive forward consumes only current/past EEG and explicit state. Future latents are constructed exclusively in `predictive_loss`; teacher parameters are frozen, targets use `detach()` inside `no_grad()`, and teacher EMA is explicit rather than a forward side effect.
- **M3 masks/state:** future-pair masks include all intervening indices, reject gaps and leading padding, allow trailing padding, and require valid support for every horizon. Loss rejects outputs started from carried state; this honestly prevents an incorrect silently reset teacher cache. Streaming downstream context is separate and tested for full/chunk agreement.
- **M3 stage controls:** downstream freezes predictive heads; frozen mode freezes encoder/context and leaves the envelope head trainable. The scaffold never claims pretraining has occurred. Variance penalties on features/context are explicit, but their existence alone does not prove noncollapse or useful neural content.
- **M4:** every lag index is nonnegative; feature bank uses causal encoder history. Adaptive/fixed controls share bank and readout, while adaptive gate uses EEG only. No target/audio enters the forward API. State contains exact lag cache and gate state, with declared parameter/compute differences between controls.
- **M5 prior-only purity:** prior-only skips the EEG encoder completely, uses autonomous latent dynamics, outputs zero innovation/gain, and rejects state tagged as another control mode. Tests explicitly install an encoder that raises if called and perturb both prefix and current EEG. EEG-only retains finite local convolution history but no recurrent latent state; full mode computes an explicit observed-minus-predicted feature residual.
- **Integration:** `StreamRunner` resets on any subject/session/record/split identity change and rejects offset discontinuity within an identity. Whole-batch reset is documented. Loss computes one correlation on the full valid sequence, handles masked target NaNs without using them in the reduction, and rejects constant targets. The smoother only consumes model outputs and is causal.
- **Execution scope:** reviewed factory, config and smoke contain no dataset loader, checkpoint load, optimizer, training loop or new-download operation. Config explicitly sets zero optimizer steps and absent data/checkpoint fields.

## Limitations that must remain explicit

- `compare_histories` replaces the whole prefix state, including the common local encoder cache. Consequently, the first 16 current samples can differ simply because the 17-sample local receptive field sees a different prefix. A later claim about added long-memory benefit must hold the encoder cache fixed or exclude/model this boundary effect with matched local controls. Same raw current EEG alone does not isolate recurrent memory. The current helper is valid as an entire-history intervention, not by itself as proof of a long-memory mechanism.
- M3 future targets are accepted via a separate tensor with a caller-owned identity contract. Equal shape does not certify that the caller supplied the same trajectory. There is no real loader yet to enforce provenance, split membership or normalization. No test supports a scientific leakage-free experiment claim at this stage.
- M3 pretraining currently requires complete boundary-started segments; teacher version-consistent streaming, cross-chunk future targets and production TBPTT are not implemented. The variance penalty can be satisfied by non-auditory structure and is not a guarantee against representational collapse.
- M5 prior-only is an autonomous time-since-reset baseline, not a useful stimulus-conditioned acoustic prior. Future predicted-envelope-history and trained smoothing controls remain necessary to attribute any advantage to new EEG evidence. No such advantage has been measured.
- Main smoke parameter counts distinguish registered/trainable/graph-active paths, but neither those counts nor isolated CPU smoke timings establish fair trained model budgets. Formal lag choice, data split, activation/runtime matching and test eligibility are pending future authorization.

## Acceptance recommendation

No blocking causality, target-gradient or prior-only-contamination defect was found in the inspected M3/M4/M5 code. Resolve the concrete reporting/documentation findings, then use actual server2203 numerical tests for engineering acceptance. If those pass, `PASS_WITH_LIMITATION` is appropriate for the basic prototype package only. This does not authorize or validate real EEG experiments.

## Review closure — 2026-10-23 Asia/Shanghai

**Final engineering decision: `PASS_WITH_LIMITATION`.** The three concrete initial findings are resolved in the current source/taskbook. No blocking defect remains within this review's prototype scope. The unresolved scientific/loader/budget limitations above remain applicable; this decision does not start or validate training.

Actual evidence inspected, rather than relying on the execution summary:

| Evidence file in this report folder | Verified content |
|---|---|
| `pytest_initial.txt`, `pytest_initial.xml` | 38 test methods and 71 subtests passed; XML records zero failures/errors/skips, hostname `nercn`, timestamp `2026-10-23T02:29:14.413042+08:00`. The XML aggregate `tests=109` includes subtests and must not be described as 109 separate test methods. One scalar-conversion warning was present. |
| `pytest_de_final.txt`, `pytest_de_final.xml` | After the DE test-only `detach()` edits, all 10 DE methods and 26 subtests passed with no reported warning; XML records zero failures/errors/skips and timestamp `2026-10-23T02:32:22.103626+08:00`. Inspected source places detach only in diagnostic float conversion, preserving test computations and checks. |
| `SMOKE_FINAL.json` | Five default models, 64 input channels, hidden=32, short shape `[2,256,1]`, long shape `[1,3840,1]`, finite values, unchanged model state, CPU/PyTorch `2.11.0+cu128`, hostname `nercn`. Config and report specify zero optimizer steps, no real data reads, and `NOT_RUN` for pretraining, real training and formal test. |

Inspected `pub_01_s03_ss06_smoke.py` performs the long comparison by splitting the same 3840-sample random input into 257-sample chunks with carried explicit state. It checks equality with the whole forward, records maximum absolute error, and checks model state against a pre-smoke clone. The observed errors in the JSON are:

| Model | Long whole/chunk max absolute difference |
|---|---:|
| M1 dense multiscale | 9.685754776000977e-08 |
| M2 persistent state | 2.9802322387695312e-08 |
| M3 predictive | 5.960464477539063e-08 |
| M4 adaptive alignment | 7.450580596923828e-08 |
| M5 innovation state | 8.940696716308594e-08 |

The script now sums `int(torch.count_nonzero(p.grad).item())` as `nonzero_gradient_scalars`. The output distinguishes M1's 35,361 nonzero scalar gradients from 40,481 graph-participating parameters, confirming that the former ambiguous tensor-level count is no longer reported. This remains a synthetic backward measurement, not a claim about trained budget fairness.

The taskbook now expressly permits fixed-perturbation/explicit-EMA unit tests while excluding data-dependent optimization, and accurately gives M5's elementwise correction as `tanh(prior + gain ⊙ innovation)`. Findings 1–3 are therefore closed.

The local-encoder-history concern also received a concrete control implementation. Inspected `compare_state_histories` preserves `real["encoder"]` in all three conditions, changes only post-encoder state keys, and keeps control-mode metadata. Its integration test hooks the common encoder and asserts exact equality (`atol=rtol=0`) of all current local features in real/reset/mismatched conditions for all five default models. That test is explicitly present in the passed initial XML. The existing `compare_histories` retains its separate entire-prefix meaning, and the taskbook distinguishes the two.

Scope of closure: the reviewed M3/M4/M5 implementations, common intervention/objective integration, reporting and taskbook consistency meet the declared engineering acceptance. The full test report includes M1/M2 outcomes but does not replace another reviewer's independent inspection of those modules. No real EEG accuracy, training improvement, physiologic interpretation, new holdout eligibility, teacher streaming training support or production loader correctness is established by these results.
