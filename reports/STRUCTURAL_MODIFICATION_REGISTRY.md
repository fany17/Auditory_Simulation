# M6A-PUBLIC-003 Structural Modification Registry

本 registry 只覆盖本轮三组：downsampling、RF growth/multiscale/decoupling、explicit change/event branch。所有模型均为小型、从零训练的 1D temporal models；不含 pretrained backbone。

| family | modification | computational meaning | temporal-resolution effect | RF effect | parameter/compute change | matched control | falsifiable endpoint | status |
|---|---|---|---|---|---|---|---|---|
| downsampling | early vs late stride | same total factor 4, move stride timing | intermediate frame step changes | RF changes as a consequence of stride timing | exact same convolutional parameterization | early vs late | seen-rate, omission, onset metrics | RUN + RF_CONFOUND noted |
| RF growth | uniform/local dilation | retain local windows | 1 ms output step | slow RF growth | exact schedule-parameter match | exponential/delayed | RF table and task metrics | RUN |
| RF growth | exponential dilation | grow context 1,2,4,8 | 1 ms output step | fast RF growth | exact schedule-parameter match | uniform/delayed | RF table and task metrics | RUN |
| RF growth | delayed-growth dilation | keep early layers local | 1 ms output step | delayed RF growth | exact schedule-parameter match | uniform/exponential | RF table and task metrics | RUN |
| RF growth | parallel multiscale k=3/7/15 | concurrent temporal windows then 1x1 fusion | 1 ms output step | max branch RF per block | width-adjusted; must be checked | uniform_local | RF table and task metrics | RUN/CONFOUNDED if outside ±10% |
| RF/downsampling | stride-coupled vs dilation-decoupled | same target RF, change source of RF | 4 ms vs 1 ms final step | matched theoretical RF | exact convolutional parameterization | pairwise | fast-event metrics at matched RF | RUN |
| RF diagnostic | kernel 3/7/15/31 | change direct local window | 1 ms output step | direct kernel growth | width-adjusted; confound retained if >±10% | kernel_3 | RF and performance diagnostic | RUN/CONFOUNDED if needed |
| event branch | explicit Δx branch | expose first temporal difference | same final resolution | same branch RF | branch width 11 vs baseline width 16 | ordinary second raw branch | onset/omission/phase metrics | RUN |

参数匹配状态以 `m6a_public_003_model_parameters.csv` 为准；任何 `CONFOUNDED` 比较只能作为工程描述，不能被写成单一结构因果证据。

## 10/20/50 ms Supplementary Perturbation Evaluation

This additive evaluation was run after the formal results and does not replace
or overwrite them. The supplement retrained the same 15 architecture variants
because the formal run saved no checkpoints; it reused the formal train,
validation, test-seen and test-unseen generation rules, AdamW optimizer,
8-epoch budget, batch size 64, gradient clip 5.0 and seeds 11/22/33.

Perturbation definition:

- **Localization:** the same test-seen signal is translated by a positive
  global shift of exactly 10, 20 or 50 ms with zero fill. Only the onset target
  is translated by the same amount. `localization_onset_mae_ms` measures the
  shifted-onset error; `localization_shift_recovery_mae_ms` measures the error
  in recovering the shift from the model's predicted onset change.
- **Discrimination:** each magnitude is an independent mixed regular/jitter
  probe generated with `unseen_jitter_ms=[m]`. Rates and phase magnitudes are
  reset to the training ranges, so `discrimination_balanced_accuracy` isolates
  regular-versus-jitter classification.
- **Generalization:** the positive-jitter subset of the same probe is used for
  `generalization_jitter_mae_ms` and its magnitude-normalized counterpart.

The 10/20/50 ms probes are all **unseen/extrapolative**: training contains
2/4/6 ms jitter and the original unseen split contains 3/5/7 ms. None of the
three supplementary magnitudes is used for training or tuning. This is a
small synthetic perturbation benchmark, not localization or discrimination
evidence for real audio or neural recordings.

Supplement run status: `45/45` seed runs PASS. The
formal parameter-match labels are retained unchanged: early_downsample=PASS, late_downsample=PASS, uniform_local=PASS, exponential_growth=PASS, delayed_growth=PASS, parallel_multiscale=PASS, rf_stride_coupled=PASS, rf_dilation_decoupled=PASS, kernel_3=PASS, kernel_7=PASS, kernel_15=PASS, kernel_31=CONFOUNDED, event_baseline=PASS, explicit_change=PASS, ordinary_second_branch=PASS.

Traceability files: `m6a_public_003_perturbation_metrics_by_seed.csv`,
`m6a_public_003_perturbation_summary.csv`,
`m6a_public_003_perturbation_run_status.csv` and
`m6a_public_003_perturbation_manifest.json`.

The registry now includes an additive 10/20/50 ms magnitude-axis evaluation; no fourth structural group was introduced.
