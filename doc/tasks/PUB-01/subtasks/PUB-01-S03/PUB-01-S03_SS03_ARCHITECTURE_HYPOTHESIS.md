# SUBTASK — PUB-01-S03_SS03 — EEG Direct Auditory Decoder Architecture Hypothesis

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S03_SS03` |
| Parent Node | `PUB-01-S03` / `PUB-01` |
| Status | **`COMPLETED / PASS_WITH_LIMITATION`** (2026-10-19; real4×3 formal train/val/test and same-agent separate review completed) |
| Upstream | `PUB-01-S03_SS02` `COMPLETED / PASS_WITH_LIMITATION`; baseline source `c7b23648b5a443ac9d4cba534493c5ae7d3a4c74` |
| Research data | `PUBLIC_ONLY`; patient data and STN-derived embeddings **FORBIDDEN** |
| Model prototype | `scripts/pub_01_s03_temporal_scales.py` |
| Config | `configs/pub_01_s03_ss03_multiscale_20261019.json` |
| Completion | Equal-data/equal-budget four-variant three-seed train/val/test and separate review achieved; restricted prototype H1 supported, no physiology/SOTA claim; SS04 remains BACKLOG |

## 1. Core scientific hypothesis

Does an **explicit mixture of temporal receptive fields** preserve EEG time-dependent information better than a **single receptive-field scale**, when decoding an aligned continuous auditory envelope?

The first target is not a SOTA claim, a new audio-generation model, or a STN statement. We test one falsifiable hypothesis on the same public SparrKULee **available-author-fold4** 64-Hz recordings, with the same EEG→envelope target and without candidates at inference. Any improvement can also reflect longer contextual coverage, which is why we must include a **uniform-long** control.

### Controlled structural comparison (same parameters, kernel taps, branch widths)

- `short`: dilations `(1,1,1)`; 9-sample effective support per branch.
- `middle`: `(4,4,4)`; 33-sample support.
- `long`: `(16,16,16)`; 129-sample support.
- `multiscale`: `(1,4,16)`; 9/33/129-sample supports simultaneously.

Each variant uses: 64-channel EEG → shared 48-channel pointwise spatial projection → three 16-channel **depthwise** convolutions (9 taps each) → pointwise mixing → 1-channel envelope. Dilation alone changes; thus parameters and nominal multiply-add count are matched. Effective receptive field is `(kernel-1)*dilation+1`, giving approximately 0.14/0.52/2.02 s at 64 Hz by sample count. **All are offline/noncausal** and see identical 3-s input windows.

**H1:** `multiscale` outperforms `long` as well as `middle` on held-out per-subject reconstruction; different temporal scales contribute beyond having a single long field.

**H0 / negative evidence:** `multiscale` does not improve, or only improves against `short/middle` but not `long`; do not rename this as evidence for a multiscale mechanism.

A 64-Hz sampling interval is 15.625 ms; these are **computational receptive-field scales**. They do not establish biological spike timing or neural processing latencies.

## 2. Frozen input contract

Reuse ONLY the public 2203 inputs already prepared under `pub_01/s03_strong_baselines_20261015/transformer_available01`:

1. `prepared/manifest_frozen.json`, original `scope` and `train_only_normalization.npz`.
2. `envelope_readout_seed1024/manifest_frozen.json` and `{train,val,test_seen,test_heldout}_envelope.npy` (the actual existing aligned targets).
3. For each manifest record and readout window index `i`, read raw 64-Hz EEG over `[i*96 : i*96+192]`, apply **the same frozen train-only channel normalization**, and pair with the existing envelope target at `phase_offset+i`.
4. Verify phase, story, subject, support, window count, scale, and array orientation with **strict assertions**; reject ambiguous records. Do not infer new labels from test outcomes, and do not resplit data.
5. Create a **new**, write-once server-only cache under `pub_01/s03_architecture_20261019`; never modify the SS02 recordings, selected checkpoints, queries, or report.
6. `test_seen`: seen subjects on specified held-out story IDs. `test_heldout`: unseen subjects, but **not necessarily unseen audio stories**. Maintain both independently.

On Windows local Git keep only own code/config/aggregate CSV/JSON/PNG. All raw EEG, audio, envelope tensors, model states and per-window prediction arrays remain on server2203. No patient/STN material enters this repository.

## 3. Staged execution for Codex

### SS03-A — verify inputs + CPU/GPU smoke (first executable action)

- Read `doc/CODEX_START_HERE.md`, this spec, SS02 `REPRO_REPORT.md` and config.
- Fetch the feature branch in a **separate worktree** or use a non-destructive workflow; preserve existing local uncommitted CNN/TRF/stash. No reset/checkout of the occupied worktree.
- Inspect SS02 server-only source and provenance. Make a per-phase, per-subject window-count manifest; check train/val/test exact story IDs, held-out persons, envelope alignment and finite/shape conditions.
- Execute `python -m unittest tests.test_pub_01_s03_temporal_scales` (or repository-supported discovery). Run actual 2203 CPU and GPU forward/backprop with public 64-Hz windows for all four variants.
- Record parameters, memory, inference latency and receptive-field tests. **No scientific test scores** and no reading test labels to choose architecture.
- Build the direct EEG→envelope train/validation loader. The **prior frozen Transformer EEG encoder is not to be inserted** into these from-scratch variants.

### SS03-B — training feasibility and protocol freeze

- Conduct a **maximum 100-step 1-seed train/val pilot** to catch loss instability, target scaling and resource issues; only train/val may guide changes. Save failures, CPU/GPU environment, input hashes and exact pipeline.
- Freeze one common full optimizer-step budget/evaluation cadence, negative-Pearson training objective, validation checkpoint rule, all four architectures and **three seeds (1024, 2026, 31415)** *before any formal test evaluation*.
- Reuse one windowing/normalization/split and identical batches across variants for each seed. Report any actual parameter/FLOPs or budget mismatch rather than concealing it.
- Primary validation metric: 60-s segment Pearson after 50%-overlap Hann reconstruction → per-subject mean → seen/heldout groups separately. Use validation only for checkpoint/selection; avoid tuning based on held-out queries.

### SS03-C — formal tests, once after freeze

- Run the 4 × 3 planned architecture/seed experiments within the same budget.
- Evaluate each selected model one time on the fixed `test_seen` and `test_heldout` partitions. Primary result: each group's **subject-macro 60-s Pearson r**; report paired per-subject multiscale-minus-long differences and subject bootstrap intervals. Weighted 2:1 score is secondary.
- Include same-record wrong-EEG/time-shift negative control, constant/mean envelope control and identity canary: model inference must depend on EEG only, not a candidate waveform or answer.
- Predeclared validation-only temporal diagnostics may stratify reconstruction error by envelope modulation frequency, onset transients and synthetic jitter. Do **not** assume this proves temporal physiology; `physical_sync=UNKNOWN` remains.
- If H1 is unsupported, document the negative result and revisit the scientific explanation rather than searching another architecture on the test set.
- **No** 2AFC/5AFC/large-pool post-hoc optimization or new GPT/TimeMixer++/SaShiMi model here. Retrieval is the separate `SS04` acceptance stage.

## 4. Baseline interpretation and comparison boundaries

- SS02 HDPS 100-Hz Mel-only Top-1 12.03%, Top-5 30.45%, Top-10 42.10%: different task and signal sampling; **not** a direct numeric comparison.
- SS02 Transformer-CLIP 64-Hz 2AFC weighted 82.38%: contrastive candidate recognition; **not** direct Pearson benchmark.
- Its fixed EEG encoder + Conv32 envelope readout weighted r=0.1593 is a **context reference** on the available fold and 60-s aggregation, with a different pretrained/contrastive objective. Treat as an external architecture reference, not automatically a compute-matched control.
- Old SS01 TRF r≈0.1819 used 10 subjects/93 recordings and a different split; do **not** rank against present experiments until a fair same-fold rerun.
- If an internal linear ridge/TRF same-fold comparison is viable, add it **as an explicitly labeled reference** with identical splits and aggregation. Keep original S02/SS01 frozen.
- Do not claim original-paper-equivalent cohort, original challenge scores, or physically verified synchronization.

## 5. Deliverables and stop rule

Required outputs, all in a new `reports/pub_01/s03/architecture/` path:

1. `DATA_CONTRACT.md`: exact public subjects/records, split, timestamps/support, feature/target definitions, exclusions and hashes.
2. `EXPERIMENT_MATRIX.md`: frozen hypothesis, four variants, seeds, parameter count, nominal compute and training budget.
3. `RESULTS.csv` and readable `REPORT.md`: aggregate and all per-subject metrics, paired comparisons, nulls, plots and limitations; NOT_RUN remains empty.
4. `REVIEW.md`: independent technical, acceptance, and parent consistency checks; no false third-party signoff.
5. Reproducible run commands and source/config hashes; server-only path manifest without raw array/weight payload in Git.

Status progression: `ACTIVE / PROTOTYPE` → `SMOKE_PASS` → `TRAIN_VAL_FROZEN` → `TEST_REVIEW` → `COMPLETED / PASS[_WITH_LIMITATION]`. **Do not auto-complete or start SS04.** Retain every negative/failure outcome.

## 6. Boundary with the STN project

`Auditory_Simulation` does public representations and methods (M6A). `STN_Decoding_Encoding` handles patients, STN recordings and individual adaptation (M6B). Transfer only through a **frozen versioned artifact** if later approved; no patient data or embedding travels back.

## 7. Execution Record — 2026-10-19

PR#3 head9ffef26在独立managed worktree `C:/Users/fanyu/.codex/worktrees/pub-01-s03-ss03/Auditory_Simulation` 接续，分支work/pub-01-s03-ss03-multiscale；原main/CNN/TRF/stash未动。2203新根`pub_01/s03_architecture_20261019`，619记录全部同坐标target float32逐元素一致，70train/49val/69seen/13heldout，窗口252454/30500/36573/48119。原数据与SS02结果只读。

5结构tests、四组真实window CPU/GPU forward/backward PASS；每组6001参数/1124352 nominal conv MACC每3s窗。seed1024 multiscale100步pilot仅train/val，有限且loss可下降，不复用其权重。正式方案在任何SS03 test成绩前write-once冻结：每组25000 optimizer steps，每2500步完整val49人60s subject-macro r选最高、相同值选最早，无early-stop；3seed1024/2026/31415，共12次，全部训练后统一核验预算再开放一次test。常数train mean也在test前保存，其Pearson零方差未定义而非0。

4个计划contrast（两个组×multiscale对long/middle）先人内平均3seed，再person bootstrap10000；给95%CI及Bonferroni family95%区间（各98.75%），不以只胜短感受野声称H1。源/config/contract身份在formal_protocol及DATA_CONTRACT中限定记录，不引入全量raw hash审计。可选线性参考/时序jitter诊断NOT_RUN，不扩大预设架构。

正式配置`configs/pub_01_s03_ss03_multiscale_formal_20261019.json`，数据、预算、命令与预实验轻量证据见`reports/pub_01/s03/architecture/{DATA_CONTRACT.md,EXPERIMENT_MATRIX.md,COMMANDS.md,evidence/bootstrap01/}`。后续实际4×3已完成，见以下Completion Record；不启动SS04。

## 8. Completion Record — 2026-10-19

- 真实输入、normalization、619记录与窗口身份见上一节；4variant×3seed全部从头、每次25000 optimizer steps、10次完整val，无early-stop；所有12训练完并检查预算后才打开test，各selected checkpoint一次评分。训练源码/config按formal_protocol SHA256校验，主执行代码已提交4661439，基于PR head9ffef26；最后报告/图表提交见PR#3，不将初始commit冒充新增源码。
- 3seed平均seen/heldout r：short .078403/.084156，middle .092966/.104134，long .064112/.069788，multiscale .103053/.113798；multiscale加权.106635。其对long Δ=.038941/.044010、对middle Δ=.010087/.009664，四计划contrast Bonferroni family95%区间下界均>0。H1仅在当前四原型/fold/预算内获支持；长组最差有稀疏采样等替代解释；原型绝对r仍低于SS02参考.1593，不称超过强基线或生理机制。
- 错误EEG近0，constant train mean零方差Pearson未定义（非填0）；12个EEG-only canary通过。测试完整段无效排除0，所有人保留；未满60s尾段沿冻结规则不评价。
- 技术/审阅：5结构unit tests和真实CPU/GPU梯度/latency smoke PASS，100步pilot无test；全部源hash/步数/val selection/test_once/逐段逐人macro/weighted独立重算通过（汇总误差0）；另从原始EEG独立CPU重建24个首60s段，maxr误差7.41e-6。审阅为同一agent separate post-execution pass，非第三方独立签署。最终包检查见package_review.json。
- 输出：`reports/pub_01/s03/architecture/{REPORT.md,REVIEW.md,RESULTS.csv,SUMMARY.csv,per_subject_all_seeds.csv,paired_comparisons.csv,SERVER_MANIFEST.csv}`、两张静态图、freeze/pilot/12run轻量证据及源码/config。36结果行含每组每seed和加权，未选择性保留。新增执行代码为scripts/pub_01_s03_ss03_*.py；只维护相关task/入口/登记状态，SS02及旧CNN/TRF资料未覆盖。raw/cache/target/weight/逐窗数组仅2203，Windows仅轻量文件。
- 失败/偏差：本轮没有工程失败重启，保留全部12结果；一个available public fold、三seed/13heldout人、共享故事/有限预算、物理同步UNKNOWN和原型容量限制保留。可选同fold linear/jitter等未跑，未擅自增加数据、权重、模型或retrieval。
- 三级验收：Technical PASS，Acceptance PASS_WITH_LIMITATION，Parent consistency PASS。SS03 COMPLETED不等于S03/PUB完成；S03/PUB仍ACTIVE、SS04/S04/S05 BACKLOG，无ART-AUDREP-v1。当前停止，PR保持Draft、不自行合并main；后续由研究者决定。
