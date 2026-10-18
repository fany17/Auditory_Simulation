# SUBTASK — PUB-01-S03_SS03 — EEG Direct Auditory Decoder Architecture Hypothesis

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S03_SS03` |
| Parent Node | `PUB-01-S03` / `PUB-01` |
| Status | **`ACTIVE — CODE/PROTOCOL BOOTSTRAP`** (2026-10-19; actual 2203 training **NOT_RUN**) |
| Upstream | `PUB-01-S03_SS02` `COMPLETED / PASS_WITH_LIMITATION`; baseline source `c7b23648b5a443ac9d4cba534493c5ae7d3a4c74` |
| Research data | `PUBLIC_ONLY`; patient data and STN-derived embeddings **FORBIDDEN** |
| Model prototype | `scripts/pub_01_s03_temporal_scales.py` |
| Config | `configs/pub_01_s03_ss03_multiscale_20261019.json` |
| Completion | Requires equal-data/equal-budget multi-seed train/val and held-out test, ablations, and independent review; **not achieved by this planning commit** |

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
