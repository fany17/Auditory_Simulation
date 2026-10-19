# SUBTASK — PUB-01-S03_SS05 — History-Aware Predictive and Associative Memory

| Field | Value |
|---|---|
| ID | `PUB-01-S03_SS05` |
| Parent | `PUB-01-S03` / `PUB-01` |
| Status | **`ACTIVE / SYNTHETIC_PROBE_COMPLETED`** |
| Prior | `PUB-01-S03_SS03` completed (PR#3 head `43cea34749a985b4e3ac051860be193b5dc379f4`) |
| Patient Data | **FORBIDDEN** |
| Execution | Local CPU proof-of-implementation complete; server2203 public EEG training **NOT_RUN** |
| Branch | `work/pub-01-s03-ss05-predictive-memory`, separated from SS03 frozen results |
| Source | `scripts/pub_01_s03_ss05_predictive_memory.py`, `scripts/pub_01_s03_ss05_window_decoder.py` |
| Controls | `local`, `state`, `predictive_state`, `assoc_no_pred`, `full`, standard GRU |
| Evidence | `reports/pub_01/s03/predictive_memory/{REPORT.md,SYNTHETIC_RESULTS.json}` |
| Next mandatory read | `reports/pub_01/s03/predictive_memory/NEXT_PUBLIC_EEG_PROTOCOL.md` |
| SS04 status | BACKLOG, not automatically authorized |

## Core question

Can a direct EEG→continuous auditory feature decoder make **content-specific use of remote temporal history**, rather than merely increasing convolutional receptive field or smoothing the output? Is the improvement attributable to **selective associative read/write**, and does a separate **prediction-error update** offer a further, falsifiable gain?

These are three different claims. Do not blur memory capacity, sample efficiency, EEG performance and algorithmic novelty.

## Hypotheses and design

- H1: remote genuine history gives more benefit than wrong-record/shuffled/reset histories when **the current input remains fixed**.
- H2: `assoc_no_pred` exceeds gated multi-rate `state` with otherwise comparable training conditions.
- H3: `full` exceeds `assoc_no_pred`; if not, predictive-error update is unnecessary in its current form.
- R: standard GRU with similar active parameters must be compared; superior toy convergence alone does not demonstrate a generally better memory architecture.

Architecture (prototype): time-local feature extraction, three different retention/update-rate state groups, previous-state prediction of the present feature, innovation `f_t - prediction_t`, innovation-conditioned gates, a content-addressable associative matrix with gated delta-rule update, and memory feedback to current readout.

**Novelty limitation:** gates, selective state propagation, retrieval and delta rules all have extensive prior art (Mamba, Titans, Gated Delta Networks). PI-MRAM is a working combination/hypothesis, **not yet a demonstrably new computation operator**.

## Stage completed 2026-10-19: two purely synthetic probes

1. Delayed one-cue memory: cue at t2–6, query at t25–30. A 3-tap local model performs near chance; full model and GRU both 100% under 160 steps.
2. Two-key interference: early key0/key1 values followed by overwrite of one key, query at t39–46. Under the fixed **160-step** toy budget, `full` and `assoc_no_pred` 100%, GRU62.5%, no-memory near50.8%; extra GRU **600-step** capacity diagnostic 100%. Only one seed, single synthetic family; 600-step comparison is NOT same budget.
3. **Prediction-gate incremental evidence absent.** `assoc_no_pred` performs as well as `full`. Must explicitly carry this result forward.
4. Local tests: ten CPU-only shape, gradient, causality, state reset and streaming-equivalence checks PASS, in local torch2.10.0+cpu; **not an independent server2203 execution**.
5. No biological data, model weights, published EEG scores or artifact have been produced by this subtask.

## Next execution

Codex may run SS05 on 2203 only after reading `NEXT_PUBLIC_EEG_PROTOCOL.md`. First inspect existing public train manifests and determine a truly non-overlapping family/story grouping. The old SS03 test_seen/test_heldout are used already and must **not** guide any new structure, lag, training budget or evaluation threshold.

For practical 64Hz EEG, treat each3s 192×64 record segment as a summary token and maintain memory across contiguous windows. The window head emits192-sample envelope directly (without candidate audio), with proper train-only standardization and record-by-record state reset.

Formal public evaluation must use a new fixed internal heldout from previously untouched SS05 model fitting data, and should be labeled **internal only**, or be delayed for an independent new dataset. New source-family grouping/availability has not yet been verified.

## Stop / completion criteria

Current status stays ACTIVE. No one may mark EEG improvement, PI-MRAM novelty or SS05 COMPLETED until real public EEG data loading, validated train/validation, fixed prospective evaluation, strong GRU control, history corruption, parameter/runtime sensitivity, and review are finished. Do not start SS04, S04, or make ART-AUDREP-v1; preserve patient boundary and all negative outcomes. 
