# SS06 M3 implementation report

- TASK_ID: PUB-01-S03_SS06
- SESSION_ID: prototype01_C
- STATUS: IMPLEMENTED / RUNTIME_NOT_RUN
- COMPLETED: causal LocalEncoder → GRUCell → envelope scaffold; separate future-latent objective with explicitly updated frozen EMA teacher; no_pretrain/frozen/finetune downstream modes; auxiliary-head stage switch; parameter accounting and synthetic tests.
- OUTPUTS: `src/m6a_public/continuity/predictive.py`; `tests/test_ss06_predictive.py`; this report.
- SELF_CHECK: source reviewed against shared API. No numerical Python execution, no optimizer steps, no model/data/weight loading or training. Runtime checks delegated to root on server2203.
- EVIDENCE_GAPS: RUNTIME_NOT_RUN until integrated server tests. Predictive efficacy, non-collapse after training, suitable horizons/regularizer weights, real EEG accuracy and comparative compute are unverified. Mode names do not imply a pretrained checkpoint exists.
- BLOCKERS: none for prototype implementation; depends on worker AB's `LocalEncoder`, `DecoderOutput` and `detach_state` contract.
- QUESTIONS_FOR_AGENT_CHECK: root should review explicit whole-segment loss limitation below and include it in the future training taskbook.
- STOP_REASON: assigned implementation finished; waiting for integration/runtime review only. Real experiment phase not started.

## Architecture and controls implemented

Default input `[B,T,64]`, LocalEncoder hidden 32 with dense causal features, causal per-sample GRUCell context 32, linear envelope 32→1. Explicit state contains encoder cache and last GRU state. Default horizon offsets `(4,16,64)` are sample offsets (at 64 Hz: 62.5 ms, 250 ms, 1 s), scaffolding choices rather than an experimentally selected configuration. Future predictors are independent linear 32→32 heads and are called only in explicit `pretrain` stage; default downstream forward omits them.

The prediction objective averages squared L2-normalized latent distance equally over horizons. Detached future targets come from a frozen EMA copy of LocalEncoder. A variance hinge on both online features and GRU context penalizes low per-channel standard deviation. Default regularizer weight 0.1 and standard-deviation target 1.0 remain proposed scaffold values; no selection or training occurred.

`no_pretrain` and `finetune` downstream modes train encoder, context and envelope; `frozen` freezes encoder/context and trains only the envelope. Frozen/finetune comparisons must later load the same approved encoder/context checkpoint; constructor initialization is random and does not establish pretraining. EMA updates occur only via explicit `update_teacher`; teacher stays frozen/eval and receives no gradient. Registered, trainable, downstream active, pretraining online, teacher and auxiliary counts are reported separately.

## Deliberate objective scope

`predictive_loss(output, target_source, valid_mask)` requires the same complete boundary-started source trajectory as forward. This identity is the caller's responsibility; shape checks cannot prove it. Mask rows permit a contiguous valid prefix followed by trailing padding only; internal gaps are rejected. Each horizon requires at least two valid source/target pairs, and every intermediate position must be valid.

Streaming forward supports persistent state and chunk equivalence. Predictive loss on carried chunks is explicitly rejected because a correct future training implementation must additionally specify EMA teacher cache/version handling across updates. This prevents silently encoding online and target representations with inconsistent history. No real loader or training entry point is added.

## Synthetic verification authored (not yet executed)

Eight tests cover full/chunk equivalence and no state mutation; causal envelope and latent heads; reset vs detach; predictive gradients and isolated target gradients; teacher immutability and explicit EMA; collapse penalty; all downstream modes and active counts; mask support and discontinuity/short-support rejection. The EMA test deliberately perturbs an encoder parameter by a fixed constant to verify copying, with no optimizer or data-dependent fitting.
