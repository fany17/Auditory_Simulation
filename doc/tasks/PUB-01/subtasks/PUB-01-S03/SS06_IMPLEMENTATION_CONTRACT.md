# SS06 implementation contract

Current authorization (2026-10-23): taskbooks + five basic prototypes + synthetic forward/backward checks only. No real EEG reads, optimizer steps, pilot, fitting, pretrained weights, new downloads, formal evaluation or result claims. All numerical model checks run on server2203, under a new isolated folder below /home/fanyu/auditory_simulation_m6a/pub_01/. Windows stores code and light reports only.

## Shared API

- Package `src/m6a_public/continuity/` uses PyTorch only (plus standard library).
- Inputs are float `x[B,T,C]`, defaults C=64, hidden=32. T>=1. Output `DecoderOutput(prediction[B,T,1], state=dict, aux=dict)`.
- `forward(x, state=None)`; `state=None` means clean boundary. Models never retain hidden state internally. Explicit returned state carries the exact prefix into the next chunk. State must not be modified in place.
- All five basic prototypes are causal in EEG coordinates: no future input, no temporal batch statistics, no dropout in default prototype. Future acoustic offset is separate from EEG causality; this scaffold makes no zero-latency biological claim.
- `common.py` exports `DecoderOutput`, `CausalConv(in_channels,out_channels,kernel_size,dilation=1,groups=1)`, and `LocalEncoder(in_channels=64,hidden=32)`. Both CausalConv and LocalEncoder forward return `(features[B,T,H], cache)`; cache is a tensor for CausalConv and a dict for LocalEncoder. CausalConv preserves last dilation*(kernel-1) samples. LocalEncoder: causal k9 Conv C→H, GELU, causal k9 depthwise H→H, GELU. No pooling.
- Shared helper `detach_state(tree)` detaches nested tensors without reset. Metadata boundary enforcement and history intervention live separately in `streaming.py` (root-owned).
- M1 `DenseMultiscaleDecoder`: LocalEncoder + dense causal branches (dilations 1,8,64,256; k5) + residual fusion and linear envelope head. `branch_mode` allows `multi`, `short`, `long`. Same shape in every mode; report active parameter differences.
- M2 `PersistentStateDecoder`: LocalEncoder + fast GRUCell + learned slow exponential state; per-sample state update; concat local/fast/slow -> output. Parameters support gradients; slow coefficient bounded strictly (0,1). Explicit fast/slow/encoder state.
- M3 `PredictiveContinuityDecoder`: LocalEncoder + causal context, envelope head and multi-horizon latent predictors. EMA target encoder detached; future targets only in a separate `predictive_loss` API, never forward. Include non-collapse regularizer and declared pretraining modes (no pretrain/frozen/finetune); no actual fitting.
- M4 `AdaptiveAlignmentDecoder`: LocalEncoder + nonnegative lag bank + slow EEG-driven lag weights + envelope readout. `adaptive` vs `fixed` mode; fixed lag bank/output delay shared; never audio/target inputs in forward.
- M5 `InnovationStateDecoder`: prior latent dynamics + predicted EEG observation + actual EEG feature residual + learned gain correction -> envelope. `full`, `prior_only`, `eeg_only` controls. No true past envelope inputs.
- Unit tests use seeded random/synthetic tensors only. Check full vs chunk equivalence; causal suffix perturbation; state reset/detach; finite gradients for actually used parameters; shape; control behavior. Backward is allowed, optimizer updates are not.

## Parallel ownership

- Worker AB: common.py, dense_state.py, tests/test_ss06_dense_state.py; report reports/pub_01/s03/continuity/prototype01/WORKER_AB.md.
- Worker C: predictive.py, tests/test_ss06_predictive.py; report WORKER_C.md in same report folder.
- Worker DE: alignment_innovation.py, tests/test_ss06_alignment_innovation.py; report WORKER_DE.md.
- Root: taskbook, config, streaming/objectives/registry/smoke, integration tests, runtime, review, Git.
- Workers do not edit one another's files, global registries, frozen SS02/03/05 files, or Git state. Do not execute models locally; root runs all tests on 2203 after integration.
- Completion report fields: TASK_ID, SESSION_ID, STATUS, COMPLETED, OUTPUTS, SELF_CHECK, EVIDENCE_GAPS, BLOCKERS, QUESTIONS_FOR_AGENT_CHECK, STOP_REASON. Report NOT_RUN until actual runtime evidence exists. Engineering acceptance is distinct from scientific acceptance.
