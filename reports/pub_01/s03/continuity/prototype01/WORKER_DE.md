# SS06 M4/M5 implementation handoff

- TASK_ID: PUB-01-S03_SS06
- SESSION_ID: prototype01_DE
- STATUS: IMPLEMENTED / NOT_RUN
- COMPLETED: M4 adaptive EEG-only causal lag mixture and fixed-weight control; M5 explicit latent prior, predicted EEG observation, EEG innovation, sigmoid correction gain and envelope head; strict prior-only and local EEG-only controls; synthetic contract tests.
- OUTPUTS: `src/m6a_public/continuity/alignment_innovation.py`; `tests/test_ss06_alignment_innovation.py`; this report.
- SELF_CHECK: Manual API/shape/state-flow inspection completed. No numerical execution, no optimizer step, no datasets, no weights loaded, no Git mutation by this worker. Root must run tests on server2203 before claiming engineering validation.
- EVIDENCE_GAPS: Runtime is NOT_RUN. No real EEG performance or evidence of improvement exists. Parameter counts/runtime/fairness are not measured here. Neither model is a claim of physiological timing or Bayesian optimality.
- BLOCKERS: None for implementation. Runtime evidence belongs to root integration on server2203.
- QUESTIONS_FOR_AGENT_CHECK: None requiring user input. Verify shared LocalEncoder returns [B,T,H] plus explicit cache and root treats input state as record-boundary-sensitive.
- STOP_REASON: Assigned implementation and honest handoff complete; awaiting root integration. No follow-on training authorized.

## Precise behavior

M4: `AdaptiveAlignmentDecoder(in_channels=64, hidden=32, lags=(0,4,8,16), mode='adaptive', fixed_lag_weights=None, slow_decay=0.98)`. Dense local EEG features feed the same nonnegative lag bank for both controls. Adaptive mode slowly filters EEG-derived logits and applies softmax; fixed mode uses a prespecified normalized nonnegative vector (uniform default, one-hot available). Lag and gate caches persist across chunks. Audio targets are absent from the forward API. Learned decay remains in (0,1), with a small numerical margin. Aux exposes weights, aligned_features and slow_decay. Fixed mode retains unused gate parameters for convenient paired state_dict loading; report active rather than nominal parameter counts.

M5: `InnovationStateDecoder(in_channels=64, hidden=32, mode='full')`. For each sample, prior=tanh(A previous_latent); predicted_observation=H prior; innovation=encoded_EEG-predicted_observation; gain=sigmoid(G[prior,innovation]); corrected_latent=tanh(prior+gain*innovation). Shared readout predicts an envelope sample. Aux exposes all of these values. The residual has an operational meaning in encoded-EEG space; it is not established as neural prediction error.

`prior_only` skips EEG encoder and observation correction, including during prefix processing. It only uses the autonomous latent dynamics and readout; same-length inputs with different EEG values produce identical outputs from identical initial state. `eeg_only` uses tanh(local_EEG_features) and readout, retaining only the finite causal encoder cache, never a recurrent latent state. Controls keep unused modules in their parameter inventory to allow exact paired state_dict loading; tests distinguish used parameters for gradient validation.

M5 states include a control-mode tag, and a cache produced by a different control mode is rejected. In particular a full-model EEG-conditioned latent cannot silently be reused as a prior-only cache.

## Synthetic checks written, not yet executed

Tests cover all five M4/M5 control modes: shape, finite outputs/gradients, one-sample and arbitrary chunk partition equivalence, causal suffix intervention, input-state immutability, detach vs reset, fixed/adaptive bounded lag weights, zero-lag bank edge case, prior-only independence of all EEG values and encoder bypass, EEG-only finite local receptive field, explicit innovation/gain identity, invalid arguments/state shape. These checks are engineering tests and contain no scientific training result.
