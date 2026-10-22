# SS06 worker AB implementation report

- TASK_ID: `PUB-01-S03_SS06`
- SESSION_ID: `prototype01_AB`
- STATUS: `IMPLEMENTED / NOT_RUN`
- COMPLETED: common causal blocks, M1 dense multiscale decoder, M2 persistent fast/slow decoder, synthetic unit checks.
- OUTPUTS: `src/m6a_public/continuity/common.py`, `src/m6a_public/continuity/dense_state.py`, `tests/test_ss06_dense_state.py`.
- SELF_CHECK: source reviewed against shared API. No model execution, data reads, fitting, optimizer updates, model download or Git mutation performed by this worker.
- EVIDENCE_GAPS: numerical tests remain `NOT_RUN`; root executes on server2203. Scientific accuracy/improvement is unverified.
- BLOCKERS: none identified by static inspection.
- QUESTIONS_FOR_AGENT_CHECK: runtime integration must verify arbitrary chunk equality and actual activated parameter counts. Short/long M1 controls have equal instantiated inventory but fewer activated parameters; do not claim compute/parameter-budget matching.
- STOP_REASON: assigned prototype implementation complete; awaiting root numerical verification and independent review. Real EEG experiments remain outside this round's authorization.

## Interface and assumptions

`CausalConv` caches exactly `dilation*(kernel-1)` raw input samples, with zero left padding at clean boundaries. `LocalEncoder` returns cache keys `spatial_temporal`, `depthwise`. It preserves all time points and has a 17-sample causal receptive field.

M1 state keys are `encoder` and `branches`; branch keys are string dilations. Modes `multi`, `short`, `long` use dilations `(1,8,64,256)`, `(1,)`, `(256,)` respectively, while preserving output shape. The selected branch features are averaged, projected and added to local features before the envelope head. Largest receptive field is 1041 samples (about 16.27 seconds at 64 Hz); this architecture's finite receptive field is not a promise of 60-second effective memory. Mode must be held fixed during a stream.

M2 state keys are `encoder`, `fast`, `slow`. At every sample, a GRUCell updates fast state; slow state follows `s_t=(1-alpha)*s_(t-1)+alpha*tanh(W*z_t+b)`, with each learned `alpha` strictly bounded to `[0.0001,0.9999]`. Local, fast and slow features jointly feed the output head. Initial alpha is 0.01, an engineering default requiring future train/validation selection. `state=None` clears all histories; `detach_state(state)` preserves their values while truncating gradients. No state is stored on modules or modified in place.

## Intended synthetic checks

Tests cover default 64-channel input, one-sample chunks, convolution dilation up to 256 across its complete receptive field, full/chunk equality, suffix causality, cache immutability, state detach/reset, fixed-current-EEG recurrent-history intervention, branch selection, finite gradients of all active parameters, no gradients in inactive branches, input/cache validation and bounded slow coefficients. Tests include backward only, with no optimizer or fitting loop.
