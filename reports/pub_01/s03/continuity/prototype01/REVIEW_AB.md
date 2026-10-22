# SS06 independent review — M1/M2 and shared stream/objective contracts

- TASK_ID: PUB-01-S03_SS06
- SESSION_ID: prototype01_review_AB_by_DE
- STATUS: STATIC_REVIEW_COMPLETE / RUNTIME_PENDING
- Reviewer independence: reviewer implemented M4/M5, not the M1/M2 or common/streaming/objective files reviewed here.
- Engineering verdict: **No blocking correctness defect found by static inspection in M1/M2. Runtime acceptance remains pending server2203 results.** One interpretation limitation in whole-prefix history intervention must be retained or addressed before claiming a selective long-memory effect.
- No local model execution, dataset read, training or optimizer step was performed for this review. Implementations were not edited.

## Files actually inspected

Worktree root: `E:/BCI_LFP_EEG/Project/0_EEG_Explanation_Auditory/_worktrees/Auditory_Simulation_ss06/`.

- `src/m6a_public/continuity/common.py`
- `src/m6a_public/continuity/dense_state.py`
- `tests/test_ss06_dense_state.py`
- `src/m6a_public/continuity/streaming.py`
- `src/m6a_public/continuity/objectives.py`
- `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS06_CONTINUITY_FRAMEWORKS.md` for interpretation, with root's declared pending corrections to the M1 full-convolution description and M2 update-rate equation.

## Static evidence and reasoning

1. **Causality and exact cache length.** CausalConv prepends exactly dilation×(kernel−1) raw input samples and uses no right padding, yielding T outputs from T new samples. LocalEncoder composes two causal k9 layers with separate raw-input caches for a 17-sample receptive field. No temporal normalization, dropout, future indexing or temporal pooling appears in these paths.
2. **M1 continuous dense features.** Each active k5 branch processes the complete local feature trajectory with its own dilation cache; all branch outputs remain [B,T,H]. The largest branch has 1041 samples of combined receptive field. Fusion and readout operate independently at each output time. Short/long controls execute only their selected branch; code explicitly acknowledges unequal active parameter counts despite an equal registered inventory.
3. **M2 persistent recurrence.** Fast GRU and slow state both update once per new sample, beginning from explicit caller-provided state. The slow coefficient is an update rate: slow=(1−alpha)×previous+alpha×tanh(W local), with alpha initialized to .01. Its bounded sigmoid parameterization includes a numerical margin. `detach_state` recursively detaches tensors while preserving values and metadata; it does not reset the stream.
4. **No input-state mutation observed.** Convolution uses fresh concatenations; recurrent updates assign new tensors rather than in-place mutation. Nested caches are returned explicitly. These properties support chunk equivalence and preserve caller-owned history, subject to numerical confirmation.
5. **Stream boundaries.** StreamRunner keys by subject/session/record/split for each lane, resets the full batch on any identity change, and rejects offset gaps within an unchanged identity. It updates held state only after the model returns. Full-batch reset is a documented prototype limitation; future production loading must use aligned boundaries or add per-lane state management. One scalar start_sample is shared by the batch, so asynchronous lane offsets are also outside this prototype contract.
6. **Continuous objective.** dense_objective computes one centered Pearson correlation over all valid times in each passed segment, then averages segments; it does not average independently scored short windows. Masked nonfinite samples are eliminated before reductions; finite valid values, at least two valid samples, nonconstant targets and nonnegative finite MSE weighting are checked. The caller still must guarantee that a segment never joins separate records and that normalization was fit on train only; the function does not establish dataset provenance.
7. **Output-only smoothing control.** causal_smooth uses previous output state and current prediction only, with a scalar EMA coefficient. It cannot access target audio. Its state is explicit and its formula is partition-invariant when carried correctly; runtime tests still needed.

## Finding / interpretation limitation

**Whole-prefix intervention is not a selective recurrent-memory intervention.** `streaming.py::compare_histories` creates real/mismatched state from different prefixes and uses state=None for reset. Consequently it changes the local encoder cache as well as long-memory state. Holding current raw EEG fixed is satisfied, but the local features in the initial 16 current samples can differ. For M1, finite branch history also changes by design. A nonzero output difference from this helper alone therefore does not prove use of slow/recurrent information beyond local context.

Acceptable handling: label the existing helper as a whole-prefix intervention and add a complementary state-only intervention that holds the real local encoder cache fixed while replacing/resetting recurrent state; alternatively a future scoring protocol must explicitly exclude the local transient and explain which memory range remains intervened upon. M2's dedicated unit test already demonstrates the selective version by preserving encoder cache and zeroing only fast/slow state. This is an interpretation limitation, not a discovered violation of causality or the currently declared fixed-raw-EEG API.

## Test coverage inspected; not executed by reviewer

The M1/M2 test file includes dilation-256 convolution testing beyond its entire 1025-sample kernel span, one-sample chunks, k1 and depthwise edge cases, local cache immutability, nested detach behavior, invalid shape/dtype caches, all branch controls, explicit recurrent history intervention, finite gradients for active paths and the slow-state update equation. Forward/chunk comparison is appropriately tolerant of floating-point differences. Dense-model chunk testing itself is shorter than the longest receptive field, but the separate causal-convolution test exercises every delayed tap; a longer full M1 integration check would strengthen coverage if integration runtime is inexpensive.

Root streaming/objectives tests were not yet present in the inspected tests directory at review time. Boundary-reset, offset-gap, continuous-loss/mask and smoother chunk tests should be included in root integration before claiming those components runtime-validated. Runtime acceptance and the five-model 60-second smoke must be based on actual server2203 evidence, not this static review.

## Remaining limits

No real EEG improvement, physiological delay, parameter/compute fairness, normalization validity, data split independence or scientific acceptance is established. These are intentionally outside this prototype-only authorization. No further training should start on the basis of this review.
