# SS06 worker AB

Task PUB-01-S03_SS06; session prototype01_AB. You are agent_work, implementing one disjoint part of the five continuity prototypes.

Worktree: E:/BCI_LFP_EEG/Project/0_EEG_Explanation_Auditory/_worktrees/Auditory_Simulation_ss06/
Read AGENTS.md and doc/tasks/PUB-01/subtasks/PUB-01-S03/SS06_IMPLEMENTATION_CONTRACT.md before changes. Read existing SS05 only if needed; it is frozen. Parent scope remains public EEG auditory direct decoding, patient data forbidden.

Implement common.py and dense_state.py for M1/M2, plus tests/test_ss06_dense_state.py. Follow exact shared exports. Include common input validation and detach_state. Implement all per-time convolution caches accurately for arbitrary chunks including 1 sample. Use per-sample fast GRUCell and slow learned stable exponential update. Return explicit DecoderOutput. Test default 64 channels and smaller hidden for speed, causality, chunk equality, reset, gradients, branch controls. No local numerical execution.

Allowed reads: code/taskbooks inside this worktree. Allowed writes: only owned paths listed in implementation contract, including WORKER_AB.md. Do not change Git state or shared registry. Do not read datasets, start real training, load weights or run numerical tests on Windows. Root will run synthetic tests on server2203. No optimizer steps. Preserve all old files. Existing user authorization covers prototype work; no additional approval needed.

Completion requires runnable modules and meaningful unit tests, explicit assumptions and honest NOT_RUN runtime status. Follow report field contract. If interface missing, proceed against contract, communicate changes to root; do not silently alter API. Block only for unavoidable missing specification, preserve partial code. Stop after implementation/report; do not start follow-on experiments. Scientific efficacy is unverified.
