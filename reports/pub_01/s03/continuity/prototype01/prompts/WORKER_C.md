# SS06 worker C

Task PUB-01-S03_SS06; session prototype01_C. You are agent_work, implementing one disjoint part of the five continuity prototypes.

Worktree: E:/BCI_LFP_EEG/Project/0_EEG_Explanation_Auditory/_worktrees/Auditory_Simulation_ss06/
Read AGENTS.md and doc/tasks/PUB-01/subtasks/PUB-01-S03/SS06_IMPLEMENTATION_CONTRACT.md before changes. Read existing SS05 only if needed; it is frozen. Parent scope remains public EEG auditory direct decoding, patient data forbidden.

Implement predictive.py and tests/test_ss06_predictive.py for M3 only. Depend on common.py contract; if not yet present write against specified API. Use LocalEncoder and causal GRUCell context with explicit state. Teacher deepcopy of LocalEncoder, requires_grad False; EMA update only explicit method (not automatically forward). Forward returns envelope, latent features and per-horizon predictions in aux. Separate predictive_loss receives a tensor target source, detaches target, handles horizons masks and rejects insufficient valid support, combines normalized latent prediction and variance regularizer; no future into forward. Implement no_pretrain/frozen/finetune toggles precisely (aux heads unused downstream should not be counted active). Test causality, chunking, teacher isolation, no future-target gradient, gradients/collapse loss, frozen encoder semantics. No actual pretraining.

Allowed reads: code/taskbooks inside this worktree. Allowed writes: only owned paths listed in implementation contract, including WORKER_C.md. Do not change Git state or shared registry. Do not read datasets, start real training, load weights or run numerical tests on Windows. Root will run synthetic tests on server2203. No optimizer steps. Preserve all old files. Existing user authorization covers prototype work; no additional approval needed.

Completion requires runnable modules and meaningful unit tests, explicit assumptions and honest NOT_RUN runtime status. Follow report field contract. If interface missing, proceed against contract, communicate changes to root; do not silently alter API. Block only for unavoidable missing specification, preserve partial code. Stop after implementation/report; do not start follow-on experiments. Scientific efficacy is unverified.
