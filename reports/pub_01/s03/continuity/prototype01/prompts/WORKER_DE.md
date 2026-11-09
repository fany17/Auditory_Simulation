# SS06 worker DE

Task PUB-01-S03_SS06; session prototype01_DE. You are agent_work, implementing one disjoint part of the five continuity prototypes.

Worktree: E:/BCI_LFP_EEG/Project/0_EEG_Explanation_Auditory/_worktrees/Auditory_Simulation_ss06/
Read AGENTS.md and doc/tasks/PUB-01/subtasks/PUB-01-S03/SS06_IMPLEMENTATION_CONTRACT.md before changes. Read existing SS05 only if needed; it is frozen. Parent scope remains public EEG auditory direct decoding, patient data forbidden.

Implement alignment_innovation.py and tests/test_ss06_alignment_innovation.py for M4/M5 only. Depend on common.py specified API. M4 lag samples nonnegative and configurable default (0,4,8,16), slow bounded recursive weights, fixed vs adaptive control, explicit encoder/lag/gate state. M5 predicted latent prior, EEG observation innovation, learned gain corrects and readout; controls full/prior_only/eeg_only. Prior-only must be completely independent of EEG values incl current and prefix and ideally skip encoder; EEG-only has no recurrent latent history (local causal encoder cache allowed). Output aux must expose innovation/prior/gain/weights as applicable. Test exact chunk semantics, causal output, fixed/adaptive, prior-only input independence, finite gradients for used modules. No datasets, execution or optimization.

Allowed reads: code/taskbooks inside this worktree. Allowed writes: only owned paths listed in implementation contract, including WORKER_DE.md. Do not change Git state or shared registry. Do not read datasets, start real training, load weights or run numerical tests on Windows. Root will run synthetic tests on server2203. No optimizer steps. Preserve all old files. Existing user authorization covers prototype work; no additional approval needed.

Completion requires runnable modules and meaningful unit tests, explicit assumptions and honest NOT_RUN runtime status. Follow report field contract. If interface missing, proceed against contract, communicate changes to root; do not silently alter API. Block only for unavoidable missing specification, preserve partial code. Stop after implementation/report; do not start follow-on experiments. Scientific efficacy is unverified.
