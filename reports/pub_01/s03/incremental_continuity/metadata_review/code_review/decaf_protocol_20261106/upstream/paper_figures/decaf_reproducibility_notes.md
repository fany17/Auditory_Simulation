## DECAF Reproducibility Notes

### Hardware and runtime (training run: `two_branch_recursive_context_3.0sec_dynamic`)
- GPU: NVIDIA GeForce RTX 4090 (24 GB)
- CPU: 16 cores / 32 logical threads
- RAM: 125 GB
- CUDA: 12.2
- Total training time: 2.20 hours
- Mean epoch time: 876.93 seconds

Source: `wandb/run-20250910_170516-rpet24uy/files/wandb-metadata.json` and `wandb-summary.json`.

### Loss weights (Pearson + L1)
We use `PearsonL1Loss` with `alpha=0.2`:
- Loss: \( L = (1 - \rho) + 0.2 \times L_1 \)
- Pearson term optimizes correlation with the target envelope.
- L1 term stabilizes amplitude/scale and reduces envelope drift.

This weighting keeps the primary objective (correlation) dominant while providing an auxiliary amplitude constraint.
