# ExpoComm-B: Project Status Update

**Project**: Adapting Sparse Communication Graph Learning for Multi-Agent Coordination in Bandwidth-Constrained Settings  
**Course**: COMP 559 – Machine Learning with Graphs  
**Team**: Aneesh Thatiparti (vt20), Ansh Dabral (ad258), Madhu Thatiparti (mt180)  
**Date**: April 4, 2026

---

## 1. Project Overview

Our project combines two recent papers in multi-agent reinforcement learning (MARL):

- **ExpoComm** (Li et al., ICLR 2025) — learns *who* to communicate with using a sparse exponential graph topology (O(log N) connections per agent).
- **BVME** (AAMAS 2026) — learns *how much* information to send using a variational message encoding bottleneck.

Neither paper addresses both problems simultaneously. We integrate BVME's variational compression into ExpoComm's architecture to create **ExpoComm-B**, which jointly optimizes communication topology and bandwidth usage.

---

## 2. Progress Summary

### Implementation (Complete)

We built the ExpoComm-B architecture from scratch on top of the open-source ExpoComm codebase (Apache 2.0 license). Since BVME has no public code, we implemented it directly from the paper. The key components we created are:

- **BVME Module** (`bvme.py`): Variational encoder that maps messages to Gaussian distributions (μ, σ), samples compressed representations using the reparameterization trick, and computes KL divergence loss against a tunable prior.
- **ExpoComm-B Agent** (`ExpoComm_bvme_agent.py`): Extends the ExpoComm one-peer agent by inserting the BVME bottleneck after message aggregation and before the Q-network.
- **ExpoComm-B Learner** (`bvme_q_learner.py`): Custom training loop that optimizes a combined objective: TD loss + auxiliary state-prediction loss + KL regularization loss.
- **Unit Tests** (`test_bvme.py`): Verified BVME module correctness — encoding dimensions, KL computation, and deterministic evaluation mode.

### Infrastructure (Complete)

- **Compute**: Deployed on the Rice NOTS HPC cluster using SLURM batch scheduling with GPU acceleration.
- **Experiment Tracking**: Integrated Weights & Biases (W&B) for real-time remote monitoring of training metrics.
- **Repository**: Code is hosted on GitHub with full documentation, including a knowledge transfer document for the team.

### Training Results (In Progress)

We have completed our first full training run using the **MAgent Battle** environment (a large-scale grid-world with ~81 cooperative agents fighting a pretrained opponent team).

**ExpoComm-B (σ₀ = 0.01) — completed 3.35M timesteps over 24 hours:**

| Metric | Start | Final |
|--------|-------|-------|
| Test Win Rate | 0% | **100%** |
| Test Return (mean) | -5.5 | **-0.85** |
| Enemy Survivors (avg) | 81 | **~50** |
| Auxiliary Loss | — | 1.15 (decreasing) |
| KL Loss | — | 31.08 (stable) |

The agents learned to coordinate and consistently defeat the pretrained opponent while communicating through the BVME bottleneck, demonstrating that the compressed messages retain sufficient information for effective coordination.

---

## 3. Current Status

As of today, we have **5 additional experiments running in parallel** on the NOTS GPU cluster:

| Experiment | What It Tests |
|-----------|--------------|
| QMIX Baseline | No communication — each agent acts independently |
| ExpoComm (original) | Sparse topology, uncompressed messages |
| ExpoComm-B (σ₀ = 0.005) | High compression (tight bandwidth) |
| ExpoComm-B (σ₀ = 0.02) | Moderate compression |
| ExpoComm-B (σ₀ = 0.05) | Low compression (relaxed bandwidth) |

These runs will produce the two core results for our paper:

1. **4-method comparison table**: QMIX vs ExpoComm vs ExpoComm-B — does adding compression to a sparse topology help or hurt?
2. **Bandwidth sensitivity analysis**: How does ExpoComm-B's performance change as we vary the compression strength (σ₀)?

---

## 4. Challenges Encountered

| Challenge | Resolution |
|-----------|-----------|
| NOTS has a 10 GB home directory quota, too small for StarCraft II (3.5 GB) | Pivoted to MAgent/MPE benchmarks for initial experiments; will use AWS credits for SMACv2 if needed |
| ExpoComm codebase had hardcoded dependencies on unavailable packages | Patched imports and configuration files to work with MAgent |
| BVME paper has no public code | Implemented the full module from scratch based on the paper's equations |
| Training jobs exceeded 24-hour SLURM time limit | Saved model checkpoints periodically; can resume from latest checkpoint if needed |
| NOTS prohibits job I/O on home directory (NFS) | SLURM scripts copy code to `$SHARED_SCRATCH` before execution |

---

## 5. Timeline & Next Steps

| Week | Tasks | Owner |
|------|-------|-------|
| **This week** (Apr 4–10) | Collect results from 5 running experiments; build comparison table | Aneesh |
| **Next week** (Apr 11–17) | Set up MPE environment; run 4-method comparison on MPE | Ansh |
| **Week 3** (Apr 18–24) | Scalability experiments (N = 5, 10, 20 agents); generate figures | Madhu |
| **Week 4** (Apr 25–27) | Draft and finalize report; prepare presentation | All |

---

## 6. Repository & Resources

- **GitHub**: [github.com/vt20-crypto/ExpoComm-B](https://github.com/vt20-crypto/ExpoComm-B)
- **W&B Dashboard**: [wandb.ai/vt20-rice-university/ExpoComm-B](https://wandb.ai/vt20-rice-university/ExpoComm-B)
- **ExpoComm Paper**: [arxiv.org/abs/2502.19717](https://arxiv.org/abs/2502.19717)
- **BVME Paper**: [arxiv.org/abs/2512.11179](https://arxiv.org/abs/2512.11179)
