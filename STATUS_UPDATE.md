# ExpoComm-B: Project Status Update

> **Project**: Adapting Sparse Communication Graph Learning for Multi-Agent Coordination in Bandwidth-Constrained Settings  
> **Course**: COMP 559 – Machine Learning with Graphs (Prof. Arlei Silva, Rice University, Spring 2026)  
> **Team**: Aneesh Thatiparti (vt20), Ansh Dabral (ad258), Madhu Thatiparti (mt180)  
> **Date**: April 7, 2026

---

## 1. Executive Summary

The ExpoComm-B architecture has been fully implemented and tested on the MAgent Battle environment. We ran 6 experiments on the Rice NOTS GPU cluster: 2 baselines (QMIX, ExpoComm) and 4 ExpoComm-B variants with different compression levels. All model checkpoints are saved and metrics are tracked on Weights & Biases.

---

## 2. MAgent Battle Results (All Experiments Complete)

### Baseline Comparison

| Method | Steps | Test Win Rate | Test Enemy Survivors | Test Return |
|--------|:---:|:---:|:---:|:---:|
| QMIX (no communication) | 4.95M | 100% | 72.4 / 81 | -0.998 |
| ExpoComm (sparse topology, no compression) | 4.35M | 100% | 60.6 / 81 | -0.894 |
| ExpoComm-B (σ₀=0.01) | 3.35M | 100% | 50.6 / 81 | -0.853 |

### Bandwidth Ablation (σ₀ sweep)

| σ₀ | Steps | Test Win Rate | Test Enemy Survivors | Test Return | KL Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.005 (high compression) | 3.85M | 100% | 70.4 / 81 | -0.980 | 131.5 |
| 0.01 | 3.35M | 100% | 50.6 / 81 | -0.853 | 31.1 |
| 0.02 | 4.05M | 100% | 64.5 / 81 | -0.883 | 6.5 |
| 0.05 (low compression) | 4.46M | 100% | 3.6 / 81 | -0.407 | 0.35 |

### Caveats
- No run completed the full 5.05M steps (24h SLURM limit). QMIX got closest at 98%.
- All runs used a single seed (123). Multi-seed runs are needed for statistical significance.
- Results are on MAgent Battle only. MPE experiments are pending.

---

## 3. Aneesh's Work — Status: COMPLETE ✅

| Deliverable | Status |
|------------|--------|
| BVME module implementation (from paper, no public code) | ✅ Done |
| ExpoComm-B agent, controller, learner | ✅ Done |
| Unit tests | ✅ Done |
| MAgent Battle — 2 baselines + 4 ablation runs | ✅ Done |
| All model checkpoints saved to NOTS | ✅ Done |
| Experiment results document (`EXPERIMENT_RESULTS.md`) | ✅ Done |
| Knowledge transfer document | ✅ Done |
| GitHub repo with proper attribution | ✅ Done |
| W&B dashboard live | ✅ Done |
| SLURM scripts (parameterized, NOTS-compliant) | ✅ Done |

---

## 4. Pending Work — Team Assignments

### Ansh (ad258) — MPE Experiments

| Task | Priority | Details |
|------|----------|---------|
| Read Knowledge Transfer doc | 🔴 Critical | `ExpoComm-B-Knowledge-Transfer.md` in repo |
| Clone repo | 🔴 Critical | `github.com/vt20-crypto/ExpoComm-B` |
| Build MPE wrapper for EPyMARL | High | Create `src/envs/mpe_wrappers.py` + `src/config/envs/MPE_Tag.yaml` |
| Run 4-method comparison on MPE | High | Use existing configs: `QMIX_baseline`, `ExpoComm_qmix`, `ExpoComm_B_qmix` |
| Run bandwidth ablation on MPE | Medium | Use `ExpoComm_B_qmix_s005/s02/s05` configs |
| Use `run_experiment.slurm` for submissions | — | `sbatch run_experiment.slurm <config> <run_name>` |

### Madhu (mt180) — Analysis & Report

| Task | Priority | Details |
|------|----------|---------|
| Read Knowledge Transfer doc | 🔴 Critical | `ExpoComm-B-Knowledge-Transfer.md` in repo |
| Download W&B data | High | Export CSVs from `wandb.ai/vt20-rice-university/ExpoComm-B` |
| Generate figures | High | Learning curves (return vs timestep), bar charts (enemy survivors), σ₀ sensitivity plot |
| Scalability experiments (if time) | Medium | Modify `map_size` in `MAgent_Battle.yaml` for different agent counts |
| Draft report | High | Introduction, Related Work, Method (Sec 4 of KT doc), Experiments, Conclusion |

---

## 5. Key Resources

| Resource | Location |
|---------|---------|
| GitHub Repo | `github.com/vt20-crypto/ExpoComm-B` |
| W&B Dashboard | `wandb.ai/vt20-rice-university/ExpoComm-B` |
| Raw Results | `EXPERIMENT_RESULTS.md` in repo |
| Knowledge Transfer | `ExpoComm-B-Knowledge-Transfer.md` in repo |
| SLURM Script | `run_experiment.slurm` in repo |
| Saved Models (NOTS) | `~/ExpoComm/saved_models/` |

---

*Last updated: April 7, 2026*
