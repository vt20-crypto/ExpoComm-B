# ExpoComm-B: Project Status Update

> **Project**: Adapting Sparse Communication Graph Learning for Multi-Agent Coordination in Bandwidth-Constrained Settings  
> **Course**: COMP 559 – Machine Learning with Graphs (Prof. Arlei Silva, Rice University, Spring 2026)  
> **Team**: Aneesh Thatiparti (vt20), Ansh Dabral (ad258), Madhu Thatiparti (mt180)  
> **Date**: March 31, 2026

---

## 1. Executive Summary

Till now, we successfully built the core **ExpoComm-B** architecture from scratch, deployed it on the Rice NOTS HPC cluster, and launched our first training run on the MAgent Battle environment. The BVME variational bottleneck module was implemented entirely from the paper (no public code exists), integrated into the ExpoComm codebase, and verified end-to-end. Training metrics are being tracked live on Weights & Biases.

---

## 2. What Has Been Completed

### Environment & Infrastructure Setup ✅

| Task | Status | Notes |
|------|--------|-------|
| Clone ExpoComm repo & explore architecture | ✅ Done | Apache 2.0 licensed, built on EPyMARL framework |
| Set up `expocomm-b` conda environment on NOTS | ✅ Done | Python 3.8, PyTorch 1.13.1, einops, torch-scatter |
| Configure Weights & Biases (W&B) for remote monitoring | ✅ Done | Live dashboard at `wandb.ai/vt20-rice-university/ExpoComm-B` |
| Set up MAgent Battle environment | ✅ Done | PettingZoo 1.14.0 + SuperSuit 3.3.0, custom map files injected |
| Create SLURM batch scripts for NOTS GPU nodes | ✅ Done | `run_gpu.slurm` — uses `$SHARED_SCRATCH` for I/O compliance |
| Push codebase to team GitHub repo | ✅ Done | `github.com/vt20-crypto/ExpoComm-B` with proper attribution |

### Core BVME Implementation (From Paper) ✅

| File Created | What It Does |
|-------------|-------------|
| `src/modules/bvme.py` | The standalone BVME module — variational encoder (μ, σ), reparameterization trick, KL divergence loss computation |
| `src/modules/agents/ExpoComm_bvme_agent.py` | ExpoComm-B agent — plugs BVME bottleneck after `_communicate()`, before Q-network |
| `src/controllers/ExpoComm_bvme_controller.py` | Multi-agent controller that accumulates KL loss across all agents and handles test-mode (deterministic z = μ) |
| `src/learners/bvme_q_learner.py` | Custom learner that combines TD loss + Auxiliary loss + KL loss into a single training objective |
| `src/config/algs/ExpoComm_B_qmix.yaml` | Algorithm config registering all ExpoComm-B components |
| `tests/test_bvme.py` | Unit tests verifying BVME module correctness (encoding, KL computation, deterministic eval mode) |

### Documentation ✅

| Document | Purpose |
|---------|---------|
| `ExpoComm-B-Knowledge-Transfer.md` | Comprehensive knowledge transfer doc for the team — covers background, architecture, BVME math, codebase walkthrough, team roles |
| `README.md` | Rewritten for our project with proper attribution and citation of original ExpoComm authors |

### First Training Run ✅

- **Environment**: MAgent Battle (45×45 grid, ~81 blue agents vs pretrained red team)
- **Algorithm**: ExpoComm-B with QMIX mixer
- **Duration**: Ran for ~23 hours on NOTS CPU node (`commons` partition)
- **Timesteps reached**: 451,233 / 5,050,000 (~9%)
- **Key metrics at termination**:
  - `test_return_mean`: improved from **-5.5** → **-0.76** (agents learning to survive)
  - `test_blue_team_win_mean`: **0.80** (80% win rate in test episodes!)
  - `test_red_team_alives_mean`: **39** (killing ~42 of 81 enemy agents on average)
- **Model checkpoint saved**: `work_dirs/.../models/451233`

---

## 3. Issues & Challenges Faced

### Resolved Issues

| Issue | Root Cause | Fix Applied |
|-------|-----------|-------------|
| `ModuleNotFoundError: imp_marl` | ExpoComm hardcodes IMP-MARL imports (3.5GB package we don't use) | Commented out imports in `src/envs/__init__.py` |
| `ImportError: Renderer` | PettingZoo version mismatch; custom map files reference a removed rendering module | Surgically removed dead `Renderer` imports from `battle_v3_view7.py` and `adversarial_pursuit_view8_v3.py` |
| `AttributeError: 'NoneType' has no attribute 'items'` | `--exp-config` flag is optional but code crashes when it's missing | Added `None` guard in `src/main.py` before `recursive_dict_update()` |
| `KeyError: 'seed'` | Missing required parameters in `default.yaml` | Added `seed`, `exp_name`, `run_name` to config |
| `TypeError: join() argument must be str, not NoneType` | `pretrained_ckpt` was set to `null` in YAML config | Set to `"battle.pt"` (the file already existed in the repo) |
| `TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'` | `bvme_compressed_dim: null` in YAML passes `None` to Python, but `getattr()` doesn't fall back to default when attribute exists as `None` | Changed to `getattr(args, "bvme_compressed_dim", None) or args.hidden_dim` |
| OOM Kill on first SLURM job | Requested only 16GB RAM; MAgent replay buffer exceeds that | Increased to `--mem=64G` |
| SLURM job killed at 24h | `commons` partition has a strict 24-hour time limit | Noted for future runs; will use `long` partition (72h) or GPU acceleration |

### Open Issue: Training Speed on CPU

- At 451k/5M timesteps in 23 hours, the estimated total time is **~8 days on CPU**.
- **Solution**: We have created a new SLURM script (`run_gpu.slurm`) that explicitly requests an NVIDIA GPU via `--gres=gpu:1`. This should reduce training time from days to hours. **This job is currently running (Job ID: 7966302).**

### Open Issue: NOTS I/O Compliance

- Rice's documentation strictly prohibits using `$HOME` (NFS) for job I/O.
- **Solution**: The new `run_gpu.slurm` script copies the entire codebase to `$SHARED_SCRATCH` before training, fully complying with NOTS rules.

---

## 4. Current Status (as of March 31, 2026)

| Item | Status |
|------|--------|
| ExpoComm-B codebase | ✅ Fully implemented and tested |
| MAgent Battle training | 🔄 GPU job currently running on NOTS (Job 7966302) |
| W&B Dashboard | ✅ Live at `wandb.ai/vt20-rice-university/ExpoComm-B` |
| GitHub Repository | ✅ Pushed to `vt20-crypto/ExpoComm-B` |
| MPE Environment | ⬜ Not yet started |
| SMACv2 Environment | ⬜ Deferred to AWS (Need some credits to do this) |
| Baseline comparisons | ⬜ Pending MPE wrapper |
| Final report | ⬜ Not yet started |

---

## 5. What's Pending & Team Assignments

### Aneesh (vt20) — Integration Lead

| Task | Priority | When |
|------|----------|------|
| Monitor current GPU training run and collect final MAgent results | High | This week |
| Run bandwidth ablation experiments (vary compression ratio `r`) | High | After MAgent baseline completes |
| Set up AWS EC2 instance for SMACv2 experiments (if needed) | Medium | Week 2 |
| Run ExpoComm-B on SMACv2 benchmarks | Medium | After AWS setup |

### Ansh (ad258) — Baselines & MPE

| Task | Priority | When |
|------|----------|------|
| **Read the Knowledge Transfer document** (`ExpoComm-B-Knowledge-Transfer.md`) | 🔴 Critical | - |
| **Clone the repo** from `github.com/vt20-crypto/ExpoComm-B` | 🔴 Critical | - |
| Set up MPE (Multi-Agent Particle Environment) wrapper for EPyMARL | High | This week |
| Run the 4-method baseline comparison on MPE (Full-comm, ExpoComm, BVME-only, ExpoComm-B) | High | Week 1–2 |
| Conduct ablation studies (KL weight λ, compression ratio r) | Medium | Week 2 |

### Madhu (mt180) — Analysis & Report

| Task | Priority | When |
|------|----------|------|
| **Read the Knowledge Transfer document** (`ExpoComm-B-Knowledge-Transfer.md`) | 🔴 Critical | - |
| Design scalability experiments (N = 5, 10, 20 agents) | Medium | Week 2 |
| Run zero-shot generalization tests (train N=5, evaluate N=10) | Medium | Week 2 |
| Generate publication-quality figures and tables from W&B data | High | Week 2–3 |
| Draft the final report (Introduction, Related Work, Method, Experiments) | High | Week 3 |
| Prepare presentation slides | Medium | Week 3 |

---

## 6. Key Resources

| Resource | Location |
|---------|---------|
| GitHub Repository | `github.com/vt20-crypto/ExpoComm-B` |
| Knowledge Transfer Doc | `ExpoComm-B-Knowledge-Transfer.md` (in repo root) |
| W&B Dashboard | `wandb.ai/vt20-rice-university/ExpoComm-B` |
| NOTS Cluster Login | `ssh YOUR_NETID@nots.rice.edu` |
| ExpoComm Paper | `arxiv.org/abs/2502.19717` |
| BVME Paper | `arxiv.org/abs/2512.11179` |
| SLURM GPU Script | `run_gpu.slurm` (in repo root) |

---

*This status update covers work completed till March 31, 2026.*
