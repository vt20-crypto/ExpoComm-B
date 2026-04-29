# ExpoComm-B: Complete Experiment Results

**Last Updated**: April 29, 2026  

---

## 1. MAgent Battle Results (81 blue vs 81 pretrained red agents)

**Environment**: MAgent Battle, 45×45 grid, max 200 steps, seed=123  
**Compute**: Rice NOTS GPU cluster (commons partition, 24h limit)  
**Target**: 5,050,000 timesteps (none fully completed due to time limit)

### Baseline Comparison

| Method | Config | Steps | % Done | Test Win Rate | Test Enemy Survivors ↓ | Test Return ↑ |
|--------|--------|:---:|:---:|:---:|:---:|:---:|
| QMIX (no comm) | `QMIX_baseline` | 4.95M | 98% | 100% | 72.4 / 81 | -0.998 |
| ExpoComm (sparse, no compress) | `ExpoComm_qmix` | 4.35M | 86% | 100% | 60.6 / 81 | -0.894 |
| ExpoComm-B (σ₀=0.01) | `ExpoComm_B_qmix` | 3.35M | 66% | 100% | 50.6 / 81 | -0.853 |

### σ₀ Ablation (MAgent)

| σ₀ | Steps | Test Win Rate | Test Enemy Survivors ↓ | Test Return ↑ | KL Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.005 (tight) | 3.85M (76%) | 100% | 70.4 | -0.980 | 131.5 |
| 0.01 | 3.35M (66%) | 100% | 50.6 | -0.853 | 31.1 |
| 0.02 | 4.05M (80%) | 100% | 64.5 | -0.883 | 6.5 |
| 0.05 (loose) | 4.46M (88%) | 100% | 3.6 | -0.407 | 0.35 |

**Caveats**: Single seed (123). No run completed full 5.05M steps. Different runs reached different step counts.

---

## 2. MPE Simple Spread Results (3 agents, cooperative navigation)

**Environment**: MPE simple_spread_v3, 3 agents, 25 steps/episode, seed=0  
**Compute**: NOTS CPU (ad258 account) + Ansh's local Mac  
**Target**: 500,000 timesteps — **all runs completed (500K/500K)**

### 4-Method Baseline Comparison

> **⚠️ DATA GAP**: The MPE baseline results for QMIX, ExpoComm, and BVME-only were NOT committed to the repo as log files. Ansh marked them as complete in `STATUS_UPDATE.md` but the raw output logs are missing. We only have the ExpoComm-B baseline (λ=1.0) result from the ablation log. Ansh needs to provide these numbers or re-run.

| Method | Config | Final test_return_mean | Source |
|--------|--------|:---:|---|
| QMIX (full-comm, no topology) | `qmix_fullcomm_mpe` | **Missing** | Log not in repo |
| ExpoComm (sparse, no compress) | `ExpoComm_mpe` | **Missing** | Log not in repo |
| BVME-only (full graph + compress) | `bvme_only_mpe` | **Missing** | Log not in repo |
| ExpoComm-B (ours) | `ExpoComm_B_mpe` | **-8.160** | `ablation_nots.log` |

### KL Weight Ablation (λ sweep) — ALL COMPLETE ✅

All runs completed 500K steps. σ₀ = 0.01, compressed_dim = 64 (no dimensional reduction), seed = 0.

| λ (KL weight) | Final test_return_mean ↑ | KL Loss | Source |
|:---:|:---:|:---:|---|
| 0.01 (weak) | **-8.507** | 40.27 | `ablation_output.log` (Ansh's Mac) |
| 0.1 | **-8.163** | 31.09 | `ablation_nots.log` |
| 1.0 (default) | **-8.160** | 31.09 | `ablation_nots.log` |
| 5.0 | **-8.555** | 31.09 | `ablation_nots.log` |
| 10.0 | **-8.138** | 31.09 | `ablation_nots.log` |

### Compression Ratio Ablation (dim sweep) — ALL COMPLETE ✅

All runs completed 500K steps. λ = 1.0, σ₀ = 0.01, seed = 0.

| Compression Ratio | compressed_dim | Final test_return_mean ↑ | Source |
|:---:|:---:|:---:|---|
| 1.0 (no reduction) | 64 | **-8.160** | `ablation_nots.log` |
| 0.5 | 32 | **-8.205** | `ablation_nots.log` |
| 0.25 | 16 | **-8.163** | `ablation_nots.log` |
| 0.125 | 8 | **-8.219** | `ablation_nots.log` |

---

## 3. Observations (from data only)

### MAgent Battle
- All methods achieve 100% win rate — not a differentiating metric.
- Key metric is enemy survivors. QMIX: 72.4, ExpoComm: 60.6, ExpoComm-B σ₀=0.01: 50.6.
- σ₀=0.05 has dramatically fewer survivors (3.6) and shorter episodes (145 steps).
- σ₀=0.005 performs close to the QMIX baseline (70.4 vs 72.4).

### MPE KL Ablation
- λ = 10.0 achieved the best test_return_mean (-8.138), slightly better than λ = 1.0 (-8.160).
- λ = 0.01 performed worst (-8.507).
- λ = 5.0 also performed poorly (-8.555).
- The range across all λ values is narrow: -8.138 to -8.555 (Δ = 0.417).

### MPE Compression Ratio Ablation
- All compression ratios achieved similar performance: -8.160 to -8.219.
- The range is extremely narrow (Δ = 0.059), suggesting that for 3-agent MPE, dimensional compression has minimal impact.
- Even 87.5% compression (64→8 dims) only reduced performance by 0.059.

---

## 4. Data Gaps — Action Required

| Gap | Owner | Action |
|-----|-------|--------|
| MPE 4-method baseline results (QMIX, ExpoComm, BVME-only) | Ansh | Provide log files or final test_return_mean numbers |
| Multi-seed runs | All | Not done — single seed only |
| Scalability experiments | Madhu | Not started |

---

## 5. Saved Checkpoints

### MAgent (on NOTS ~/ExpoComm/saved_models/)
| Experiment | Checkpoint |
|-----------|:---:|
| QMIX Baseline | `QMIX_baseline/4950662/` |
| ExpoComm | `ExpoComm_qmix/4353243/` |
| ExpoComm-B σ₀=0.01 | `MAgent_Battle_GPU_Run/3354276/` |
| ExpoComm-B σ₀=0.005 | `ExpoComm_B_s005/3804690/` |
| ExpoComm-B σ₀=0.02 | `ExpoComm_B_s02/4053243/` |
| ExpoComm-B σ₀=0.05 | `ExpoComm_B_s05/4456919/` |

### MPE (on NOTS /home/ad258/ExpoComm-B/work_dirs/)
All MPE checkpoints are on Ansh's NOTS account.
