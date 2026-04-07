# ExpoComm-B: MAgent Battle Experiment Results

**Environment**: MAgent Battle (45×45 grid, 81 blue agents vs 81 pretrained red agents)  
**Max Episode Steps**: 200  
**Training Target**: 5,050,000 timesteps  
**Date**: April 4–7, 2026

---

## 1. Raw Results

All experiments were run on Rice NOTS GPU nodes (`commons` partition, 24h limit). None completed the full 5.05M steps due to the time limit, but all ran for 76–98% of the target.

### Baseline Comparison

| Method | Config | Steps Reached | % Complete | Test Win Rate | Test Return | Test Enemy Survivors | Test Ep Length |
|--------|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| QMIX (no comm) | `QMIX_baseline` | 4,950,662 | 98% | 100% | -0.998 | 72.4 | 200 |
| ExpoComm (sparse, no compress) | `ExpoComm_qmix` | 4,353,243 | 86% | 100% | -0.894 | 60.6 | 200 |
| ExpoComm-B (σ₀=0.01) | `ExpoComm_B_qmix` | 3,354,276 | 66% | 100% | -0.853 | 50.6 | 200 |

### Bandwidth Ablation (σ₀ sweep)

| σ₀ | Compression Level | Steps Reached | Test Win Rate | Test Return | Test Enemy Survivors | Test Ep Length | KL Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.005 | High | 3,854,690 (76%) | 100% | -0.980 | 70.4 | 200 | 131.5 |
| 0.01 | Medium-High | 3,354,276 (66%) | 100% | -0.853 | 50.6 | 200 | 31.1 |
| 0.02 | Medium | 4,053,243 (80%) | 100% | -0.883 | 64.5 | 200 | 6.5 |
| 0.05 | Low | 4,456,919 (88%) | 100% | -0.407 | 3.6 | 145 | 0.35 |

---

## 2. Observations (From Data Only)

### Baseline Comparison
- All three methods achieve a 100% test win rate. The win rate alone does not differentiate them.
- The differentiating metric is `test_red_team_alives_mean` (how many enemies survive at episode end). Lower = more enemies killed = better coordination.
- **QMIX** (no communication): 72.4 enemies survive out of 81. Agents win but kill very few opponents.
- **ExpoComm** (sparse topology, no compression): 60.6 enemies survive. Adding the exponential communication topology reduces enemy survivors by 11.8 compared to QMIX.
- **ExpoComm-B σ₀=0.01** (sparse topology + BVME): 50.6 enemies survive. Adding BVME compression further reduces enemy survivors by 10.0 compared to plain ExpoComm.

### Bandwidth Ablation
- **σ₀=0.005** (tightest compression): 70.4 enemies survive for ExpoComm-B, which is close to the QMIX baseline (72.4). KL loss is very high (131.5), indicating the bottleneck is heavily restricting information flow. This variant performed worst among all ExpoComm-B settings.
- **σ₀=0.01**: 50.6 enemies — better than both baselines.
- **σ₀=0.02**: 64.5 enemies — worse than σ₀=0.01, better than σ₀=0.005.
- **σ₀=0.05** (loosest compression): 3.6 enemies survive. This is the best result across all experiments by a wide margin. KL loss is only 0.35. Episodes end at step 145 on average instead of 200, meaning the blue team eliminates nearly all enemies before time runs out.

### Additional Training Metrics
- **aux_loss** (state prediction from messages) decreases as σ₀ increases: 1.77 (σ₀=0.005) → 1.44 (0.02) → 1.03 (0.05). This indicates messages carry more useful state information at lower compression levels.
- **grad_norm** is highest for σ₀=0.005 (7.8) and lowest for ExpoComm baseline (0.45). High KL loss from aggressive compression may cause gradient instability.

---

## 3. Caveats

1. **No run completed 5.05M steps.** The QMIX baseline got closest (98%). ExpoComm-B σ₀=0.01 only reached 66%. Results may not reflect fully converged performance.
2. **Single seed.** All experiments used seed=123. Results may vary across seeds. Standard practice requires 3–5 seeds for statistical significance.
3. **One environment only.** All results are on MAgent Battle. Generalization to MPE or SMACv2 has not been tested.
4. **Non-uniform training progress.** Different methods reached different step counts, making direct comparison imperfect. The QMIX baseline had ~30% more training than ExpoComm-B σ₀=0.01.

---

## 4. Saved Checkpoints

All final model checkpoints are saved in `~/ExpoComm/saved_models/` on NOTS:

| Experiment | Checkpoint Step | Location |
|-----------|:---:|----------|
| QMIX Baseline | 4,950,662 | `saved_models/QMIX_baseline/4950662/` |
| ExpoComm | 4,353,243 | `saved_models/ExpoComm_qmix/4353243/` |
| ExpoComm-B σ₀=0.01 | 3,354,276 | `saved_models/MAgent_Battle_GPU_Run/3354276/` |
| ExpoComm-B σ₀=0.005 | 3,804,690 | `saved_models/ExpoComm_B_s005/3804690/` |
| ExpoComm-B σ₀=0.02 | 4,053,243 | `saved_models/ExpoComm_B_s02/4053243/` |
| ExpoComm-B σ₀=0.05 | 4,456,919 | `saved_models/ExpoComm_B_s05/4456919/` |

---

## 5. W&B Dashboard

All runs are logged at: `wandb.ai/vt20-rice-university/ExpoComm-B`
