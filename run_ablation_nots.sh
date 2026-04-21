#!/bin/bash
# =============================================================================
# ExpoComm-B: Ablation Studies on NOTS (Rice HPC)
# =============================================================================
# Runs the remaining ablation experiments (skips kl_001 which completed locally).
#
# Usage on NOTS:
#   # 1. Clone/pull repo
#   git clone https://github.com/vt20-crypto/ExpoComm-B.git  # or git pull
#   cd ExpoComm-B
#
#   # 2. Set up conda env (one-time)
#   conda create -n expocomm-b python=3.10 -y
#   conda activate expocomm-b
#   pip install torch torchvision 'pettingzoo[mpe]' pygame 'numpy<2' pyyaml wandb setproctitle einops
#
#   # 3. Run
#   WANDB_MODE=offline nohup bash run_ablation_nots.sh > ablation_nots.log 2>&1 &
#   disown
#   # Then you can close your SSH session safely
# =============================================================================

set -e
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════"
echo "  ExpoComm-B: Ablation Study (NOTS Resume)"
echo "  Skipping: ablation_kl_001 (completed locally)"
echo "  Running: 8 remaining experiments"
echo "═══════════════════════════════════════════════"

run_method() {
    local label=$1
    local config=$2
    local exp_config=$3

    echo ""
    echo ">>> Running: ${label}"
    echo "    Config: --config=${config} --env-config=MPE_Spread --exp-config=${exp_config}"
    echo "    Started at: $(date)"

    python src/main.py \
        --config="${config}" \
        --env-config=MPE_Spread \
        --exp-config="${exp_config}"

    echo ">>> Finished: ${label} at $(date)"
    echo ""
}

echo ""
echo "=== KL Weight Sweep (skipping kl_001) ==="

run_method "λ = 0.1"              "ablation_kl_01"   "MPE_ablation_kl_01_s0"
run_method "λ = 1.0 (baseline)"   "ExpoComm_B_mpe"   "MPE_Spread_ExpoComm_B_s0"
run_method "λ = 5.0"              "ablation_kl_5"    "MPE_ablation_kl_5_s0"
run_method "λ = 10.0 (strong)"    "ablation_kl_10"   "MPE_ablation_kl_10_s0"

echo ""
echo "=== Compression Ratio Sweep ==="

run_method "r = 1.0 (baseline)"   "ExpoComm_B_mpe"   "MPE_Spread_ExpoComm_B_s0"
run_method "r = 0.5  (dim 64→32)" "ablation_cr_050"  "MPE_ablation_cr_050_s0"
run_method "r = 0.25 (dim 64→16)" "ablation_cr_025"  "MPE_ablation_cr_025_s0"
run_method "r = 0.125 (dim 64→8)" "ablation_cr_0125" "MPE_ablation_cr_0125_s0"

echo ""
echo "═══════════════════════════════════════════════"
echo "  All ablation studies completed!"
echo "  Finished at: $(date)"
echo "═══════════════════════════════════════════════"
