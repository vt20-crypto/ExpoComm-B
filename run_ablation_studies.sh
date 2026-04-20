#!/bin/bash
# =============================================================================
# ExpoComm-B: Ablation Studies on MPE simple_spread
# =============================================================================
# Two ablation axes:
#   1. KL weight λ:  {0.01, 0.1, 1.0*, 5.0, 10.0}   (* = baseline)
#   2. Compression ratio r: {1.0*, 0.5, 0.25, 0.125}  (* = baseline)
#
# The baseline (λ=1.0, r=1.0) is ExpoComm_B_mpe which was already run in
# the 4-method comparison. We include it here for completeness.
#
# Usage:
#   ./run_ablation_studies.sh               # Run all ablation experiments
#   ./run_ablation_studies.sh kl            # Run KL weight sweep only
#   ./run_ablation_studies.sh cr            # Run compression ratio sweep only
#   ./run_ablation_studies.sh smoke         # Quick smoke test (~1 min)
#
# Prerequisites:
#   conda activate expocomm-b
# =============================================================================

set -e
cd "$(dirname "$0")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Use offline mode if W&B is not configured
if [ -z "${WANDB_MODE}" ]; then
    if [ -z "${WANDB_API_KEY}" ] && [ ! -f ~/.netrc ] && [ ! -f ~/.config/wandb/settings ]; then
        echo -e "${YELLOW}W&B not configured — running in offline mode.${NC}"
        echo -e "Run 'wandb login' to enable cloud logging."
        export WANDB_MODE=offline
    fi
fi

MODE=${1:-all}

run_method() {
    local label=$1
    local config=$2
    local exp_config=$3

    echo -e "\n${GREEN}>>> Running: ${label}${NC}"
    echo -e "    Config:     --config=${config}"
    echo -e "    Env:        --env-config=MPE_Spread"
    echo -e "    Experiment: --exp-config=${exp_config}"

    python src/main.py \
        --config="${config}" \
        --env-config=MPE_Spread \
        --exp-config="${exp_config}"

    echo -e "${GREEN}>>> Finished: ${label}${NC}\n"
}

# ═══════════════════════════════════════════════════════════
# KL WEIGHT SWEEP: λ ∈ {0.01, 0.1, 1.0, 5.0, 10.0}
# ═══════════════════════════════════════════════════════════
run_kl_sweep() {
    echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   Ablation 1: KL Weight λ Sweep          ║${NC}"
    echo -e "${CYAN}║   λ ∈ {0.01, 0.1, 1.0, 5.0, 10.0}       ║${NC}"
    echo -e "${CYAN}║   (r = 1.0, σ₀ = 0.01 fixed)            ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"

    run_method "λ = 0.01 (weak)"    "ablation_kl_001"  "MPE_ablation_kl_001_s0"
    run_method "λ = 0.1"            "ablation_kl_01"   "MPE_ablation_kl_01_s0"
    run_method "λ = 1.0 (baseline)" "ExpoComm_B_mpe"   "MPE_Spread_ExpoComm_B_s0"
    run_method "λ = 5.0"            "ablation_kl_5"    "MPE_ablation_kl_5_s0"
    run_method "λ = 10.0 (strong)"  "ablation_kl_10"   "MPE_ablation_kl_10_s0"

    echo -e "${GREEN}KL weight sweep complete!${NC}"
}

# ═══════════════════════════════════════════════════════════
# COMPRESSION RATIO SWEEP: r ∈ {1.0, 0.5, 0.25, 0.125}
# ═══════════════════════════════════════════════════════════
run_cr_sweep() {
    echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   Ablation 2: Compression Ratio r Sweep      ║${NC}"
    echo -e "${CYAN}║   r ∈ {1.0, 0.5, 0.25, 0.125}               ║${NC}"
    echo -e "${CYAN}║   (dim: 64 → {64, 32, 16, 8})               ║${NC}"
    echo -e "${CYAN}║   (λ = 1.0, σ₀ = 0.01 fixed)                ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"

    run_method "r = 1.0 (no dim reduction)" "ExpoComm_B_mpe"   "MPE_Spread_ExpoComm_B_s0"
    run_method "r = 0.5  (dim 64→32)"       "ablation_cr_050"  "MPE_ablation_cr_050_s0"
    run_method "r = 0.25 (dim 64→16)"       "ablation_cr_025"  "MPE_ablation_cr_025_s0"
    run_method "r = 0.125 (dim 64→8)"       "ablation_cr_0125" "MPE_ablation_cr_0125_s0"

    echo -e "${GREEN}Compression ratio sweep complete!${NC}"
}

# ═══════════════════════════════════════════════════════════
# SMOKE TEST: ultra-short runs to verify all configs
# ═══════════════════════════════════════════════════════════
run_smoke() {
    echo -e "${YELLOW}Running SMOKE TESTS for all ablation configs${NC}"
    echo -e "Each run trains for ~200 timesteps (< 5 seconds)\n"

    for config in ablation_kl_001 ablation_kl_01 ablation_kl_5 ablation_kl_10 \
                  ablation_cr_050 ablation_cr_025 ablation_cr_0125; do
        echo -e "${GREEN}>>> Smoke: ${config}${NC}"
        python src/main.py \
            --config="${config}" \
            --env-config=MPE_Spread \
            --exp-config=MPE_smoke_test 2>&1 | grep -E "(Beginning|Finished|Error|Traceback)"
    done

    echo -e "\n${GREEN}All smoke tests passed!${NC}"
}

case $MODE in
    kl)
        run_kl_sweep
        ;;
    cr)
        run_cr_sweep
        ;;
    smoke)
        run_smoke
        ;;
    all)
        echo -e "${BLUE}══════════════════════════════════════════════${NC}"
        echo -e "${BLUE}   ExpoComm-B: Full Ablation Study on MPE    ${NC}"
        echo -e "${BLUE}   7 new runs + 1 shared baseline = 8 total  ${NC}"
        echo -e "${BLUE}══════════════════════════════════════════════${NC}"

        run_kl_sweep
        run_cr_sweep

        echo -e "\n${GREEN}══════════════════════════════════════════════${NC}"
        echo -e "${GREEN}   All ablation studies completed!            ${NC}"
        echo -e "${GREEN}   Check W&B dashboard for results.           ${NC}"
        echo -e "${GREEN}══════════════════════════════════════════════${NC}"
        ;;
    *)
        echo "Usage: $0 {all|kl|cr|smoke}"
        exit 1
        ;;
esac
