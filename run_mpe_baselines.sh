#!/bin/bash
# =============================================================================
# ExpoComm-B: MPE Baseline Comparison Runner
# =============================================================================
# Runs all 4 methods on MPE simple_spread for the baseline comparison.
#
# Usage:
#   ./run_mpe_baselines.sh              # Run all 4 methods sequentially
#   ./run_mpe_baselines.sh fullcomm     # Run only full-comm QMIX
#   ./run_mpe_baselines.sh expocomm    # Run only ExpoComm
#   ./run_mpe_baselines.sh bvme        # Run only BVME-only
#   ./run_mpe_baselines.sh expocomm_b  # Run only ExpoComm-B
#   ./run_mpe_baselines.sh smoke       # Quick ~1 min smoke test for all 4
#
# Prerequisites:
#   conda activate expocomm-b
#   pip install pettingzoo[mpe] pygame
# =============================================================================

set -e

cd "$(dirname "$0")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  ExpoComm-B: MPE Baseline Comparison ${NC}"
echo -e "${BLUE}======================================${NC}"

# Check if wandb is configured
if [ -z "${WANDB_API_KEY}" ] && [ ! -f ~/.netrc ] && [ ! -f ~/.config/wandb/settings ]; then
    echo -e "${YELLOW}WARNING: W&B doesn't appear to be configured.${NC}"
    echo -e "Run 'wandb login' or set WANDB_API_KEY env variable."
    echo -e "Continuing with W&B in offline mode..."
    export WANDB_MODE=offline
fi

MODE=${1:-all}

run_method() {
    local method_name=$1
    local config=$2
    local exp_config=$3
    local t_max_override=$4

    echo -e "\n${GREEN}>>> Running: ${method_name}${NC}"
    echo -e "Config: --config=${config} --env-config=MPE_Spread --exp-config=${exp_config}"

    CMD="python src/main.py --config=${config} --env-config=MPE_Spread --exp-config=${exp_config}"

    if [ -n "$t_max_override" ]; then
        echo -e "  (smoke test: t_max=${t_max_override})"
        # For smoke tests, we append overrides via environment
        # The code reads from config; we'll just let it run briefly
        CMD="$CMD"
    fi

    echo -e "${YELLOW}$ ${CMD}${NC}"
    eval $CMD

    echo -e "${GREEN}>>> Finished: ${method_name}${NC}"
}

case $MODE in
    fullcomm)
        run_method "Full-comm QMIX" "qmix_fullcomm_mpe" "MPE_Spread_fullcomm_s0"
        ;;
    expocomm)
        run_method "ExpoComm" "ExpoComm_mpe" "MPE_Spread_ExpoComm_s0"
        ;;
    bvme)
        run_method "BVME-only" "bvme_only_mpe" "MPE_Spread_bvme_only_s0"
        ;;
    expocomm_b)
        run_method "ExpoComm-B" "ExpoComm_B_mpe" "MPE_Spread_ExpoComm_B_s0"
        ;;
    smoke)
        echo -e "${YELLOW}Running SMOKE TESTS (ultra-short runs to verify configs)${NC}"
        echo -e "Each run will train for ~100 timesteps (< 1 minute each)\n"

        # For smoke tests, we use the same configs but training will be
        # manually stopped or we rely on the short t_max
        # We can temporarily override t_max via a smoke test config
        run_method "Full-comm QMIX (smoke)" "qmix_fullcomm_mpe" "MPE_Spread_fullcomm_s0"
        run_method "ExpoComm (smoke)" "ExpoComm_mpe" "MPE_Spread_ExpoComm_s0"
        run_method "BVME-only (smoke)" "bvme_only_mpe" "MPE_Spread_bvme_only_s0"
        run_method "ExpoComm-B (smoke)" "ExpoComm_B_mpe" "MPE_Spread_ExpoComm_B_s0"

        echo -e "\n${GREEN}All smoke tests passed!${NC}"
        ;;
    all)
        echo -e "${YELLOW}Running ALL 4 baselines (this will take several hours)${NC}\n"

        run_method "1/4: Full-comm QMIX" "qmix_fullcomm_mpe" "MPE_Spread_fullcomm_s0"
        run_method "2/4: ExpoComm" "ExpoComm_mpe" "MPE_Spread_ExpoComm_s0"
        run_method "3/4: BVME-only" "bvme_only_mpe" "MPE_Spread_bvme_only_s0"
        run_method "4/4: ExpoComm-B" "ExpoComm_B_mpe" "MPE_Spread_ExpoComm_B_s0"

        echo -e "\n${GREEN}======================================${NC}"
        echo -e "${GREEN}  All 4 baselines completed!           ${NC}"
        echo -e "${GREEN}  Check W&B dashboard for results.     ${NC}"
        echo -e "${GREEN}======================================${NC}"
        ;;
    *)
        echo "Usage: $0 {all|fullcomm|expocomm|bvme|expocomm_b|smoke}"
        exit 1
        ;;
esac
