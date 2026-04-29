#!/bin/bash
# =============================================================================
# Run 3 Missing MPE Baselines (QMIX, ExpoComm, BVME-only)
# Submit to NOTS via nohup so it runs unattended (~4.5h total)
# =============================================================================

set -e
cd ~/ExpoComm

echo "============================================"
echo "  Running 3 Missing MPE Baselines"
echo "  Started: $(date)"
echo "============================================"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate expocomm-b

# Install pettingzoo MPE if not present
pip install "pettingzoo[mpe]" pygame 2>/dev/null || true

echo ""
echo ">>> [1/3] QMIX Full-Comm Baseline"
echo "    Started: $(date)"
python src/main.py \
    --config=qmix_fullcomm_mpe \
    --env-config=MPE_Spread \
    exp_name="MPE_baselines" \
    run_name="MPE_QMIX_fullcomm_s0" \
    seed=0
echo ">>> [1/3] QMIX Done: $(date)"

echo ""
echo ">>> [2/3] ExpoComm (sparse topology, no compression)"
echo "    Started: $(date)"
python src/main.py \
    --config=ExpoComm_mpe \
    --env-config=MPE_Spread \
    exp_name="MPE_baselines" \
    run_name="MPE_ExpoComm_s0" \
    seed=0
echo ">>> [2/3] ExpoComm Done: $(date)"

echo ""
echo ">>> [3/3] BVME-only (full graph + compression)"
echo "    Started: $(date)"
python src/main.py \
    --config=bvme_only_mpe \
    --env-config=MPE_Spread \
    exp_name="MPE_baselines" \
    run_name="MPE_BVME_only_s0" \
    seed=0
echo ">>> [3/3] BVME-only Done: $(date)"

echo ""
echo "============================================"
echo "  All 3 baselines complete!"
echo "  Finished: $(date)"
echo "============================================"
