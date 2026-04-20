#!/bin/bash
#SBATCH --job-name=ExpoComm-Exp
#SBATCH --partition=commons
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ========================================================
# Parameterized ExpoComm-B Experiment Runner
# Usage: sbatch run_experiment.slurm <ALG_CONFIG> <RUN_NAME>
# Example: sbatch run_experiment.slurm ExpoComm_B_qmix "ExpoComm-B-Default"
# ========================================================

ALG_CONFIG=${1:?"ERROR: Provide algorithm config name as 1st argument"}
RUN_NAME=${2:?"ERROR: Provide run name as 2nd argument"}

echo "============================================"
echo "ExpoComm-B Experiment Runner"
echo "Algorithm Config: $ALG_CONFIG"
echo "Run Name: $RUN_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "============================================"

# Initialize Conda
eval "$(conda shell.bash hook)"
conda activate expocomm-b

# NOTS Compliance: Use scratch for all I/O
RUN_DIR=$SHARED_SCRATCH/$USER/ExpoComm_JOB_$SLURM_JOB_ID
mkdir -p $RUN_DIR
echo "Copying codebase to $RUN_DIR ..."
cp -a $HOME/ExpoComm/* $RUN_DIR/
cd $RUN_DIR

echo "Launching training: --config=$ALG_CONFIG --env-config=MAgent_Battle"
python src/main.py \
    --config=$ALG_CONFIG \
    --env-config=MAgent_Battle \
    run_name="$RUN_NAME"

echo "Training Complete!"
