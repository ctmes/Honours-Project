#!/bin/bash
# ============================================================================
# 20-seed production sweep for one arm of the adversarial MM study.
#
# Usage:
#   sbatch slurm_sweep.sh kaya_config1_baseline
#   sbatch slurm_sweep.sh kaya_config2_adversarial
#   sbatch slurm_sweep.sh kaya_config3_full
#
# Each array task trains one seed (SEED = SLURM_ARRAY_TASK_ID, 0..19) and
# checkpoints under checkpoints/MARLCheckpoints/<PROJECT>/seed_<SEED>/ —
# the seed-derived run name keeps array tasks from colliding and gives the
# eval pipeline its by-index seed pairing across arms.
#
# Wall-time: TOTAL_TIMESTEPS is sized to ~19h (1000 updates @ ~68s) so a task
# normally finishes inside 24h. If a task is killed at the wall, RESUBMIT THE
# SAME COMMAND: training resumes from the latest checkpoint (model + reward-
# normalizer state), and completed seeds exit immediately ("checkpoint at
# update >= NUM_UPDATES") without burning GPU time.
#
# Throttle (%8) keeps a margin of the 34 V100s free for other users; raise it
# if the queue is empty.
# ============================================================================
#SBATCH --account=pmc097
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-19%8
#SBATCH --job-name=honours-sweep
#SBATCH --output=/group/pmc097/cmelville/logs/sweep_%A_%a.out
#SBATCH --error=/group/pmc097/cmelville/logs/sweep_%A_%a.err

CONFIG_NAME=${1:?usage: sbatch slurm_sweep.sh <config-name, e.g. kaya_config3_full>}

module load cuda/12.6.3

NVIDIA_LIBS=/home/cmelville/.conda/envs/honours/lib/python3.11/site-packages/nvidia
for dir in $NVIDIA_LIBS/*/lib; do
    export LD_LIBRARY_PATH=$dir:$LD_LIBRARY_PATH
done

export PYTHONPATH="/group/pmc097/cmelville/Honours-Project:$PYTHONPATH"
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONUNBUFFERED=1

cd /group/pmc097/cmelville/Honours-Project

/home/cmelville/.conda/envs/honours/bin/python \
    gymnax_exchange/jaxrl/MARL/ippo_adversarial.py \
    --config-name=${CONFIG_NAME} \
    "SEED=${SLURM_ARRAY_TASK_ID}"
