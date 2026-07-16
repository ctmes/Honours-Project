#!/bin/bash
# ============================================================================
# 20-seed production sweep for one arm — one QUARTER phase per submission.
#
# The full Jan-Sep 2024 training pool (~22 GB on-device) cannot fit a 16 GB
# V100, so each seed trains as a CHAIN of three resumed phases over quarter
# caches (~7-9 GB each): Q1 (updates 0-333) -> Q2 (334-667) -> Q3 (668-1001).
# The checkpoint (model + reward-normalizer state) carries across phases, and
# ANNEAL_TOTAL_UPDATES in the yaml pins the LR anneal to the whole chain.
#
# Usage (normally via submit_chain.sh, which wires the per-seed dependencies):
#   sbatch slurm_sweep.sh kaya_config3_full q1
#   sbatch --dependency=aftercorr:<q1_jobid> slurm_sweep.sh kaya_config3_full q2
#   sbatch --dependency=aftercorr:<q2_jobid> slurm_sweep.sh kaya_config3_full q3
#
# Each array task trains one seed (SEED = SLURM_ARRAY_TASK_ID, 0..19) into
# checkpoints/MARLCheckpoints/<PROJECT>/seed_<SEED>/ — shared by all phases of
# that seed. Wall-killed tasks: resubmit the same phase; training resumes from
# the latest checkpoint, and already-complete phases exit immediately.
#
# Throttle (%8) keeps a margin of the 34 V100s free; raise it if queue is empty.
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

CONFIG_NAME=${1:?usage: sbatch slurm_sweep.sh <config-name> [q1|q2|q3]}
PHASE=${2:-}

# Per-phase hydra overrides. TOTAL_TIMESTEPS is CUMULATIVE (the resume logic
# trains from the checkpoint up to this config's NUM_UPDATES):
#   334 updates/phase x 512 steps x 16 envs = 2,736,128 timesteps per phase.
case "$PHASE" in
  q1) OVERRIDES=("TimePeriod=2024_q1" "WINDOW_TO_DATE_PATH=window_to_date_2024_q1.json" "TOTAL_TIMESTEPS=2736128") ;;
  q2) OVERRIDES=("TimePeriod=2024_q2" "WINDOW_TO_DATE_PATH=window_to_date_2024_q2.json" "TOTAL_TIMESTEPS=5472256") ;;
  q3) OVERRIDES=("TimePeriod=2024_q3" "WINDOW_TO_DATE_PATH=window_to_date_2024_q3.json" "TOTAL_TIMESTEPS=8208384") ;;
  "") OVERRIDES=() ;;   # no phase: run the yaml exactly as written
  *)  echo "unknown phase '$PHASE' (expected q1|q2|q3)"; exit 2 ;;
esac

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
    "SEED=${SLURM_ARRAY_TASK_ID}" "${OVERRIDES[@]}"
