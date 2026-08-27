#!/bin/bash
# ============================================================================
# Production sweep, CPU single-phase. One array task = one seed = one full
# 1002-update run over the whole 2024_train pool (22.23 GB, held in host RAM).
#
#   sbatch --array=0-19%10 slurm_cpu_sweep.sh kaya_config1_baseline
#   sbatch --array=0-19%10 slurm_cpu_sweep.sh kaya_config4_detection
#
# Any arguments after the config name are forwarded as hydra overrides, so a
# short validation run can be written to a throwaway PROJECT instead of leaving
# a partial checkpoint in the real one:
#
#   sbatch --array=0-0 slurm_cpu_sweep.sh kaya_config1_baseline \
#          TOTAL_TIMESTEPS=491520 PROJECT=smoke_A
#
# The Q1->Q2->Q3 chaining of the GPU era is gone: it existed only to fit a 16 GB
# V100, and these nodes have 1.5 TB. Measured ~30 s/update on 8 cores, so a full
# run is ~8.3 h; --time=24:00:00 leaves roughly 3x margin for node contention.
# Wall-killed tasks: just resubmit, training resumes from the latest checkpoint
# and a finished seed exits immediately with "already complete, nothing to do".
#
# BEFORE SUBMITTING, check nothing is already queued for this arm — sbatch is
# not idempotent and two array tasks writing one checkpoint dir corrupt it:
#   squeue -u $USER -h -o %F | sort -u
# ============================================================================
#SBATCH --account=pmc097
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-19%10
#SBATCH --job-name=cpu-sweep
#SBATCH --output=/group/pmc097/cmelville/logs/cpusweep_%A_%a.out
#SBATCH --error=/group/pmc097/cmelville/logs/cpusweep_%A_%a.err

CONFIG_NAME=${1:?usage: sbatch slurm_cpu_sweep.sh <config-name> [hydra overrides...]}
shift

cd /group/pmc097/cmelville/Honours-Project
export PYTHONPATH="/group/pmc097/cmelville/Honours-Project:$PYTHONPATH"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

/home/cmelville/.conda/envs/honours/bin/python \
    gymnax_exchange/jaxrl/MARL/ippo_adversarial.py \
    --config-name=${CONFIG_NAME} \
    "SEED=${SLURM_ARRAY_TASK_ID}" \
    TimePeriod=2024_train \
    WINDOW_TO_DATE_PATH=window_to_date_2024_train.json \
    TOTAL_TIMESTEPS=8208384 "$@"
