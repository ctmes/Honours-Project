#!/bin/bash
# check_adversary_sidedness.py on the CPU `work` partition (gpu partition down
# since 2026-07-22; matches slurm_eval_cpu.sh's backend choice).
#
#   sbatch slurm_adversary_sidedness.sh                                              # default: v4_config3_full, seeds 0-4
#   sbatch slurm_adversary_sidedness.sh "--project v4_config6_unconstrained --yaml config/rl_configs/eval_2024_test_v4_config6.yaml --seeds 0-4"
#
# Much lighter than run_production_eval.py: builds the env twice total (attack
# on/off), not "3x per seed per arm" -- 128G is a conservative margin below the
# proven-safe 256G eval template, not a re-guess. Time kept short (not 48h)
# because a long --time hurts backfill scheduling priority on this cluster;
# this job is a handful of seeds against one checkpoint, not a 120-arm sweep.
#
# Output: stdout only (see slurm_*.out) -- no results/ file is written.
#SBATCH --account=pmc097
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=adv-sidedness
#SBATCH --output=/group/pmc097/cmelville/logs/sidedness_%j.out
#SBATCH --error=/group/pmc097/cmelville/logs/sidedness_%j.err

EXTRA_ARGS=${1:-"--project v4_config3_full --yaml config/rl_configs/eval_2024_test_v4_config3.yaml --seeds 0-4"}

cd /group/pmc097/cmelville/Honours-Project
export PYTHONPATH="/group/pmc097/cmelville/Honours-Project:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

/home/cmelville/.conda/envs/honours/bin/python check_adversary_sidedness.py ${EXTRA_ARGS}
