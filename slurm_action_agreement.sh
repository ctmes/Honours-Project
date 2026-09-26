#!/bin/bash
# check_action_agreement.py on the CPU `work` partition (gpu partition down
# since 2026-07-22; matches slurm_eval_cpu.sh's backend choice).
#
#   sbatch slurm_action_agreement.sh                                     # smoke test: v4_config3_full, seed 0 only
#   sbatch slurm_action_agreement.sh "--project v4_config3_full --yaml config/rl_configs/eval_2024_test_v4_config3.yaml --seeds 0-19"
#   sbatch slurm_action_agreement.sh "--project v4_config1_baseline --yaml config/rl_configs/eval_2024_test_v4_config1.yaml --seeds 0-19"
#   sbatch slurm_action_agreement.sh "--project v4_config6_unconstrained --yaml config/rl_configs/eval_2024_test_v4_config6.yaml --seeds 0-19"
#
# Run the smoke test (default args, no quoted override) FIRST and check the
# output for NaNs / an out-of-range flip rate before submitting the three full
# 20-seed runs above -- this script has not been executed against a real
# checkpoint yet, only signature/syntax-checked, so the smoke test is the
# first real validation of its shape assumptions (see the nper[mm_idx]==1
# guard in the script itself, which will raise loudly rather than silently
# miscompute if that assumption doesn't hold).
#
# Builds the env once (single attack-mode="on"), so materially lighter than
# run_production_eval.py; 128G is a conservative margin below the proven-safe
# 256G eval template. Time kept short, not 48h, since a long --time hurts
# backfill scheduling priority and this is a handful of seeds against one
# checkpoint, not a sweep.
#
# Output: stdout only (see slurm_*.out) -- no results/ file is written.
#SBATCH --account=pmc097
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --job-name=action-agreement
#SBATCH --output=/group/pmc097/cmelville/logs/actionagree_%j.out
#SBATCH --error=/group/pmc097/cmelville/logs/actionagree_%j.err

EXTRA_ARGS=${1:-"--project v4_config3_full --yaml config/rl_configs/eval_2024_test_v4_config3.yaml --seeds 0"}

cd /group/pmc097/cmelville/Honours-Project
export PYTHONPATH="/group/pmc097/cmelville/Honours-Project:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

/home/cmelville/.conda/envs/honours/bin/python check_action_agreement.py ${EXTRA_ARGS}
