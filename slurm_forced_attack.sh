#!/bin/bash
# eval_forced_attack.py on the CPU `work` partition (gpu partition down since
# 2026-07-22; matches slurm_eval_cpu.sh's backend so 'off' numbers can reproduce
# the production eval exactly).
#
#   sbatch slurm_forced_attack.sh     # smoke test: baseline, seed 0, bid, + reproduction check
#   sbatch slurm_forced_attack.sh "--arms full --seeds 0 --side bid --budget-override 1e6 --reference-eval results/eval_17313.json"
#   sbatch slurm_forced_attack.sh "--arms full --seeds 0 --side ask --budget-override 1e6"
#   sbatch slurm_forced_attack.sh "--arms all --seeds 0-19 --side bid --budget-override 1e6 --out results/forced_attack_v4_bid"
#   sbatch slurm_forced_attack.sh "--arms all --seeds 0-19 --side ask --budget-override 1e6 --out results/forced_attack_v4_ask"
#
# Do NOT submit the two --arms all runs until the smoke tests pass the checks in
# docs/note_forced_attack_amendment.md (attack_rate ~1.0, budget not exhausted,
# off reproduces eval_17313 MATCH, |QI| moves and flips sign between bid/ask).
# One side per submission, so a failure on one does not lose the other.
#
# Memory/time: n_envs=64 like the production eval, so 256G as in slurm_eval_cpu.sh
# (64G OOM'd there). This script builds the env twice per arm (14 builds for 7 arms)
# against the production eval's once per seed per mode, so a full side should take
# well under the 20h20m job 17313 did; 16h keeps it under the 24h wall that blocks
# backfill on this cluster. The single-seed smoke tests finish in minutes.
#
# Output: stdout always; results/forced_attack_*.{json,txt} only when --out is given.
#SBATCH --account=pmc097
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --job-name=forced-attack
#SBATCH --output=/group/pmc097/cmelville/logs/forcedattack_%j.out
#SBATCH --error=/group/pmc097/cmelville/logs/forcedattack_%j.err

EXTRA_ARGS=${1:-"--arms baseline --seeds 0 --side bid --budget-override 1e6 --reference-eval results/eval_17313.json"}

cd /group/pmc097/cmelville/Honours-Project
export PYTHONPATH="/group/pmc097/cmelville/Honours-Project:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

/home/cmelville/.conda/envs/honours/bin/python eval_forced_attack.py ${EXTRA_ARGS}
