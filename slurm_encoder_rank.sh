#!/bin/bash
# check_encoder_rank.py (RC4) on the CPU `work` partition, same backend as the
# production eval and the forced-attack runs.
#
#   sbatch slurm_encoder_rank.sh     # smoke test: 2 arms, seed 0, builds the fixed obs set
#   sbatch slurm_encoder_rank.sh "--arms all --seeds 0-19 --own-steps 128 --eval-ref results/eval_17313.json --out results/encoder_rank_v4"
#
# The smoke test builds and caches the fixed observation set
# (outputs/encoder_rank_obs_v4.npz, a clean v4 baseline seed-0 episode, 64 envs x
# 512 steps); the full run reuses it, so both runs score every checkpoint on the
# SAME observations. Delete the cache only if the reference arm/seed/step changes.
#
# Memory/time: one env build per arm at n_envs=64 (as slurm_forced_attack.sh,
# which needed 256G); forward passes through a 128-d encoder are negligible. The
# forced-attack grid does 14 builds and 280 episode rollouts in <16h; this job does
# 8 builds, 1 episode rollout and (with --own-steps 128) 140 quarter-episodes, so
# 10h is generous and short enough for backfill (see kaya_backfill note).
#SBATCH --account=pmc097
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --job-name=encoder-rank
#SBATCH --output=/group/pmc097/cmelville/logs/encoderrank_%j.out
#SBATCH --error=/group/pmc097/cmelville/logs/encoderrank_%j.err

EXTRA_ARGS=${1:-"--arms baseline,detection --seeds 0"}

cd /group/pmc097/cmelville/Honours-Project
export PYTHONPATH="/group/pmc097/cmelville/Honours-Project:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

/home/cmelville/.conda/envs/honours/bin/python check_encoder_rank.py ${EXTRA_ARGS}
