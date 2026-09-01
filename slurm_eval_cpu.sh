#!/bin/bash
# Production evaluation on the CPU `work` partition (the `gpu` partition has been
# down since 2026-07-22; training moved to CPU, so evaluation follows it — mixing
# backends across arms would confound the H1 contrast).
#
#   sbatch slurm_eval_cpu.sh                                       # full 4-arm pass
#   sbatch slurm_eval_cpu.sh "--arms full --n-seeds 2 --n-envs 4"  # smoke test
#
# Memory: every attack mode rebuilds the env, which holds the whole 2024_test
# message array (~8.8 GB) plus its device buffer. rollout.py frees the previous
# mode's env before building the next; 256G still leaves generous margin on
# nodes with 1.5 TB, and the earlier 64G attempt died OUT_OF_MEMORY (job 1069659).
#
# Runtime is dominated by env construction (~2-4 min cache load), which happens
# 3x per seed per arm. Budget accordingly and smoke-test with --n-seeds first.
#
# Output: results/eval_<jobid>.{json,txt}. ESTIMATION-ONLY until
# preregistration.json is signed off.
#SBATCH --account=pmc097
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --job-name=eval-cpu
#SBATCH --output=/group/pmc097/cmelville/logs/evalcpu_%j.out
#SBATCH --error=/group/pmc097/cmelville/logs/evalcpu_%j.err

EXTRA_ARGS=${1:-}

cd /group/pmc097/cmelville/Honours-Project
export PYTHONPATH="/group/pmc097/cmelville/Honours-Project:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

/home/cmelville/.conda/envs/honours/bin/python run_production_eval.py \
    --out "results/eval_${SLURM_JOB_ID}" ${EXTRA_ARGS}
