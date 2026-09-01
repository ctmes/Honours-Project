#!/bin/bash
# Production evaluation job: run AFTER the sweep completes (60x "1002"
# checkpoints). One GPU, ~a workday for the full 4-arm x 20-seed pass.
#
#   sbatch slurm_eval.sh                              # full pass
#   sbatch slurm_eval.sh "--arms baseline,as --n-seeds 2"   # quick partial dry-run
#
# Output lands in results/eval_<jobid>.{json,txt}. The run is ESTIMATION-ONLY
# until preregistration.json is signed off — sign off with supervisors BEFORE
# the first confirmatory pass, then never edit the values.
#SBATCH --account=pmc097
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=honours-eval
#SBATCH --output=/group/pmc097/cmelville/logs/eval_%j.out
#SBATCH --error=/group/pmc097/cmelville/logs/eval_%j.err

EXTRA_ARGS=${1:-}

module load cuda/12.6.3

NVIDIA_LIBS=/home/cmelville/.conda/envs/honours/lib/python3.11/site-packages/nvidia
for dir in $NVIDIA_LIBS/*/lib; do
    export LD_LIBRARY_PATH=$dir:$LD_LIBRARY_PATH
done

export PYTHONPATH="/group/pmc097/cmelville/Honours-Project:$PYTHONPATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONUNBUFFERED=1

cd /group/pmc097/cmelville/Honours-Project

/home/cmelville/.conda/envs/honours/bin/python run_production_eval.py \
    --out "results/eval_${SLURM_JOB_ID}" ${EXTRA_ARGS}
