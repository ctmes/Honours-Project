#!/bin/bash
#SBATCH --account=pmc097
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=honours-adversarial
#SBATCH --output=/group/pmc097/cmelville/logs/adversarial_%j.out
#SBATCH --error=/group/pmc097/cmelville/logs/adversarial_%j.err

module load Anaconda3/2024.06 cuda/12.6.3
source activate honours

export PYTHONPATH="/group/pmc097/cmelville/Honours-Project:$PYTHONPATH"
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

cd /group/pmc097/cmelville/Honours-Project

export PYTHONUNBUFFERED=1
python gymnax_exchange/jaxrl/MARL/ippo_adversarial.py --config-name=ippo_adversarial
