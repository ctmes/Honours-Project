#!/bin/bash
#SBATCH --account=pmc097
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=honours-stability
#SBATCH --output=/group/pmc097/cmelville/logs/stability_%j.out
#SBATCH --error=/group/pmc097/cmelville/logs/stability_%j.err

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

# Stability-validation run: single seed, V100-sized, reward normalization ON.
# Config: config/rl_configs/kaya_stability.yaml (200 updates @ NUM_ENVS=16).
/home/cmelville/.conda/envs/honours/bin/python gymnax_exchange/jaxrl/MARL/ippo_adversarial.py --config-name=kaya_stability "ENV_CONFIG=config/env_configs/adversarial_mm_cluster.json"
