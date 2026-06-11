#!/bin/bash
#SBATCH --account=pmc097
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=honours-baseline
#SBATCH --output=/group/pmc097/cmelville/logs/baseline_%j.out
#SBATCH --error=/group/pmc097/cmelville/logs/baseline_%j.err

module load cuda/12.6.3

NVIDIA_LIBS=/home/cmelville/.conda/envs/honours/lib/python3.11/site-packages/nvidia
for dir in $NVIDIA_LIBS/*/lib; do
    export LD_LIBRARY_PATH=$dir:$LD_LIBRARY_PATH
done

export PYTHONPATH="/group/pmc097/cmelville/Honours-Project:$PYTHONPATH"
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

cd /group/pmc097/cmelville/Honours-Project

/home/cmelville/.conda/envs/honours/bin/python gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py --config-name=ippo_rnn_JAXMARL_2player
