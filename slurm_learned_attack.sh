#!/bin/bash
# eval_learned_attack.py (WS10a) on the CPU `work` partition, same backend as the
# production, forced-attack and training runs.
#
#   sbatch slurm_learned_attack.sh     # smoke test: baseline, seed 0, + reproduction check
#   sbatch slurm_learned_attack.sh "--arms baseline,adversarial,full --seeds 0-19 --reference-eval results/eval_17313.json --out results/learned_attack_v4"
#
# Run only after the three kaya_v4opt_* training arrays have finished. The smoke
# test must show: attack_rate 1.000 in the 'learned' condition, and 'off'
# reproducing eval_17313.json EXACTLY (MATCH) -- the victim's params are
# bit-identical to the v4 checkpoint, so anything else means the loop is wrong.
#
# Memory/time: as slurm_forced_attack.sh (two env builds per arm, n_envs 64);
# three arms instead of seven, so 10h is generous.
#SBATCH --account=pmc097
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --job-name=learned-attack
#SBATCH --output=/group/pmc097/cmelville/logs/learnedattack_%j.out
#SBATCH --error=/group/pmc097/cmelville/logs/learnedattack_%j.err

EXTRA_ARGS=${1:-"--arms baseline --seeds 0 --reference-eval results/eval_17313.json"}

cd /group/pmc097/cmelville/Honours-Project
export PYTHONPATH="/group/pmc097/cmelville/Honours-Project:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

/home/cmelville/.conda/envs/honours/bin/python eval_learned_attack.py ${EXTRA_ARGS}
