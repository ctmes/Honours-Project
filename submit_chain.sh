#!/bin/bash
# Submit one arm's full 20-seed CHAINED sweep: Q1 -> Q2 -> Q3 per seed.
#
#   ./submit_chain.sh kaya_config1_baseline
#   ./submit_chain.sh kaya_config2_adversarial
#   ./submit_chain.sh kaya_config3_full
#
# aftercorr makes array task i of each phase wait for task i of the previous
# phase to complete OK — so seeds chain independently and a single failed seed
# only holds up its own chain, not the other 19.
set -euo pipefail
CONFIG=${1:?usage: ./submit_chain.sh <config-name, e.g. kaya_config3_full>}

j1=$(sbatch --parsable slurm_sweep.sh "$CONFIG" q1)
j2=$(sbatch --parsable --dependency=aftercorr:"$j1" slurm_sweep.sh "$CONFIG" q2)
j3=$(sbatch --parsable --dependency=aftercorr:"$j2" slurm_sweep.sh "$CONFIG" q3)
echo "$CONFIG chain submitted: q1=$j1 -> q2=$j2 -> q3=$j3"
echo "watch: squeue -u \$USER -o '%.10i %.20j %.8T %.10M %R'"
