#!/bin/bash
# One-shot CPU job: slice the Q1/Q2/Q3 quarter caches out of the uploaded
# master 2024_train npz, on the cluster (no raw CSVs there — dates come from
# the committed window_to_date_2024_train.json).
#
#   sbatch slurm_slice_quarters.sh
#
# Expected output (must match the locally-built quarters exactly):
#   2024_q1: 61 days, 289 windows, 205,192,091 msgs
#   2024_q2: 63 days, 300 windows, 212,241,308 msgs
#   2024_q3: 64 days, 399 windows, 277,102,170 msgs
#SBATCH --account=pmc097
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=slice-quarters
#SBATCH --output=/group/pmc097/cmelville/logs/slice_%j.out
#SBATCH --error=/group/pmc097/cmelville/logs/slice_%j.err

set -euo pipefail
cd /group/pmc097/cmelville/Honours-Project
P=/home/cmelville/.conda/envs/honours/bin/python

$P slice_period_cache.py --master-period 2024_train --period 2024_q1 \
   --from-date 2024-01-01 --to-date 2024-03-31 \
   --dates-from-window-map window_to_date_2024_train.json \
   --data-path /group/pmc097/cmelville/Honours-Project/data \
   --atpath /group/pmc097/cmelville/Honours-Project

$P slice_period_cache.py --master-period 2024_train --period 2024_q2 \
   --from-date 2024-04-01 --to-date 2024-06-30 \
   --dates-from-window-map window_to_date_2024_train.json \
   --data-path /group/pmc097/cmelville/Honours-Project/data \
   --atpath /group/pmc097/cmelville/Honours-Project

$P slice_period_cache.py --master-period 2024_train --period 2024_q3 \
   --from-date 2024-07-01 --to-date 2024-09-30 \
   --dates-from-window-map window_to_date_2024_train.json \
   --data-path /group/pmc097/cmelville/Honours-Project/data \
   --atpath /group/pmc097/cmelville/Honours-Project

echo "ALL QUARTERS SLICED OK"
