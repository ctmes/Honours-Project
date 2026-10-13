#!/bin/bash
# DOSE-RESPONSE / UNBOUNDED ATTACK: does ANY observation-space spoof hurt the undefended MM?
# Pre-registered: preregistration.json -> amendments -> dose_response_amendment_2026-10-09
#
# Carlini et al. 2019 ("On Evaluating Adversarial Robustness", sec. 5.2): attack success
# must rise with the perturbation budget, and an unbounded attack must eventually succeed,
# or the evaluation is suspect. The thesis's forced attack stops at inject_mult = 2.0
# (2x best-quote depth). This raises it on ONE arm (baseline, the undefended H1 victim),
# bid side, maximal scale, no retraining:
#
#   task  inject_mult   one-sided QI on a balanced book   MM depth input vs clean
#   0     2     (= the existing forced run; must REPRODUCE it)   0.50        3x
#   1     10                                                     0.83       11x
#   2     50                                                     0.96       51x
#   3     200                                                    0.99      201x
#   4     1000                                                   1.00     1001x
#
# QI saturates at 1, but the depth inputs (volume / 1e4, no clip, mm_env.get_obs) grow
# without bound, so the top rungs are an out-of-distribution input attack, not a
# stronger version of the economic signal.
#
# STEP 1 — smoke test FIRST (1 seed, top rung, no --out; minutes):
#   sbatch --array=4 --time=02:00:00 slurm_dose_response.sh smoke
# Check in the .out file:
#   * banner says DOSE-RESPONSE and inject_mult=1000
#   * attack_rate 1.0000, bid_vol ~500x the inject_mult=2 run's ~1340, ask_vol 0
#   * budget_final close to 1 (BUDGET 1e10 must not bind)
#   * no NaN in sharpe; reproduction check prints MATCH for every *_off metric
#
# STEP 2 — full ladder, 20 seeds per rung:
#   sbatch --array=0-4 slurm_dose_response.sh
#
# STEP 3 — analyse LOCALLY after copying results/dose_response_m*.json back (analysis/
# is gitignored, so it is not on Kaya):
#   python -m analysis.dose_response --out results/dose_response
#
# Time: one arm, two env builds per rung, like slurm_h3m_generalisation.sh (6 h);
# kept far under the 24 h wall that blocks backfill.
#SBATCH --account=pmc097
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --job-name=dose-resp
#SBATCH --output=/group/pmc097/cmelville/logs/dose_%A_%a.out
#SBATCH --error=/group/pmc097/cmelville/logs/dose_%A_%a.err

set -euo pipefail

MODE=${1:-full}              # "smoke" = 1 seed, no --out; "full" = 20 seeds, writes files
BUDGET=${BUDGET:-1e10}       # 1e7 left 93% unused at inject_mult 2; 1e10 covers 1000x
ARM=baseline
REF=results/forced_attack_v4_bid.json   # its *_off must reproduce (and rung 0 its *_forced)

TASK=${SLURM_ARRAY_TASK_ID:-0}
case "${TASK}" in
  0) MULT=2 ;;
  1) MULT=10 ;;
  2) MULT=50 ;;
  3) MULT=200 ;;
  4) MULT=1000 ;;
  *) echo "unknown array task ${TASK}; use --array=0-4" >&2; exit 2 ;;
esac

if [ "${MODE}" = "smoke" ]; then
  SEEDS=0
  OUT_ARG=""
else
  SEEDS=0-19
  OUT_ARG="--out results/dose_response_m${MULT}"
fi

cd /group/pmc097/cmelville/Honours-Project
export PYTHONPATH="/group/pmc097/cmelville/Honours-Project:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

echo "=== dose-response | task ${TASK} | mode ${MODE} | inject_mult ${MULT} ==="
echo "=== arm ${ARM} | side bid | seeds ${SEEDS} | budget ${BUDGET} | out '${OUT_ARG}' ==="

/home/cmelville/.conda/envs/honours/bin/python eval_forced_attack.py \
    --arms "${ARM}" \
    --seeds "${SEEDS}" \
    --side bid \
    --inject-scale 1.0 \
    --inject-mult "${MULT}" \
    --budget-override "${BUDGET}" \
    --n-envs 64 \
    --reference-eval "${REF}" \
    ${OUT_ARG}
