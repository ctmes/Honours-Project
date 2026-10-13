#!/bin/bash
# H3m GENERALISATION: does the detection head detect attacks OUTSIDE its training family?
# Pre-registered: preregistration.json -> amendments -> h3m_generalisation_amendment_2026-10-06
# Design and how to read each outcome: docs/note_h3m_generalisation.md
#
# One arm (scripted_detection), 20 seeds, no retraining. Three conditions, as a job
# array so each lands in a separate backfill slot:
#
#   0  SYMMETRIC maximal     --side both --inject-scale 1.0   (out of family: SIDEDNESS)
#   1  SUB-THRESHOLD bid     --side bid  --inject-scale 0.10  (out of family: MAGNITUDE)
#   2  SUB-THRESHOLD ask     --side ask  --inject-scale 0.10  (out of family: MAGNITUDE)
#
# STEP 1 — smoke test FIRST (preconditions in the amendment; ~minutes, 1 seed):
#   sbatch --array=0 --time=02:00:00 slurm_h3m_generalisation.sh smoke
# Check in the .out file before going further:
#   * attack rate 1.000 on every step
#   * budget_final_frac high / budget_exhausted_frac 0  (the symmetric attack injects on
#     10 levels, not 5, so it uses ~2x the volume; the one-sided maximal run left 92.5%
#     of 1e7, so 1e7 should hold -- but it is CHECKED, not assumed. If it binds, raise
#     BUDGET below and rerun this smoke test.)
#   * bid_vol ~= ask_vol  (that is what "symmetric" means)
#   * the reproduction check prints MATCH for sharpe_off / sortino_off / cvar_off /
#     inventory_sd_off against results/forced_attack_v4s_bid.json
#
# STEP 2 — full grid, all three conditions, 20 seeds each:
#   sbatch --array=0-2 slurm_h3m_generalisation.sh
#
# STEP 3 — analyse (locally or here; see the bottom of this file for the exact calls).
#
# Memory/time: one arm = 2 env builds, against 7 arms (14 builds) for the original
# forced grid, so this is far cheaper than the 16h that script asks for. 6h keeps it
# well under the 24h wall that blocks backfill on this cluster.
#SBATCH --account=pmc097
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --job-name=h3m-general
#SBATCH --output=/group/pmc097/cmelville/logs/h3mgen_%A_%a.out
#SBATCH --error=/group/pmc097/cmelville/logs/h3mgen_%A_%a.err

set -euo pipefail

MODE=${1:-full}                 # "smoke" = 1 seed, no --out; "full" = 20 seeds, writes files
BUDGET=${BUDGET:-1e7}           # raise if the smoke test shows the budget binding
ARM=scripted_detection
REF=results/forced_attack_v4s_bid.json   # the in-family H3m run: its *_off must reproduce

TASK=${SLURM_ARRAY_TASK_ID:-0}
case "${TASK}" in
  0) SIDE=both; SCALE=1.0;  TAG=sym ;;
  1) SIDE=bid;  SCALE=0.10; TAG=bid_s010 ;;
  2) SIDE=ask;  SCALE=0.10; TAG=ask_s010 ;;
  *) echo "unknown array task ${TASK}; use --array=0-2" >&2; exit 2 ;;
esac

if [ "${MODE}" = "smoke" ]; then
  SEEDS=0
  OUT_ARG=""
else
  SEEDS=0-19
  OUT_ARG="--out results/forced_attack_v4s_${TAG}"
fi

cd /group/pmc097/cmelville/Honours-Project
# ${PYTHONPATH:-}, not $PYTHONPATH: under set -u an unset PYTHONPATH aborts the job
# before python starts (job 89584 died this way in 0 s).
export PYTHONPATH="/group/pmc097/cmelville/Honours-Project:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

echo "=== H3m generalisation | task ${TASK} | mode ${MODE} | side ${SIDE} scale ${SCALE} ==="
echo "=== arm ${ARM} | seeds ${SEEDS} | budget ${BUDGET} | out '${OUT_ARG}' ==="

/home/cmelville/.conda/envs/honours/bin/python eval_forced_attack.py \
    --arms "${ARM}" \
    --seeds "${SEEDS}" \
    --side "${SIDE}" \
    --inject-scale "${SCALE}" \
    --budget-override "${BUDGET}" \
    --n-envs 64 \
    --reference-eval "${REF}" \
    ${OUT_ARG}

# ---------------------------------------------------------------------------
# STEP 3, after the full grid finishes. Two SEPARATE analysis calls: the symmetric
# condition answers the sidedness question and the two sub-threshold runs answer the
# magnitude question, so they are not pooled into one Holm family.
#
#   # (A) symmetric, alone -- it has no sides, so its single cell is the verdict
#   python -m analysis.h3_material_check results/forced_attack_v4s_sym.json \
#       --ref-bid results/forced_attack_v4s_bid.json \
#       --out results/h3m_generalisation_sym
#
#   # (B) sub-threshold, both sides, exactly like the in-family test
#   python -m analysis.h3_material_check \
#       results/forced_attack_v4s_bid_s010.json results/forced_attack_v4s_ask_s010.json \
#       --ref-bid results/forced_attack_v4s_bid.json \
#       --ref-ask results/forced_attack_v4s_ask.json \
#       --out results/h3m_generalisation_s010
#
# Both print "!! GENERALISATION TEST" and name the out-of-family conditions.
