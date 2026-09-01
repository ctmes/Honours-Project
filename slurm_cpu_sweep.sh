#!/bin/bash
# ============================================================================
# Production sweep, CPU single-phase. One array task = one seed = one full
# 1002-update run over the whole 2024_train pool (22.23 GB, held in host RAM).
#
#   sbatch --array=0-19%10 slurm_cpu_sweep.sh kaya_config1_baseline
#   sbatch --array=0-19%10 slurm_cpu_sweep.sh kaya_config4_detection
#
# Any arguments after the config name are forwarded as hydra overrides, so a
# short validation run can be written to a throwaway PROJECT instead of leaving
# a partial checkpoint in the real one:
#
#   sbatch --array=0-0 slurm_cpu_sweep.sh kaya_config1_baseline \
#          TOTAL_TIMESTEPS=491520 PROJECT=smoke_A
#
# The Q1->Q2->Q3 chaining of the GPU era is gone: it existed only to fit a 16 GB
# V100, and these nodes have 1.5 TB. Measured ~30 s/update on 8 cores, so a full
# run is ~8.3 h; --time=24:00:00 leaves roughly 3x margin for node contention.
# Wall-killed tasks: just resubmit, training resumes from the latest checkpoint
# and a finished seed exits immediately with "already complete, nothing to do".
#
# BEFORE SUBMITTING, check nothing is already queued for this arm — sbatch is
# not idempotent and two array tasks writing one checkpoint dir corrupt it:
#   squeue -u $USER -h -o %F | sort -u
# ============================================================================
#SBATCH --account=pmc097
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-19%10
#SBATCH --job-name=cpu-sweep
#SBATCH --output=/group/pmc097/cmelville/logs/cpusweep_%A_%a.out
#SBATCH --error=/group/pmc097/cmelville/logs/cpusweep_%A_%a.err

CONFIG_NAME=${1:?usage: sbatch slurm_cpu_sweep.sh <config-name> [hydra overrides...]}
shift

cd /group/pmc097/cmelville/Honours-Project
export PYTHONPATH="/group/pmc097/cmelville/Honours-Project:$PYTHONPATH"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

# --- refuse to be the second writer of this seed's checkpoint dir ----------
# The header above used to just SAY "check nothing is already queued" and leave
# it to the operator. On 2026-08-27 that check was skipped: arrays 1137496-500
# and 1137535-539 were both submitted for arms 2-6, and seeds 0-5 of each arm
# ended up with two concurrent orbax writers. A comment is not an enforcement
# mechanism, so this is one.
#
# The lock is per SEED, not per arm, because that is the granularity orbax
# corrupts at, and because a resume after a wall-kill must still be allowed:
# a lock whose owning job is no longer in squeue is stale and gets taken over.
PROJECT=$(sed -n 's/^"\?PROJECT"\?: *"\?\([^"]*\)"\?.*/\1/p' \
          "config/rl_configs/${CONFIG_NAME}.yaml" | tail -1)
for a in "$@"; do
    case "$a" in PROJECT=*) PROJECT=${a#PROJECT=} ;; esac
done
if [[ -z "$PROJECT" ]]; then
    echo "FATAL: could not resolve PROJECT from ${CONFIG_NAME}.yaml or overrides" >&2
    exit 1
fi

SEED_DIR="checkpoints/MARLCheckpoints/${PROJECT}/seed_${SLURM_ARRAY_TASK_ID}"
LOCK="${SEED_DIR}/.writer"
mkdir -p "$SEED_DIR"
if [[ -f "$LOCK" ]]; then
    owner=$(awk '{print $1}' "$LOCK")
    # A sibling task of the SAME array is not a clash - only a different array
    # job is. squeue returning a line means that job is still alive.
    if [[ "${owner%%_*}" != "${SLURM_ARRAY_JOB_ID}" ]] \
       && [[ -n "$(squeue -j "${owner%%_*}" -h -o %i 2>/dev/null)" ]]; then
        echo "REFUSING TO RUN: ${SEED_DIR} is already being written by job ${owner}," >&2
        echo "which is still in squeue. Two concurrent orbax writers corrupt the" >&2
        echo "directory silently. Cancel one of them, then resubmit." >&2
        exit 1
    fi
    echo "stale lock from job ${owner} (no longer queued) - taking over"
fi
echo "${SLURM_JOB_ID} $(hostname) $(date -Is)" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
echo "WRITER-LOCK ok: ${SEED_DIR} held by ${SLURM_JOB_ID}"

/home/cmelville/.conda/envs/honours/bin/python \
    gymnax_exchange/jaxrl/MARL/ippo_adversarial.py \
    --config-name=${CONFIG_NAME} \
    "SEED=${SLURM_ARRAY_TASK_ID}" \
    TimePeriod=2024_train \
    WINDOW_TO_DATE_PATH=window_to_date_2024_train.json \
    TOTAL_TIMESTEPS=8208384 "$@"
