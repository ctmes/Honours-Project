#!/bin/bash
# ============================================================================
# collect_runs.sh - bundle what happened into one file to take off Kaya.
#
#   ./collect_runs.sh                  # everything that ran in the last 7 days
#   ./collect_runs.sh -d 30            # ... in the last 30 days
#   ./collect_runs.sh 1234567 1234599  # just these job / array ids
#
# eta.sh answers "when will this finish". This answers "what happened", and
# writes ONE tarball so a single scp brings the whole picture to a laptop.
#
# It deliberately does NOT copy training logs whole: twenty seeds x 1002 updates
# is tens of MB of progress lines that say nothing a tail does not. What decides
# whether a run is usable is:
#
#   State/ExitCode - COMPLETED vs TIMEOUT vs OUT_OF_MEMORY. A seed wall-killed
#                    at update 900 left a valid checkpoint and needs only a
#                    resubmit; one that died OUT_OF_MEMORY at update 3 left
#                    nothing, and the two look identical from the log tail.
#   MaxRSS         - how close --mem came to the edge, for the next submit.
#   PROJECT= line  - which arm the task ACTUALLY trained, which a resubmitted
#                    hydra override can make differ from the submitted config.
#   last Update N/M- the only evidence that a TIMEOUT task is worth resuming
#                    rather than rerunning from zero.
#   tracebacks     - the one part of a failed log worth reading.
#   results/eval_* - the actual thesis numbers; copied whole.
#
# Checkpoints are gigabytes and stay here; only the tree listing comes down,
# which is enough to see which seeds have weights left to evaluate.
#
# The CLASH section is the post-hoc form of eta.sh's live warning. It flags two
# tasks that wrote ONE checkpoint_dir while both were running - concurrent
# orbax writers corrupt the directory silently. Sequential writers of the same
# dir are NOT flagged: that is exactly what a resume after a wall-kill is.
# ============================================================================
set -u

REPO=${REPO:-/group/pmc097/cmelville/Honours-Project}
LOGDIR=${LOGDIR:-/group/pmc097/cmelville/logs}
DAYS=7
JOBS=()

while (( $# )); do
    case "$1" in
        -d) DAYS=${2:?-d needs a number of days}; shift 2 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *)  JOBS+=("$1"); shift ;;
    esac
done

SINCE=$(date -d "-${DAYS} days" +%Y-%m-%dT00:00:00)
STAMP=$(date +%Y%m%d_%H%M)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/results"
D="$STAGE/digest.txt"

if (( ${#JOBS[@]} )); then
    SEL=(-j "$(IFS=,; echo "${JOBS[*]}")")
    SCOPE="jobs ${JOBS[*]}"
else
    SEL=(-u "$USER" -S "$SINCE")
    SCOPE="all jobs for $USER since $SINCE"
fi

{
    echo "collect_runs.sh  $(date '+%a %d %b %Y %H:%M %Z')"
    echo "host=$(hostname)  repo=$REPO"
    echo "scope: $SCOPE"
    echo "commit: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')" \
         "$(git -C "$REPO" log -1 --format=%s 2>/dev/null)"
    # A dirty tree means the code that ran is not the code in git, so the digest
    # cannot be matched to a commit later. Say so at the top, not in a footnote.
    dirty=$(git -C "$REPO" status --porcelain 2>/dev/null | head -20)
    if [[ -n "$dirty" ]]; then
        echo "UNCOMMITTED CHANGES IN THE RUNNING TREE:"
        echo "$dirty"
    fi
} > "$D"

# --- 1. what SLURM thinks -------------------------------------------------
{
    echo
    echo "==== JOBS ===================================================="
    sacct "${SEL[@]}" -X -o JobID%16,JobName%12,State%18,ExitCode%9,Elapsed%11,Start%17,End%17,NodeList%12
    echo
    echo "==== MEMORY HIGH-WATER (batch steps) ========================="
    echo "ReqMem is what --mem asked for; MaxRSS is what the task actually"
    echo "touched. MaxRSS near ReqMem means the next submit needs more."
    sacct "${SEL[@]}" -o JobID%20,State%16,ReqMem%9,MaxRSS%10,MaxVMSize%10 --units=G \
        | awk 'NR<=2 || /\.batch/'
} >> "$D"

sacct "${SEL[@]}" -X -n -P -o JobID,JobName,State,ExitCode,Elapsed,Start,End > "$STAGE/jobs.psv"

logfor() { # $1 = job/task id, $2 = job name  (mirrors eta.sh)
    case "${2:-}" in
        cpu-sweep) echo "$LOGDIR/cpusweep_${1}.out" ;;
        eval-cpu)  echo "$LOGDIR/evalcpu_${1}.out"  ;;
        *)         ls -t "$LOGDIR"/*"${1}"*.out 2>/dev/null | head -1 ;;
    esac
}

# --- 2. per-task digest ---------------------------------------------------
: > "$STAGE/dirs.tsv"
{
    echo
    echo "==== PER-TASK ================================================"
} >> "$D"

while IFS='|' read -r jid name state code elapsed start end; do
    log=$(logfor "$jid" "$name")
    err="${log%.out}.err"
    {
        echo
        echo "--------------------------------------------------------------"
        echo "JOB $jid  ($name)  $state  exit=$code  elapsed=$elapsed"
        echo "log: ${log:-<none found>}"
    } >> "$D"

    if [[ ! -f "${log:-}" ]]; then
        echo "  (no log file)" >> "$D"
        continue
    fi

    pline=$(grep -am1 '^PROJECT=' "$log")
    upd=$(grep -a '^Update ' "$log" | tail -1)
    evl=$(grep -a '^\[eval\] ' "$log" | tail -1)
    fin=$(grep -a 'already complete, nothing to do' "$log" | tail -1)
    res=$(grep -ao 'Resuming from checkpoint: update [0-9]*' "$log" | tail -1)

    {
        [[ -n "$pline" ]] && echo "  arm:      $pline"
        [[ -n "$res"   ]] && echo "  resumed:  $res"
        [[ -n "$upd"   ]] && echo "  progress: $upd"
        [[ -n "$evl"   ]] && echo "  progress: $evl"
        [[ -n "$fin"   ]] && echo "  note:     $fin"
    } >> "$D"

    # checkpoint_dir + run window, for the clash check below
    cdir=$(sed -n 's/.*checkpoint_dir=\([^ ]*\).*/\1/p' <<< "$pline")
    if [[ -n "$cdir" && "$start" != "Unknown" && -n "$start" ]]; then
        s=$(date -d "$start" +%s 2>/dev/null || echo 0)
        if [[ "$end" == "Unknown" || -z "$end" ]]; then
            e=$(date +%s)          # still running: treat the window as open now
        else
            e=$(date -d "$end" +%s 2>/dev/null || echo 0)
        fi
        printf '%s\t%s\t%s\t%s\n' "$cdir" "$jid" "$s" "$e" >> "$STAGE/dirs.tsv"
    fi

    {
        echo "  --- head ---"
        head -20 "$log" | sed 's/^/  | /'
        echo "  --- tail ---"
        tail -20 "$log" | sed 's/^/  | /'
    } >> "$D"

    # Last traceback only: a dying worker can print dozens of identical ones.
    tb=$(grep -an 'Traceback (most recent call last)' "$log" | tail -1 | cut -d: -f1)
    if [[ -n "$tb" ]]; then
        {
            echo "  --- last traceback (line $tb) ---"
            sed -n "${tb},$((tb+40))p" "$log" | sed 's/^/  | /'
        } >> "$D"
    fi

    if [[ -s "$err" ]]; then
        {
            echo "  --- stderr tail ($(wc -c < "$err") bytes) ---"
            tail -25 "$err" | sed 's/^/  | /'
        } >> "$D"
    fi
done < "$STAGE/jobs.psv"

# --- 3. concurrent writers to one checkpoint dir --------------------------
{
    echo
    echo "==== CHECKPOINT-DIR CLASHES =================================="
    if [[ -s "$STAGE/dirs.tsv" ]]; then
        sort -k1,1 -k3,3n "$STAGE/dirs.tsv" | awk -F'\t' '
            $1 == pd && $3 < pe {
                print "OVERLAP " $1
                print "    " pj " and " $2 " ran concurrently - dir may be corrupt"
                n++
            }
            { pd = $1; pj = $2; pe = $4 }
            END { if (!n) print "none - no two tasks wrote one dir at the same time" }'
    else
        echo "no PROJECT= lines found (jobs predate that print, or logs are gone)"
    fi
} >> "$D"

# --- 4. what is on disk ---------------------------------------------------
{
    echo
    echo "==== CHECKPOINT TREE ========================================="
    echo "PROJECT / seed / orbax step. A seed with no step dir has no weights."
    find "$REPO/checkpoints/MARLCheckpoints" -mindepth 1 -maxdepth 3 -type d 2>/dev/null \
        | sed "s|$REPO/checkpoints/MARLCheckpoints/||" | sort
    echo
    echo "==== EVAL RESULTS COPIED ====================================="
} >> "$D"

find "$REPO/results" -maxdepth 1 -name 'eval_*' -newermt "$SINCE" -print 2>/dev/null \
    | while read -r f; do
          cp "$f" "$STAGE/results/" && echo "  $(basename "$f") ($(wc -c < "$f") bytes)"
      done >> "$D"
[[ -n "$(ls -A "$STAGE/results")" ]] || echo "  none since $SINCE" >> "$D"

# --- 5. ship it -----------------------------------------------------------
OUT="$HOME/runs_${STAMP}.tar.gz"
tar czf "$OUT" -C "$STAGE" digest.txt jobs.psv results
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo
echo "from the laptop:"
echo "  scp ${USER}@kaya.hpc.uwa.edu.au:${OUT} D:/tmp/"
