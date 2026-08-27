#!/bin/bash
# ============================================================================
# eta.sh - when will my jobs actually finish?
#
#   ./eta.sh              # every queued/running job
#   ./eta.sh 1234567      # just that array job
#
# Training tasks print "Update N/M" (ippo_adversarial.py) and eval jobs print
# "[eval] D/G arm=... seed i/n". Neither line is time-stamped, so the rate comes
# from SLURM's elapsed clock divided by the work done IN THIS TASK - a resumed
# run subtracts its "Resuming from checkpoint: update X" offset, or it looks
# several times faster than it is. Elapsed also includes the one-off JAX compile
# (2-4 min), so early readings are pessimistic and tighten as the run goes.
#
# The ARRAY line is the number that matters for scheduling: with --array=0-19%10
# half the seeds have not started, and the array is not finished until they run
# too. WALL flags any task whose remaining work exceeds its --time limit; those
# get wall-killed and need a resubmit (training resumes from the last checkpoint).
# ============================================================================
set -u

LOGDIR=${LOGDIR:-/group/pmc097/cmelville/logs}
FILTER=${1:-}
NOW=$(date +%s)
TMP=$(mktemp)
trap 'rm -f "$TMP" "$TMP.tasks"' EXIT

squeue -u "$USER" -h -o "%i|%j|%T|%M|%L|%R" | sort > "$TMP"
[[ -n "$FILTER" ]] && grep "^${FILTER}" "$TMP" > "$TMP.f" && mv "$TMP.f" "$TMP"
if [[ ! -s "$TMP" ]]; then echo "no jobs queued or running for $USER"; exit 0; fi

secs() {   # SLURM D-HH:MM:SS / HH:MM:SS / MM:SS -> seconds
    awk -F'[-:]' '{ if (NF==4) print $1*86400+$2*3600+$3*60+$4;
                    else if (NF==3) print $1*3600+$2*60+$3;
                    else if (NF==2) print $1*60+$2; else print 0 }' <<< "${1:-0}"
}
hms() {    # seconds -> "6h 12m"
    awk -v s="${1:-0}" 'BEGIN{ if (s<0||s=="") s=0;
        d=int(s/86400); h=int((s%86400)/3600); m=int((s%3600)/60);
        if (d) printf "%dd %dh", d, h; else if (h) printf "%dh %dm", h, m;
        else printf "%dm", m }'
}

declare -A ARM
armof() {  # config name from the submitted command line, cached per array id
    local a=${1%%_*}
    if [[ -z "${ARM[$a]:-}" ]]; then
        ARM[$a]=$(scontrol show job "$a" 2>/dev/null \
                  | tr ' ' '\n' | grep -m1 -o 'kaya_config[0-9a-z_]*')
        ARM[$a]=${ARM[$a]:-"-"}
    fi
    echo "${ARM[$a]}"
}

printf "%-13s %-22s %-8s %-11s %-8s %-10s %-16s %s\n" \
       JOB ARM STATE PROGRESS RATE REMAINING FINISH FLAG
printf '%.0s-' {1..105}; echo

: > "$TMP.tasks"
while IFS='|' read -r jid name state elapsed left reason; do
    case "$name" in
        cpu-sweep) log="$LOGDIR/cpusweep_${jid}.out" ;;
        eval-cpu)  log="$LOGDIR/evalcpu_${jid}.out"  ;;
        *)         log=$(ls -t "$LOGDIR"/*"${jid}"*.out 2>/dev/null | head -1) ;;
    esac
    arm=$(armof "$jid")

    if [[ "$state" != "RUNNING" ]]; then
        printf "%-13s %-22s %-8s %-11s %-8s %-10s %-16s %s\n" \
               "$jid" "$arm" "$state" "-" "-" "-" "-" "${reason:0:20}"
        echo "$jid|$state|0|0|0|0" >> "$TMP.tasks"
        continue
    fi

    cur=0; tot=0; start=0
    if [[ -f "$log" ]]; then
        if [[ "$name" == "eval-cpu" ]]; then
            ln=$(grep -a '^\[eval\] ' "$log" | tail -1)
            cur=$(awk '{split($2,a,"/"); print a[1]}' <<< "${ln:-x 0/0}")
            tot=$(awk '{split($2,a,"/"); print a[2]}' <<< "${ln:-x 0/0}")
        else
            ln=$(grep -a '^Update ' "$log" | tail -1)
            cur=$(awk '{split($2,a,"/"); print a[1]}' <<< "${ln:-x 0/0}")
            tot=$(awk '{split($2,a,"/"); print a[2]}' <<< "${ln:-x 0/0}")
            start=$(grep -ao 'Resuming from checkpoint: update [0-9]*' "$log" \
                    | tail -1 | awk '{print $NF}')
        fi
    fi
    cur=${cur:-0}; tot=${tot:-0}; start=${start:-0}
    el=$(secs "$elapsed"); lf=$(secs "$left")
    done_now=$(( cur - start ))

    if (( tot == 0 || done_now < 1 )); then
        # No completed unit yet - still loading the message cache or compiling.
        printf "%-13s %-22s %-8s %-11s %-8s %-10s %-16s %s\n" \
               "$jid" "$arm" "$state" "starting" "-" "-" "-" "up $(hms "$el")"
        echo "$jid|$state|0|0|0|$lf" >> "$TMP.tasks"
        continue
    fi

    rate=$(awk -v e="$el" -v d="$done_now" 'BEGIN{printf "%.1f", e/d}')
    rem=$(awk -v t="$tot" -v c="$cur" -v r="$rate" 'BEGIN{printf "%d", (t-c)*r}')
    full=$(awk -v t="$tot" -v r="$rate" 'BEGIN{printf "%d", t*r}')
    flag=""
    (( rem > lf )) && flag="WALL (limit $left)"
    printf "%-13s %-22s %-8s %-11s %-8s %-10s %-16s %s\n" \
           "$jid" "$arm" "$state" "$cur/$tot" "${rate}s" "$(hms "$rem")" \
           "$(date -d "@$((NOW+rem))" '+%a %d %b %H:%M')" "$flag"
    echo "$jid|$state|$rem|$full|$rate|$lf" >> "$TMP.tasks"
done < "$TMP"

# --- array roll-up ---------------------------------------------------------
# A throttled array (%N) finishes in waves: the running wave has to drain before
# the pending tasks start, so the array ETA is the slowest running task plus one
# full run per remaining wave.
echo
for a in $(cut -d'|' -f1 "$TMP.tasks" | cut -d'_' -f1 | sort -u); do
    running=$(awk -F'|' -v a="$a" '$1 ~ "^"a && $2=="RUNNING"' "$TMP.tasks" | wc -l)
    pending=$(awk -F'|' -v a="$a" '$1 ~ "^"a && $2!="RUNNING"' "$TMP.tasks" | wc -l)
    (( running + pending == 0 )) && continue
    slowest=$(awk -F'|' -v a="$a" '$1 ~ "^"a {if ($3+0>m) m=$3+0} END{print m+0}' "$TMP.tasks")
    full=$(awk -F'|' -v a="$a" '$4+0>0 {n++; s+=$4} END{print (n? int(s/n) : 0)}' "$TMP.tasks")
    ok=$(sacct -j "$a" -X -n -o State 2>/dev/null | grep -c COMPLETED)
    if (( full > 0 && running > 0 )); then
        waves=$(( (pending + running - 1) / running ))
        eta=$(( slowest + waves * full ))
        when=$(date -d "@$((NOW+eta))" '+%a %d %b %H:%M')
    else
        eta=0; when="unknown (nothing running yet)"
    fi
    printf "array %-9s %-22s %2d done  %2d running  %2d pending   -> ~%s  (%s)\n" \
           "$a" "$(armof "$a")" "$ok" "$running" "$pending" "$(hms "$eta")" "$when"
done
