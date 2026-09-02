"""Why did the market maker stop quoting?

The v2 sweep (eval 1151304 / 1151370) left 10-14 of 20 seeds per arm at
quote_presence = 0.00 exactly, with nothing between 0 and 0.3. Two stories fit
that shape and they have opposite fixes:

  (A) EXPLORATION COLLAPSE. Policy entropy crashes early, the categorical locks
      onto the ladder's non-two-sided tail entries before the value function
      ever learns what quoting is worth. Fix: ENT_COEF, ladder design, warmup.

  (B) NO-TRADE OPTIMUM. Entropy stays healthy and the policy *deliberately*
      migrates to not quoting, because the Spooner-family reward is not
      volume-normalised and refusing to trade dodges inventory risk and adverse
      selection. Fix: the reward. This is the failure preregistration.json
      already names in interpretation_notes/no_trade_equilibrium.

Both are visible in the training logs, which print entropy and avg_reward every
update -- no re-rollout needed. The discriminator is not entropy alone but
entropy TOGETHER WITH where the reward went: if the seeds that stopped quoting
ended up with HIGHER reward than the ones that kept quoting, the objective is
paying for the collapse, and exploring harder only finds a worse policy faster.

  # on Kaya -- one TSV for the whole arm
  python diagnose_collapse.py extract --job 1137481 -o traces_baseline.tsv

  # locally, after scp
  python diagnose_collapse.py analyse traces_baseline.tsv \
      --eval results/eval_1151370.json --arm baseline
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

# "Update 412/1002  phase=mm"
_UPDATE = re.compile(r"^Update\s+(\d+)/(\d+)\s+phase=(\w+)")
# "  avg_reward_AdversarialMM: -0.1234"   (type names come from env.type_names,
# so match whatever is there rather than hard-coding them)
_REWARD = re.compile(r"^\s+avg_reward_(\S+):\s+(-?[\d.eE+]+|N/A)")
# "  MM ppo_loss: 0.1  value_loss: 0.2  entropy: 1.4  bce_loss: 0.0  adv_label_rate: 0.5"
_MMLINE = re.compile(
    r"^\s+MM\s+ppo_loss:\s*(-?[\d.eE+]+)\s+value_loss:\s*(-?[\d.eE+]+)\s+"
    r"entropy:\s*(-?[\d.eE+]+)\s+bce_loss:\s*(-?[\d.eE+]+)\s+"
    r"adv_label_rate:\s*(-?[\d.eE+]+)")

_COLS = ["seed", "update", "phase", "reward_mm", "reward_adv",
         "entropy", "ppo_loss", "value_loss", "adv_label_rate"]

_LN5 = float(np.log(5.0))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def extract(logdir: str, job: str, out: str) -> None:
    """Parse every array task's log for one job into a single tidy TSV.

    A wall-killed-and-resumed seed writes the SAME update numbers twice, across
    two logs. Keyed on (seed, update) with later records winning, so the
    resumed run's values are the ones kept.
    """
    paths = sorted(glob.glob(os.path.join(logdir, "cpusweep_%s_*.out" % job)))
    if not paths:
        sys.exit("no logs matching cpusweep_%s_*.out under %s" % (job, logdir))

    rows = {}
    for path in paths:
        m = re.search(r"cpusweep_%s_(\d+)\.out$" % job, path)
        if not m:
            continue
        seed = int(m.group(1))
        cur = None
        with open(path, errors="replace") as fh:
            for line in fh:
                mu = _UPDATE.match(line)
                if mu:
                    cur = {"seed": seed, "update": int(mu.group(1)),
                           "phase": mu.group(3)}
                    rows[(seed, cur["update"])] = cur
                    continue
                if cur is None:
                    continue
                mr = _REWARD.match(line)
                if mr:
                    # type_names index 0 is the MM, 1 the adversary, so the
                    # first avg_reward line of an update is the MM's.
                    key = "reward_mm" if "reward_mm" not in cur else "reward_adv"
                    cur[key] = _f(mr.group(2))
                    continue
                mm = _MMLINE.match(line)
                if mm:
                    cur["ppo_loss"] = _f(mm.group(1))
                    cur["value_loss"] = _f(mm.group(2))
                    cur["entropy"] = _f(mm.group(3))
                    cur["adv_label_rate"] = _f(mm.group(5))

    with open(out, "w") as fh:
        fh.write("\t".join(_COLS) + "\n")
        for key in sorted(rows):
            r = rows[key]
            fh.write("\t".join(str(r.get(c, "")) for c in _COLS) + "\n")

    seeds = sorted({s for s, _ in rows})
    print("wrote %s: %d update records, %d seeds (%d..%d) from %d logs"
          % (out, len(rows), len(seeds), seeds[0], seeds[-1], len(paths)))


def _load_tsv(path):
    raw = np.genfromtxt(path, delimiter="\t", names=True, dtype=None, encoding="utf-8")
    return {n: raw[n] for n in raw.dtype.names}


def _split_by_terminal_entropy(t, mm_phase, seeds):
    """Classify seeds without an eval JSON, using end-of-run policy entropy.

    A pilot has no evaluated checkpoints, so quote_presence does not exist yet.
    Terminal entropy stands in for it: on the v2 baseline the collapsed seeds
    ended at 0.02-0.06 and the quoting ones at 0.93-1.15, cleanly separated.
    The cut is the LARGEST GAP in the sorted values rather than a fixed
    threshold, because the separating value moves with how long the run went
    (at update 200 the two groups sat at 0.39 and 1.23, at update 1000 at 0.02
    and 1.15) and a hard-coded number would silently mis-split a short pilot.

    Validated against the v2 baseline, where quote_presence is known: the split
    recovers 19 of 20 seeds. The miss is the failure mode to remember as seed 10
    ends at entropy 0.00 with quote_presence 1.00, i.e. a policy can be
    DETERMINISTIC AND QUOTING. Low entropy means "committed", not "committed to
    doing nothing", so this proxy over-reports collapse and its verdict is
    provisional until the pilot's checkpoints are evaluated.
    """
    term = {}
    for s in seeds:
        m = mm_phase & (t["seed"] == s)
        e = np.asarray(t["entropy"][m], dtype=float)
        e = e[np.isfinite(e)]
        if e.size:
            term[int(s)] = float(e[-20:].mean())
    if len(term) < 3:
        return sorted(term), [], term
    order = sorted(term, key=term.get)
    vals = np.array([term[s] for s in order])
    gaps = np.diff(vals)
    cut = int(np.argmax(gaps))
    # A split is only meaningful if the gap actually dominates the spread.
    if gaps[cut] < 0.25 * (vals[-1] - vals[0] + 1e-9):
        return [], sorted(term), term
    return sorted(order[:cut + 1]), sorted(order[cut + 1:]), term


def analyse(tsv: str, eval_json, arm: str, qp_cut: float) -> None:
    t = _load_tsv(tsv)
    mm_phase = np.asarray([str(p) == "mm" for p in t["phase"]])
    seeds = np.unique(t["seed"]).astype(int)

    if eval_json:
        qp = np.asarray(json.load(open(eval_json))["per_seed"][arm]["quote_presence_off"],
                        dtype=float)
        collapsed = sorted(int(s) for s in seeds if s < len(qp) and qp[s] < qp_cut)
        quoting = sorted(int(s) for s in seeds if s < len(qp) and qp[s] >= qp_cut)
        print("arm=%s  collapsed (qp<%g): %d seeds  %s"
              % (arm, qp_cut, len(collapsed), collapsed))
        print("        quoting:            %d seeds  %s\n" % (len(quoting), quoting))
        if not collapsed or not quoting:
            print("  (only one group present -- group comparisons below are skipped)\n")
    else:
        qp = None
        collapsed, quoting, term = _split_by_terminal_entropy(t, mm_phase, seeds)
        print("no --eval given: seeds split on TERMINAL ENTROPY, not quote_presence")
        print("  low-entropy  (%d): %s" % (len(collapsed), collapsed))
        print("  high-entropy (%d): %s" % (len(quoting), quoting))
        print("  per seed: " + "  ".join("%d=%.2f" % (s, term[s]) for s in sorted(term)))
        if not collapsed:
            print("  no bimodal split -- every seed retained entropy, which is what a")
            print("  fixed ratchet looks like. Read WITHIN-EPISODE DRIFT below.")
        print()

    def series(seed, col):
        m = mm_phase & (t["seed"] == seed)
        order = np.argsort(t["update"][m])
        u = np.asarray(t["update"][m], dtype=float)[order]
        v = np.asarray(t[col][m], dtype=float)[order]
        ok = np.isfinite(v)
        return u[ok], v[ok]

    # --- 1. entropy: does it crash, and does it crash ONLY in collapsed seeds?
    print("ENTROPY over MM-phase updates.  Max for a 5-action categorical = ln5 = %.3f" % _LN5)
    print("  %-10s%10s%10s%11s%16s"
          % ("group", "first 50", "last 50", "frac lost", "upd @50% drop"))
    for name, grp in (("collapsed", collapsed), ("quoting", quoting)):
        early, late, drop = [], [], []
        for s in grp:
            u, e = series(s, "entropy")
            if e.size < 60:
                continue
            e0 = e[:50].mean()
            early.append(e0)
            late.append(e[-50:].mean())
            below = np.flatnonzero(e < 0.5 * e0)
            drop.append(u[below[0]] if below.size else np.nan)
        if not early:
            print("  %-10s(no seed had enough MM-phase updates)" % name)
            continue
        lost = 1.0 - np.mean(late) / np.mean(early)
        d = np.asarray(drop, dtype=float)
        dtxt = ("%.0f" % np.nanmedian(d)) if np.isfinite(d).any() else "never"
        print("  %-10s%10.3f%10.3f%10.1f%%%16s"
              % (name, np.mean(early), np.mean(late), 100 * lost, dtxt))

    # --- 2. the discriminator: where did the reward go?
    print("\nMM REWARD, mean over each seed's last 100 MM-phase updates")
    finals = {}
    for name, grp in (("collapsed", collapsed), ("quoting", quoting)):
        vals = []
        for s in grp:
            _, r = series(s, "reward_mm")
            if r.size:
                finals[s] = float(r[-100:].mean())
                vals.append(finals[s])
        if vals:
            print("  %-10s n=%-3d mean=%10.4f  median=%10.4f  sd=%9.4f"
                  % (name, len(vals), np.mean(vals), np.median(vals), np.std(vals)))

    # Only meaningful against measured quote_presence. Without --eval the seed
    # grouping is itself derived from the traces, so correlating reward against
    # it would just be measuring the split back out of the same data.
    common = sorted(finals)
    if qp is not None and len(common) >= 4:
        r = np.array([finals[s] for s in common])
        q = qp[common]
        rank = lambda v: np.argsort(np.argsort(v)).astype(float)
        rho = np.corrcoef(rank(r), rank(q))[0, 1]
        print("\n  Spearman(final MM reward, quote_presence) = %+.3f  (n=%d)"
              % (rho, len(common)))
        print("  Negative => the less a seed quoted, the more it earned.")

    _cv = [finals[s] for s in collapsed if s in finals]
    _qv = [finals[s] for s in quoting if s in finals]
    cm = float(np.mean(_cv)) if _cv else np.nan
    qm = float(np.mean(_qv)) if _qv else np.nan

    # --- 3. within-episode drift ------------------------------------------
    # The (A)/(B) split above compares seeds to each other at the END of
    # training and is blind to what happens WITHIN an episode. On the baseline
    # arm that blindness was the whole story: reward falls monotonically from
    # roughly -2 to -45 across each episode and snaps back at the reset, a
    # sawtooth whose period is episode_time / NUM_STEPS. That is inventory
    # accumulating with nothing in the reward pushing it back to flat, and no
    # amount of exploration fixes it.
    drift = _episode_drift(t, collapsed + quoting, mm_phase)
    print("\nWITHIN-EPISODE DRIFT (all seeds, reward_mm averaged per update)")
    if drift is None:
        print("  no clean sawtooth found -- reward is not dominated by a")
        print("  within-episode ratchet, so read the two sections above.")
    else:
        period, start_r, end_r = drift
        print("  reset period      %.2f updates   (episode_time/NUM_STEPS)" % period)
        print("  reward at episode start  %10.2f" % start_r)
        print("  reward at episode end    %10.2f" % end_r)
        print("  within-episode decay     %10.2f  (end - start)" % (end_r - start_r))

    print("\n" + "=" * 74)
    ratchet = drift is not None and (drift[2] - drift[1]) < -1.0
    if ratchet:
        print("VERDICT: (C) WITHIN-EPISODE RATCHET -- the dominant effect.")
        print("Reward decays %.1f per episode and resets with it, so the policy is"
              % (drift[1] - drift[2]))
        print("bleeding on a position it accumulates and never flattens.")
        print("")
        print("Check the ACTION SPACE before the reward. The bob_v0 ladder emits two")
        print("LIMIT orders at best_bid/best_ask, so the MM can only stop adding to a")
        print("position, never reduce one -- it has to wait to be filled on the other")
        print("side. auto_liquidate_threshold is the only mechanism that actively")
        print("unwinds, and 0 DISABLES it (mm_env.py:1103), against a default of 10000.")
        print("With it off and a passive-only ladder, inventory is a one-way ratchet no")
        print("reward term can undo: the 2026-09-02 pilots added a quadratic penalty at")
        print("F=1000 and F=250 and the decay went from -32 to -1130 and -5610, scaling")
        print("with 1/F while the inventory path barely moved. Punishing an agent for a")
        print("state it has no action to escape only adds a state-dependent constant")
        print("that swamps credit assignment.")
        print("")
        print("So: give it a way out first (auto_liquidate_threshold, or aggressive")
        print("entries in the ladder), and only then ask what the reward should weigh.")
    elif np.isfinite(cm) and np.isfinite(qm) and cm > qm:
        print("VERDICT: (B) NO-TRADE OPTIMUM.")
        print("Seeds that stopped quoting earn MORE (%.4f) than seeds that kept" % cm)
        print("quoting (%.4f), so the objective is paying for the collapse." % qm)
        print("Raising ENT_COEF would only explore harder toward a worse policy.")
        print("Fix the reward -- volume-normalise, or add an explicit quoting")
        print("term -- before spending the cluster on another 120 seeds.")
    else:
        print("VERDICT: leans (A) EXPLORATION.")
        print("Collapsed seeds do not out-earn quoting ones (%.4f vs %.4f) and there"
              % (cm, qm))
        print("is no within-episode ratchet, so the collapse is neither what the")
        print("objective rewards nor an inventory bleed. Read the entropy table:")
        print("a crash confined to the collapsed group points at ENT_COEF and the")
        print("action ladder.")
    print("=" * 74)


def _episode_drift(t, seeds, mm_phase):
    """Find the episode reset period and the reward decay across an episode.

    Episodes are longer than a rollout (episode_time 6400 vs NUM_STEPS 512),
    so one episode spans several updates and its boundary shows up as reward
    snapping back to near zero. Returns (period, reward_at_start, reward_at_end)
    or None when no regular sawtooth is present.
    """
    by_update = {}
    for s in seeds:
        m = t["seed"] == s
        for u, v in zip(t["update"][m], np.asarray(t["reward_mm"][m], dtype=float)):
            if np.isfinite(v):
                by_update.setdefault(int(u), []).append(v)
    if len(by_update) < 60:
        return None
    us = np.array(sorted(by_update))
    rs = np.array([np.mean(by_update[u]) for u in us])
    # Skip the first few updates: the policy is still near initialisation and
    # the ratchet has not had time to build.
    keep = us >= 20
    us, rs = us[keep], rs[keep]
    if us.size < 40:
        return None
    med = np.median(rs)
    peaks = [i for i in range(1, len(rs) - 1)
             if rs[i] > rs[i - 1] and rs[i] > rs[i + 1] and rs[i] > med]
    if len(peaks) < 4:
        return None
    gaps = np.diff(us[peaks])
    period = float(np.mean(gaps))
    if not np.isfinite(period) or period < 2 or np.std(gaps) > 0.35 * period:
        return None  # not regular enough to call a sawtooth
    # Reward at the reset (episode start) vs just before the next one.
    starts = rs[peaks]
    ends = np.array([rs[p - 1] for p in peaks])
    return period, float(np.mean(starts)), float(np.mean(ends))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="parse Kaya training logs into a TSV")
    e.add_argument("--logdir", default="/group/pmc097/cmelville/logs")
    e.add_argument("--job", required=True, help="array job id, e.g. 1137481")
    e.add_argument("-o", "--out", required=True)

    a = sub.add_parser("analyse", help="classify the collapse from the TSV")
    a.add_argument("tsv")
    a.add_argument("--eval", default=None,
                   help="eval_*.json with per_seed metrics; omit for a pilot with no "
                        "evaluated checkpoints (seeds then split on terminal entropy)")
    a.add_argument("--arm", default="baseline")
    a.add_argument("--qp-cut", type=float, default=0.01)

    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.logdir, args.job, args.out)
    else:
        analyse(args.tsv, args.eval, args.arm, args.qp_cut)


if __name__ == "__main__":
    main()
