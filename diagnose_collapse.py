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


def analyse(tsv: str, eval_json: str, arm: str, qp_cut: float) -> None:
    t = _load_tsv(tsv)
    per_seed = json.load(open(eval_json))["per_seed"][arm]
    qp = np.asarray(per_seed["quote_presence_off"], dtype=float)

    mm_phase = np.asarray([str(p) == "mm" for p in t["phase"]])
    seeds = np.unique(t["seed"]).astype(int)
    collapsed = sorted(int(s) for s in seeds if s < len(qp) and qp[s] < qp_cut)
    quoting = sorted(int(s) for s in seeds if s < len(qp) and qp[s] >= qp_cut)
    if not collapsed or not quoting:
        sys.exit("need both groups; collapsed=%s quoting=%s" % (collapsed, quoting))

    print("arm=%s  collapsed (qp<%g): %d seeds  %s" % (arm, qp_cut, len(collapsed), collapsed))
    print("        quoting:            %d seeds  %s\n" % (len(quoting), quoting))

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

    common = sorted(finals)
    if len(common) >= 4:
        r = np.array([finals[s] for s in common])
        q = qp[common]
        rank = lambda v: np.argsort(np.argsort(v)).astype(float)
        rho = np.corrcoef(rank(r), rank(q))[0, 1]
        print("\n  Spearman(final MM reward, quote_presence) = %+.3f  (n=%d)"
              % (rho, len(common)))
        print("  Negative => the less a seed quoted, the more it earned.")

    cm = np.mean([finals[s] for s in collapsed if s in finals]) if collapsed else np.nan
    qm = np.mean([finals[s] for s in quoting if s in finals]) if quoting else np.nan

    print("\n" + "=" * 74)
    if np.isfinite(cm) and np.isfinite(qm) and cm > qm:
        print("VERDICT: (B) NO-TRADE OPTIMUM.")
        print("Seeds that stopped quoting earn MORE (%.4f) than seeds that kept" % cm)
        print("quoting (%.4f), so the objective is paying for the collapse." % qm)
        print("Raising ENT_COEF would only explore harder toward a worse policy.")
        print("Fix the reward -- volume-normalise, or add an explicit quoting")
        print("term -- before spending the cluster on another 120 seeds.")
    else:
        print("VERDICT: leans (A) EXPLORATION.")
        print("Collapsed seeds do not out-earn quoting ones (%.4f vs %.4f), so the"
              % (cm, qm))
        print("collapse is not what the objective rewards. Read the entropy table:")
        print("a crash confined to the collapsed group points at ENT_COEF and the")
        print("action ladder rather than at the reward.")
    print("Entropy retention is the cross-check either way -- a drop confined to")
    print("the collapsed group indicates (A) even when the reward gap is small.")
    print("=" * 74)


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
    a.add_argument("--eval", required=True, help="eval_*.json carrying per_seed metrics")
    a.add_argument("--arm", default="baseline")
    a.add_argument("--qp-cut", type=float, default=0.01)

    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.logdir, args.job, args.out)
    else:
        analyse(args.tsv, args.eval, args.arm, args.qp_cut)


if __name__ == "__main__":
    main()
