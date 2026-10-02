"""How far should a rational market maker move its quotes when the book's imbalance moves?

The mechanism behind the H1 null, quantified. A market maker that prices off the book
uses an imbalance-adjusted fair value, E[mid_{t+h} | QI_t] (Stoikov's micro-price idea;
Gould & Bonart 2016; Cartea, Donnelly & Jaimungal 2018). This script estimates that
relation on the 2024 TRAINING days only, at the environment's decision horizon
(h = 100 messages = one env step), using the same queue-imbalance definition as the MM
observation (mm_env.get_adversarial_observation):

    QI = (bid_v1 - ask_v1) / (bid_v1 + ask_v1)

and reports the slope beta (ticks of expected mid move per unit QI) with a
day-clustered bootstrap CI, then the quote shift a spoof would induce in a rational
imbalance-pricing MM:  shift = beta * dQI, for the forced attack's measured dQI
(+0.40 bid side, -0.44 ask side; docs/note_h1_forced_attack.md). If that shift is a
small fraction of a tick, a rational MM's (tick-rounded) quotes cannot be moved by an
observation-only spoof in this market.

    python -m analysis.qi_fair_value --data data/rawLOBSTER/AMZN/2024_train --out results/qi_fair_value
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

TICK = 100.0          # LOBSTER price units ($ x 1e4); AMZN tick = $0.01
FORCED_DQI = {"bid": 0.40, "ask": -0.44}


def day_pairs(path: str, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ob = pd.read_csv(path, header=None, usecols=[0, 1, 2, 3],
                     names=["ap", "av", "bp", "bv"], dtype=np.float64)
    ob = ob[(ob.ap > 0) & (ob.bp > 0) & (ob.ap < 9e9) & (ob.bv > 0) & (ob.av > 0)]
    mid = (ob.ap.values + ob.bp.values) / 2.0
    qi = (ob.bv.values - ob.av.values) / (ob.bv.values + ob.av.values)
    spread = ob.ap.values - ob.bp.values
    idx = np.arange(0, len(mid) - horizon, horizon)          # non-overlapping horizons
    return qi[idx], (mid[idx + horizon] - mid[idx]) / TICK, spread[idx] / TICK


def run(data_dir: str, every: int, horizon: int, n_boot: int, seed: int) -> dict:
    files = sorted(glob.glob(os.path.join(data_dir, "*_orderbook_10.csv")))[::every]
    if not files:
        raise SystemExit(f"no *_orderbook_10.csv under {data_dir}")
    per_day = []
    for f in files:
        x, y, s = day_pairs(f, horizon)
        ok = np.isfinite(x) & np.isfinite(y)
        per_day.append((os.path.basename(f).split("_")[1], x[ok], y[ok], s[ok]))
    x = np.concatenate([d[1] for d in per_day])
    y = np.concatenate([d[2] for d in per_day])
    s = np.concatenate([d[3] for d in per_day])
    beta, alpha = np.polyfit(x, y, 1)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):                                    # day-clustered bootstrap
        pick = rng.integers(0, len(per_day), len(per_day))
        bx = np.concatenate([per_day[i][1] for i in pick])
        by = np.concatenate([per_day[i][2] for i in pick])
        boots.append(np.polyfit(bx, by, 1)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    bins = {}
    for a, b in ((-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.01)):
        m = (x >= a) & (x < b)
        bins[f"[{a:+.1f},{min(b, 1.0):+.1f})"] = {"n": int(m.sum()),
                                                  "mean_dmid_ticks": float(y[m].mean())}
    return {
        "data_dir": data_dir, "days": [d[0] for d in per_day], "n_obs": int(x.size),
        "horizon_messages": horizon,
        "beta_ticks_per_unit_qi": float(beta), "beta_ci95_day_bootstrap": [float(lo), float(hi)],
        "alpha_ticks": float(alpha), "corr": float(np.corrcoef(x, y)[0, 1]),
        "median_spread_ticks": float(np.median(s)),
        "share_one_tick_spread": float(np.mean(np.isclose(s, 1.0))),
        "mean_dmid_by_qi_bin": bins,
        "implied_quote_shift_ticks": {k: float(beta * v) for k, v in FORCED_DQI.items()},
        "implied_quote_shift_ci95": {k: sorted([float(lo * v), float(hi * v)])
                                     for k, v in FORCED_DQI.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/rawLOBSTER/AMZN/2024_train")
    ap.add_argument("--every", type=int, default=12, help="use every k-th trading day")
    ap.add_argument("--horizon", type=int, default=100, help="messages = one env step")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(args.data, args.every, args.horizon, args.n_boot, args.seed)
    lo, hi = res["beta_ci95_day_bootstrap"]
    lines = [f"QI fair-value calibration: {len(res['days'])} training days, {res['n_obs']} obs, "
             f"h = {res['horizon_messages']} messages",
             f"  beta = {res['beta_ticks_per_unit_qi']:.3f} ticks per unit QI "
             f"(95% day-bootstrap CI {lo:.3f} to {hi:.3f}); corr {res['corr']:.3f}",
             f"  median spread {res['median_spread_ticks']:.2f} ticks; 1-tick spread share "
             f"{res['share_one_tick_spread']:.2f}"]
    for k, v in res["mean_dmid_by_qi_bin"].items():
        lines.append(f"  QI {k:<12} n={v['n']:7d}  mean mid move {v['mean_dmid_ticks']:+.3f} ticks")
    for k, v in res["implied_quote_shift_ticks"].items():
        c = res["implied_quote_shift_ci95"][k]
        lines.append(f"  rational quote shift under the forced {k}-side spoof: {v:+.3f} ticks "
                     f"(CI {c[0]:+.3f} to {c[1]:+.3f})")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(f"{args.out}.json", "w") as f:
            json.dump(res, f, indent=2)
        with open(f"{args.out}.txt", "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
