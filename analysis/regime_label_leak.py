"""How much look-ahead is in regime_labels.json? (literature audit sec 3.13)

The committed labels come from build_regime_labels.label_regimes' default rule,
which has two leaks relative to a point-in-time rule (Fonseca 2026: at decision
time t the information set must lie in F_t):

  1. same-day: rv on day t includes day t's own close-to-close return, and the
     label is the MM's observation DURING day t;
  2. threshold: the high/low cut is the median of rv over the whole of 2024, so
     the held-out Q4's own volatility helps set the cut used to label it.

This script recomputes the labels under three stricter rules and counts how many
change, over the year and over the held-out test period. The numbers quoted in
the thesis limitations come from here, not from a notebook.

  lag          rv through day t-1, same full-year threshold        (removes leak 1)
  train_thr    same-day rv, threshold = median rv over the training period only
                                                                     (removes leak 2 for the test period)
  point_in_time  label_regimes(point_in_time=True): lagged rv vs expanding median
                 of lagged rv                                        (removes both)

    python -m analysis.regime_label_leak                     # reads regime_labels.csv
    python -m analysis.regime_label_leak --out results/regime_label_leak.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from build_regime_labels import label_regimes

TEST_START = "2024-10-01"   # holdout = 2024-10-01..2024-12-31 (preregistration training_schedule_note)


def variants(df: pd.DataFrame, test_start: str = TEST_START, window: int = 20) -> dict:
    base = label_regimes(df[["date", "close"]], window=window)
    rv = base["rv"]
    out = {"committed": base["regime"].to_numpy()}

    thr_full = rv.median()
    lag = (rv.shift(1) > thr_full).astype(int)
    lag[rv.shift(1).isna()] = 0
    out["lag"] = lag.to_numpy()

    thr_train = rv[base["date"] < test_start].median()
    tt = (rv > thr_train).astype(int)
    tt[rv.isna()] = 0
    out["train_thr"] = tt.to_numpy()

    out["point_in_time"] = label_regimes(df[["date", "close"]], window=window,
                                         point_in_time=True)["regime"].to_numpy()
    return {"dates": base["date"].to_numpy(), "labels": out,
            "thresholds": {"full_year_median": float(thr_full),
                           "train_only_median": float(thr_train)}}


def leak_summary(df: pd.DataFrame, test_start: str = TEST_START) -> dict:
    v = variants(df, test_start)
    dates, lab = v["dates"], v["labels"]
    test = dates >= test_start
    base = lab["committed"]
    res = {"n_days": int(len(dates)), "n_test_days": int(test.sum()),
           "committed_test_high_frac": float(base[test].mean()),
           "thresholds": v["thresholds"], "variants": {}}
    for name in ("lag", "train_thr", "point_in_time"):
        x = lab[name]
        changed = x != base
        res["variants"][name] = {
            "changed_all_year": int(changed.sum()),
            "changed_test": int(changed[test].sum()),
            "changed_test_frac": float(changed[test].mean()),
            "test_high_frac": float(x[test].mean()),
            "changed_test_dates": [str(d) for d in dates[test & changed]],
        }
    return res


def format_text(res: dict) -> str:
    lines = [f"regime label leak: {res['n_days']} days, {res['n_test_days']} held-out test days",
             f"  committed labels: {100 * res['committed_test_high_frac']:.1f}% of test days high-vol",
             f"  thresholds: full-year median {res['thresholds']['full_year_median']:.5f}, "
             f"training-only median {res['thresholds']['train_only_median']:.5f}",
             "  variant         changed(year)  changed(test)  test high-vol"]
    for name, r in res["variants"].items():
        lines.append(f"  {name:<14} {r['changed_all_year']:>8}/{res['n_days']:<5} "
                     f"{r['changed_test']:>6}/{res['n_test_days']:<3} ({100 * r['changed_test_frac']:.1f}%)"
                     f"   {100 * r['test_high_frac']:.1f}%")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default="regime_labels.csv")
    ap.add_argument("--test-start", default=TEST_START)
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    args = ap.parse_args()
    res = leak_summary(pd.read_csv(args.csv), args.test_start)
    print(format_text(res))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
