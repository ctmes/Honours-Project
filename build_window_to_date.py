"""
Build window_to_date.json: maps the env's window_index -> source trading date.

The adversarial env (AdversarialMARLEnv._build_regime_array) needs this to look up
each training window's volatility regime. Windows (base_env.start_indeces) are the
per-window message offsets into the concatenated msgs array, grouped by day in date
order. Days are concatenated back-to-back, each spanning 34200-57600s, so a day
boundary is exactly where the message time column resets downward — no CSV re-read
needed; we read it straight from the cached .npz the loader already produced.

Pair with regime_labels.json (build_regime_labels.py): the env composes
window_index -> date (here) -> regime (there).

Usage:
  python build_window_to_date.py --stock AMZN --period 2022
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def find_cache(atpath: str, stock: str, period: str) -> str:
    pat = os.path.join(atpath, "saved_npz", f"loaded_lobster_*_{stock}_{period}_*.npz")
    matches = [m for m in glob.glob(pat)
               if f"_{stock}_{period}_small_" not in os.path.basename(m)]
    if not matches:
        raise FileNotFoundError(
            f"No cached npz for {stock} {period} under {atpath}/saved_npz/. "
            f"Run the env/loader once for this period+config to create it."
        )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple caches match {stock} {period}: {matches}. "
                           f"Pass --cache to disambiguate.")
    return matches[0]


def build_window_to_date(data_path: str, stock: str, period: str,
                         atpath: str = ".", cache: str | None = None) -> dict[int, str]:
    cache = cache or find_cache(atpath, stock, period)
    data = np.load(cache, allow_pickle=True)
    msgs, starts = data["msgs"], data["starts"]

    # Source dates in loader (sorted-filename = chronological) order.
    d = Path(data_path) / "rawLOBSTER" / stock / period
    dates = [f.name.split("_")[1] for f in sorted(d.glob(f"{stock}_*_message_10.csv"))]

    # Day boundaries = downward resets in the message time column (last-but-one col).
    t = msgs[:, -2]
    boundaries = np.where(np.diff(t) < 0)[0] + 1
    n_days = len(boundaries) + 1
    if n_days != len(dates):
        raise AssertionError(
            f"detected {n_days} days from time resets but {len(dates)} date files — "
            f"cache/config mismatch with the data directory."
        )

    day_of_window = np.searchsorted(boundaries, starts, side="right")
    return {int(i): dates[int(day_of_window[i])] for i in range(len(starts))}, dates, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data")
    ap.add_argument("--stock", default="AMZN")
    ap.add_argument("--period", default="2022")
    ap.add_argument("--atpath", default=".")
    ap.add_argument("--cache", default=None, help="explicit .npz path (overrides auto-find)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--out-file", default=None,
                    help="output filename (default: window_to_date_<period>.json, "
                         "matching the WINDOW_TO_DATE_PATH values in the production yamls)")
    args = ap.parse_args()

    w2d, dates, cache = build_window_to_date(
        args.data_path, args.stock, args.period, args.atpath, args.cache)

    out = Path(args.out) / (args.out_file or f"window_to_date_{args.period}.json")
    out.write_text(json.dumps(w2d, indent=2))

    # windows-per-day distribution (sanity)
    per_day = {}
    for d in w2d.values():
        per_day[d] = per_day.get(d, 0) + 1
    counts = np.array(list(per_day.values()))
    print(f"cache: {os.path.basename(cache)}")
    print(f"{len(w2d)} windows across {len(per_day)} days "
          f"(expected {len(dates)} date files)")
    print(f"windows/day: min={counts.min()} max={counts.max()} "
          f"mean={counts.mean():.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
