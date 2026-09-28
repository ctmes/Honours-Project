"""
Generate binary volatility-regime labels (proposal Challenge 3) from LOBSTER data.

For each trading day: take the closing mid-price (last orderbook row), compute daily
log-returns, a `window`-day rolling realised volatility (std of daily returns), and label
each day high-vol (1) / low-vol (0) by thresholding the RV series at its median.

Outputs (written next to --out, default repo root):
  regime_labels.json   {date: 0|1}        -> set as REGIME_LABELS_PATH in the train config
  regime_labels.csv    date,close,ret,rv,regime  (for inspection / plotting)

Threshold note: the proposal specifies a *trailing 252-day* median, which needs >1 year of
history. With a single year (AMZN 2022) we use the in-sample median of the RV series; the
trailing rule is a drop-in (`--trailing 252`) once multi-year data is available. This ties
to the 2022-vs-2024 data decision.

Usage:
  python build_regime_labels.py --stock AMZN --period 2022 --window 20
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

_PRICE_SCALE = 10000.0  # LOBSTER prices are dollars * 1e4


def last_line(path: Path, chunk: int = 8192) -> str:
    """Read the final non-empty line of a (possibly large) file without loading it all."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        data = b""
        while size > 0:
            read = min(chunk, size)
            size -= read
            f.seek(size)
            data = f.read(read) + data
            if data.count(b"\n") >= 2 or size == 0:
                break
    lines = data.splitlines()
    return lines[-1].decode() if lines else ""


def closing_mid(ob_path: Path) -> float:
    """Closing mid = (best_ask + best_bid) / 2 from the last book row."""
    cols = last_line(ob_path).split(",")
    ask_p1, bid_p1 = float(cols[0]), float(cols[2])
    return (ask_p1 + bid_p1) / 2.0 / _PRICE_SCALE


def build_regime_labels(data_path: str, stock: str, period: str,
                        window: int = 20, trailing: int | None = None,
                        point_in_time: bool = False) -> pd.DataFrame:
    d = Path(data_path) / "rawLOBSTER" / stock / period
    ob_files = sorted(d.glob(f"{stock}_*_orderbook_10.csv"))
    if not ob_files:
        raise FileNotFoundError(f"No orderbook files under {d}")

    rows = [(ob.name.split("_")[1], closing_mid(ob)) for ob in ob_files]
    df = pd.DataFrame(rows, columns=["date", "close"])
    return label_regimes(df, window, trailing, point_in_time)


def label_regimes(df: pd.DataFrame, window: int = 20, trailing: int | None = None,
                  point_in_time: bool = False) -> pd.DataFrame:
    """Pure labelling step: (date, close) -> ret, rv, regime. No file I/O.

    Default (point_in_time=False) is the rule that produced the committed
    regime_labels.json and is kept bit-identical. It has TWO look-ahead leaks
    (docs/literature_audit_2026-09-28.md sec 3.13):
      1. rv on day t is the std of returns up to and INCLUDING day t's close-to-close
         return, and the label is the policy's input DURING day t;
      2. without --trailing, the threshold is the median over the whole period,
         so later days' volatility sets earlier days' labels.
    point_in_time=True removes both: day t is labelled from rv through day t-1
    (rv.shift(1)), thresholded at the median of lagged rv known at t (expanding,
    or rolling over `trailing` days). The label for day t then depends on no
    close later than day t-1.
    """
    df = df[["date", "close"]].sort_values("date").reset_index(drop=True).copy()
    df["ret"] = np.log(df["close"]).diff()
    df["rv"] = df["ret"].rolling(window).std()

    if point_in_time:
        rv_used = df["rv"].shift(1)
        base = (rv_used.rolling(trailing, min_periods=window) if trailing
                else rv_used.expanding(min_periods=window))
        thr = base.median()
        df["regime"] = (rv_used > thr).astype("float")
        df.loc[rv_used.isna() | thr.isna(), "regime"] = 0.0
    elif trailing:
        # Trailing threshold on SAME-DAY rv (leak 1 remains; see docstring).
        thr = df["rv"].rolling(trailing, min_periods=window).median()
        df["regime"] = (df["rv"] > thr).astype("float")
    else:
        # Single-year fallback: in-sample median of the whole RV series (leaks 1 and 2).
        thr = df["rv"].median()
        df["regime"] = (df["rv"] > thr).astype("float")

    df.loc[df["rv"].isna(), "regime"] = 0.0   # warm-up days default to low-vol
    df["regime"] = df["regime"].fillna(0.0).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data")
    ap.add_argument("--stock", default="AMZN")
    ap.add_argument("--period", default="2022")
    ap.add_argument("--window", type=int, default=20, help="rolling RV window (trading days)")
    ap.add_argument("--trailing", type=int, default=None,
                    help="trailing-median window (e.g. 252); omit for in-sample median")
    ap.add_argument("--point-in-time", action="store_true",
                    help="label day t from rv through day t-1 against a median of past rv "
                         "only (no look-ahead); default off = the rule behind the committed "
                         "regime_labels.json")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    df = build_regime_labels(args.data_path, args.stock, args.period, args.window,
                             args.trailing, args.point_in_time)

    out = Path(args.out)
    labels = {r.date: int(r.regime) for r in df.itertuples()}
    (out / "regime_labels.json").write_text(json.dumps(labels, indent=2))
    df.to_csv(out / "regime_labels.csv", index=False)

    n = len(df)
    n_high = int(df["regime"].sum())
    print(f"{args.stock} {args.period}: {n} days, "
          f"{n_high} high-vol ({100*n_high/n:.0f}%), {n - n_high} low-vol")
    print(f"  RV median threshold: {df['rv'].median():.5f}   "
          f"(window={args.window}, trailing={args.trailing or 'in-sample'}, "
          f"point_in_time={args.point_in_time})")
    print(f"  wrote {out/'regime_labels.json'} and {out/'regime_labels.csv'}")


if __name__ == "__main__":
    main()
