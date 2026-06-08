"""
Data-QA: find which day-files the LOBSTER loader silently drops.

The loader (lobster_loader.read_pair) wraps per-file processing in a bare
`except Exception: return None`, so days whose data fails processing vanish with
only a console warning. This replicates that exact per-file path one day at a time
(low memory — no full-year concatenation) and reports every day that fails, with the
exception, so the underlying data/conversion problem can be fixed.

Usage:
  python find_dropped_days.py --stock AMZN --period 2022
"""
from __future__ import annotations

import argparse
import os
import traceback

import pandas as pd

from gymnax_exchange.jaxlobster.lobster_loader import LoadLOBSTER_resample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data")
    ap.add_argument("--atpath", default=".")
    ap.add_argument("--stock", default="AMZN")
    ap.add_argument("--period", default="2022")
    # Must match the env's loader construction (base_env.py) so the drop set matches.
    ap.add_argument("--book-depth", type=int, default=10)
    ap.add_argument("--ep-type", default="fixed_steps")
    ap.add_argument("--window-length", type=int, default=6400)
    ap.add_argument("--n-data-msg-per-step", type=int, default=100)
    ap.add_argument("--window-resolution", type=int, default=6400)
    ap.add_argument("--day-start", type=int, default=34200)
    ap.add_argument("--day-end", type=int, default=57600)
    args = ap.parse_args()

    loader = LoadLOBSTER_resample(
        args.data_path, args.atpath, args.book_depth, args.ep_type,
        window_length=args.window_length,
        n_data_msg_per_step=args.n_data_msg_per_step,
        window_resolution=args.window_resolution,
        day_start=args.day_start, day_end=args.day_end,
        stock=args.stock, time_period=args.period,
    )

    msg_files = list(loader.message_files)
    book_files = list(loader.book_files)
    print(f"{args.stock} {args.period}: {len(msg_files)} day-files discovered\n")

    ok, failed = [], []
    for mfile, bfile in zip(msg_files, book_files):
        date = os.path.basename(mfile).split("_")[1]
        try:
            dfm = pd.read_csv(mfile, usecols=range(6), header=None, engine="c",
                              low_memory=True, na_filter=False, skip_blank_lines=True)
            dfb = pd.read_csv(bfile, header=None, engine="c",
                              low_memory=True, na_filter=False, skip_blank_lines=True)
            if dfm.empty or dfb.empty:
                raise ValueError(f"empty file (msg_rows={len(dfm)}, book_rows={len(dfb)})")
            msg, book = loader._pre_process_msg_ob(dfm, dfb)
            message_day, index_s, index_e, init_OBs = loader._get_inits_day(msg, book)
            ok.append((date, len(index_s)))   # windows produced this day
        except Exception as e:
            failed.append((date, f"{type(e).__name__}: {e}"))
            print(f"  [FAIL] {date}: {type(e).__name__}: {e}")

    print(f"\n=== summary: {len(ok)} OK, {len(failed)} FAILED (of {len(msg_files)}) ===")
    if failed:
        print("failed dates:", ", ".join(d for d, _ in failed))
        # group by exception type
        from collections import Counter
        kinds = Counter(msg.split(":")[0] for _, msg in failed)
        print("by exception type:", dict(kinds))


if __name__ == "__main__":
    main()
