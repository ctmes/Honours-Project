"""
Slice a date-range sub-period cache out of an existing master npz cache —
no CSV reprocessing.

Why: the master 2024_train cache (Jan-Sep, 694.5M msgs) is ~22 GB on-device at
int32 and cannot fit a 16 GB V100, so training runs use a sub-period (e.g. Q3).
Days are stored chronologically and windows never span days, so a date range is
a contiguous row range of the master msgs array plus the subset of windows whose
[start, end) falls inside it, with offsets rebased to the slice.

Day boundaries are recovered from downward resets in the message time column
(same trick as build_window_to_date.py) and mapped to dates via the sorted
day-file names of the MASTER period directory. The slice gets its own period
directory of hardlinked day files (create it first — see --help epilog) so
build_period_cache.py (reset pickle + QA) and build_window_to_date.py work on
the sub-period unchanged.

Usage (local, dates from the raw day-file directory):
  python slice_period_cache.py --master-period 2024_train --period 2024_q3 \
      --from-date 2024-07-01 --to-date 2024-09-30

Usage (on the cluster, where no raw CSVs exist — dates from the committed
window map of the master period):
  python slice_period_cache.py --master-period 2024_train --period 2024_q3 \
      --from-date 2024-07-01 --to-date 2024-09-30 \
      --dates-from-window-map window_to_date_2024_train.json \
      --data-path /group/pmc097/cmelville/Honours-Project/data \
      --atpath /group/pmc097/cmelville/Honours-Project
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="D:/UWA/Honours/Honours-Project/data")
    ap.add_argument("--atpath", default="D:/UWA/Honours/Honours-Project")
    ap.add_argument("--stock", default="AMZN")
    ap.add_argument("--master-period", required=True)
    ap.add_argument("--period", required=True, help="name of the new sub-period")
    ap.add_argument("--from-date", required=True)  # inclusive, YYYY-MM-DD
    ap.add_argument("--to-date", required=True)    # inclusive, YYYY-MM-DD
    ap.add_argument("--suffix-tail",
                    default="10_fixed_steps_6400_6400_100_34200_57600",
                    help="loader-config part of the cache filename")
    ap.add_argument("--dates-from-window-map", default=None,
                    help="path to the MASTER period's window_to_date json; use when "
                         "the raw day-file directory is not available (cluster)")
    args = ap.parse_args()

    npz_dir = Path(args.atpath) / "saved_npz"
    master_path = npz_dir / (f"loaded_lobster_LoadLOBSTER_resample_"
                             f"{args.stock}_{args.master_period}_{args.suffix_tail}.npz")
    out_path = npz_dir / (f"loaded_lobster_LoadLOBSTER_resample_"
                          f"{args.stock}_{args.period}_{args.suffix_tail}.npz")
    if out_path.exists():
        raise SystemExit(f"already exists: {out_path}")

    print(f"loading master cache {master_path.name} ...", flush=True)
    data = np.load(master_path, allow_pickle=True)
    msgs = data["msgs"]
    starts, ends = data["starts"], data["ends"]
    obs, maxm = data["obs"], data["max_msgs_in_windows_arr"]

    t = msgs[:, -2]
    day_start_rows = np.concatenate([[0], np.where(np.diff(t) < 0)[0] + 1])

    # Master day dates, in day order. Two sources:
    #   default — the master period's raw day-file directory (local machine);
    #   --dates-from-window-map — the master's committed window_to_date json
    #     (cluster, where raw CSVs were never uploaded).
    if args.dates_from_window_map:
        with open(args.dates_from_window_map) as f:
            wmap = json.load(f)
        if len(wmap) != len(starts):
            raise SystemExit(f"window map has {len(wmap)} windows but master cache "
                             f"has {len(starts)} — wrong map for this cache")
        # date of the day each window starts in -> unique day-ordered date list
        day_of_window = np.searchsorted(day_start_rows, starts, side="right") - 1
        dates_arr = [None] * len(day_start_rows)
        for w, d in enumerate(day_of_window):
            dates_arr[int(d)] = wmap[str(w)]
        if any(d is None for d in dates_arr):
            missing = [i for i, d in enumerate(dates_arr) if d is None]
            raise SystemExit(f"days without any window (cannot date them): {missing}")
        dates = dates_arr
    else:
        master_dir = Path(args.data_path) / "rawLOBSTER" / args.stock / args.master_period
        dates = sorted(f.name.split("_")[1]
                       for f in master_dir.glob(f"{args.stock}_*_message_*.csv"))
        if not dates:
            raise SystemExit(f"no day files under {master_dir} — on the cluster, "
                             f"pass --dates-from-window-map instead")
        if len(day_start_rows) != len(dates):
            raise SystemExit(f"master cache has {len(day_start_rows)} days but "
                             f"{master_dir} has {len(dates)} day files — mismatch")

    in_range = [i for i, d in enumerate(dates)
                if args.from_date <= d <= args.to_date]
    if not in_range:
        raise SystemExit("no days in the requested date range")
    i0, i1 = in_range[0], in_range[-1]
    row_start = int(day_start_rows[i0])
    row_end = int(day_start_rows[i1 + 1]) if i1 + 1 < len(day_start_rows) else msgs.shape[0]

    wmask = (starts >= row_start) & (ends <= row_end)
    print(f"slicing {dates[i0]}..{dates[i1]}: rows [{row_start:,}, {row_end:,}) "
          f"= {row_end - row_start:,} msgs, {int(wmask.sum())} of {len(starts)} windows",
          flush=True)

    np.savez_compressed(
        out_path,
        msgs=msgs[row_start:row_end],
        starts=starts[wmask] - row_start,
        ends=ends[wmask] - row_start,
        obs=obs[wmask],
        max_msgs_in_windows_arr=maxm[wmask],
    )
    print(f"saved {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")
    print(f"days: {len(in_range)}  windows: {int(wmask.sum())}  "
          f"messages: {row_end - row_start:,}")
    print("next: create the hardlink day dir for this period, then run "
          "build_period_cache.py (reset pickle + QA) and build_window_to_date.py.")


if __name__ == "__main__":
    main()
