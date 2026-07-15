"""
Build the loader cache (saved_npz/*.npz) and reset-state pickle
(pre_reset_states/*.pkl) for a data period, without launching training.

Constructing BaseLOBEnv is exactly what MARLEnv does at training start
(marl_env.py builds BaseLOBEnv(cfg=world_config)), so the artifacts produced
here are byte-identical in naming and content to what a training run would
create on first touch. Building them ahead of time means:
  - day-drop QA happens NOW, not silently inside a cluster job;
  - the cluster only needs the ~GB-scale npz + pkl, not the raw CSVs.

Any day the loader drops is reported by its own "✗ Error processing" lines;
this script re-counts days from the built cache (time-column resets, same
trick as build_window_to_date.py) and compares against the day-files in the
period directory, failing loudly on mismatch so a silent drop can't reach
window_to_date construction or training.

Usage:
  python build_period_cache.py --period 2024_test
  python build_period_cache.py --period 2024_train
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="D:/UWA/Honours/Honours-Project/data")
    ap.add_argument("--atpath", default="D:/UWA/Honours/Honours-Project")
    ap.add_argument("--stock", default="AMZN")
    ap.add_argument("--period", required=True)
    # Loader params — must match the production world_config or the cache
    # filename suffix will not be the one training looks for.
    ap.add_argument("--book-depth", type=int, default=10)
    ap.add_argument("--ep-type", default="fixed_steps")
    ap.add_argument("--episode-time", type=int, default=6400)
    ap.add_argument("--start-resolution", type=int, default=6400)
    ap.add_argument("--n-data-msg-per-step", type=int, default=100)
    ap.add_argument("--day-start", type=int, default=34200)
    ap.add_argument("--day-end", type=int, default=57600)
    args = ap.parse_args()

    from gymnax_exchange.jaxob.jaxob_config import World_EnvironmentConfig
    from gymnax_exchange.jaxen.base_env import BaseLOBEnv
    import jax

    cfg = World_EnvironmentConfig(
        alphatradePath=args.atpath,
        dataPath=args.data_path,
        stock=args.stock,
        timePeriod=args.period,
        book_depth=args.book_depth,
        ep_type=args.ep_type,
        episode_time=args.episode_time,
        start_resolution=args.start_resolution,
        n_data_msg_per_step=args.n_data_msg_per_step,
        day_start=args.day_start,
        day_end=args.day_end,
        use_pickles_for_init=True,
    )

    env = BaseLOBEnv(cfg=cfg, key=jax.random.PRNGKey(0))

    # ---- Day-count QA: cache days (time resets) vs directory day files ----
    t = np.asarray(env.messages[:, -2])
    n_days_cache = int(np.sum(np.diff(t) < 0)) + 1
    day_dir = Path(args.data_path) / "rawLOBSTER" / args.stock / args.period
    dates = sorted(f.name.split("_")[1]
                   for f in day_dir.glob(f"{args.stock}_*_message_*.csv"))
    print(f"\n=== build summary: {args.stock} {args.period} ===")
    print(f"windows: {env.n_windows}")
    print(f"messages: {env.messages.shape[0]:,}")
    print(f"days in cache: {n_days_cache}  |  day files in {day_dir}: {len(dates)}")
    if n_days_cache != len(dates):
        print("MISMATCH: the loader dropped day(s). Check the '✗ Error processing'"
              " lines above, remove those day files from the period directory"
              " (the cache already excludes them), then window_to_date will align.")
        sys.exit(1)
    print("OK: no dropped days — cache and directory are aligned.")


if __name__ == "__main__":
    main()
