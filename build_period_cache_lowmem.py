"""
Low-memory builder for the loader cache (saved_npz/*.npz) + reset pickle.

Why this exists: LoadLOBSTER_resample.run_loading holds every day's message
array in RAM and then np.concatenate's them — for AMZN 2024 that is int64 and
peaks at ~2x the year's footprint (>100 GB for the 188-day train split), which
does not fit on this machine. This builder processes days SEQUENTIALLY with the
loader's own per-day methods (_pre_process_msg_ob, _get_inits_day — no
reimplementation of parsing logic), streams messages to a raw int32 file, and
then writes the standard npz from a memmap, so peak RAM is one day's dataframes.

int32 note: JAX runs with x64 disabled, so jnp.asarray(messages) already
downcasts int64 -> int32 on device; an int32 cache is therefore bit-identical
downstream while halving host RAM and GPU residency. Every day's columns are
asserted within int32 range — order ids reset daily and observed maxima are
~0.86e9 vs the 2.15e9 cap — so an overflow fails the build loudly instead of
wrapping silently.

After writing the cache, BaseLOBEnv is constructed once (it now hits the cache)
to generate the pre_reset_states pickle, and the same day-count QA as
build_period_cache.py runs: days detected from time-column resets must equal
the day files in the period directory.

Resumable: per-day metadata (starts/ends/obs/max) is checkpointed to a sidecar
.meta.npz after every day, and the raw .bin is truncated back to the last
complete day on restart — so a sleep/reboot mid-build (which killed the first
188-day attempt at day 135 with metadata only in RAM) costs at most one day of
work. Re-run the same command to resume. While running, the process also asks
Windows not to system-sleep (SetThreadExecutionState; display may still sleep,
and a manual shutdown still interrupts — hence the resume support).

Usage:
  python build_period_cache_lowmem.py --period 2024_train
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

INT32_MIN, INT32_MAX = np.iinfo(np.int32).min, np.iinfo(np.int32).max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="D:/UWA/Honours/Honours-Project/data")
    ap.add_argument("--atpath", default="D:/UWA/Honours/Honours-Project")
    ap.add_argument("--stock", default="AMZN")
    ap.add_argument("--period", required=True)
    ap.add_argument("--book-depth", type=int, default=10)
    ap.add_argument("--ep-type", default="fixed_steps")
    ap.add_argument("--episode-time", type=int, default=6400)
    ap.add_argument("--start-resolution", type=int, default=6400)
    ap.add_argument("--n-data-msg-per-step", type=int, default=100)
    ap.add_argument("--day-start", type=int, default=34200)
    ap.add_argument("--day-end", type=int, default=57600)
    args = ap.parse_args()

    from gymnax_exchange.jaxlobster.lobster_loader import LoadLOBSTER_resample

    loader = LoadLOBSTER_resample(
        args.data_path, args.atpath, args.book_depth, args.ep_type,
        window_length=args.episode_time,
        n_data_msg_per_step=args.n_data_msg_per_step,
        window_resolution=args.start_resolution,
        day_start=args.day_start, day_end=args.day_end,
        stock=args.stock, time_period=args.period,
    )
    save_path = loader._get_save_filename(
        f"{args.stock}_{args.period}_{args.book_depth}_{args.ep_type}_"
        f"{args.episode_time}_{args.start_resolution}_{args.n_data_msg_per_step}_"
        f"{args.day_start}_{args.day_end}"
    )
    if os.path.exists(save_path):
        print(f"cache already exists: {save_path}")
        sys.exit(0)

    # Keep the SYSTEM awake for the lifetime of this process (reverts on exit).
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED = 0x80000000 | 0x00000001.
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    raw_bin = save_path + ".msgs.int32.bin"
    meta_path = raw_bin + ".meta.npz"
    n_total = 0
    n_days_done = 0
    day_rows = []
    starts_all, ends_all, obs_all, maxm_all = [], [], [], []

    # ---- Resume from a previous interrupted run --------------------------
    if os.path.exists(raw_bin) and os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True)
        day_rows = list(meta["day_rows"])          # rows appended per completed day
        starts_all = list(meta["starts_all"])
        ends_all = list(meta["ends_all"])
        obs_all = list(meta["obs_all"])
        maxm_all = list(meta["maxm_all"])
        n_days_done = len(day_rows)
        n_total = int(sum(day_rows))
        expected = n_total * 8 * 4                 # rows x 8 cols x int32
        actual = os.path.getsize(raw_bin)
        if actual < expected:
            raise RuntimeError(f"raw bin smaller than metadata claims "
                               f"({actual} < {expected}) — delete both and restart")
        if actual > expected:
            print(f"truncating partial day: {actual} -> {expected} bytes")
            with open(raw_bin, "r+b") as f:
                f.truncate(expected)
        print(f"RESUMING after {n_days_done} completed days ({n_total:,} msgs)")
    elif os.path.exists(raw_bin):
        print("stale .bin without metadata sidecar (pre-resume-support run) — "
              "restarting from scratch")
        os.remove(raw_bin)

    def _checkpoint_meta(day_rows):
        np.savez(meta_path,
                 day_rows=np.asarray(day_rows, dtype=np.int64),
                 starts_all=np.array(starts_all, dtype=object),
                 ends_all=np.array(ends_all, dtype=object),
                 obs_all=np.array(obs_all, dtype=object),
                 maxm_all=np.array(maxm_all, dtype=object))

    t0 = time.time()
    with open(raw_bin, "ab") as out:
        for i, (mfile, bfile) in enumerate(zip(loader.message_files, loader.book_files)):
            if i < n_days_done:
                continue
            date = os.path.basename(mfile).split("_")[1]
            t1 = time.time()
            dfm = pd.read_csv(mfile, usecols=range(6), header=None, engine="c",
                              low_memory=True, na_filter=False, skip_blank_lines=True)
            dfb = pd.read_csv(bfile, header=None, engine="c",
                              low_memory=True, na_filter=False, skip_blank_lines=True)
            if dfb.shape[1] != loader.n_Levels * 4:
                raise ValueError(f"{date}: orderbook has {dfb.shape[1]} cols, "
                                 f"expected {loader.n_Levels * 4}")
            msg, book = loader._pre_process_msg_ob(dfm, dfb)
            message_day, index_s, index_e, init_OBs = loader._get_inits_day(msg, book)
            del dfm, dfb, msg, book

            lo, hi = message_day.min(), message_day.max()
            if lo < INT32_MIN or hi > INT32_MAX:
                raise OverflowError(f"{date}: message values [{lo}, {hi}] exceed int32")
            message_day = message_day.astype(np.int32)

            index_s = np.asarray(index_s, dtype=np.int64) + n_total
            index_e = np.asarray(index_e, dtype=np.int64) + n_total
            starts_all.append(index_s)
            ends_all.append(index_e)
            maxm_all.append(index_e - index_s)
            obs_all.append(np.asarray(init_OBs, dtype=np.int32))

            message_day.tofile(out)
            out.flush()
            n_total += message_day.shape[0]
            day_rows.append(int(message_day.shape[0]))
            _checkpoint_meta(day_rows)
            print(f"[{i + 1}/{len(loader.message_files)}] {date}: "
                  f"{message_day.shape[0]:,} msgs, {len(index_s)} windows, "
                  f"{time.time() - t1:.1f}s (total {n_total:,})", flush=True)
            del message_day

    starts = np.concatenate(starts_all)
    ends = np.concatenate(ends_all)
    maxm = np.concatenate(maxm_all)
    obs = np.concatenate(obs_all, axis=0)

    print(f"\nstreaming {n_total:,} messages into compressed npz "
          f"({time.time() - t0:.0f}s so far)...", flush=True)
    msgs_mm = np.memmap(raw_bin, dtype=np.int32, mode="r", shape=(n_total, 8))
    np.savez_compressed(save_path, msgs=msgs_mm, starts=starts, ends=ends,
                        obs=obs, max_msgs_in_windows_arr=maxm)
    del msgs_mm
    os.remove(raw_bin)
    if os.path.exists(meta_path):
        os.remove(meta_path)
    print(f"saved {save_path} ({os.path.getsize(save_path) / 1e9:.2f} GB)")

    # ---- Reset pickle + day-count QA (cache hit -> no reprocessing) ----
    from gymnax_exchange.jaxob.jaxob_config import World_EnvironmentConfig
    from gymnax_exchange.jaxen.base_env import BaseLOBEnv
    import jax
    cfg = World_EnvironmentConfig(
        alphatradePath=args.atpath, dataPath=args.data_path,
        stock=args.stock, timePeriod=args.period,
        book_depth=args.book_depth, ep_type=args.ep_type,
        episode_time=args.episode_time, start_resolution=args.start_resolution,
        n_data_msg_per_step=args.n_data_msg_per_step,
        day_start=args.day_start, day_end=args.day_end,
        use_pickles_for_init=True,
    )
    env = BaseLOBEnv(cfg=cfg, key=jax.random.PRNGKey(0))

    t = np.asarray(env.messages[:, -2])
    n_days_cache = int(np.sum(np.diff(t) < 0)) + 1
    day_dir = Path(args.data_path) / "rawLOBSTER" / args.stock / args.period
    n_files = len(list(day_dir.glob(f"{args.stock}_*_message_*.csv")))
    print(f"\n=== build summary: {args.stock} {args.period} ===")
    print(f"windows: {env.n_windows}  messages: {n_total:,}")
    print(f"days in cache: {n_days_cache}  |  day files: {n_files}")
    if n_days_cache != n_files:
        print("MISMATCH: dropped day(s) — see per-day lines above.")
        sys.exit(1)
    print("OK: no dropped days — cache and directory are aligned.")


if __name__ == "__main__":
    main()
