"""Download Databento XNAS.ITCH MBO for the shallow-book beta calibration.

Pre-registered: preregistration.json -> amendments -> shallow_book_beta_amendment_2026-10-09.
One file per stock per day (the same 16 days used for the AMZN calibration):

    data/databento/<TICKER>/2024_calib/<TICKER>_<date>_mbo.dbn.zst

Existing files are skipped, so a rerun never pays twice. The API key is read from the
DATABENTO_API_KEY environment variable, or from a DATABENTO_* line in .env.

    python download_calib_mbo.py                 # BYND, ZM, INTC, NFLX
    python download_calib_mbo.py --symbols LYFT  # any others
    python download_calib_mbo.py --quote         # cost only, downloads nothing
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import time
from pathlib import Path

DAYS = ["2024-01-02", "2024-01-19", "2024-02-06", "2024-02-23", "2024-03-12", "2024-03-28",
        "2024-04-16", "2024-05-02", "2024-05-20", "2024-06-06", "2024-06-25", "2024-07-12",
        "2024-07-30", "2024-08-15", "2024-09-03", "2024-09-19"]
ROOT = Path(__file__).resolve().parent


def api_key() -> str:
    if os.environ.get("DATABENTO_API_KEY"):
        return os.environ["DATABENTO_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(?:export\s+)?[A-Za-z_]*DATABENTO[A-Za-z_]*\s*=\s*['\"]?([^'\"\s#]+)", line)
            if m:
                return m.group(1)
    raise SystemExit("no Databento API key: set DATABENTO_API_KEY or add it to .env")


def retry(fn, **kw):
    for attempt in range(5):
        try:
            return fn(**kw)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--symbols", default="BYND,ZM,INTC,NFLX")
    ap.add_argument("--quote", action="store_true", help="print the cost and download nothing")
    args = ap.parse_args()

    import databento as db
    client = db.Historical(api_key())
    total = 0.0
    for sym in args.symbols.split(","):
        out_dir = ROOT / "data" / "databento" / sym / "2024_calib"
        out_dir.mkdir(parents=True, exist_ok=True)
        for day in DAYS:
            path = out_dir / f"{sym}_{day}_mbo.dbn.zst"
            end = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
            kw = dict(dataset="XNAS.ITCH", symbols=[sym], schema="mbo", start=day, end=end)
            if path.exists() and path.stat().st_size > 0:
                print(f"skip  {path.name} (exists)")
                continue
            cost = retry(client.metadata.get_cost, **kw)
            total += cost
            if args.quote:
                print(f"quote {sym} {day} ${cost:.3f}")
                continue
            tmp = path.with_suffix(".part")
            retry(client.timeseries.get_range, path=str(tmp), **kw)
            tmp.replace(path)
            print(f"got   {path.name} ${cost:.3f} {path.stat().st_size / 1e6:.1f} MB", flush=True)
    print(f"{'QUOTED' if args.quote else 'SPENT'} USD: {total:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
