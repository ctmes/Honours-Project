"""Download AMZN MBO data from DataBento as compressed DBN.

Parameterised by year so the same script pulls 2024 (primary), 2023 (regime
trailing-median warm-up), or 2022. Prints DataBento's cost estimate and asks for
confirmation before the (paid) download unless --yes is passed.

Usage:
  python download_data.py --year 2024                 # full 2024 trading year
  python download_data.py --year 2023                 # warm-up year for trailing-252 regime median
  python download_data.py --start 2024-01-02 --end 2024-12-31 --symbol AMZN
  python download_data.py --year 2024 --estimate-only # cost/size estimate, no download
"""
import argparse
from pathlib import Path

import databento as db

from db_key import get_databento_key

# First session .. day AFTER the last session per year. DataBento get_range `end` is
# EXCLUSIVE, so to include the final trading day the end must be the next calendar day
# (else you silently lose Dec 31, as happened on the first 2024 pull -> 251/252 days).
YEAR_RANGES = {
    "2022": ("2022-01-03", "2022-12-31"),  # last session 2022-12-30
    "2023": ("2023-01-03", "2023-12-30"),  # last session 2023-12-29
    "2024": ("2024-01-02", "2025-01-01"),  # last session 2024-12-31
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", choices=sorted(YEAR_RANGES), help="convenience preset for start/end")
    ap.add_argument("--start", help="ISO date, overrides --year")
    ap.add_argument("--end", help="ISO date, overrides --year")
    ap.add_argument("--symbol", default="AMZN")
    ap.add_argument("--dataset", default="XNAS.ITCH")
    ap.add_argument("--schema", default="mbo")
    ap.add_argument("--estimate-only", action="store_true", help="print cost/size, do not download")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    if args.start and args.end:
        start, end = args.start, args.end
        year_tag = start[:4]
    elif args.year:
        start, end = YEAR_RANGES[args.year]
        year_tag = args.year
    else:
        ap.error("provide --year or both --start and --end")

    client = db.Historical(key=get_databento_key())

    # Enterprise rigor: estimate the (paid) query cost + size BEFORE pulling.
    query = dict(dataset=args.dataset, symbols=[args.symbol], schema=args.schema,
                 stype_in="raw_symbol", start=start, end=end)
    cost = client.metadata.get_cost(**query)
    size = client.metadata.get_billable_size(**query)
    print(f"{args.symbol} {args.schema} {start}..{end}  "
          f"est. cost=${cost:.2f}  billable={size/1e9:.2f} GB")
    if args.estimate_only:
        return
    if not args.yes:
        if input("proceed with paid download? [y/N] ").strip().lower() != "y":
            print("aborted.")
            return

    output_dir = Path(f"data/databento/{args.symbol}/{year_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Full-year presets use the year in the filename; custom ranges use the range,
    # so a partial pull (e.g. a single missing day) never overwrites a year file.
    file_tag = year_tag if args.year and not (args.start or args.end) else f"{start}_to_{end}"
    out_path = output_dir / f"{args.symbol}_{file_tag}_{args.schema}.dbn.zst"

    client.timeseries.get_range(**query, path=out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
