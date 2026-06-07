#!/usr/bin/env python3
"""
Convert Databento MBO DBN file to LOBSTER-format CSV files.

Streams records one-by-one (avoids loading 9.5 GB into RAM).

Input:  data/databento/AMZN/2022/AMZN_2022_mbo.dbn.zst
Output: data/rawLOBSTER/AMZN/2022/AMZN_{date}_34200000_57600000_{message|orderbook}_10.csv

Usage:
    python convert_dbn_to_lobster.py              # full conversion
    python convert_dbn_to_lobster.py --sample     # show first 20 records, exit
    python convert_dbn_to_lobster.py --days 3     # convert first 3 trading days only
    python convert_dbn_to_lobster.py --verify     # run JAX cross-check after conversion
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sortedcontainers import SortedList

N_LEVELS    = 10
DAY_START_S = 34200   # 09:30:00 Eastern
DAY_END_S   = 57600   # 16:00:00 Eastern

# Actions to ignore
SKIP_ACTIONS = frozenset({'T', 'R', 'N'})   # T=aggressor trade (dup of E/F), R=reset, N=none

ACTION_TO_LOBSTER_TYPE = {
    'A': 1,  # Add         → New limit order
    'C': 2,  # Cancel      → Partial cancellation
    'D': 3,  # Delete      → Full deletion
    'E': 4,  # Execute     → Visible execution (passive, partial)
    'F': 4,  # Fill        → Visible execution (passive, complete)
}

SIDE_TO_DIR = {'B': 1, 'A': -1}   # Bid=buy=1, Ask=sell=-1

# Price units: Databento int64 in 1e-9 USD; LOBSTER int in 1e-4 USD
PRICE_SCALE = 100_000      # lobster_price = databento_price // PRICE_SCALE
DATABENTO_NAN_PRICE = 9223372036854775807   # INT64_MAX = invalid price


# ---------------------------------------------------------------------------
# Efficient timestamp conversion (cache eastern midnight per UTC day)
# ---------------------------------------------------------------------------

_midnight_cache: dict = {}   # utc_day_int → (eastern_midnight_ns, eastern_date)


def _eastern_info(ts_ns: int):
    """Return (time_s_eastern, time_ns_sub, eastern_date) for ts_ns (UTC ns since epoch)."""
    utc_day = ts_ns // 86_400_000_000_000
    if utc_day not in _midnight_cache:
        ts    = pd.Timestamp(ts_ns, unit='ns', tz='UTC').tz_convert('US/Eastern')
        mid   = ts.normalize()
        _midnight_cache[utc_day] = (int(mid.value), mid.date())
    east_mid_ns, east_date = _midnight_cache[utc_day]
    delta_ns = ts_ns - east_mid_ns
    return delta_ns // 1_000_000_000, delta_ns % 1_000_000_000, east_date


# ---------------------------------------------------------------------------
# Python order book (accurate historical MBO replay)
# ---------------------------------------------------------------------------

class PythonOrderBook:
    """
    Dict-based order book with SortedList price levels for O(log k) updates
    and O(N_LEVELS) snapshots (vs O(k log k) with plain dict + sorted()).
    """

    def __init__(self, n_levels: int = N_LEVELS):
        self.n_levels = n_levels
        self.orders: dict = {}    # order_id → [price, qty, side]
        self.bid_qty: dict = {}   # price → total qty on bid side
        self.ask_qty: dict = {}   # price → total qty on ask side
        # SortedList maintains insertion order efficiently
        self.bid_prices = SortedList(key=lambda x: -x)  # highest first
        self.ask_prices = SortedList()                   # lowest first
        self._l2_buf = np.zeros(4 * n_levels, dtype=np.int64)

    def reset(self):
        self.orders.clear()
        self.bid_qty.clear()
        self.ask_qty.clear()
        self.bid_prices.clear()
        self.ask_prices.clear()

    def _add_to_side(self, price: int, qty: int, side: int):
        if side == 1:
            if price not in self.bid_qty:
                self.bid_prices.add(price)
            self.bid_qty[price] = self.bid_qty.get(price, 0) + qty
        else:
            if price not in self.ask_qty:
                self.ask_prices.add(price)
            self.ask_qty[price] = self.ask_qty.get(price, 0) + qty

    def _remove_from_side(self, price: int, qty: int, side: int):
        if side == 1:
            cur = self.bid_qty.get(price)
            if cur is None:
                return  # Price level unknown (hidden/pre-existing order)
            new_qty = cur - qty
            if new_qty <= 0:
                del self.bid_qty[price]
                try:
                    self.bid_prices.remove(price)
                except ValueError:
                    pass
            else:
                self.bid_qty[price] = new_qty
        else:
            cur = self.ask_qty.get(price)
            if cur is None:
                return
            new_qty = cur - qty
            if new_qty <= 0:
                del self.ask_qty[price]
                try:
                    self.ask_prices.remove(price)
                except ValueError:
                    pass
            else:
                self.ask_qty[price] = new_qty

    def add(self, order_id: int, price: int, qty: int, side: int):
        if order_id in self.orders:
            old_price, old_qty, old_side = self.orders[order_id]
            self._remove_from_side(old_price, old_qty, old_side)
        self.orders[order_id] = [price, qty, side]
        self._add_to_side(price, qty, side)

    def _reduce(self, order_id: int, qty: int):
        if order_id not in self.orders:
            return
        price, cur_qty, side = self.orders[order_id]
        self._remove_from_side(price, qty, side)
        remaining = cur_qty - qty
        if remaining <= 0:
            del self.orders[order_id]
        else:
            self.orders[order_id][1] = remaining

    def cancel(self, order_id: int, qty: int):
        self._reduce(order_id, qty)

    def delete(self, order_id: int):
        if order_id in self.orders:
            self._reduce(order_id, self.orders[order_id][1])

    def execute(self, order_id: int, qty: int):
        self._reduce(order_id, qty)

    def apply(self, action: str, order_id: int, price: int, qty: int, side: int):
        if action == 'A':
            self.add(order_id, price, qty, side)
        elif action == 'C':
            self.cancel(order_id, qty)
        elif action == 'D':
            self.delete(order_id)
        elif action in ('E', 'F'):
            self.execute(order_id, qty)

    def l2_snapshot(self) -> np.ndarray:
        """Return (4*n_levels,) int64: [ask_p1,ask_s1,bid_p1,bid_s1,...]. O(N_LEVELS)."""
        buf = self._l2_buf
        buf[:] = 0
        ask_sl = self.ask_prices
        bid_sl = self.bid_prices
        n = self.n_levels
        for i in range(min(n, len(ask_sl))):
            p = ask_sl[i]
            buf[4 * i]     = p
            buf[4 * i + 1] = self.ask_qty[p]
        for i in range(min(n, len(bid_sl))):
            p = bid_sl[i]
            buf[4 * i + 2] = p
            buf[4 * i + 3] = self.bid_qty[p]
        return buf.copy()


# ---------------------------------------------------------------------------
# Write one day's data
# ---------------------------------------------------------------------------

def write_day(date_str: str, msg_rows: list, ob_rows: list, out_dir: Path):
    if not msg_rows:
        return
    msg_arr = np.array(msg_rows)
    ob_arr  = np.array(ob_rows, dtype=np.int64)
    fname   = f"AMZN_{date_str}_34200000_57600000"
    # Message CSV: time as float (9dp), remaining columns as plain integers
    msg_df = pd.DataFrame({
        0: msg_arr[:, 0],
        1: msg_arr[:, 1].astype(np.int64),
        2: msg_arr[:, 2].astype(np.int64),
        3: msg_arr[:, 3].astype(np.int64),
        4: msg_arr[:, 4].astype(np.int64),
        5: msg_arr[:, 5].astype(np.int64),
    })
    msg_df.to_csv(
        out_dir / f"{fname}_message_10.csv",
        header=False, index=False, float_format='%.9f'
    )
    pd.DataFrame(ob_arr).to_csv(
        out_dir / f"{fname}_orderbook_10.csv",
        header=False, index=False
    )


# ---------------------------------------------------------------------------
# Main conversion loop (streaming)
# ---------------------------------------------------------------------------

def convert(store, out_dir: Path, n_days_limit: int = 0):
    ob = PythonOrderBook(N_LEVELS)
    current_date = None
    msg_rows: list = []
    ob_rows:  list = []
    days_written = 0

    total_recs = 0
    try:
        from tqdm import tqdm
        it = tqdm(store, desc="Records", unit="M rec",
                  unit_scale=1e-6, mininterval=5, dynamic_ncols=True)
    except ImportError:
        it = store

    for rec in it:
        action = str(rec.action)
        side   = str(rec.side)

        if action in SKIP_ACTIONS:
            continue
        if side not in ('A', 'B'):
            continue
        if action not in ACTION_TO_LOBSTER_TYPE:
            continue

        price_raw = rec.price
        if price_raw >= DATABENTO_NAN_PRICE or price_raw < 0:
            continue

        ts_ns   = int(rec.ts_event)
        time_s, time_ns, east_date = _eastern_info(ts_ns)

        # Day boundary
        if east_date != current_date:
            if current_date is not None:
                write_day(str(current_date), msg_rows, ob_rows, out_dir)
                days_written += 1
                print(f"  Wrote {str(current_date)}: {len(msg_rows):,} trading-hours msgs")
                if n_days_limit > 0 and days_written >= n_days_limit:
                    break
            current_date = east_date
            msg_rows = []
            ob_rows  = []
            ob.reset()

        price    = int(price_raw) // PRICE_SCALE
        size     = int(rec.size)
        order_id = int(rec.order_id)
        direction = SIDE_TO_DIR[side]

        # Always update the book (needed for accurate pre-open state)
        ob.apply(action, order_id, price, size, direction)

        # Only record trading-hours events in the LOBSTER CSVs
        if DAY_START_S <= time_s <= DAY_END_S:
            time_float = time_s + time_ns / 1e9
            msg_rows.append([time_float, ACTION_TO_LOBSTER_TYPE[action],
                             order_id, size, price, direction])
            ob_rows.append(ob.l2_snapshot())

        total_recs += 1

    # Write final day
    if current_date is not None and msg_rows and (n_days_limit == 0 or days_written < n_days_limit):
        write_day(str(current_date), msg_rows, ob_rows, out_dir)
        days_written += 1
        print(f"  Wrote {str(current_date)}: {len(msg_rows):,} trading-hours msgs")

    print(f"\nTotal records scanned: {total_recs:,}")
    print(f"Trading days written:  {days_written}")


# ---------------------------------------------------------------------------
# JAX cross-check
# ---------------------------------------------------------------------------

def jax_cross_check(out_dir: Path, n_msgs: int = 5000):
    """
    Feed the first n_msgs of the first trading day through the JAX OrderBook
    (with the same LOBSTER-loader preprocessing applied) and compare best
    bid/ask to the last row of our Python-book orderbook CSV.
    """
    import jax.numpy as jnp
    from gymnax_exchange.jaxob.jorderbook import OrderBook
    from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration

    msg_files = sorted(out_dir.glob("*_message_10.csv"))
    ob_files  = sorted(out_dir.glob("*_orderbook_10.csv"))
    if not msg_files:
        print("[cross-check] No CSV files found.")
        return

    print(f"[cross-check] First {n_msgs} msgs of {msg_files[0].name}")

    msg_df = pd.read_csv(msg_files[0], header=None,
                         names=['time', 'type', 'order_id', 'size', 'price', 'direction'])
    ob_df  = pd.read_csv(ob_files[0],  header=None)
    msg_df = msg_df.head(n_msgs)
    ob_df  = ob_df.head(n_msgs)

    # Apply same preprocessing as lobster_loader._pre_process_msg_ob:
    #   type 3 → type 2 (full delete treated as cancel)
    #   type 4 → type 1 with flipped direction (execution = aggressive limit on opposite side)
    msgs = msg_df.copy()
    msgs.loc[msgs['type'] == 3, 'type'] = 2
    exec_mask = msgs['type'] == 4
    msgs.loc[exec_mask, 'direction'] *= -1
    msgs.loc[exec_mask, 'type'] = 1

    time_s_arr  = msgs['time'].astype(int).values
    time_ns_arr = ((msgs['time'] - time_s_arr) * 1e9).astype(int).values

    # JAX message format: [type, dir, qty, price, order_id, trader_id, time_s, time_ns]
    # order_id may overflow int32 — clip to int32 range for JAX
    oid = np.clip(msgs['order_id'].values, -(2**31), 2**31 - 1).astype(np.int32)
    jax_msgs = jnp.array(np.column_stack([
        msgs['type'].values.astype(np.int32),
        msgs['direction'].values.astype(np.int32),
        msgs['size'].values.astype(np.int32),
        msgs['price'].values.astype(np.int32),
        oid, oid,
        time_s_arr.astype(np.int32),
        time_ns_arr.astype(np.int32),
    ]))

    cfg   = JAXLOB_Configuration(nOrders=200_000, nTrades=50_000)
    ob    = OrderBook(cfg)
    state = ob.init()
    state = ob.process_orders_array(state, jax_msgs)

    best_bid_jax = int(ob.get_best_bid(state))
    best_ask_jax = int(ob.get_best_ask(state))

    last_row     = ob_df.iloc[-1].values
    best_ask_py  = int(last_row[0])
    best_bid_py  = int(last_row[2])

    print(f"  Python-book  best_bid={best_bid_py:,}  best_ask={best_ask_py:,}")
    print(f"  JAX-book     best_bid={best_bid_jax:,}  best_ask={best_ask_jax:,}")

    bid_ok = (best_bid_jax == best_bid_py)
    ask_ok = (best_ask_jax == best_ask_py)
    if bid_ok and ask_ok:
        print("  Result: PASS")
    else:
        print(f"  Result: MISMATCH  bid_diff={abs(best_bid_jax-best_bid_py)}  ask_diff={abs(best_ask_jax-best_ask_py)}")
        print("  (Small differences may reflect JAX cancel-mode semantics vs exact replay)")


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def validate_output(out_dir: Path):
    msg_files = sorted(out_dir.glob("*_message_10.csv"))
    ob_files  = sorted(out_dir.glob("*_orderbook_10.csv"))
    print(f"\n=== Validation ===")
    print(f"Message files  : {len(msg_files)}")
    print(f"Orderbook files: {len(ob_files)}")
    if not msg_files:
        return

    sample = list(zip(msg_files[:5], ob_files[:5]))
    errors = 0
    for mf, of in sample:
        msg = pd.read_csv(mf, header=None)
        ob  = pd.read_csv(of, header=None)
        cols_ok   = msg.shape[1] == 6 and ob.shape[1] == 4 * N_LEVELS
        rows_ok   = msg.shape[0] == ob.shape[0]
        head      = ob.head(500)
        crossed   = ((head.iloc[:,0] > 0) & (head.iloc[:,2] > 0) & (head.iloc[:,0] < head.iloc[:,2])).any()
        status = "OK" if (cols_ok and rows_ok and not crossed) else "WARN"
        if status == "WARN":
            errors += 1
        print(f"  {mf.name[:50]}: msg_cols={msg.shape[1]} ob_cols={ob.shape[1]} "
              f"rows_match={rows_ok} crossed={crossed} [{status}]")
    if len(msg_files) > 5:
        print(f"  ... ({len(msg_files)} days total, showing first 5)")
    if errors == 0:
        print("All sampled files look correct.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default='data/databento/AMZN/2022/AMZN_2022_mbo.dbn.zst')
    parser.add_argument('--output', default='data/rawLOBSTER/AMZN/2022')
    parser.add_argument('--sample', action='store_true', help='Print first 20 records and exit')
    parser.add_argument('--verify', action='store_true', help='Run JAX cross-check after conversion')
    parser.add_argument('--days',   type=int, default=0, help='Limit to first N trading days (0=all)')
    args = parser.parse_args()

    import databento as db
    dbn_path = Path(args.input)
    out_dir  = Path(args.output)

    if not dbn_path.exists():
        print(f"Error: {dbn_path} not found.")
        sys.exit(1)

    store = db.DBNStore.from_file(dbn_path)

    if args.sample:
        print(f"Schema: {store.schema}")
        print(f"Symbols: {store.symbols}")
        print(f"\nFirst 20 records:")
        for i, rec in enumerate(store):
            if i >= 20:
                break
            ts_ns = int(rec.ts_event)
            time_s, time_ns, east_date = _eastern_info(ts_ns)
            price_lob = int(rec.price) // PRICE_SCALE if rec.price < DATABENTO_NAN_PRICE else -1
            print(f"  {east_date} {time_s:5d}s action={rec.action} side={rec.side} "
                  f"price={price_lob} size={rec.size} order_id={rec.order_id}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {dbn_path}\n  -> {out_dir}")
    if args.days > 0:
        print(f"  (first {args.days} trading days only)")

    convert(store, out_dir, n_days_limit=args.days)
    validate_output(out_dir)

    if args.verify:
        print()
        try:
            jax_cross_check(out_dir)
        except ImportError as e:
            print(f"[cross-check] Skipped (JAX not available): {e}")
        except Exception as e:
            import traceback
            print(f"[cross-check] Error: {e}")
            traceback.print_exc()

    print("\nDone.")


if __name__ == '__main__':
    main()
