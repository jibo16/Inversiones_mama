"""Bulk backfill historical OHLCV from yfinance for an expanded universe.

yfinance is free and rate-limit-tolerant — this CLI pulls the full SP500
(or any other curated universe) in one batch call per chunk, writes each
ticker's bars to the parquet cache, and prints a per-ticker coverage
summary.

Unlike ``scripts/backfill_ibkr_history.py``, this script needs NO 2FA,
NO Gateway, NO vendor subscription. It's the default "get data now"
path while the IBKR pipeline waits on your session login.

Usage
-----
    # Default: full curated universe (~230 tickers, fastest path)
    .venv\\Scripts\\python.exe scripts\\backfill_yfinance.py

    # S&P 500 (runtime Wikipedia fetch ~500 tickers):
    .venv\\Scripts\\python.exe scripts\\backfill_yfinance.py --kind sp500 --years 5

    # S&P 500 plus all our curated ETFs:
    .venv\\Scripts\\python.exe scripts\\backfill_yfinance.py --kind sp500_plus_etfs --years 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from inversiones_mama.config import CACHE_DIR
from inversiones_mama.data.cache import ParquetCache
from inversiones_mama.data.liquid_universe import build_liquid_universe

log = logging.getLogger("backfill_yf")


def _batch_download(
    tickers: list[str],
    start: datetime,
    end: datetime,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Fetch ``tickers`` in chunks of ``batch_size`` to respect yfinance limits."""
    frames: list[pd.DataFrame] = []
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i : i + batch_size]
        log.info("batch %d-%d / %d", i + 1, min(i + batch_size, len(tickers)), len(tickers))
        data = yf.download(
            chunk,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if data is None or len(data) == 0:
            log.warning("batch returned empty: %s", chunk[:3])
            continue
        if isinstance(data.columns, pd.MultiIndex):
            closes = {
                t: data[t]["Close"] for t in chunk
                if t in data.columns.get_level_values(0)
            }
            frames.append(pd.DataFrame(closes))
        else:
            # Single ticker fallback
            frames.append(pd.DataFrame({chunk[0]: data["Close"]}))
    if not frames:
        return pd.DataFrame()
    wide = pd.concat(frames, axis=1)
    # Normalize index dtype to match data/prices.py
    wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index).values.astype("datetime64[ns]"))
    wide.index.name = "date"
    return wide.sort_index()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", default="all",
                        help="Universe: all / sp100 / nasdaq100 / etfs / sp500 / sp500_plus_etfs")
    parser.add_argument("--years", type=float, default=5.0,
                        help="History window in years (default 5).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap ticker count for a fast smoke test.")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Tickers per yfinance.download() call (default 100).")
    parser.add_argument("--out", default="results/yfinance_backfill_summary.csv",
                        help="Destination for the per-ticker coverage summary.")
    args = parser.parse_args(argv)

    universe = build_liquid_universe(args.kind, limit=args.limit)
    tickers = universe.all_tickers()
    log.info("universe=%s tickers=%d years=%.1f batch=%d",
             args.kind, len(tickers), args.years, args.batch_size)

    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=int(args.years * 365) + 14)

    start_wall = time.monotonic()
    prices = _batch_download(tickers, start, end, batch_size=args.batch_size)
    elapsed = time.monotonic() - start_wall
    if prices.empty:
        log.error("no data returned; check ticker list and network.")
        return 2

    # Coverage report
    coverage = prices.notna().sum()
    full_coverage_threshold = int(0.95 * len(prices))
    rows = []
    for t in tickers:
        if t not in prices.columns:
            rows.append({"ticker": t, "n_bars": 0, "start": None, "end": None, "status": "missing"})
            continue
        s = prices[t].dropna()
        if s.empty:
            rows.append({"ticker": t, "n_bars": 0, "start": None, "end": None, "status": "empty"})
            continue
        status = "ok" if len(s) >= full_coverage_threshold else "sparse"
        rows.append({
            "ticker": t,
            "n_bars": len(s),
            "start": s.index.min().date(),
            "end": s.index.max().date(),
            "status": status,
        })

    summary = pd.DataFrame(rows).sort_values("ticker")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    # Also cache the wide prices panel so subsequent load_prices() calls can
    # benefit via the data/prices.py cache naming convention. Each ticker's
    # closes are stored in a single-column parquet keyed like yfinance pulls.
    cache = ParquetCache(CACHE_DIR)
    sorted_tickers = sorted(prices.columns.tolist())
    key = f"prices_yfinance_{'-'.join(sorted_tickers)}_{start.date()}_{end.date()}"
    # Windows path length safety check: the cache sanitizes and truncates
    # the key internally, but exceeding 200+ chars on Windows can still bite
    # — prefer the ticker-per-file pattern.
    for t in prices.columns:
        try:
            col_df = prices[[t]].dropna()
            if not col_df.empty:
                cache.put(f"prices_yfinance_{t}_{start.date()}_{end.date()}", col_df)
        except Exception as exc:  # noqa: BLE001
            log.warning("cache put failed for %s: %s", t, exc)

    # Summary print
    ok = int((summary["status"] == "ok").sum())
    sparse = int((summary["status"] == "sparse").sum())
    missing = int((summary["status"] == "missing").sum())
    empty = int((summary["status"] == "empty").sum())
    log.info("")
    log.info("=" * 70)
    log.info("BACKFILL DONE (%.1fs)", elapsed)
    log.info("  universe:  %s  (%d tickers requested)", args.kind, len(tickers))
    log.info("  returned:  %d trading days covering %d tickers", len(prices), prices.shape[1])
    log.info("  ok (>=95%% coverage): %d", ok)
    log.info("  sparse (<95%%):       %d", sparse)
    log.info("  empty:               %d", empty)
    log.info("  missing:             %d", missing)
    log.info("  summary csv:         %s", out_path)
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
