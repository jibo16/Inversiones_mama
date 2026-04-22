"""Bulk-backfill historical OHLCV from the IBKR Client Portal Gateway.

Pulls daily bars for a configurable universe (default: the full curated
SP100 + NASDAQ100 + LIQUID_ETFS union, ~230 tickers) over the deepest
period IBKR allows (10 years). Each ticker's bars are written to the
project parquet cache so subsequent backtests can read them without
re-hitting the API.

Flow
----
1. You launch the Gateway + complete 2FA in the browser (see
   ``docs/IBKR_SETUP.md``).
2. You run this CLI. It verifies the session, iterates the universe,
   respects the rate limit, writes each result to parquet, prints a
   per-ticker progress line, and produces a summary CSV.
3. On interruption (Ctrl-C) the partial progress is preserved; re-run
   with ``--resume`` to pick up where you left off.

Usage
-----
    .venv\\Scripts\\python.exe scripts\\backfill_ibkr_history.py
    .venv\\Scripts\\python.exe scripts\\backfill_ibkr_history.py --kind etfs --period 5y
    .venv\\Scripts\\python.exe scripts\\backfill_ibkr_history.py --resume --limit 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from inversiones_mama.config import CACHE_DIR
from inversiones_mama.data.cache import ParquetCache
from inversiones_mama.data.ibkr_historical import (
    IBKR_PERIOD_ALIASES,
    IBKRHistoricalLoader,
)
from inversiones_mama.data.liquid_universe import (
    LIQUID_ETFS,
    NASDAQ100_CORE,
    SP100_CORE,
    all_curated_tickers,
)
from inversiones_mama.execution.ibkr import IBKRConnectionError, IBKRDataError

log = logging.getLogger("backfill")


KIND_CHOICES: dict[str, list[str]] = {
    "all": all_curated_tickers(),
    "sp100": list(SP100_CORE),
    "nasdaq100": list(NASDAQ100_CORE),
    "etfs": list(LIQUID_ETFS),
}


def _cache_key(ticker: str, period: str, bar: str) -> str:
    return f"ibkr_hist_{ticker}_{period}_{bar}"


def _backfill_one(
    loader: IBKRHistoricalLoader,
    cache: ParquetCache,
    ticker: str,
    period: str,
    bar: str,
    resume: bool,
) -> dict:
    """Fetch + cache bars for one ticker, returning a summary row."""
    key = _cache_key(ticker, period, bar)
    if resume and cache.exists(key):
        try:
            df = cache.get(key)
            return {
                "ticker": ticker,
                "status": "cached",
                "n_bars": len(df),
                "start": df.index.min().date() if not df.empty else None,
                "end": df.index.max().date() if not df.empty else None,
                "error": None,
            }
        except Exception:  # noqa: BLE001
            # Cache corrupt -> refetch
            pass

    try:
        df = loader.fetch_bars(ticker, period=period, bar=bar)
        if df.empty:
            return {"ticker": ticker, "status": "empty", "n_bars": 0,
                    "start": None, "end": None, "error": "no bars returned"}
        cache.put(key, df)
        return {
            "ticker": ticker,
            "status": "ok",
            "n_bars": len(df),
            "start": df.index.min().date(),
            "end": df.index.max().date(),
            "error": None,
        }
    except (IBKRConnectionError, IBKRDataError) as exc:
        return {"ticker": ticker, "status": "error", "n_bars": 0,
                "start": None, "end": None, "error": str(exc)}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=list(KIND_CHOICES), default="all",
                        help="Which curated universe to backfill (default: all).")
    parser.add_argument("--period", choices=list(IBKR_PERIOD_ALIASES), default="10y",
                        help="IBKR period string (default: 10y).")
    parser.add_argument("--bar", default="1d", help="Bar size (default: 1d).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N tickers (for smoke tests).")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between requests (default: 0.5).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tickers already in the parquet cache.")
    parser.add_argument("--out", default="results/ibkr_backfill_summary.csv",
                        help="Summary CSV destination.")
    args = parser.parse_args(argv)

    tickers = KIND_CHOICES[args.kind]
    if args.limit is not None:
        tickers = tickers[:args.limit]

    log.info("Backfill plan: kind=%s period=%s bar=%s tickers=%d delay=%.2fs resume=%s",
             args.kind, args.period, args.bar, len(tickers), args.delay, args.resume)

    loader = IBKRHistoricalLoader.from_env()
    loader.inter_request_seconds = float(args.delay)
    try:
        loader.ensure_authenticated()
    except IBKRConnectionError as exc:
        log.error("IBKR Gateway not reachable / authenticated: %s", exc)
        log.error("See docs/IBKR_SETUP.md for the 2FA flow.")
        return 2

    cache = ParquetCache(CACHE_DIR)
    rows: list[dict] = []
    start_wall = time.monotonic()

    try:
        for i, ticker in enumerate(tickers):
            row = _backfill_one(loader, cache, ticker, args.period, args.bar, args.resume)
            rows.append(row)
            marker = {"ok": "✓", "cached": "•", "empty": "–", "error": "✗"}.get(row["status"], "?")
            log.info("[%3d/%3d] %s %-6s n=%-4s %s",
                     i + 1, len(tickers), marker, ticker, row["n_bars"], row.get("error") or "")
            if row["status"] != "cached" and i < len(tickers) - 1:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        log.warning("Interrupted — preserving partial progress.")

    # Summary
    summary = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    elapsed = time.monotonic() - start_wall

    ok = int((summary["status"] == "ok").sum()) if not summary.empty else 0
    cached = int((summary["status"] == "cached").sum()) if not summary.empty else 0
    errs = int((summary["status"] == "error").sum()) if not summary.empty else 0
    empty = int((summary["status"] == "empty").sum()) if not summary.empty else 0

    log.info("Done. elapsed=%.1fs  ok=%d  cached=%d  empty=%d  error=%d", elapsed, ok, cached, empty, errs)
    log.info("Summary -> %s", out_path)

    if errs > 0 and errs == len(rows):
        return 3  # total failure
    return 0


if __name__ == "__main__":
    sys.exit(main())
