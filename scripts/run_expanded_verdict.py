"""Run the v1a validation on an arbitrary universe, from IBKR or yfinance.

Lets us test the 6-factor + Risk-Constrained Kelly pipeline against
universes bigger than the 10-ETF v1a baseline (SP100, NASDAQ100,
LIQUID_ETFS, or the full curated union of ~230 tickers) using whichever
data source is reachable.

Usage
-----
    # Default: curated LIQUID_ETFS (~80 tickers), yfinance, 5y:
    .venv\\Scripts\\python.exe scripts\\run_expanded_verdict.py

    # All ~230 curated tickers from IBKR Gateway:
    .venv\\Scripts\\python.exe scripts\\run_expanded_verdict.py --universe all --source ibkr

    # Custom 20-ticker subset from yfinance:
    .venv\\Scripts\\python.exe scripts\\run_expanded_verdict.py --tickers SPY,QQQ,IWM,TLT,GLD

Design
------
The core ``run_full_validation`` already accepts arbitrary prices +
factors. This script is just CLI sugar to select the universe, pull
data via the right source, and persist the verdict.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from inversiones_mama.backtest.engine import BacktestConfig
from inversiones_mama.data.factors import load_factor_returns
from inversiones_mama.data.liquid_universe import (
    LIQUID_ETFS,
    NASDAQ100_CORE,
    SP100_CORE,
    all_curated_tickers,
)
from inversiones_mama.data.prices import load_prices
from inversiones_mama.validation.gates import render_report, run_full_validation


def _sp500_tickers() -> list[str]:
    """Lazy SP500 fetch so the script doesn't make a network call at import."""
    from inversiones_mama.data.liquid_universe import fetch_sp500_tickers  # noqa: PLC0415

    return list(fetch_sp500_tickers())


UNIVERSES: dict[str, list[str] | None] = {
    "v1a":              ["AVUV", "AVDV", "AVEM", "MTUM", "IMTM", "USMV", "GLD", "DBC", "TLT", "SPY"],
    "etfs":             list(LIQUID_ETFS),
    "sp100":            list(SP100_CORE),
    "nasdaq100":        list(NASDAQ100_CORE),
    "all":              all_curated_tickers(),
    "sp500":            None,  # fetched lazily by _sp500_tickers()
    "sp500_plus_etfs":  None,
}


def _parse_tickers(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("--tickers was provided but empty")
    return tickers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run v1a validation on an arbitrary universe.")
    parser.add_argument("--universe", choices=list(UNIVERSES), default="etfs",
                        help="Curated universe name (default: etfs).")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated ticker list overriding --universe.")
    parser.add_argument("--source", choices=["yfinance", "ibkr"], default="yfinance",
                        help="Price source (default: yfinance).")
    parser.add_argument("--years", type=float, default=5.0,
                        help="History window in years (default: 5.0).")
    parser.add_argument("--capital", type=float, default=5000.0,
                        help="Starting capital (default: 5000).")
    parser.add_argument("--kelly", type=float, default=None,
                        help="Override Kelly fraction (default: from config).")
    parser.add_argument("--cap", type=float, default=None,
                        help="Override per-name cap (default: from config).")
    parser.add_argument("--covariance", choices=["sample", "lw_diagonal", "lw_constant_correlation"],
                        default=None,
                        help="Covariance estimator. Default: 'sample' for universes "
                             "<= 15 assets, 'lw_diagonal' for larger (auto-switch).")
    parser.add_argument("--mc-paths", type=int, default=3000,
                        help="Monte Carlo path count (default: 3000).")
    parser.add_argument("--out", default=None,
                        help="Verdict text destination (default: results/<universe>_verdict.txt).")
    parser.add_argument("--oos-split", type=str, default="2023-01-01",
                        help="OOS split date YYYY-MM-DD (default: 2023-01-01).")
    args = parser.parse_args(argv)

    # Resolve ticker list
    tickers_override = _parse_tickers(args.tickers)
    if tickers_override:
        tickers = tickers_override
        universe_name = f"custom_{len(tickers)}"
    else:
        if args.universe == "sp500":
            tickers = _sp500_tickers()
        elif args.universe == "sp500_plus_etfs":
            tickers = sorted(set(_sp500_tickers()) | set(LIQUID_ETFS))
        else:
            tickers = UNIVERSES[args.universe] or []
        universe_name = args.universe

    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=int(args.years * 365) + 14)

    print(f"[1/4] Universe: {universe_name} ({len(tickers)} tickers)")
    print(f"       Window: {start.date()} -> {end.date()} ({args.years}y)")
    print(f"       Source: {args.source}")

    print("[2/4] Loading prices...")
    prices = load_prices(tickers, start, end, source=args.source, use_cache=True)
    n_unreachable = len(tickers) - prices.shape[1]
    # Drop tickers with sparse coverage — new ETFs that didn't trade for the
    # full window will otherwise poison the walk-forward drift calculation
    # with NaN returns. We keep only tickers with >= 95% coverage of the
    # dense trading-day grid.
    coverage = prices.notna().sum()
    min_coverage = int(0.95 * len(prices))
    sparse = coverage[coverage < min_coverage].index.tolist()
    if sparse:
        print(f"       dropping {len(sparse)} sparse tickers (< 95% coverage): {sparse[:8]}{'...' if len(sparse) > 8 else ''}")
        prices = prices.drop(columns=sparse)
    # Forward-fill any remaining interior NaNs (dividend days, holidays elsewhere)
    prices = prices.ffill().dropna(how="any")
    print(f"       shape: {prices.shape}  (dropped {n_unreachable} unreachable + {len(sparse)} sparse)")
    if prices.shape[1] < 5:
        print(f"ERROR: only {prices.shape[1]} tickers returned. Check tickers / source.")
        return 2

    print("[3/4] Loading factors...")
    factors = load_factor_returns(start=start, end=end, use_cache=True)
    print(f"       shape: {factors.shape}")

    # Build config
    cfg_kwargs = dict(initial_capital=float(args.capital))
    if args.kelly is not None:
        cfg_kwargs["kelly_fraction"] = float(args.kelly)
    if args.cap is not None:
        cfg_kwargs["per_name_cap"] = float(args.cap)
    # Auto-pick covariance method: shrinkage kicks in once the universe is large
    # enough that sample covariance gets ill-conditioned under 252-day windows.
    if args.covariance is not None:
        cfg_kwargs["covariance_method"] = args.covariance
    elif prices.shape[1] > 15:
        cfg_kwargs["covariance_method"] = "lw_diagonal"
    else:
        cfg_kwargs["covariance_method"] = "sample"
    print(f"       Covariance: {cfg_kwargs['covariance_method']}")
    config = BacktestConfig(**cfg_kwargs)

    print(f"[4/4] Running full validation (MC paths={args.mc_paths}, Kelly={config.kelly_fraction})...")
    try:
        report = run_full_validation(
            prices.dropna(axis=1, how="all"),  # drop entirely-missing tickers
            factors,
            config=config,
            oos_split_date=datetime.fromisoformat(args.oos_split),
            mc_n_paths=int(args.mc_paths),
            mc_horizon_days=252,
            rng=np.random.default_rng(20260422),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"VALIDATION FAILED: {type(exc).__name__}: {exc}")
        return 3

    text = render_report(report)
    print()
    print(text)

    # Diagnose swallowed failures — crucial for large universes where some
    # rebalances silently fail and distort the realized wealth path.
    if report.rebalance_failures:
        from collections import Counter
        print(f"\n[DIAGNOSTICS] {len(report.rebalance_failures)} rebalance(s) failed:")
        by_stage = Counter(f.stage for f in report.rebalance_failures)
        for stage, count in by_stage.most_common():
            print(f"  stage={stage:20s} count={count}")
        print("  first 5 failures:")
        for f in report.rebalance_failures[:5]:
            print(f"    {f.date.date()} [{f.stage}] {f.error_type}: {f.error_message[:120]}")

    # Persist
    out = Path(args.out) if args.out else Path("results") / f"{universe_name}_verdict.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved -> {out}")

    return 0 if report.all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
