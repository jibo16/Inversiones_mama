"""Run the full v1a validation on the live 10-ETF universe and print the verdict.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_v1a_verdict.py

This is the canonical "does v1a ship?" command. It runs the full pipeline:
  1. Pull 5y of yfinance prices for the 10-ETF universe.
  2. Pull Ken French FF5F + Momentum factor panel.
  3. Walk-forward backtest (monthly rebalance, Kelly 0.65, 35% cap).
  4. Compute performance metrics + Deflated Sharpe Ratio.
  5. In-sample / out-of-sample split.
  6. Monte Carlo drawdown validation on the last rebalance's weights.
  7. Print every sanity-gate verdict and the overall PASS/FAIL.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import numpy as np

from inversiones_mama.config import UNIVERSE
from inversiones_mama.data.factors import load_factor_returns
from inversiones_mama.data.prices import load_prices
from inversiones_mama.validation.gates import render_report, run_full_validation


def main() -> int:
    tickers = list(UNIVERSE.keys())
    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=5 * 365 + 14)

    print(f"Loading prices for {len(tickers)} tickers: {start.date()} -> {end.date()}...")
    prices = load_prices(tickers, start, end, use_cache=True)
    print(f"  shape: {prices.shape}")

    print("Loading Ken French factors...")
    factors = load_factor_returns(start=start, end=end, use_cache=True)
    print(f"  shape: {factors.shape}")

    print("Running full v1a validation (walk-forward + MC + gates)...")
    # v1a hardening (2026-04-22):
    #   * oos_split_date = 2023-01-01 (chronological) — not median-of-series.
    #   * MC bootstrap source = full history (done in gates.run_full_validation).
    report = run_full_validation(
        prices,
        factors,
        oos_split_date=datetime(2023, 1, 1),
        mc_n_paths=5_000,
        mc_horizon_days=252,
        rng=np.random.default_rng(20260422),
    )

    out = render_report(report)
    print()
    print(out)

    # Persist to results/
    os.makedirs("results", exist_ok=True)
    with open("results/v1a_verdict.txt", "w", encoding="utf-8") as fh:
        fh.write(out + "\n")
    print("\nSaved -> results/v1a_verdict.txt")

    return 0 if report.all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
