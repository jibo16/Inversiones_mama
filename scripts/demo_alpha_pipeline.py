"""End-to-end demo of Agent 1's Data + Alpha Engine on the approved 10-ETF universe.

Steps:
  1. Pull 5y of daily prices via yfinance (cached parquet).
  2. Pull Fama-French 5-factor + Momentum panel from Ken French (cached parquet).
  3. Fit the 6-factor OLS per ETF.
  4. Compute the composite mu vector (annualized for readability).
  5. Print a summary table and dump a CSV so Agent 2 can feed it into Kelly.

Run:  .venv\\Scripts\\python.exe scripts\\demo_alpha_pipeline.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd

from inversiones_mama.config import UNIVERSE, LOOKBACK_DAYS
from inversiones_mama.data.factors import load_factor_returns
from inversiones_mama.data.prices import load_prices, returns_from_prices
from inversiones_mama.models.factor_regression import (
    FACTOR_COLS,
    compute_composite_mu,
    factor_premia,
    fit_factor_loadings,
)


def main() -> int:
    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=5 * 365 + 14)

    tickers = list(UNIVERSE.keys())
    print(f"[1/4] Fetching prices for {len(tickers)} ETFs ({start.date()} -> {end.date()})...")
    prices = load_prices(tickers, start, end, use_cache=True)
    rets = returns_from_prices(prices, method="simple")
    print(f"       prices shape: {prices.shape}, returns shape: {rets.shape}")

    print("[2/4] Fetching Fama-French 5F + Momentum factors...")
    factors = load_factor_returns(start=start, end=end, use_cache=True)
    print(f"       factors shape: {factors.shape}, columns: {list(factors.columns)}")

    print("[3/4] Fitting 6-factor OLS per ETF...")
    loadings = fit_factor_loadings(rets, factors)
    print(f"       n_obs used: {loadings.n_obs}")

    # LOOKBACK_DAYS-based premia (the same window the rebalance will use)
    premia_rolling = factor_premia(factors, lookback_days=LOOKBACK_DAYS)
    premia_full = factor_premia(factors)
    mu_rolling = compute_composite_mu(loadings, premia_rolling)
    mu_full = compute_composite_mu(loadings, premia_full)

    # Annualize for human readability
    table = pd.DataFrame(
        {
            "alpha_ann": loadings.alpha * 252,
            "r_squared": loadings.r_squared,
            "resid_vol_ann": loadings.residual_std * (252**0.5),
            "mu_ann_fullhist": mu_full * 252,
            "mu_ann_252d": mu_rolling * 252,
        }
    )
    # Join key factor loadings for readability
    for k in FACTOR_COLS:
        table[f"b_{k}"] = loadings.betas[k]

    table = table.round(4)

    print("[4/4] Summary (annualized where applicable):")
    print()
    print(table.to_string())

    os.makedirs("results", exist_ok=True)
    table.to_csv("results/alpha_pipeline_summary.csv")
    print("\nSaved -> results/alpha_pipeline_summary.csv")

    # Print premia for context
    print("\nFactor premia used:")
    print("  full history (daily):")
    for k, v in premia_full.items():
        print(f"    {k:>7s}: {v*252:+.4f} ann   ({v:+.6f} daily)")
    print(f"  trailing {LOOKBACK_DAYS} days (daily):")
    for k, v in premia_rolling.items():
        print(f"    {k:>7s}: {v*252:+.4f} ann   ({v:+.6f} daily)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
