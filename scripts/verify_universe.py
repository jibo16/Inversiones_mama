"""
Verify the proposed ETF universe for Inversiones_mama.

Pulls 5 years of daily adjusted prices via yfinance, computes:
  annualized return, annualized volatility, Sharpe (r_f=0), max drawdown,
  correlation with SPY, and a full pairwise correlation matrix.

This is a *verification* script — it exists to replace any ETF statistics
I quoted from memory with live, checked numbers before committing to the
universe. Not part of production strategy code.

Run (from repo root):
    .venv\\Scripts\\python.exe scripts\\verify_universe.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


# Proposed 15-ETF candidate universe. Kelly+RCK will pick 6-12 actual holdings
# at any time; the rest get zero weight when they don't improve the objective.
UNIVERSE: dict[str, str] = {
    # --- Equity factor core (9) ---
    "AVUV": "Avantis US Small-Cap Value (RMW+HML tilt)",
    "AVDV": "Avantis International Small-Cap Value",
    "AVEM": "Avantis Emerging Markets Equity",
    "MTUM": "iShares MSCI USA Momentum",
    "IMTM": "iShares MSCI International Momentum",
    "QUAL": "iShares MSCI USA Quality (RMW proxy)",
    "IQLT": "iShares MSCI International Quality",
    "VLUE": "iShares MSCI USA Value",
    "USMV": "iShares MSCI USA Minimum Volatility",
    # --- Macro diversifiers (4) ---
    "TLT":  "iShares 20+Y Treasury",
    "IEF":  "iShares 7-10Y Treasury",
    "GLD":  "SPDR Gold Shares",
    "DBC":  "Invesco DB Commodity Index",
    # --- Higher-vol / regional sleeves (2) ---
    "VWO":  "Vanguard Emerging Markets",
    "VNQ":  "Vanguard REIT",
}
BENCHMARK = "SPY"


def fetch_prices(tickers: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch auto-adjusted close prices for `tickers` between `start` and `end`."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    # yfinance shapes output differently for single vs multi-ticker requests
    if isinstance(data.columns, pd.MultiIndex):
        closes = pd.DataFrame({t: data[t]["Close"] for t in tickers if t in data.columns.levels[0]})
    else:
        closes = data[["Close"]].rename(columns={"Close": tickers[0]})
    return closes.dropna(how="all")


def compute_stats(prices: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Per-ticker annualized stats + correlation to benchmark."""
    rets = prices.pct_change().dropna(how="all")
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    # Sharpe with r_f=0 (we'll refine with actual T-bill rate later)
    sharpe = ann_ret / ann_vol.replace(0, np.nan)
    cum = (1 + rets.fillna(0)).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()
    # Correlation to the benchmark (only where both present)
    if benchmark in rets.columns:
        corr = rets.corrwith(rets[benchmark])
    else:
        corr = pd.Series(np.nan, index=rets.columns)
    out = pd.DataFrame(
        {
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "corr_SPY": corr,
            "obs_days": rets.count(),
        }
    )
    return out


def main() -> int:
    end = datetime.today()
    start = end - timedelta(days=5 * 365 + 14)  # ~5 years with slack
    tickers = list(UNIVERSE.keys()) + [BENCHMARK]
    print(f"Fetching {len(tickers)} tickers from {start.date()} to {end.date()}...")
    prices = fetch_prices(tickers, start, end)
    print(f"Fetched: {prices.shape[0]} trading days, {prices.shape[1]} columns alive.")

    missing = set(tickers) - set(prices.columns)
    if missing:
        print(f"\n[WARN] Missing data for tickers (possibly delisted/new): {sorted(missing)}")

    stats = compute_stats(prices, BENCHMARK)
    stats["name"] = stats.index.map(lambda t: UNIVERSE.get(t, f"({t})"))
    stats = stats[["name", "ann_return", "ann_vol", "sharpe", "max_drawdown", "corr_SPY", "obs_days"]]
    stats = stats.sort_values("ann_vol", ascending=False).round(4)

    os.makedirs("results", exist_ok=True)
    stats.to_csv("results/universe_stats.csv")

    print("\n" + "=" * 90)
    print("ETF UNIVERSE — 5-YEAR TRAILING STATS (annualized)")
    print("=" * 90)
    print(stats.to_string())
    print("\nSaved -> results/universe_stats.csv")

    # Pairwise correlation matrix (universe only, excl. benchmark)
    rets = prices[list(UNIVERSE.keys())].pct_change().dropna(how="all")
    corr_matrix = rets.corr().round(2)
    corr_matrix.to_csv("results/universe_corr.csv")
    print("\n" + "=" * 90)
    print("PAIRWISE CORRELATION MATRIX (daily returns, 5y)")
    print("=" * 90)
    print(corr_matrix.to_string())
    print("\nSaved -> results/universe_corr.csv")

    # Quick diversification sanity check: average pairwise correlation
    off_diag = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).stack()
    print(f"\nMean pairwise correlation: {off_diag.mean():.3f}")
    print(f"Median pairwise correlation: {off_diag.median():.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
