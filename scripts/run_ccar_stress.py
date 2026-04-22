"""Run the CCAR Severely Adverse stress projection on the current portfolio.

Loads:
  * 20+ years of FRED macros (captures 2008 crisis)
  * Ken French factor panel over the same window
  * The engine's most recent rebalance (for weights + asset loadings)

Then projects 9 quarters of the Fed's Severely Adverse scenario through
our factor-macro regression and applies it to the held weights. Reports
terminal wealth, max drawdown, and the quarter-by-quarter wealth path.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_ccar_stress.py
    .venv\\Scripts\\python.exe scripts\\run_ccar_stress.py --universe sp500
    .venv\\Scripts\\python.exe scripts\\run_ccar_stress.py --initial-wealth 48000

The default ``--initial-wealth`` comes from the current backtest's
final_wealth so the stress starts from where the strategy actually ended.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from inversiones_mama.backtest.engine import walk_forward_backtest
from inversiones_mama.data.factors import load_factor_returns
from inversiones_mama.data.liquid_universe import LIQUID_ETFS, SP100_CORE, fetch_sp500_tickers
from inversiones_mama.data.macro import FRED_MACRO_SERIES, load_macro_panel
from inversiones_mama.data.prices import load_prices, returns_from_prices
from inversiones_mama.models.factor_regression import fit_factor_loadings
from inversiones_mama.simulation.ccar_stress import run_ccar_stress, summarize_result


V1A_TICKERS: list[str] = [
    "AVUV", "AVDV", "AVEM", "MTUM", "IMTM",
    "USMV", "GLD", "DBC", "TLT", "SPY",
]


def _tickers_for(kind: str) -> list[str]:
    if kind == "v1a":
        return list(V1A_TICKERS)
    if kind == "sp100":
        return list(SP100_CORE)
    if kind == "etfs":
        return list(LIQUID_ETFS)
    if kind == "sp500":
        return list(fetch_sp500_tickers())
    raise ValueError(f"Unknown universe kind: {kind!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="v1a",
                        choices=["v1a", "sp100", "etfs", "sp500"],
                        help="Strategy universe to stress-test (default: v1a).")
    parser.add_argument("--initial-wealth", type=float, default=None,
                        help="Override starting wealth. Default = current backtest terminal_wealth.")
    parser.add_argument("--macro-years", type=int, default=25,
                        help="Historical macro window in years (default 25; captures 2008).")
    parser.add_argument("--scenario", default="severely_adverse_2025",
                        help="CCAR scenario name (default: severely_adverse_2025).")
    parser.add_argument("--out", default="results/ccar_stress.txt",
                        help="Destination for the human-readable report.")
    args = parser.parse_args(argv)

    tickers = _tickers_for(args.universe)
    end = datetime.today() - timedelta(days=1)
    backtest_start = end - timedelta(days=5 * 365 + 14)
    macro_start = end - timedelta(days=args.macro_years * 365 + 14)

    # --- 1. Prices + factors --------------------------------------------------
    print(f"[1/5] Loading prices for {len(tickers)} {args.universe} tickers (5y)...")
    prices = load_prices(tickers, backtest_start, end, use_cache=True)
    coverage = prices.notna().sum()
    sparse = coverage[coverage < int(0.95 * len(prices))].index.tolist()
    if sparse:
        print(f"       dropping {len(sparse)} sparse tickers")
        prices = prices.drop(columns=sparse)
    prices = prices.ffill().dropna(how="any")
    print(f"       prices: {prices.shape}")

    print("[2/5] Loading Ken French factors (5y for regression; longer history below)...")
    factors_5y = load_factor_returns(start=backtest_start, end=end, use_cache=True)

    # --- 2. Long-window factors + macros for the macro regression -------------
    print(f"[3/5] Loading factors + FRED macros over {args.macro_years}y (captures 2008)...")
    factors_long = load_factor_returns(start=macro_start, end=end, use_cache=True)
    macro_panel = load_macro_panel(
        series_map=FRED_MACRO_SERIES,
        start=macro_start, end=end,
        freq="QE", use_cache=True,
    )
    print(f"       factors_long: {factors_long.shape}, macros: {macro_panel.shape}")

    # --- 3. Current portfolio state -------------------------------------------
    print("[4/5] Running walk-forward backtest to fetch current weights + loadings...")
    # Use the SAME factors panel for backtest (5y) so the engine's internal
    # factor regression matches the macro regression's factor vocabulary
    result = walk_forward_backtest(prices, factors_5y)
    if not result.rebalance_records:
        print("ERROR: backtest produced no rebalances; can't stress-test")
        return 2
    last_rebal = result.rebalance_records[-1]
    current_weights = last_rebal.target_weights
    backtest_final_wealth = result.final_wealth
    print(f"       latest rebalance: {last_rebal.date.date()}")
    print(f"       backtest final wealth: ${backtest_final_wealth:,.2f}")
    print(f"       holdings: {int((current_weights > 1e-4).sum())} non-zero positions")

    # Fit asset-level factor loadings over the backtest window (same as
    # the engine does at rebalance time) — we need them to project
    # asset-level returns from projected factor returns.
    train_returns = returns_from_prices(prices.tail(252 + 1))
    common = train_returns.index.intersection(factors_5y.index)
    loadings = fit_factor_loadings(
        train_returns.loc[common],
        factors_5y.loc[common],
    )

    # --- 4. Project stress ----------------------------------------------------
    initial_wealth = (
        args.initial_wealth
        if args.initial_wealth is not None
        else backtest_final_wealth
    )
    print(f"[5/5] Projecting {args.scenario} from wealth=${initial_wealth:,.2f}...")
    try:
        stress = run_ccar_stress(
            factors_daily=factors_long,
            macro_panel_quarterly=macro_panel,
            current_weights=current_weights,
            asset_loadings=loadings,
            initial_wealth=initial_wealth,
            scenario_name=args.scenario,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] stress projection failed: {type(exc).__name__}: {exc}")
        return 3

    text = summarize_result(stress)
    print()
    print(text)

    # --- Persist --------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved -> {out_path}")

    # Flag any hard failures: full wealth loss or DD > 80%
    if stress.terminal_wealth <= 0 or stress.max_drawdown >= 0.80:
        print("\n[!] SEVERE outcome: terminal wealth <= 0 or drawdown >= 80%")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
