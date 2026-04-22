"""Compute explicit strategy stats across all four production universes.

For each of v1a / etfs / sp100 / sp500 runs the walk-forward backtest and
emits the granular metrics Jorge asked for by name:

* max drawdown (value + peak/trough dates)
* max single-day gain / max single-day loss (value + date)
* max month / min month (value + month)
* daily winrate  (% of days with strictly positive return)
* monthly winrate (% of calendar months with strictly positive return)
* rebalance winrate (% of inter-rebalance spans with strictly positive
  compound return)
* final sector composition of the latest rebalance target weights

Writes both a per-strategy text block to ``results/strategy_stats.txt``
and a consolidated CSV to ``results/strategy_stats.csv``.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from inversiones_mama.backtest.engine import BacktestConfig, walk_forward_backtest
from inversiones_mama.data.factors import load_factor_returns
from inversiones_mama.data.liquid_universe import (
    LIQUID_ETFS,
    SP100_CORE,
    fetch_sp500_tickers,
)
from inversiones_mama.data.prices import load_prices
from inversiones_mama.data.sectors import build_sector_map


V1A_TICKERS = ["AVUV", "AVDV", "AVEM", "MTUM", "IMTM", "USMV", "GLD", "DBC", "TLT", "SPY"]


def _resolve_universe(kind: str) -> tuple[str, list[str]]:
    kind = kind.lower()
    if kind == "v1a":
        return "v1a", list(V1A_TICKERS)
    if kind == "etfs":
        return "etfs", list(LIQUID_ETFS)
    if kind == "sp100":
        return "sp100", list(SP100_CORE)
    if kind == "sp500":
        return "sp500", list(fetch_sp500_tickers())
    raise ValueError(f"unknown universe: {kind}")


def _load_inputs(tickers: list[str], years: float = 5.0):
    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=int(years * 365) + 14)
    prices = load_prices(tickers, start, end, use_cache=True)

    # Drop sparse tickers (<95% coverage) and forward-fill then dropna
    coverage = prices.notna().sum()
    min_cov = int(0.95 * len(prices))
    sparse = coverage[coverage < min_cov].index.tolist()
    if sparse:
        prices = prices.drop(columns=sparse)
    prices = prices.ffill().dropna(how="any")

    factors = load_factor_returns(start=start, end=end, use_cache=True)
    return prices, factors


def _stats_for_result(result, universe_name: str, universe_size: int) -> dict:
    daily = result.daily_returns.dropna()
    wealth = result.wealth.dropna()

    # Max drawdown with dates
    peak = wealth.cummax()
    dd = (wealth / peak - 1.0)
    max_dd = float(-dd.min())
    trough_date = dd.idxmin()
    peak_date = wealth.loc[:trough_date].idxmax()

    # Daily extremes
    max_day_ret = float(daily.max())
    max_day_date = daily.idxmax()
    min_day_ret = float(daily.min())
    min_day_date = daily.idxmin()

    # Monthly aggregation
    monthly = (1.0 + daily).resample("ME").prod() - 1.0
    max_month_ret = float(monthly.max())
    max_month_date = monthly.idxmax()
    min_month_ret = float(monthly.min())
    min_month_date = monthly.idxmin()

    # Winrates
    n_days = int(len(daily))
    n_days_win = int((daily > 0).sum())
    daily_winrate = n_days_win / n_days if n_days else 0.0

    n_months = int(len(monthly))
    n_months_win = int((monthly > 0).sum())
    monthly_winrate = n_months_win / n_months if n_months else 0.0

    # Rebalance-to-rebalance winrate: compound wealth between consecutive
    # rebalance dates and count how many spans were positive.
    rebal_dates = [r.date for r in result.rebalance_records]
    rebal_wins = 0
    rebal_total = 0
    span_returns: list[float] = []
    for i in range(len(rebal_dates) - 1):
        t0, t1 = rebal_dates[i], rebal_dates[i + 1]
        w0 = float(wealth.loc[wealth.index.asof(t0)])
        w1 = float(wealth.loc[wealth.index.asof(t1)])
        if w0 > 0:
            r = w1 / w0 - 1.0
            span_returns.append(r)
            rebal_total += 1
            if r > 0:
                rebal_wins += 1
    rebalance_winrate = rebal_wins / rebal_total if rebal_total else 0.0
    span_returns_arr = np.array(span_returns) if span_returns else np.array([0.0])

    # Final picks + sector distribution
    final_w = (
        result.rebalance_records[-1].target_weights
        if result.rebalance_records
        else pd.Series(dtype=float)
    )
    active = final_w[final_w > 1e-4].sort_values(ascending=False)
    tickers = list(active.index)
    sector_map = build_sector_map(tickers) if tickers else {}
    by_sector: dict[str, float] = {}
    for t in tickers:
        s = sector_map.get(t, "Unmapped")
        by_sector[s] = by_sector.get(s, 0.0) + float(active[t])
    cash = 1.0 - float(active.sum())

    total_return = float(wealth.iloc[-1] / wealth.iloc[0] - 1.0) if len(wealth) else 0.0

    return {
        "strategy": universe_name,
        "universe_size": int(universe_size),
        "n_rebalances": len(result.rebalance_records),
        "start_date": str(wealth.index[0].date()) if len(wealth) else "",
        "end_date": str(wealth.index[-1].date()) if len(wealth) else "",
        "initial_wealth": float(wealth.iloc[0]) if len(wealth) else 0.0,
        "final_wealth": float(wealth.iloc[-1]) if len(wealth) else 0.0,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "max_dd_peak_date": str(peak_date.date()) if len(wealth) else "",
        "max_dd_trough_date": str(trough_date.date()) if len(wealth) else "",
        "best_day": max_day_ret,
        "best_day_date": str(max_day_date.date()),
        "worst_day": min_day_ret,
        "worst_day_date": str(min_day_date.date()),
        "best_month": max_month_ret,
        "best_month_period": str(max_month_date.to_period("M")),
        "worst_month": min_month_ret,
        "worst_month_period": str(min_month_date.to_period("M")),
        "daily_winrate": daily_winrate,
        "daily_winrate_n": f"{n_days_win}/{n_days}",
        "monthly_winrate": monthly_winrate,
        "monthly_winrate_n": f"{n_months_win}/{n_months}",
        "rebalance_winrate": rebalance_winrate,
        "rebalance_winrate_n": f"{rebal_wins}/{rebal_total}",
        "best_rebalance_span": float(span_returns_arr.max()),
        "worst_rebalance_span": float(span_returns_arr.min()),
        "median_rebalance_span": float(np.median(span_returns_arr)),
        "final_active_positions": int(len(active)),
        "final_cash": float(cash),
        "final_weights": {str(k): float(v) for k, v in active.items()},
        "final_sector_weights": by_sector,
    }


def _render(stats: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"STRATEGY: {stats['strategy']}   universe_size={stats['universe_size']}")
    lines.append("=" * 78)
    lines.append(
        f"  period:           {stats['start_date']} -> {stats['end_date']}  "
        f"(n_rebalances = {stats['n_rebalances']})"
    )
    lines.append(
        f"  wealth:           ${stats['initial_wealth']:,.2f} -> "
        f"${stats['final_wealth']:,.2f}   (total return = {stats['total_return']:+.2%})"
    )
    lines.append("")
    lines.append("  --- Drawdown ---")
    lines.append(
        f"  max drawdown:     {stats['max_drawdown']:.2%}   "
        f"(peak {stats['max_dd_peak_date']} -> trough {stats['max_dd_trough_date']})"
    )
    lines.append("")
    lines.append("  --- Single-day extremes ---")
    lines.append(
        f"  best day:         {stats['best_day']:+.2%}   on {stats['best_day_date']}"
    )
    lines.append(
        f"  worst day:        {stats['worst_day']:+.2%}   on {stats['worst_day_date']}"
    )
    lines.append("")
    lines.append("  --- Monthly extremes ---")
    lines.append(
        f"  best month:       {stats['best_month']:+.2%}   ({stats['best_month_period']})"
    )
    lines.append(
        f"  worst month:      {stats['worst_month']:+.2%}   ({stats['worst_month_period']})"
    )
    lines.append("")
    lines.append("  --- Winrates ---")
    lines.append(
        f"  daily winrate:    {stats['daily_winrate']:.2%}   "
        f"({stats['daily_winrate_n']} trading days)"
    )
    lines.append(
        f"  monthly winrate:  {stats['monthly_winrate']:.2%}   "
        f"({stats['monthly_winrate_n']} months)"
    )
    lines.append(
        f"  rebalance winrate:{stats['rebalance_winrate']:.2%}   "
        f"({stats['rebalance_winrate_n']} inter-rebalance spans)"
    )
    lines.append(
        f"  best span:        {stats['best_rebalance_span']:+.2%}"
    )
    lines.append(
        f"  worst span:       {stats['worst_rebalance_span']:+.2%}"
    )
    lines.append(
        f"  median span:      {stats['median_rebalance_span']:+.2%}"
    )
    lines.append("")
    lines.append(
        f"  --- Final portfolio (rebalance {stats['end_date']}) ---"
    )
    lines.append(
        f"  active positions: {stats['final_active_positions']}   "
        f"cash: {stats['final_cash']:.2%}"
    )
    lines.append("  weights:")
    for t, w in sorted(stats["final_weights"].items(), key=lambda kv: -kv[1]):
        lines.append(f"     {t:8s} {w:+.2%}")
    lines.append("  by sector:")
    for sec, w in sorted(stats["final_sector_weights"].items(), key=lambda kv: -kv[1]):
        lines.append(f"     {sec:<28s} {w:+.2%}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[dict] = []
    all_text: list[str] = []

    for kind in ("v1a", "etfs", "sp100", "sp500"):
        print(f"[{kind}] loading inputs ...", flush=True)
        name, tickers = _resolve_universe(kind)
        prices, factors = _load_inputs(tickers, years=5.0)
        print(f"[{kind}] prices={prices.shape}, factors={factors.shape}", flush=True)
        # sp500 large-N needs lw_diagonal covariance
        cov_method = "sample" if prices.shape[1] <= 15 else "lw_diagonal"
        config = BacktestConfig(covariance_method=cov_method)
        print(f"[{kind}] running walk_forward_backtest ({cov_method}) ...", flush=True)
        result = walk_forward_backtest(prices, factors, config)
        print(f"[{kind}] rebalances={len(result.rebalance_records)}, "
              f"failures={len(result.rebalance_failures)}", flush=True)
        stats = _stats_for_result(result, name, prices.shape[1])
        all_stats.append(stats)
        all_text.append(_render(stats))
        print(_render(stats))

    # Persist
    txt_path = out_dir / "strategy_stats.txt"
    txt_path.write_text("\n".join(all_text), encoding="utf-8")

    # Flattened CSV (drop dict columns, keep scalars)
    rows: list[dict] = []
    scalar_keys = [k for k in all_stats[0] if not isinstance(all_stats[0][k], dict)]
    for s in all_stats:
        rows.append({k: s[k] for k in scalar_keys})
    pd.DataFrame(rows).to_csv(out_dir / "strategy_stats.csv", index=False)

    print(f"\nWROTE: {txt_path}")
    print(f"WROTE: {out_dir / 'strategy_stats.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
