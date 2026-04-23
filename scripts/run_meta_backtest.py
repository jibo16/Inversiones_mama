"""Historical walk-forward backtest for every strategy in STRATEGY_CATALOG.

For each of the 20 bags, loads its universe's prices + factors, invokes
the same strategy-runner functions used by the CPCV tournament, and
derives full performance metrics from the resulting daily-return
series. Saves per-strategy CSVs of the daily wealth curve plus a
summary CSV + JSON with Sharpe, Sortino, MaxDD, CVaR95, hit rate,
profit factor, monthly win rate, rebalance-span win rate, and DSR.

Intentionally reuses the existing runners from
``scripts.run_cpcv_rigorous`` to guarantee the backtest matches what
the CPCV tournament measured (no forking the allocator math).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inversiones_mama.data.factors import load_factor_returns
from inversiones_mama.data.prices import load_prices
from inversiones_mama.execution.strategy_catalog import STRATEGY_CATALOG, StrategySpec
from inversiones_mama.simulation.metrics import compute_all_metrics


UNIVERSE_RESOLVERS = {
    "v1a":   lambda: ["AVUV", "AVDV", "AVEM", "MTUM", "IMTM", "USMV", "GLD", "DBC", "TLT", "SPY"],
    "etfs":  lambda: None,   # populated lazily
    "sp100": lambda: None,
    "sp500": lambda: None,
}


def _resolve_universe(key: str) -> list[str]:
    if key == "v1a":
        return UNIVERSE_RESOLVERS["v1a"]()
    from inversiones_mama.data.liquid_universe import (
        LIQUID_ETFS, SP100_CORE, fetch_sp500_tickers,
    )
    if key == "etfs":
        return list(LIQUID_ETFS)
    if key == "sp100":
        return list(SP100_CORE)
    if key == "sp500":
        return list(fetch_sp500_tickers())
    raise ValueError(f"unknown universe key: {key}")


def _load_inputs(tickers: list[str], years: float = 5.0):
    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=int(years * 365) + 14)
    prices = load_prices(tickers, start, end, use_cache=True)
    coverage = prices.notna().sum()
    min_cov = int(0.95 * len(prices))
    sparse = coverage[coverage < min_cov].index.tolist()
    if sparse:
        prices = prices.drop(columns=sparse)
    prices = prices.ffill().dropna(how="any")
    try:
        factors = load_factor_returns(start=start, end=end, use_cache=True)
    except Exception:  # noqa: BLE001
        factors = pd.DataFrame()
    return prices, factors


def _run_exploration(strategy, prices: pd.DataFrame) -> pd.Series:
    weights = strategy.generate_signals(prices)
    if weights.empty:
        return pd.Series(dtype=float)
    returns = prices.pct_change().iloc[1:]
    common_dates = weights.index.intersection(returns.index)
    common_tickers = weights.columns.intersection(returns.columns)
    w = weights.loc[common_dates, common_tickers].fillna(0.0)
    r = returns.loc[common_dates, common_tickers].fillna(0.0)
    return (w * r).sum(axis=1).dropna()


def _run_rck_walk_forward(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
    from inversiones_mama.backtest.engine import BacktestConfig, walk_forward_backtest
    cov_method = "sample" if prices.shape[1] <= 15 else "lw_diagonal"
    result = walk_forward_backtest(prices, factors, BacktestConfig(covariance_method=cov_method))
    return result.daily_returns.dropna()


def _run_monthly_allocator(
    prices: pd.DataFrame,
    weight_fn,
    lookback: int = 252,
    require_lookback: bool = True,
) -> pd.Series:
    """Monthly rebalance driver for allocators whose weight_fn takes a
    trailing price window and returns the target weight vector."""
    returns = prices.pct_change().iloc[1:]
    if returns.empty:
        return pd.Series(dtype=float)
    rebal_dates = set(returns.groupby(pd.Grouper(freq="ME")).tail(1).index.tolist())
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    current_w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    for i, d in enumerate(returns.index):
        if d in rebal_dates and (not require_lookback or i >= lookback):
            try:
                w = weight_fn(prices.iloc[: i + 1 + 1], returns.iloc[: i + 1])
                current_w = w.reindex(weights.columns).fillna(0.0)
                s = current_w.sum()
                if s > 0:
                    current_w = current_w / s
            except Exception:  # noqa: BLE001
                pass
        weights.loc[d] = current_w.values
    port_ret = (weights.values * returns.values).sum(axis=1)
    return pd.Series(port_ret, index=returns.index, name="daily_return").dropna()


def _run_strategy(spec: StrategySpec, prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
    """Dispatch to the right runner based on the allocator key."""
    from inversiones_mama.exploration.strategies.dual_momentum import DualMomentum
    from inversiones_mama.exploration.strategies.mean_reversion import RSIMeanReversion
    from inversiones_mama.exploration.strategies.momentum_ts import TimeSeriesMomentum
    from inversiones_mama.exploration.strategies.momentum_xsec import CrossSectionalMomentum
    from inversiones_mama.exploration.strategies.vol_targeting import VolatilityTargeting

    alloc = spec.allocator
    if alloc == "rck_6factor":
        return _run_rck_walk_forward(prices, factors)
    if alloc == "vol_targeting":
        return _run_exploration(VolatilityTargeting(vol_lookback=60, target_vol=0.15), prices)
    if alloc == "inverse_vol":
        from inversiones_mama.sizing.inverse_vol import inverse_vol_allocator
        _, port_ret = inverse_vol_allocator(prices, vol_lookback=60, rebal_freq="ME",
                                             per_name_cap=0.15, equity_floor=None)
        return port_ret.dropna()
    if alloc == "invvol_eqfloor":
        from inversiones_mama.sizing.inverse_vol import inverse_vol_allocator
        _, port_ret = inverse_vol_allocator(prices, vol_lookback=60, rebal_freq="ME",
                                             per_name_cap=0.15, equity_floor=0.40)
        return port_ret.dropna()
    if alloc in ("hrp_capped", "hrp_eqfloor"):
        from inversiones_mama.sizing.hrp import hrp_weights
        from inversiones_mama.sizing.inverse_vol import _apply_equity_floor
        per_cap = 0.15
        eq_floor = 0.40 if alloc == "hrp_eqfloor" else None

        def hrp_fn(px_win, ret_win):
            window = ret_win.tail(252)
            active = window.dropna(axis=1, how="any")
            var = active.var()
            active = active[var[var > 0].index]
            if active.shape[1] < 2:
                return pd.Series(1.0 / ret_win.shape[1], index=ret_win.columns)
            w = hrp_weights(active)
            out = pd.Series(0.0, index=ret_win.columns)
            out.loc[w.index] = w.values
            # Per-name cap
            for _ in range(10):
                em = out > per_cap
                if not em.any():
                    break
                excess = (out[em] - per_cap).sum()
                out[em] = per_cap
                below = out[~em]
                if below.sum() > 0:
                    out[~em] += excess * below / below.sum()
            if eq_floor is not None:
                out = _apply_equity_floor(out, eq_floor)
            return out

        return _run_monthly_allocator(prices, hrp_fn, lookback=252)
    if alloc == "equal_weight":
        returns = prices.pct_change().iloc[1:]
        n = returns.shape[1]
        w = np.ones(n) / n
        return pd.Series(returns.values @ w, index=returns.index).dropna()
    if alloc == "sixty_forty":
        returns = prices.pct_change().iloc[1:]
        if "SPY" not in returns.columns or "AGG" not in returns.columns:
            return pd.Series(dtype=float)
        w = pd.Series(0.0, index=returns.columns)
        w["SPY"] = 0.6
        w["AGG"] = 0.4
        return (returns * w).sum(axis=1).dropna()
    if alloc == "spy_hold":
        returns = prices.pct_change().iloc[1:]
        if "SPY" not in returns.columns:
            return pd.Series(dtype=float)
        return returns["SPY"].dropna()
    if alloc in ("momentum_ts_l60", "momentum_ts_l120", "momentum_ts_l252"):
        lb = int(alloc.split("_l")[1])
        return _run_exploration(TimeSeriesMomentum(lookback=lb), prices)
    if alloc == "momentum_xsec":
        return _run_exploration(CrossSectionalMomentum(lookback=120, top_k=3), prices)
    if alloc == "dual_momentum":
        return _run_exploration(DualMomentum(lookback=120, top_k=3, risk_off_asset="TLT"), prices)
    if alloc == "mean_reversion":
        return _run_exploration(RSIMeanReversion(rsi_period=14, oversold=30.0, overbought=70.0), prices)
    raise ValueError(f"no runner for allocator: {alloc}")


def _derive_metrics(daily: pd.Series, strategy_id: str, n_trials: int) -> dict[str, Any]:
    """Compute a dense set of metrics for the strategy's daily returns."""
    if len(daily) < 30:
        return {"strategy_id": strategy_id, "n_days": len(daily), "error": "too_short"}
    m = compute_all_metrics(daily, n_trials=n_trials)
    wealth = (1.0 + daily).cumprod()

    # Monthly aggregation
    monthly = (1.0 + daily).resample("ME").prod() - 1.0
    # Rebalance-span win rate: pct positive months
    rebal_winrate = float((monthly > 0).mean()) if len(monthly) else np.nan

    # CVaR 95
    q05 = float(np.percentile(daily, 5))
    tail = daily[daily <= q05]
    cvar95 = float(-tail.mean()) if tail.size else np.nan

    # Best / worst day + month
    best_day = float(daily.max())
    worst_day = float(daily.min())
    best_month = float(monthly.max()) if len(monthly) else np.nan
    worst_month = float(monthly.min()) if len(monthly) else np.nan

    return {
        "strategy_id":           strategy_id,
        "n_days":                int(len(daily)),
        "start_date":            str(daily.index[0].date()),
        "end_date":              str(daily.index[-1].date()),
        "total_return":          float(wealth.iloc[-1] - 1.0),
        "annualized_return":     float(m.annualized_return),
        "annualized_volatility": float(m.annualized_volatility),
        "sharpe_ratio":          float(m.sharpe_ratio),
        "sortino_ratio":         float(m.sortino_ratio),
        "calmar_ratio":          float(m.calmar_ratio),
        "max_drawdown":          float(m.max_drawdown),
        "avg_drawdown":          float(m.avg_drawdown),
        "skewness":              float(m.skewness),
        "excess_kurtosis":       float(m.excess_kurtosis),
        "deflated_sharpe":       float(m.deflated_sharpe),
        "hit_rate":              float(m.hit_rate),
        "profit_factor":         float(m.profit_factor),
        "tail_ratio":            float(m.tail_ratio),
        "cvar_95":               cvar95,
        "best_day":              best_day,
        "worst_day":             worst_day,
        "best_month":            best_month,
        "worst_month":           worst_month,
        "monthly_winrate":       rebal_winrate,
        "n_trials":              int(n_trials),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=float, default=5.0,
                        help="Years of history to backtest (default: 5).")
    parser.add_argument("--out-dir", type=str, default="results/meta_backtest")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated subset of strategy_ids.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    (out_dir / "daily").mkdir(parents=True, exist_ok=True)

    catalog = STRATEGY_CATALOG
    if args.strategies:
        wanted = {s.strip() for s in args.strategies.split(",") if s.strip()}
        catalog = [s for s in catalog if s.strategy_id in wanted]

    print("=" * 80)
    print(f"META-PORTFOLIO HISTORICAL BACKTEST  |  {args.years}y")
    print("=" * 80)
    print(f"  strategies: {len(catalog)}")
    print(f"  out_dir:    {out_dir}")
    print("=" * 80)

    n_trials = len(STRATEGY_CATALOG)  # 20-strategy deflation baseline
    summary_rows: list[dict[str, Any]] = []

    for i, spec in enumerate(catalog, 1):
        print(f"\n[{i}/{len(catalog)}] {spec.strategy_id}  ({spec.allocator} on {spec.universe})", flush=True)
        try:
            tickers = _resolve_universe(spec.universe)
        except Exception as exc:
            print(f"  [error] universe: {exc}")
            summary_rows.append({"strategy_id": spec.strategy_id, "error": f"universe:{exc}"})
            continue

        try:
            prices, factors = _load_inputs(tickers, years=args.years)
        except Exception as exc:
            print(f"  [error] data load: {exc}")
            summary_rows.append({"strategy_id": spec.strategy_id, "error": f"data:{exc}"})
            continue

        try:
            daily = _run_strategy(spec, prices, factors)
        except Exception as exc:
            print(f"  [error] runner: {exc}")
            summary_rows.append({"strategy_id": spec.strategy_id, "error": f"runner:{exc}"})
            continue

        if len(daily) < 30:
            print(f"  [skip] too few daily returns ({len(daily)})")
            summary_rows.append({"strategy_id": spec.strategy_id, "error": "too_short"})
            continue

        metrics = _derive_metrics(daily, spec.strategy_id, n_trials)
        summary_rows.append(metrics)

        # Persist daily returns + wealth curve
        wealth = (1.0 + daily).cumprod()
        out_path = out_dir / "daily" / f"{spec.strategy_id}.csv"
        pd.DataFrame({"daily_return": daily, "wealth": wealth}).to_csv(out_path)

        print(f"  days={metrics['n_days']}  SR={metrics['sharpe_ratio']:+.3f}  "
              f"MDD={metrics['max_drawdown']*100:.2f}%  "
              f"CVaR95={metrics['cvar_95']*100:.2f}%  "
              f"DSR={metrics['deflated_sharpe']:.3f}")

    # Save summary
    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        df.to_json(orient="records", indent=2), encoding="utf-8",
    )

    # Leaderboard by DSR
    valid = df[df.get("error").isna() if "error" in df.columns else df.index == df.index]
    if "sharpe_ratio" in valid.columns:
        print("\n" + "=" * 80)
        print("LEADERBOARD (by DSR median)")
        print("=" * 80)
        print(f'{"strategy_id":<30} {"SR":>7} {"MDD":>8} {"CVaR95":>8} {"DSR":>7}')
        print("-" * 80)
        for r in valid.sort_values("deflated_sharpe", ascending=False).to_dict("records"):
            print(f'{r["strategy_id"]:<30} '
                  f'{r.get("sharpe_ratio", float("nan")):>+7.3f} '
                  f'{r.get("max_drawdown", 0)*100:>7.2f}% '
                  f'{r.get("cvar_95", 0)*100:>7.2f}% '
                  f'{r.get("deflated_sharpe", 0):>7.3f}')
    print(f"\nSaved: {out_dir}/summary.csv + daily/*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
