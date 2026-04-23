"""CPCV strategy tournament — 6-factor RCK vs. 5 exploration strategies.

For each (universe, strategy) pair, produces a distribution of OOS
Sharpe / MaxDD / DSR across ``C(N, k)`` CPCV splits. Sharpes are
DSR-deflated for the total number of trials (strategies * splits).

Methodology
-----------
1. Load 5y of prices and factor returns for each universe (SP100 and
   LIQUID_ETFS, per Jorge's CPCV sweep spec).
2. Compute a deterministic daily-return series per strategy over the
   full history:
   - 6-factor RCK: ``walk_forward_backtest`` with current (v1a) params.
   - Exploration strategies: ``generate_signals`` then
     ``sum(weights * returns)`` per day.
   Since strategy parameters are frozen per Jorge's instruction #3, no
   per-fold re-fitting is needed — CPCV here gives a **distribution of
   OOS Sharpes across different market segments**, not a hyperparameter
   search.
3. For each CPCV split (C(N, k) per universe):
   - Slice the daily-return series to ``test_idx`` dates.
   - Compute Sharpe, Sortino, MaxDD, hit_rate, DSR on the slice.
4. Aggregate per (strategy, universe):
   - Mean / median / p5 / p95 Sharpe across splits.
   - DSR at each quantile.
   - Best-split / worst-split Sharpe with their test windows.

Outputs
-------
- ``results/cpcv/<universe>_<strategy>_splits.csv`` — one row per split
- ``results/cpcv/aggregate.csv`` — one row per (universe, strategy)
- ``results/cpcv/report.txt`` — human-readable summary

Usage
-----
    python scripts/run_cpcv_tournament.py
    python scripts/run_cpcv_tournament.py --n-groups 10 --test-groups 2
    python scripts/run_cpcv_tournament.py --universes sp100    # subset
    python scripts/run_cpcv_tournament.py --strategies rck,momentum_xsec
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from inversiones_mama.backtest.engine import BacktestConfig, walk_forward_backtest
from inversiones_mama.data.factors import load_factor_returns
from inversiones_mama.data.liquid_universe import (
    LIQUID_ETFS,
    SP100_CORE,
)
from inversiones_mama.data.prices import load_prices
from inversiones_mama.exploration.base import Strategy
from inversiones_mama.exploration.strategies.dual_momentum import DualMomentum
from inversiones_mama.exploration.strategies.mean_reversion import RSIMeanReversion
from inversiones_mama.exploration.strategies.momentum_ts import TimeSeriesMomentum
from inversiones_mama.exploration.strategies.momentum_xsec import CrossSectionalMomentum
from inversiones_mama.exploration.strategies.vol_targeting import VolatilityTargeting
from inversiones_mama.simulation.cpcv import PurgedKFold
from inversiones_mama.simulation.metrics import compute_all_metrics

log = logging.getLogger(__name__)


# --- Universes --------------------------------------------------------------

UNIVERSES: dict[str, list[str]] = {
    "sp100": list(SP100_CORE),
    "etfs":  list(LIQUID_ETFS),
}


# --- Strategy runners -------------------------------------------------------


def _run_rck(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
    """Run the production 6-factor + RCK walk-forward; return daily returns."""
    # Auto-pick covariance method per universe size (matches run_expanded_verdict)
    cov_method = "sample" if prices.shape[1] <= 15 else "lw_diagonal"
    cfg = BacktestConfig(covariance_method=cov_method)
    result = walk_forward_backtest(prices, factors, cfg)
    return result.daily_returns.dropna()


def _run_exploration(strategy: Strategy, prices: pd.DataFrame) -> pd.Series:
    """Call generate_signals, then compute daily portfolio returns."""
    weights = strategy.generate_signals(prices)
    if weights.empty:
        raise RuntimeError(f"{strategy.name}: generate_signals produced empty weights")
    returns = prices.pct_change().iloc[1:]
    common_dates = weights.index.intersection(returns.index)
    common_tickers = weights.columns.intersection(returns.columns)
    w = weights.loc[common_dates, common_tickers].fillna(0.0)
    r = returns.loc[common_dates, common_tickers].fillna(0.0)
    port_ret = (w * r).sum(axis=1)
    port_ret.name = "daily_return"
    return port_ret.dropna()


@dataclass(frozen=True)
class StrategySpec:
    name: str
    runner: Callable[[pd.DataFrame, pd.DataFrame], pd.Series]


def _build_strategy_specs() -> list[StrategySpec]:
    """Return the tournament roster: 6-factor RCK + 5 exploration strategies."""

    def make_xsec(prices, factors):
        return _run_exploration(CrossSectionalMomentum(lookback=120, top_k=3), prices)

    def make_ts(prices, factors):
        return _run_exploration(TimeSeriesMomentum(lookback=120), prices)

    def make_meanrev(prices, factors):
        return _run_exploration(
            RSIMeanReversion(rsi_period=14, oversold=30.0, overbought=70.0),
            prices,
        )

    def make_dual(prices, factors):
        return _run_exploration(
            DualMomentum(lookback=120, top_k=3, risk_off_asset="TLT"),
            prices,
        )

    def make_vol(prices, factors):
        return _run_exploration(
            VolatilityTargeting(vol_lookback=60, target_vol=0.15),
            prices,
        )

    return [
        StrategySpec("rck_6factor",     _run_rck),
        StrategySpec("momentum_xsec",   make_xsec),
        StrategySpec("momentum_ts",     make_ts),
        StrategySpec("mean_reversion",  make_meanrev),
        StrategySpec("dual_momentum",   make_dual),
        StrategySpec("vol_targeting",   make_vol),
    ]


# --- Data loading -----------------------------------------------------------


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
    factors = load_factor_returns(start=start, end=end, use_cache=True)
    return prices, factors


# --- Per-split metrics ------------------------------------------------------


def _metrics_for_slice(
    daily_returns: pd.Series,
    total_n_trials: int,
) -> dict:
    """Compute per-split metrics including the DSR-deflated Sharpe."""
    if len(daily_returns) < 10:
        return {
            "n_days": len(daily_returns),
            "sharpe": np.nan, "sortino": np.nan, "max_drawdown": np.nan,
            "hit_rate": np.nan, "dsr": np.nan, "skew": np.nan, "excess_kurt": np.nan,
            "total_return": np.nan,
        }
    m = compute_all_metrics(daily_returns, n_trials=total_n_trials)
    wealth = (1.0 + daily_returns).cumprod()
    return {
        "n_days": int(len(daily_returns)),
        "sharpe": float(m.sharpe_ratio),
        "sortino": float(m.sortino_ratio),
        "max_drawdown": float(m.max_drawdown),
        "hit_rate": float(m.hit_rate),
        "dsr": float(m.deflated_sharpe),
        "skew": float(m.skewness),
        "excess_kurt": float(m.excess_kurtosis),
        "total_return": float(wealth.iloc[-1] - 1.0),
    }


# --- Main loop --------------------------------------------------------------


def run_tournament(
    universes: list[str],
    strategies: list[str] | None,
    n_groups: int,
    test_groups: int,
    embargo_pct: float,
    years: float,
    out_dir: Path,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _build_strategy_specs()
    if strategies:
        wanted = {s.strip() for s in strategies if s.strip()}
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            print(f"[error] none of --strategies matched: {wanted}", file=sys.stderr)
            return 2

    pk = PurgedKFold(n_groups=n_groups, test_groups=test_groups, embargo_pct=embargo_pct)
    total_trials = len(specs) * len(universes) * pk.n_splits

    print("=" * 78)
    print(f"CPCV TOURNAMENT  |  {len(specs)} strategies x {len(universes)} universes "
          f"x {pk.n_splits} splits = {total_trials} trials")
    print(f"  n_groups={n_groups}  test_groups={test_groups}  "
          f"embargo_pct={embargo_pct*100:.1f}%  phi(N,k) paths={pk.n_paths}")
    print("=" * 78)

    all_split_rows: list[dict] = []
    all_agg_rows: list[dict] = []

    for universe_name in universes:
        tickers = UNIVERSES[universe_name]
        print(f"\n[{universe_name}] loading {len(tickers)} tickers ...", flush=True)
        try:
            prices, factors = _load_inputs(tickers, years=years)
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] data load failed: {exc}", file=sys.stderr)
            continue
        print(f"[{universe_name}] prices={prices.shape}  factors={factors.shape}", flush=True)

        # Run each strategy ONCE on the full history.
        for spec in specs:
            tag = f"{universe_name}/{spec.name}"
            print(f"\n  [{tag}] running full walk-forward ...", flush=True)
            try:
                daily = spec.runner(prices, factors)
            except Exception as exc:  # noqa: BLE001
                print(f"    [error] runner failed: {exc}", file=sys.stderr)
                continue
            if len(daily) < 2 * n_groups:
                print(f"    [skip] only {len(daily)} daily obs "
                      f"(need >= {2 * n_groups}).")
                continue
            print(f"    daily returns: {len(daily)} obs "
                  f"[{daily.index[0].date()} -> {daily.index[-1].date()}]")

            # Iterate CPCV splits on the daily-returns index
            split_rows: list[dict] = []
            for split in pk.split(n_obs=len(daily)):
                test_dates = daily.index[split.test_idx]
                slice_rets = daily.loc[test_dates]
                stats = _metrics_for_slice(slice_rets, total_n_trials=total_trials)
                stats.update({
                    "universe":       universe_name,
                    "strategy":       spec.name,
                    "test_groups":    ",".join(map(str, split.test_group_ids)),
                    "test_start":     str(test_dates[0].date()),
                    "test_end":       str(test_dates[-1].date()),
                })
                split_rows.append(stats)
                all_split_rows.append(stats)

            if not split_rows:
                continue

            # Persist per-strategy splits
            df = pd.DataFrame(split_rows)
            csv_path = out_dir / f"{universe_name}_{spec.name}_splits.csv"
            df.to_csv(csv_path, index=False)

            # Aggregate across splits
            sharpes = df["sharpe"].dropna().to_numpy()
            dsrs = df["dsr"].dropna().to_numpy()
            dds = df["max_drawdown"].dropna().to_numpy()
            best_row = df.loc[df["sharpe"].idxmax()] if sharpes.size else None
            worst_row = df.loc[df["sharpe"].idxmin()] if sharpes.size else None
            agg = {
                "universe":         universe_name,
                "strategy":         spec.name,
                "n_splits":         int(len(df)),
                "sharpe_mean":      float(np.mean(sharpes)) if sharpes.size else np.nan,
                "sharpe_median":    float(np.median(sharpes)) if sharpes.size else np.nan,
                "sharpe_p05":       float(np.percentile(sharpes, 5)) if sharpes.size else np.nan,
                "sharpe_p95":       float(np.percentile(sharpes, 95)) if sharpes.size else np.nan,
                "sharpe_std":       float(np.std(sharpes)) if sharpes.size else np.nan,
                "dsr_median":       float(np.median(dsrs)) if dsrs.size else np.nan,
                "dsr_mean":         float(np.mean(dsrs)) if dsrs.size else np.nan,
                "dsr_p95":          float(np.percentile(dsrs, 95)) if dsrs.size else np.nan,
                "dsr_frac_over_95": float(np.mean(dsrs > 0.95)) if dsrs.size else np.nan,
                "maxdd_median":     float(np.median(dds)) if dds.size else np.nan,
                "maxdd_p95":        float(np.percentile(dds, 95)) if dds.size else np.nan,
                "best_sharpe":      float(best_row["sharpe"]) if best_row is not None else np.nan,
                "best_window":      f"{best_row['test_start']}:{best_row['test_end']}" if best_row is not None else "",
                "worst_sharpe":     float(worst_row["sharpe"]) if worst_row is not None else np.nan,
                "worst_window":     f"{worst_row['test_start']}:{worst_row['test_end']}" if worst_row is not None else "",
            }
            all_agg_rows.append(agg)

            # Compact console summary
            print(f"    splits={agg['n_splits']}  "
                  f"SR median={agg['sharpe_median']:+.3f} [{agg['sharpe_p05']:+.3f}, {agg['sharpe_p95']:+.3f}]  "
                  f"DSR median={agg['dsr_median']:.3f}  DSR>0.95 in {agg['dsr_frac_over_95']*100:.1f}% of splits  "
                  f"MaxDD median={agg['maxdd_median']*100:.2f}%")

    if not all_agg_rows:
        print("\n[error] no strategy/universe combinations produced usable data.", file=sys.stderr)
        return 2

    # Persist aggregates
    agg_df = pd.DataFrame(all_agg_rows)
    agg_df.to_csv(out_dir / "aggregate.csv", index=False)
    splits_df = pd.DataFrame(all_split_rows)
    splits_df.to_csv(out_dir / "all_splits.csv", index=False)

    # Text report
    lines = []
    lines.append("=" * 78)
    lines.append(f"CPCV TOURNAMENT REPORT  |  {datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 78)
    lines.append(f"  Strategies x Universes x Splits = {total_trials} trials "
                 f"(DSR deflation baseline = {total_trials})")
    lines.append(f"  CPCV: N={n_groups}, k={test_groups}, embargo={embargo_pct*100:.1f}%, "
                 f"phi(N,k)={pk.n_paths} disjoint OOS paths per (strategy, universe)")
    lines.append("")

    lines.append(" universe   strategy          splits  SRmed [p05,p95]           "
                 "DSRmed  DSR>.95%  MDDmed")
    lines.append(" " + "-" * 96)
    for r in all_agg_rows:
        sr_bracket = f"[{r['sharpe_p05']:+.2f},{r['sharpe_p95']:+.2f}]"
        lines.append(
            f" {r['universe']:<9} {r['strategy']:<17} {r['n_splits']:>4}  "
            f"{r['sharpe_median']:+.3f} {sr_bracket:<18} "
            f"{r['dsr_median']:.3f}   {r['dsr_frac_over_95']*100:5.1f}%   "
            f"{r['maxdd_median']*100:5.2f}%"
        )
    lines.append("")

    # Overall DSR-deflated winner
    winner = max(all_agg_rows, key=lambda r: r["dsr_median"])
    lines.append(f"DSR-deflated winner: {winner['universe']}/{winner['strategy']}  "
                 f"(DSR median = {winner['dsr_median']:.3f}, "
                 f"Sharpe median = {winner['sharpe_median']:+.3f}, "
                 f"MaxDD median = {winner['maxdd_median']*100:.2f}%)")
    lines.append("")
    lines.append("DSR > 0.95 threshold (Bailey-Lopez de Prado) is the bar for 'real edge")
    lines.append("after multiple-testing deflation'. 'DSR>.95%' = share of CPCV splits")
    lines.append("where the strategy cleared the bar on its own OOS window.")
    lines.append("=" * 78)

    text = "\n".join(lines)
    (out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    print()
    print(text)

    # Also emit a small JSON for dashboard consumers
    meta = {
        "generated_at":   datetime.now().isoformat(),
        "n_groups":       n_groups,
        "test_groups":    test_groups,
        "embargo_pct":    embargo_pct,
        "n_paths":        pk.n_paths,
        "n_splits":       pk.n_splits,
        "total_trials":   total_trials,
        "universes":      universes,
        "strategies":     [s.name for s in specs],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return 0


# --- CLI --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universes", type=str, default="sp100,etfs",
                        help="Comma-sep universe names. Default: sp100,etfs.")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-sep subset of strategy names (default: all 6).")
    parser.add_argument("--n-groups", type=int, default=8,
                        help="CPCV N: number of partitions (default 8).")
    parser.add_argument("--test-groups", type=int, default=2,
                        help="CPCV k: groups per test set (default 2). "
                             "n_splits = C(N, k).")
    parser.add_argument("--embargo-pct", type=float, default=0.01,
                        help="Purge+embargo fraction of total observations (default 0.01).")
    parser.add_argument("--years", type=float, default=5.0,
                        help="History window in years (default 5).")
    parser.add_argument("--out-dir", type=str, default="results/cpcv",
                        help="Output directory (default results/cpcv).")
    args = parser.parse_args(argv)

    universes = [u.strip() for u in args.universes.split(",") if u.strip()]
    unknown = [u for u in universes if u not in UNIVERSES]
    if unknown:
        print(f"[error] unknown universes: {unknown}. Available: {list(UNIVERSES)}",
              file=sys.stderr)
        return 2
    strategies = [s.strip() for s in args.strategies.split(",")] if args.strategies else None

    return run_tournament(
        universes=universes,
        strategies=strategies,
        n_groups=int(args.n_groups),
        test_groups=int(args.test_groups),
        embargo_pct=float(args.embargo_pct),
        years=float(args.years),
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    sys.exit(main())
