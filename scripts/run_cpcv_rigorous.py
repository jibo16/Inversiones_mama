"""Rigorous CPCV tournament with null-strategy baseline + proper deflation.

Addresses the lenience issues in run_cpcv_tournament.py:

1. Null strategies added (equal-weight, random uniform, inverse-vol).
   If our 6 real strategies don't beat ALL THREE nulls on DSR, they
   have no demonstrable edge.

2. Per-strategy DSR with N_trials=6 (one per real strategy compared).
   The previous script used N_trials = 540 which double-counted CPCV
   slices as independent hypotheses -- wrong, those slices are not
   independent draws, they're different test windows of the same
   hypothesis.

3. Block-bootstrap confidence intervals on each strategy's Sharpe
   (1000 iterations, block size 20 business days). CI including zero
   => strategy statistically indistinguishable from noise.

4. Mann-Whitney U head-to-head: each real strategy's Sharpe
   distribution vs. the pooled null baseline. p < 0.05 with positive
   mean rank = real edge vs. noise.

5. CVaR 95% per cell = expected loss in the worst 5% of daily returns.

Outputs
-------
results/cpcv_rigorous/
  all_splits.csv            - per-split, per-(strategy,universe)
  aggregate.csv             - per-(strategy,universe) with CIs
  dsr_per_strategy.csv      - per-strategy DSR using N_trials=6
  mannwhitney.csv           - real strategies vs null baseline
  report.txt                - human-readable summary
  figures/                  - static PNGs (equity, box plots, CI dotplots)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
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
    fetch_russell3000_tickers,
    fetch_sp500_tickers,
)
from inversiones_mama.data.prices import load_prices
from inversiones_mama.exploration.strategies.dual_momentum import DualMomentum
from inversiones_mama.exploration.strategies.mean_reversion import RSIMeanReversion
from inversiones_mama.exploration.strategies.momentum_ts import TimeSeriesMomentum
from inversiones_mama.exploration.strategies.momentum_xsec import CrossSectionalMomentum
from inversiones_mama.exploration.strategies.vol_targeting import VolatilityTargeting
from inversiones_mama.simulation.cpcv import PurgedKFold
from inversiones_mama.simulation.metrics import compute_all_metrics

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _lazy_sp500() -> list[str]:
    return list(fetch_sp500_tickers())


def _lazy_russell3000() -> list[str]:
    return list(fetch_russell3000_tickers())


def _lazy_wide() -> list[str]:
    """Russell 3000 union SP500 union LIQUID_ETFS."""
    return sorted(set(fetch_russell3000_tickers())
                  | set(fetch_sp500_tickers())
                  | set(LIQUID_ETFS))


UNIVERSES: dict[str, callable] = {
    "sp100":       lambda: list(SP100_CORE),
    "etfs":        lambda: list(LIQUID_ETFS),
    "sp500":       _lazy_sp500,
    "russell3000": _lazy_russell3000,
    "wide":        _lazy_wide,
}


# ============================================================================
# Strategy runners
# ============================================================================


def _run_rck(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
    cov_method = "sample" if prices.shape[1] <= 15 else "lw_diagonal"
    cfg = BacktestConfig(covariance_method=cov_method)
    result = walk_forward_backtest(prices, factors, cfg)
    return result.daily_returns.dropna()


def _run_exploration(strategy, prices: pd.DataFrame) -> pd.Series:
    weights = strategy.generate_signals(prices)
    if weights.empty:
        raise RuntimeError(f"{strategy}: generate_signals produced empty weights")
    returns = prices.pct_change().iloc[1:]
    common_dates = weights.index.intersection(returns.index)
    common_tickers = weights.columns.intersection(returns.columns)
    w = weights.loc[common_dates, common_tickers].fillna(0.0)
    r = returns.loc[common_dates, common_tickers].fillna(0.0)
    return (w * r).sum(axis=1).dropna()


# ============================================================================
# Null strategies (baselines to beat)
# ============================================================================


def _null_equal_weight(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
    """Equal-weight all tickers, rebalanced monthly. With daily drift (no
    intraday rebalance), this is mathematically identical to holding a
    constant equal-weight vector."""
    returns = prices.pct_change().iloc[1:]
    n = returns.shape[1]
    w = np.ones(n) / n
    port = returns.values @ w
    return pd.Series(port, index=returns.index, name="daily_return").dropna()


def _null_random_uniform(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
    """Random weights from Dirichlet(1), redrawn monthly. Pure noise."""
    rng = np.random.default_rng(seed=20260422)
    returns = prices.pct_change().iloc[1:]
    n = returns.shape[1]
    # Monthly rebalance: pick rebalance dates = last business day of each month
    rebal_dates = returns.groupby(pd.Grouper(freq="ME")).tail(1).index
    # Weight matrix: hold-from-last-rebalance
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    current_w = rng.dirichlet(np.ones(n))
    for d in returns.index:
        if d in rebal_dates:
            current_w = rng.dirichlet(np.ones(n))
        weights.loc[d] = current_w
    port_ret = (weights.values * returns.values).sum(axis=1)
    return pd.Series(port_ret, index=returns.index, name="daily_return").dropna()


def _null_inverse_vol(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
    """Weights proportional to 1/sigma_i (trailing 60d), rebalanced monthly."""
    returns = prices.pct_change().iloc[1:]
    rebal_dates = set(returns.groupby(pd.Grouper(freq="ME")).tail(1).index.tolist())
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    current_w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    vol_lookback = 60
    for i, d in enumerate(returns.index):
        if d in rebal_dates and i >= vol_lookback:
            window = returns.iloc[i - vol_lookback:i]
            sigma = window.std(ddof=0).replace(0, np.nan)
            if sigma.notna().sum() >= 2:
                inv_vol = 1.0 / sigma
                inv_vol = inv_vol.fillna(0.0)
                current_w = inv_vol / inv_vol.sum()
        weights.loc[d] = current_w.reindex(weights.columns).fillna(0.0).values
    port_ret = (weights.values * returns.values).sum(axis=1)
    return pd.Series(port_ret, index=returns.index, name="daily_return").dropna()


def _null_hrp(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado 2016).

    Monthly rebalance; trailing 252 trading-day returns feed the
    correlation-distance -> clustering -> recursive-bisection pipeline.
    No expected-return signal used, so classified as a null baseline --
    but a sophisticated one (correlation-aware, variance-aware).
    """
    from inversiones_mama.sizing.hrp import hrp_weights  # noqa: PLC0415

    returns = prices.pct_change().iloc[1:]
    rebal_dates = set(returns.groupby(pd.Grouper(freq="ME")).tail(1).index.tolist())
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    current_w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    lookback = 252
    for i, d in enumerate(returns.index):
        if d in rebal_dates and i >= lookback:
            window = returns.iloc[i - lookback:i]
            # Keep only assets with full coverage + non-zero variance in the
            # window. Others are dropped for this rebalance; re-added next.
            active = window.dropna(axis=1, how="any")
            var = active.var()
            active = active[var[var > 0].index]
            if active.shape[1] >= 2:
                try:
                    w = hrp_weights(active)
                    current_w = w.reindex(weights.columns).fillna(0.0)
                    total = current_w.sum()
                    if total > 0:
                        current_w = current_w / total
                except Exception as exc:  # noqa: BLE001
                    print(f"  [hrp-warn] {d.date()}: {exc}")
        weights.loc[d] = current_w.values
    port_ret = (weights.values * returns.values).sum(axis=1)
    return pd.Series(port_ret, index=returns.index, name="daily_return").dropna()


@dataclass(frozen=True)
class StrategySpec:
    name: str
    runner: Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
    is_null: bool = False


def _build_strategy_specs() -> list[StrategySpec]:
    return [
        # Real strategies
        StrategySpec("rck_6factor",
                     _run_rck, is_null=False),
        StrategySpec("momentum_xsec",
                     lambda p, f: _run_exploration(CrossSectionalMomentum(lookback=120, top_k=3), p),
                     is_null=False),
        StrategySpec("momentum_ts",
                     lambda p, f: _run_exploration(TimeSeriesMomentum(lookback=120), p),
                     is_null=False),
        StrategySpec("mean_reversion",
                     lambda p, f: _run_exploration(RSIMeanReversion(rsi_period=14, oversold=30.0, overbought=70.0), p),
                     is_null=False),
        StrategySpec("dual_momentum",
                     lambda p, f: _run_exploration(DualMomentum(lookback=120, top_k=3, risk_off_asset="TLT"), p),
                     is_null=False),
        StrategySpec("vol_targeting",
                     lambda p, f: _run_exploration(VolatilityTargeting(vol_lookback=60, target_vol=0.15), p),
                     is_null=False),
        # Null baselines
        StrategySpec("null_equal_weight",    _null_equal_weight,    is_null=True),
        StrategySpec("null_random_uniform",  _null_random_uniform,  is_null=True),
        StrategySpec("null_inverse_vol",     _null_inverse_vol,     is_null=True),
        StrategySpec("null_hrp",             _null_hrp,             is_null=True),
    ]


# ============================================================================
# Data loading
# ============================================================================


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


def _load_from_parquet(
    parquet_path: Path,
    min_coverage_pct: float = 0.95,
    min_year: int | None = None,
):
    """Load a pre-gathered wide prices parquet and apply a coverage filter.

    ``min_year`` lets us truncate ancient rows before applying the coverage
    filter, so the parquet's deep-history veterans don't dominate the
    common-date window. E.g., ``min_year=2005`` applies the 95% coverage
    test over 2005-2026, not 1990-2026.
    """
    from pathlib import Path as _Path  # noqa: PLC0415

    path = _Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"parquet not found: {path}")
    prices = pd.read_parquet(path)
    if min_year is not None:
        prices = prices.loc[prices.index.year >= int(min_year)]
    coverage = prices.notna().sum()
    min_cov = int(min_coverage_pct * len(prices))
    sparse = coverage[coverage < min_cov].index.tolist()
    if sparse:
        prices = prices.drop(columns=sparse)
    prices = prices.ffill().dropna(how="any")
    if prices.empty:
        raise RuntimeError(
            f"After coverage filter {min_coverage_pct*100:.0f}% from {min_year}, "
            f"zero tickers survived. Loosen the filter."
        )
    # Factors over the final common window
    factors = load_factor_returns(
        start=prices.index[0].to_pydatetime(),
        end=prices.index[-1].to_pydatetime(),
        use_cache=True,
    )
    return prices, factors


# ============================================================================
# Metrics and statistics
# ============================================================================


def _cvar_95(daily_returns: np.ndarray) -> float:
    """Expected loss in the worst 5% of days. Positive = loss magnitude."""
    if len(daily_returns) < 20:
        return float("nan")
    q05 = float(np.percentile(daily_returns, 5))
    tail = daily_returns[daily_returns <= q05]
    if tail.size == 0:
        return float("nan")
    return float(-tail.mean())


def _annualized_sharpe(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 2:
        return 0.0
    mu = daily_returns.mean()
    sd = daily_returns.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(mu / sd * np.sqrt(252))


def _block_bootstrap_sharpe_ci(
    daily_returns: np.ndarray,
    n_boot: int = 1000,
    block_size: int = 20,
    seed: int = 20260422,
) -> tuple[float, float, float]:
    """Block-bootstrap 95% CI for the annualized Sharpe.

    Returns (ci_low, ci_high, frac_ci_above_zero).
    """
    rng = np.random.default_rng(seed)
    n = len(daily_returns)
    if n < block_size * 3:
        return (float("nan"), float("nan"), float("nan"))
    n_blocks = int(np.ceil(n / block_size))
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        starts = rng.integers(0, n - block_size, size=n_blocks)
        idx = np.concatenate([
            np.arange(s, s + block_size) for s in starts
        ])[:n]
        samples[i] = _annualized_sharpe(daily_returns[idx])
    lo, hi = np.percentile(samples, [2.5, 97.5])
    frac_above = float(np.mean(samples > 0.0))
    return float(lo), float(hi), frac_above


def _metrics_for_slice(
    daily_returns: pd.Series,
    total_n_trials: int,
) -> dict:
    if len(daily_returns) < 10:
        return {
            "n_days": len(daily_returns),
            "sharpe": np.nan, "sortino": np.nan, "max_drawdown": np.nan,
            "hit_rate": np.nan, "dsr": np.nan, "skew": np.nan,
            "excess_kurt": np.nan, "total_return": np.nan, "cvar_95": np.nan,
        }
    m = compute_all_metrics(daily_returns, n_trials=total_n_trials)
    wealth = (1.0 + daily_returns).cumprod()
    return {
        "n_days":        int(len(daily_returns)),
        "sharpe":        float(m.sharpe_ratio),
        "sortino":       float(m.sortino_ratio),
        "max_drawdown":  float(m.max_drawdown),
        "hit_rate":      float(m.hit_rate),
        "dsr":           float(m.deflated_sharpe),
        "skew":          float(m.skewness),
        "excess_kurt":   float(m.excess_kurtosis),
        "total_return":  float(wealth.iloc[-1] - 1.0),
        "cvar_95":       _cvar_95(daily_returns.values),
    }


# ============================================================================
# Main
# ============================================================================


def run_rigorous(
    universes: list[str],
    n_groups: int,
    test_groups: int,
    embargo_pct: float,
    years: float,
    n_boot: int,
    block_size: int,
    out_dir: Path,
    prices_parquet: Path | None = None,
    parquet_min_year: int | None = None,
    strategies_filter: list[str] | None = None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    specs = _build_strategy_specs()
    if strategies_filter:
        wanted = set(strategies_filter)
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            print(f"[error] --strategies filter matched nothing: {wanted}",
                  file=sys.stderr)
            return 2
    real_strategies = [s for s in specs if not s.is_null]
    null_strategies = [s for s in specs if s.is_null]
    pk = PurgedKFold(n_groups=n_groups, test_groups=test_groups, embargo_pct=embargo_pct)

    # Per-strategy DSR uses N_trials=len(real_strategies)=6 (not 540)
    n_trials_for_dsr = len(real_strategies)

    print("=" * 80)
    print(f"RIGOROUS CPCV TOURNAMENT")
    print("=" * 80)
    print(f"  Real strategies:  {len(real_strategies)}  (DSR deflation baseline)")
    print(f"  Null baselines:   {len(null_strategies)}  ({[s.name for s in null_strategies]})")
    print(f"  Universes:        {universes}")
    print(f"  CPCV:             N={n_groups}, k={test_groups}, embargo={embargo_pct*100:.1f}%,"
          f" splits/cell={pk.n_splits}, disjoint paths={pk.n_paths}")
    print(f"  History:          {years} years")
    print(f"  Bootstrap:        {n_boot} iterations, block size {block_size}d")
    print(f"  Per-strategy DSR: N_trials={n_trials_for_dsr} (correct deflation)")
    print("=" * 80)

    all_split_rows: list[dict] = []
    agg_rows: list[dict] = []
    daily_return_series: dict[tuple[str, str], pd.Series] = {}  # for bootstrap + plots

    # If prices_parquet is provided, override the universes loop and use it
    # as a single universe named from its stem.
    if prices_parquet is not None:
        universe_iter = [("parquet:" + prices_parquet.stem, None)]
    else:
        universe_iter = [(u, UNIVERSES[u]()) for u in universes]

    for universe_name, tickers in universe_iter:
        if prices_parquet is not None:
            print(f"\n[{universe_name}] loading from parquet {prices_parquet}"
                  f"  (min_year={parquet_min_year})...", flush=True)
            try:
                prices, factors = _load_from_parquet(
                    prices_parquet, min_coverage_pct=0.95,
                    min_year=parquet_min_year,
                )
            except Exception as exc:
                print(f"  [error] parquet load failed: {exc}", file=sys.stderr)
                continue
        else:
            print(f"\n[{universe_name}] loading {len(tickers)} tickers...",
                  flush=True)
            try:
                prices, factors = _load_inputs(tickers, years=years)
            except Exception as exc:
                print(f"  [error] data load failed: {exc}", file=sys.stderr)
                continue
        print(f"[{universe_name}] prices={prices.shape}  factors={factors.shape}  "
              f"window={prices.index[0].date()}->{prices.index[-1].date()}",
              flush=True)

        for spec in specs:
            tag = f"{universe_name}/{spec.name}"
            print(f"\n  [{tag}]  ({'NULL' if spec.is_null else 'REAL'})", flush=True)
            try:
                daily = spec.runner(prices, factors)
            except Exception as exc:
                print(f"    [error] runner failed: {exc}", file=sys.stderr)
                continue
            if len(daily) < 2 * n_groups:
                print(f"    [skip] only {len(daily)} daily obs")
                continue
            print(f"    daily returns: {len(daily)} obs"
                  f"  [{daily.index[0].date()} -> {daily.index[-1].date()}]", flush=True)

            daily_return_series[(universe_name, spec.name)] = daily

            # Per-split metrics
            split_rows: list[dict] = []
            for split in pk.split(n_obs=len(daily)):
                test_dates = daily.index[split.test_idx]
                slice_rets = daily.loc[test_dates]
                stats = _metrics_for_slice(slice_rets, total_n_trials=n_trials_for_dsr)
                stats.update({
                    "universe": universe_name,
                    "strategy": spec.name,
                    "is_null":  bool(spec.is_null),
                    "test_groups": ",".join(map(str, split.test_group_ids)),
                    "test_start":  str(test_dates[0].date()),
                    "test_end":    str(test_dates[-1].date()),
                })
                split_rows.append(stats)
                all_split_rows.append(stats)

            df = pd.DataFrame(split_rows)
            sharpes = df["sharpe"].dropna().to_numpy()
            dsrs = df["dsr"].dropna().to_numpy()
            dds = df["max_drawdown"].dropna().to_numpy()
            cvars = df["cvar_95"].dropna().to_numpy()

            # Full-series bootstrap CI (on the cell's full daily-return series)
            ci_lo, ci_hi, frac_pos = _block_bootstrap_sharpe_ci(
                daily.values, n_boot=n_boot, block_size=block_size,
            )
            full_sharpe = _annualized_sharpe(daily.values)

            agg = {
                "universe":           universe_name,
                "strategy":           spec.name,
                "is_null":            bool(spec.is_null),
                "n_splits":           int(len(df)),
                "full_sharpe":        full_sharpe,
                "boot_ci_lo":         ci_lo,
                "boot_ci_hi":         ci_hi,
                "boot_frac_ci_above_zero": frac_pos,
                "sharpe_mean":        float(np.mean(sharpes)) if sharpes.size else np.nan,
                "sharpe_median":      float(np.median(sharpes)) if sharpes.size else np.nan,
                "sharpe_p05":         float(np.percentile(sharpes, 5)) if sharpes.size else np.nan,
                "sharpe_p95":         float(np.percentile(sharpes, 95)) if sharpes.size else np.nan,
                "dsr_mean":           float(np.mean(dsrs)) if dsrs.size else np.nan,
                "dsr_median":         float(np.median(dsrs)) if dsrs.size else np.nan,
                "dsr_frac_over_95":   float(np.mean(dsrs > 0.95)) if dsrs.size else np.nan,
                "maxdd_median":       float(np.median(dds)) if dds.size else np.nan,
                "maxdd_p95":          float(np.percentile(dds, 95)) if dds.size else np.nan,
                "cvar_95_median":     float(np.median(cvars)) if cvars.size else np.nan,
            }
            agg_rows.append(agg)
            print(f"    full Sharpe={full_sharpe:+.3f}  "
                  f"boot CI95=[{ci_lo:+.3f},{ci_hi:+.3f}]  "
                  f"frac CI>0={frac_pos*100:.1f}%  "
                  f"SR median across splits={agg['sharpe_median']:+.3f}  "
                  f"DSR median={agg['dsr_median']:.3f}  "
                  f"CVaR95 median={agg['cvar_95_median']*100:.2f}%  "
                  f"MaxDD median={agg['maxdd_median']*100:.2f}%", flush=True)

    if not agg_rows:
        print("[error] no data produced.", file=sys.stderr)
        return 2

    # Persist aggregates
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "aggregate.csv", index=False)
    splits_df = pd.DataFrame(all_split_rows)
    splits_df.to_csv(out_dir / "all_splits.csv", index=False)

    # =========================================================================
    # Mann-Whitney U: each real strategy's Sharpes vs pooled-null Sharpes
    # =========================================================================
    print("\n" + "=" * 80)
    print("MANN-WHITNEY U:  REAL vs POOLED NULL  (per universe)")
    print("=" * 80)
    from scipy.stats import mannwhitneyu  # lazy

    mw_rows: list[dict] = []
    # Iterate over the universes actually run, not the CLI arg (which may be
    # shadowed when --prices-parquet overrides the universe fetchers).
    actual_universes = sorted(splits_df["universe"].unique().tolist())
    for universe_name in actual_universes:
        # Pool null sharpes across all null strategies on this universe
        null_mask = (splits_df["universe"] == universe_name) & (splits_df["is_null"])
        null_sharpes = splits_df.loc[null_mask, "sharpe"].dropna().to_numpy()
        if null_sharpes.size == 0:
            continue
        null_median = float(np.median(null_sharpes))
        print(f"\n  [{universe_name}] pooled-null (n={null_sharpes.size}): "
              f"median Sharpe={null_median:+.3f}")
        for spec in real_strategies:
            mask = (splits_df["universe"] == universe_name) & (splits_df["strategy"] == spec.name)
            strat_sharpes = splits_df.loc[mask, "sharpe"].dropna().to_numpy()
            if strat_sharpes.size == 0:
                continue
            # Alternative="greater": real median > null median
            u, p = mannwhitneyu(strat_sharpes, null_sharpes, alternative="greater")
            median_diff = float(np.median(strat_sharpes) - null_median)
            significant = "YES" if p < 0.05 else "no "
            print(f"    {spec.name:<17} median={np.median(strat_sharpes):+.3f}  "
                  f"diff_vs_null={median_diff:+.3f}  U={u:.0f}  p={p:.4f}  "
                  f"sig@5pct={significant}")
            mw_rows.append({
                "universe":        universe_name,
                "strategy":        spec.name,
                "strategy_median": float(np.median(strat_sharpes)),
                "null_median":     null_median,
                "diff":            median_diff,
                "u_stat":          float(u),
                "p_value":         float(p),
                "significant_5pct": bool(p < 0.05),
            })
    mw_df = pd.DataFrame(mw_rows)
    mw_df.to_csv(out_dir / "mannwhitney.csv", index=False)

    # =========================================================================
    # Verdict
    # =========================================================================
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Candidates that pass all three gates:
    #   (a) bootstrap CI lower bound > 0  (Sharpe distinguishable from zero)
    #   (b) Mann-Whitney p < 0.05 vs pooled null on its universe
    #   (c) DSR median > 0.95  (institutional bar after multiple-testing deflation)
    passes = []
    for r in agg_rows:
        if r["is_null"]:
            continue
        mw_hit = mw_df[(mw_df["universe"] == r["universe"]) & (mw_df["strategy"] == r["strategy"])]
        if mw_hit.empty:
            continue
        mw_row = mw_hit.iloc[0]
        gate_a = (r["boot_ci_lo"] > 0)
        gate_b = bool(mw_row["significant_5pct"])
        gate_c = (r["dsr_median"] > 0.95)
        all_pass = gate_a and gate_b and gate_c
        mark = "PASS" if all_pass else "fail"
        print(f"  [{mark}] {r['universe']}/{r['strategy']:<17}  "
              f"CI>0:{gate_a!s:<5} MW(p<0.05):{gate_b!s:<5} DSR>0.95:{gate_c!s:<5}  "
              f"(full Sharpe={r['full_sharpe']:+.3f}, bootCI=[{r['boot_ci_lo']:+.3f},{r['boot_ci_hi']:+.3f}], "
              f"DSR={r['dsr_median']:.3f})")
        if all_pass:
            passes.append(r)

    print()
    if passes:
        print(f"{len(passes)} strategy-universe cell(s) cleared ALL THREE gates:")
        for r in passes:
            print(f"  -> {r['universe']}/{r['strategy']}")
    else:
        print("NO strategy-universe cell cleared all three gates.")
        print("Interpretation: on this data, none of the 6 real strategies can be")
        print("statistically distinguished from a random/equal/invvol null baseline")
        print("at the institutional bar (DSR > 0.95 after deflation).")
        print()
        print("This is the honest answer. It does not mean the strategies are worthless")
        print("-- they may still compound positively in expectation. It means we do not")
        print("have statistical evidence of edge at the 95% confidence level on")
        print("5y of data against random baselines.")
    print("=" * 80)

    # ====================
    # Figures
    # ====================
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        _make_plots(agg_df, splits_df, daily_return_series, out_dir / "figures")
        print(f"\nStatic PNG plots -> {out_dir / 'figures'}")
    except Exception as exc:
        print(f"[warn] plotting failed: {exc}", file=sys.stderr)

    # ====================
    # Text report
    # ====================
    _write_text_report(agg_df, mw_df, passes, out_dir, n_trials_for_dsr, n_groups, test_groups)

    # ====================
    # JSON meta
    # ====================
    meta = {
        "generated_at":      datetime.now().isoformat(timespec="seconds"),
        "real_strategies":   [s.name for s in real_strategies],
        "null_strategies":   [s.name for s in null_strategies],
        "universes":         universes,
        "n_groups":          n_groups,
        "test_groups":       test_groups,
        "embargo_pct":       embargo_pct,
        "n_paths":           pk.n_paths,
        "n_splits":          pk.n_splits,
        "years_history":     years,
        "dsr_n_trials":      n_trials_for_dsr,
        "bootstrap_iters":   n_boot,
        "bootstrap_block":   block_size,
        "n_cells_passed":    len(passes),
        "gate_verdict":      "PASS" if passes else "FAIL",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return 0 if passes else 1


# ============================================================================
# Plotting and report
# ============================================================================


def _make_plots(
    agg_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    daily_series: dict,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Boxplot of Sharpe across splits, per (strategy, universe)
    universes = sorted(splits_df["universe"].unique())
    fig, axes = plt.subplots(len(universes), 1, figsize=(12, 4 * len(universes)))
    if len(universes) == 1:
        axes = [axes]
    for ax, uni in zip(axes, universes):
        sub = splits_df[splits_df["universe"] == uni]
        strategies = list(sub["strategy"].unique())
        data = [sub[sub["strategy"] == s]["sharpe"].dropna().values for s in strategies]
        colors = ["#d62728" if sub[sub["strategy"] == s]["is_null"].iloc[0] else "#1f77b4"
                  for s in strategies]
        bp = ax.boxplot(data, labels=strategies, patch_artist=True, showmeans=True)
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.5)
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
        ax.set_title(f"Per-split Sharpe distribution — universe={uni}  "
                     f"(red=null baseline, blue=real strategy)")
        ax.set_ylabel("Annualized Sharpe")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "01_sharpe_boxplots.png", dpi=110)
    plt.close()

    # 2. Bootstrap CI dot-plot
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(agg_df) + 1.0))
    y = np.arange(len(agg_df))
    colors = ["#d62728" if b else "#1f77b4" for b in agg_df["is_null"]]
    ax.hlines(y, agg_df["boot_ci_lo"], agg_df["boot_ci_hi"], colors=colors, linewidth=3)
    ax.plot(agg_df["full_sharpe"], y, "o", color="k")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.universe}/{r.strategy}" for r in agg_df.itertuples()])
    ax.axvline(0, color="r", linestyle="--", alpha=0.7, label="zero")
    ax.set_xlabel("Annualized Sharpe (block-bootstrap 95% CI)")
    ax.set_title("Bootstrap 95% CI on full-series Sharpe  (red=null baseline, blue=real)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "02_bootstrap_ci.png", dpi=110)
    plt.close()

    # 3. Cumulative equity curves
    fig, axes = plt.subplots(len(universes), 1, figsize=(12, 4 * len(universes)))
    if len(universes) == 1:
        axes = [axes]
    for ax, uni in zip(axes, universes):
        for (uu, name), series in daily_series.items():
            if uu != uni:
                continue
            wealth = (1 + series).cumprod()
            is_null = agg_df[(agg_df["universe"] == uni) & (agg_df["strategy"] == name)]["is_null"].iloc[0]
            ax.plot(wealth.index, wealth.values,
                    label=f"{'[null] ' if is_null else ''}{name}",
                    alpha=0.7, linewidth=1.2 + (0.6 if not is_null else 0))
        ax.set_title(f"Cumulative wealth ($1 base) — universe={uni}")
        ax.set_ylabel("Wealth")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "03_equity_curves.png", dpi=110)
    plt.close()


def _write_text_report(
    agg_df: pd.DataFrame,
    mw_df: pd.DataFrame,
    passes: list[dict],
    out_dir: Path,
    n_trials_for_dsr: int,
    n_groups: int,
    test_groups: int,
) -> None:
    lines = []
    lines.append("=" * 80)
    lines.append(f"RIGOROUS CPCV REPORT  |  {datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 80)
    lines.append(f"  CPCV: N={n_groups}, k={test_groups}, splits/cell={int(agg_df['n_splits'].iloc[0])}")
    lines.append(f"  DSR deflation: N_trials={n_trials_for_dsr} (per-strategy, CORRECT)")
    lines.append("")
    lines.append(" universe  strategy          null?  full_SR   bootCI95             CI>0%  "
                 "SRmed    DSRmed   CVaR95  MDD_med")
    lines.append(" " + "-" * 108)
    for r in agg_df.itertuples():
        lines.append(
            f" {r.universe:<9} {r.strategy:<17} "
            f"{'*' if r.is_null else ' ':<6}"
            f"{r.full_sharpe:+.3f}   "
            f"[{r.boot_ci_lo:+.3f},{r.boot_ci_hi:+.3f}]    "
            f"{r.boot_frac_ci_above_zero*100:5.1f}  "
            f"{r.sharpe_median:+.3f}   "
            f"{r.dsr_median:.3f}    "
            f"{r.cvar_95_median*100:5.2f}%  "
            f"{r.maxdd_median*100:5.2f}%"
        )
    lines.append("")
    lines.append("MANN-WHITNEY U (real strategy Sharpe dist. vs pooled-null dist., alt=greater):")
    for r in mw_df.itertuples():
        lines.append(f"  {r.universe:<9} {r.strategy:<17} "
                     f"diff vs null = {r.diff:+.3f}   p = {r.p_value:.4f}   "
                     f"{'SIG at 5%' if r.significant_5pct else '--'}")
    lines.append("")
    lines.append("VERDICT (all three gates required: bootCI>0, Mann-Whitney p<0.05, DSR>0.95):")
    if passes:
        lines.append(f"  {len(passes)} cell(s) cleared all three gates:")
        for p in passes:
            lines.append(f"    -> {p['universe']}/{p['strategy']}")
    else:
        lines.append("  NO cell cleared all three gates.")
        lines.append("  Interpretation: no strategy has statistical edge vs random baselines")
        lines.append("  at the 95% confidence level on this data. Strategies may still")
        lines.append("  compound positively; we simply cannot distinguish signal from noise")
        lines.append("  with the available evidence.")
    lines.append("=" * 80)
    text = "\n".join(lines)
    (out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")


# ============================================================================
# CLI
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universes", type=str, default="sp100,etfs")
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--test-groups", type=int, default=2)
    parser.add_argument("--embargo-pct", type=float, default=0.01)
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="results/cpcv_rigorous")
    parser.add_argument("--prices-parquet", type=str, default=None,
                        help="Pre-gathered prices parquet (skips universe fetchers "
                             "and uses this file as a single universe).")
    parser.add_argument("--parquet-min-year", type=int, default=None,
                        help="Truncate rows before this year before applying the "
                             "coverage filter on a parquet. Example: 2005.")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated strategy names to include "
                             "(default: all 9). Useful to skip rck_6factor on very "
                             "wide universes where CVXPY is prohibitively slow.")
    args = parser.parse_args(argv)
    return run_rigorous(
        universes=[u.strip() for u in args.universes.split(",") if u.strip()],
        n_groups=int(args.n_groups),
        test_groups=int(args.test_groups),
        embargo_pct=float(args.embargo_pct),
        years=float(args.years),
        n_boot=int(args.n_boot),
        block_size=int(args.block_size),
        out_dir=Path(args.out_dir),
        prices_parquet=Path(args.prices_parquet) if args.prices_parquet else None,
        parquet_min_year=args.parquet_min_year,
        strategies_filter=([s.strip() for s in args.strategies.split(",") if s.strip()]
                           if args.strategies else None),
    )


if __name__ == "__main__":
    sys.exit(main())
