"""Block-bootstrap forward simulation per strategy.

For each strategy's backtested daily-return series, resample 1000
forward paths of ~252 trading days each using the stationary
block-bootstrap with a 20-day mean block length. This preserves
volatility clustering + heavy tails (no Gaussian assumption) and
produces the distribution of terminal wealth, max drawdown, and
probability of X% loss under the assumption that the HISTORICAL
return-generating process persists.

Reads ``results/meta_backtest/daily/<strategy_id>.csv`` for each
strategy's daily returns, written by ``run_meta_backtest.py``.

Outputs
-------
results/meta_bootstrap/
  summary.csv                          per-strategy terminal/DD percentiles
  paths/<strategy_id>_wealth.parquet   (1000 x horizon) wealth paths
  paths/<strategy_id>_stats.json       per-strategy stats
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from inversiones_mama.execution.strategy_catalog import STRATEGY_CATALOG
from inversiones_mama.simulation.bootstrap import stationary_bootstrap


def _load_daily(strategy_id: str, backtest_dir: Path) -> pd.Series | None:
    path = backtest_dir / "daily" / f"{strategy_id}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "daily_return" not in df.columns:
        return None
    return df["daily_return"].dropna()


def _bootstrap_paths(
    daily: pd.Series,
    n_paths: int,
    horizon_days: int,
    mean_block_length: int,
    seed: int,
) -> np.ndarray:
    """Return an (n_paths, horizon_days) matrix of daily-return samples."""
    rng = np.random.default_rng(seed)
    samples = stationary_bootstrap(
        daily.values,
        horizon=horizon_days,
        mean_block_length=mean_block_length,
        n_samples=n_paths,
        rng=rng,
    )
    # Shape: (n_paths, horizon_days)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    return samples


def _path_stats(paths: np.ndarray) -> dict:
    """Compute terminal-wealth + max-DD stats across paths."""
    wealth = np.cumprod(1.0 + paths, axis=1)
    terminal = wealth[:, -1] - 1.0  # total return per path

    # Max drawdown per path
    running_max = np.maximum.accumulate(wealth, axis=1)
    dd = 1.0 - wealth / running_max
    max_dd = dd.max(axis=1)

    return {
        "n_paths":                  int(paths.shape[0]),
        "horizon_days":             int(paths.shape[1]),
        "terminal_return_mean":     float(np.mean(terminal)),
        "terminal_return_median":   float(np.median(terminal)),
        "terminal_return_p05":      float(np.percentile(terminal, 5)),
        "terminal_return_p25":      float(np.percentile(terminal, 25)),
        "terminal_return_p75":      float(np.percentile(terminal, 75)),
        "terminal_return_p95":      float(np.percentile(terminal, 95)),
        "max_dd_median":            float(np.median(max_dd)),
        "max_dd_p75":               float(np.percentile(max_dd, 75)),
        "max_dd_p95":               float(np.percentile(max_dd, 95)),
        "max_dd_p99":               float(np.percentile(max_dd, 99)),
        "prob_loss_10pct":          float(np.mean(terminal < -0.10)),
        "prob_loss_20pct":          float(np.mean(terminal < -0.20)),
        "prob_loss_40pct":          float(np.mean(terminal < -0.40)),
        "prob_dd_gte_20pct":        float(np.mean(max_dd >= 0.20)),
        "prob_dd_gte_40pct":        float(np.mean(max_dd >= 0.40)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-paths", type=int, default=1000)
    parser.add_argument("--horizon-days", type=int, default=252)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--backtest-dir", type=str, default="results/meta_backtest")
    parser.add_argument("--out-dir", type=str, default="results/meta_bootstrap")
    parser.add_argument("--save-paths", action="store_true",
                        help="Also persist the (n_paths x horizon) wealth matrix per strategy.")
    args = parser.parse_args(argv)

    backtest_dir = Path(args.backtest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paths").mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("META-PORTFOLIO BOOTSTRAP FORWARD SIM")
    print("=" * 72)
    print(f"  n_paths={args.n_paths}  horizon={args.horizon_days}d  "
          f"block_length={args.block_length}d")

    summary_rows = []
    for i, spec in enumerate(STRATEGY_CATALOG, 1):
        daily = _load_daily(spec.strategy_id, backtest_dir)
        if daily is None or len(daily) < 100:
            print(f"  [{i:>2}/20] {spec.strategy_id:<30} SKIP (no/insufficient backtest)")
            continue
        paths = _bootstrap_paths(
            daily,
            n_paths=args.n_paths,
            horizon_days=args.horizon_days,
            mean_block_length=args.block_length,
            seed=20260423 + hash(spec.strategy_id) % 1000,
        )
        stats = _path_stats(paths)
        stats["strategy_id"] = spec.strategy_id

        (out_dir / "paths" / f"{spec.strategy_id}_stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8",
        )
        if args.save_paths:
            wealth = np.cumprod(1.0 + paths, axis=1)
            pd.DataFrame(wealth).to_parquet(
                out_dir / "paths" / f"{spec.strategy_id}_wealth.parquet"
            )

        summary_rows.append(stats)
        print(f"  [{i:>2}/20] {spec.strategy_id:<30} "
              f"term p05/p50/p95 = "
              f"{stats['terminal_return_p05']*100:+6.1f}% / "
              f"{stats['terminal_return_median']*100:+6.1f}% / "
              f"{stats['terminal_return_p95']*100:+6.1f}%  "
              f"MDD p95 = {stats['max_dd_p95']*100:5.2f}%  "
              f"P(loss>20%) = {stats['prob_loss_20pct']*100:4.1f}%")

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved: {out_dir}/summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
