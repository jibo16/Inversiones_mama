"""Tests for inversiones_mama.backtest.engine.

Four classes of tests:

1. Schema / accounting sanity (synthetic data; no network).
2. Zero-cost conservation (costs off -> wealth ~ compounded daily returns).
3. Rebalance scheduling correctness.
4. Live end-to-end: real 10-ETF universe, real Ken French factors, real
   Kelly + RCK. Sanity bounds on final wealth, turnover, and Sharpe (marked
   ``@pytest.mark.live``).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    _rebalance_schedule,
    walk_forward_backtest,
)


# --------------------------------------------------------------------------- #
# Synthetic data                                                              #
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_prices_and_factors() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct plausible multi-asset prices and a matching factor panel.

    We use ``config.lookback_days`` default (252) + ~1.5 years of post-warmup
    data, so we have enough history for the engine to rebalance several times.
    """
    rng = np.random.default_rng(20260422)
    n = 600  # ~2.4 years of business days
    dates = pd.date_range("2023-01-02", periods=n, freq="B")

    # 5 assets, correlated with market factor
    factor_mkt = rng.normal(0.0004, 0.010, n)
    factor_smb = rng.normal(0.0001, 0.005, n)
    factor_hml = rng.normal(0.0001, 0.005, n)
    factor_rmw = rng.normal(0.0002, 0.004, n)
    factor_cma = rng.normal(0.0000, 0.004, n)
    factor_mom = rng.normal(0.0002, 0.007, n)
    rf = np.full(n, 0.00015)

    factors = pd.DataFrame(
        {"Mkt-RF": factor_mkt, "SMB": factor_smb, "HML": factor_hml,
         "RMW": factor_rmw, "CMA": factor_cma, "MOM": factor_mom, "RF": rf},
        index=dates,
    )

    # Planted betas per asset
    betas = {
        "A": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # pure market
        "B": [1.0, 0.5, 0.3, 0.1, 0.0, 0.0],   # value-ish
        "C": [1.0, 0.0, 0.0, 0.0, 0.0, 0.4],   # momentum-ish
        "D": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],   # low-beta
        "E": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # pure idiosyncratic
    }
    factor_mat = factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]].to_numpy()
    idio = rng.normal(0, 0.003, (n, len(betas)))
    asset_rets = {}
    for i, (t, b) in enumerate(betas.items()):
        # r = RF + sum(beta * factor) + idio
        asset_rets[t] = rf + factor_mat @ np.asarray(b) + idio[:, i]
    ret_df = pd.DataFrame(asset_rets, index=dates)

    # Build prices from returns (geometric compounding)
    prices = (1.0 + ret_df).cumprod() * 100.0
    # Prepend initial price so pct_change yields the same ret_df on index 1..n
    first = pd.DataFrame(
        {t: [100.0] for t in asset_rets},
        index=[dates[0] - pd.tseries.offsets.BDay()],
    )
    prices = pd.concat([first, prices])
    # Factors need to cover the prices range too
    first_factor = factors.iloc[[0]].copy()
    first_factor.index = [prices.index[0]]
    factors_full = pd.concat([first_factor, factors])
    return prices, factors_full


# --------------------------------------------------------------------------- #
# Schedule                                                                    #
# --------------------------------------------------------------------------- #


def test_rebalance_schedule_monthly_skips_warmup():
    dates = pd.date_range("2024-01-02", periods=400, freq="B")
    schedule = _rebalance_schedule(dates, "ME", skip_warmup=252)
    # Should produce at least 4 month-ends after skipping 252 days
    assert len(schedule) >= 4
    # All scheduled dates must be past the warmup cutoff
    cutoff = dates[252]
    assert all(d >= cutoff for d in schedule)


def test_rebalance_schedule_empty_when_too_short():
    dates = pd.date_range("2024-01-02", periods=100, freq="B")
    schedule = _rebalance_schedule(dates, "ME", skip_warmup=252)
    assert schedule == set()


def test_rebalance_schedule_quarterly():
    dates = pd.date_range("2024-01-02", periods=800, freq="B")
    monthly = _rebalance_schedule(dates, "ME", skip_warmup=252)
    quarterly = _rebalance_schedule(dates, "QE", skip_warmup=252)
    # Quarterly must produce strictly fewer dates than monthly
    assert 0 < len(quarterly) < len(monthly)


# --------------------------------------------------------------------------- #
# Engine smoke                                                                #
# --------------------------------------------------------------------------- #


def test_engine_produces_expected_schema(synthetic_prices_and_factors):
    prices, factors = synthetic_prices_and_factors
    result = walk_forward_backtest(prices, factors)
    assert isinstance(result, BacktestResult)
    assert not result.daily_returns.empty
    assert not result.wealth.empty
    assert len(result.daily_returns) == len(result.wealth)
    assert len(result.rebalance_records) >= 1


def test_engine_wealth_series_monotonic_in_time(synthetic_prices_and_factors):
    prices, factors = synthetic_prices_and_factors
    result = walk_forward_backtest(prices, factors)
    # Index must be strictly increasing
    assert result.wealth.index.is_monotonic_increasing


def test_engine_daily_returns_recover_wealth(synthetic_prices_and_factors):
    """Cumulative product of (1 + daily returns) should trace the wealth series."""
    prices, factors = synthetic_prices_and_factors
    result = walk_forward_backtest(prices, factors, BacktestConfig(apply_costs=False))
    # Without costs, wealth_t = wealth_0 * prod(1 + r_s) for s <= t
    reconstructed = result.config.initial_capital * (1 + result.daily_returns).cumprod()
    # Tolerance: floating point + the single-day snap-to-target at rebalance
    np.testing.assert_allclose(
        result.wealth.values,
        reconstructed.values,
        rtol=1e-9,
        atol=1e-6,
    )


def test_engine_with_costs_reduces_wealth(synthetic_prices_and_factors):
    """Applying costs should produce lower terminal wealth than cost-free run."""
    prices, factors = synthetic_prices_and_factors
    r_with = walk_forward_backtest(prices, factors, BacktestConfig(apply_costs=True))
    r_without = walk_forward_backtest(prices, factors, BacktestConfig(apply_costs=False))
    assert r_with.final_wealth <= r_without.final_wealth
    assert r_with.cumulative_cost > 0
    assert r_without.cumulative_cost == 0


def test_engine_rebalance_records_well_formed(synthetic_prices_and_factors):
    prices, factors = synthetic_prices_and_factors
    result = walk_forward_backtest(prices, factors)
    for rec in result.rebalance_records:
        assert isinstance(rec.date, pd.Timestamp)
        # Weights reindex must cover all tickers
        assert set(rec.target_weights.index) == set(prices.columns)
        # Target weights sum to <= 1 (cash is allowed)
        assert rec.target_weights.sum() <= 1.0 + 1e-6
        # Per-name cap (drive from config, not a magic number)
        from inversiones_mama.config import MAX_WEIGHT_PER_NAME
        assert (rec.target_weights <= MAX_WEIGHT_PER_NAME + 1e-6).all()
        # No negative weights (long-only)
        assert (rec.target_weights >= -1e-9).all()


def test_engine_annualized_turnover_cost_is_bounded(synthetic_prices_and_factors):
    prices, factors = synthetic_prices_and_factors
    result = walk_forward_backtest(prices, factors)
    # For synthetic data with 15% cap and monthly rebal, costs should be well
    # under the 1.5% annual threshold in config (sanity bound is 5%).
    assert 0 <= result.annualized_turnover_cost < 0.05


def test_engine_empty_prices_raises():
    with pytest.raises(ValueError, match="prices DataFrame is empty"):
        walk_forward_backtest(pd.DataFrame(), pd.DataFrame({"RF": [0.0]}))


def test_engine_rejects_short_history():
    """If overlap < lookback_days + 5, the engine refuses to run."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    p = pd.DataFrame({"A": np.linspace(100, 110, 100)}, index=dates)
    f = pd.DataFrame(
        {c: np.zeros(100) for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]},
        index=dates,
    )
    with pytest.raises(ValueError, match="need at least lookback_days"):
        walk_forward_backtest(p, f)


# --------------------------------------------------------------------------- #
# Live end-to-end                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_walk_forward_5y(tmp_path, monkeypatch):
    """Full pipeline on real 5y data — most important end-to-end check."""
    from inversiones_mama.config import UNIVERSE
    from inversiones_mama.data import factors as factors_mod
    from inversiones_mama.data import prices as prices_mod
    from inversiones_mama.data.factors import load_factor_returns
    from inversiones_mama.data.prices import load_prices
    from inversiones_mama.simulation.metrics import compute_all_metrics

    monkeypatch.setattr(prices_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(factors_mod, "CACHE_DIR", tmp_path)

    tickers = list(UNIVERSE.keys())
    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=5 * 365 + 14)

    prices = load_prices(tickers, start, end)
    factors = load_factor_returns(start=start, end=end)

    result = walk_forward_backtest(prices, factors)

    # At least ~36 monthly rebalances in ~4 years post-warmup
    assert len(result.rebalance_records) >= 20
    # Final wealth in a plausible band — strategy should not blow up nor 10x
    assert 500 <= result.final_wealth <= 50_000
    # Turnover-cost gate: must come in under Jorge's 1.5% annual threshold
    assert result.annualized_turnover_cost < 0.015, (
        f"cost gate violated: {result.annualized_turnover_cost:.4f}"
    )
    # Sharpe should be positive on a factor-tilted long-only portfolio over 5y
    metrics = compute_all_metrics(result.daily_returns.dropna(), n_trials=10)
    assert metrics.sharpe_ratio > 0, f"Sharpe unexpectedly non-positive: {metrics.sharpe_ratio}"
    # Deflated Sharpe: for v1a we just want a valid number in [0, 1]
    assert 0.0 <= metrics.deflated_sharpe <= 1.0
