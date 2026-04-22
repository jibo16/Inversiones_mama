"""Tests for inversiones_mama.simulation.monte_carlo.

Four classes of tests:

1. **Shape/schema sanity** on synthetic inputs.
2. **Accounting invariants**: all-zeros weights => no drawdowns, probabilities zero.
3. **RCK bound empirical property test**: planted mu, Sigma, run MC with many
   paths; empirical P(DD >= alpha) should not exceed beta by much.
4. **Live smoke**: real 10-ETF universe; weights from the live walk-forward
   engine; Jorge's drawdown gates reported (not asserted — that's Step 10's
   job).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.simulation.monte_carlo import (
    MCValidationResult,
    run_mc_rck_validation,
)


# --------------------------------------------------------------------------- #
# Synthetic fixtures                                                          #
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    """3-asset synthetic returns, 1,500 business days, with moderate correlation."""
    rng = np.random.default_rng(1234)
    n = 1500
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    mkt = rng.normal(0.0005, 0.012, n)
    return pd.DataFrame(
        {
            "A": mkt + rng.normal(0, 0.006, n),
            "B": 0.7 * mkt + rng.normal(0, 0.008, n),
            "C": rng.normal(0.0002, 0.010, n),  # uncorrelated
        },
        index=idx,
    )


# --------------------------------------------------------------------------- #
# Schema                                                                      #
# --------------------------------------------------------------------------- #


def test_mc_result_schema(synthetic_returns):
    result = run_mc_rck_validation(
        synthetic_returns,
        n_paths=200,
        horizon_days=60,
        rng=np.random.default_rng(0),
    )
    assert isinstance(result, MCValidationResult)
    assert result.n_paths == 200
    assert result.horizon_days == 60
    assert result.terminal_wealth.shape == (200,)
    assert result.max_drawdowns.shape == (200,)
    # Percentiles are ordered
    assert result.terminal_p05 <= result.terminal_p25 <= result.terminal_median
    assert result.terminal_median <= result.terminal_p75 <= result.terminal_p95
    assert 0.0 <= result.dd_median <= result.dd_p95 <= result.dd_p99
    # Probabilities in [0, 1]
    assert 0.0 <= result.prob_loss_40pct <= 1.0
    assert 0.0 <= result.prob_loss_60pct <= 1.0
    assert 0.0 <= result.prob_dd_exceeds_rck_alpha <= 1.0


def test_mc_rng_reproducibility(synthetic_returns):
    a = run_mc_rck_validation(synthetic_returns, n_paths=100, horizon_days=30,
                               rng=np.random.default_rng(99))
    b = run_mc_rck_validation(synthetic_returns, n_paths=100, horizon_days=30,
                               rng=np.random.default_rng(99))
    np.testing.assert_array_equal(a.terminal_wealth, b.terminal_wealth)
    np.testing.assert_array_equal(a.max_drawdowns, b.max_drawdowns)


# --------------------------------------------------------------------------- #
# Accounting                                                                  #
# --------------------------------------------------------------------------- #


def test_mc_all_zero_weights_no_drawdown(synthetic_returns):
    """If weights are all zero, the portfolio is all cash; no drawdown."""
    zero_w = pd.Series([0.0, 0.0, 0.0], index=["A", "B", "C"])
    result = run_mc_rck_validation(
        synthetic_returns,
        weights=zero_w,
        n_paths=300,
        horizon_days=60,
        initial_capital=5000.0,
    )
    # All terminal == initial (no loss possible when holding only cash)
    assert np.allclose(result.terminal_wealth, 5000.0)
    # No drawdowns
    assert np.allclose(result.max_drawdowns, 0.0)
    # All gates trivially satisfied
    assert result.prob_loss_40pct == 0.0
    assert result.prob_loss_60pct == 0.0
    assert result.prob_dd_exceeds_rck_alpha == 0.0


def test_mc_terminal_mean_grows_with_positive_mu(synthetic_returns):
    """Fully invested in a positive-mu asset -> expected terminal > initial."""
    w = pd.Series([1.0, 0.0, 0.0], index=["A", "B", "C"])
    result = run_mc_rck_validation(
        synthetic_returns,
        weights=w,
        n_paths=500,
        horizon_days=252,
        rng=np.random.default_rng(7),
    )
    # Asset A has mu ~0.0005 daily; 252 days compounded should show positive drift
    assert result.terminal_mean > 5000.0


def test_mc_bad_inputs(synthetic_returns):
    with pytest.raises(ValueError, match="empty"):
        run_mc_rck_validation(pd.DataFrame())
    with pytest.raises(ValueError, match="n_paths"):
        run_mc_rck_validation(synthetic_returns, n_paths=5)
    with pytest.raises(ValueError, match="horizon_days"):
        run_mc_rck_validation(synthetic_returns, horizon_days=1)
    with pytest.raises(ValueError, match="bootstrap_method"):
        run_mc_rck_validation(synthetic_returns, n_paths=100, horizon_days=30,
                               bootstrap_method="bogus")


def test_mc_gate_verdicts_from_zero_strategy(synthetic_returns):
    """A zero-allocation strategy passes every gate (no loss possible)."""
    zero_w = pd.Series([0.0, 0.0, 0.0], index=["A", "B", "C"])
    r = run_mc_rck_validation(synthetic_returns, weights=zero_w, n_paths=100, horizon_days=30)
    assert r.gate_prob_loss_40pct_pass
    assert r.gate_prob_loss_60pct_pass
    assert r.gate_dd_95th_pass
    assert r.gate_rck_bound_pass


# --------------------------------------------------------------------------- #
# Empirical RCK bound property                                                #
# --------------------------------------------------------------------------- #


def test_mc_rck_bound_approximately_honored(synthetic_returns):
    """Empirical P(DD >= alpha) should not blow past beta on a well-conditioned case.

    This is the *raison d'etre* of Step 9: if the RCK solver produces weights
    whose empirical drawdown probability exceeds the theoretical bound by a
    wide margin, the risk budget is mis-specified. We allow some empirical
    slack (bootstrap tail bias and finite-sample noise), but the RCK bound
    must be order-of-magnitude honored.
    """
    result = run_mc_rck_validation(
        synthetic_returns,
        n_paths=2000,
        horizon_days=252,
        # Loose drawdown bound so empirical probability is easy to measure
        alpha=0.30,
        beta=0.25,
        mean_block_length=10,
        rng=np.random.default_rng(42),
    )
    # Allow a 10pp empirical margin over beta (bootstrap can overstate tails)
    assert result.prob_dd_exceeds_rck_alpha <= result.rck_beta + 0.10, (
        f"P(DD>={result.rck_alpha}) = {result.prob_dd_exceeds_rck_alpha:.3f} "
        f"exceeds beta={result.rck_beta} by more than 10pp"
    )


def test_mc_prob_ordering(synthetic_returns):
    """P(loss > 60%) <= P(loss > 40%) by monotonicity of the indicator."""
    result = run_mc_rck_validation(
        synthetic_returns, n_paths=500, horizon_days=180, rng=np.random.default_rng(3)
    )
    assert result.prob_loss_60pct <= result.prob_loss_40pct


# --------------------------------------------------------------------------- #
# Live smoke: real 10-ETF universe, weights from walk-forward engine          #
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_mc_on_final_rebalance(tmp_path, monkeypatch):
    """End-to-end: run the engine, take the final rebalance's target weights,
    run MC validation on them, and assert every gate is defined and numeric."""
    from inversiones_mama.backtest.engine import walk_forward_backtest
    from inversiones_mama.config import UNIVERSE
    from inversiones_mama.data import factors as factors_mod
    from inversiones_mama.data import prices as prices_mod
    from inversiones_mama.data.factors import load_factor_returns
    from inversiones_mama.data.prices import load_prices

    monkeypatch.setattr(prices_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(factors_mod, "CACHE_DIR", tmp_path)

    tickers = list(UNIVERSE.keys())
    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=5 * 365 + 14)

    prices = load_prices(tickers, start, end)
    factors = load_factor_returns(start=start, end=end)
    engine_result = walk_forward_backtest(prices, factors)

    # Take the last rebalance's target weights
    assert engine_result.rebalance_records, "engine should have produced rebalances"
    last_weights = engine_result.rebalance_records[-1].target_weights

    returns = prices.pct_change().dropna().loc[:, tickers]
    mc = run_mc_rck_validation(
        returns.tail(252),  # use most recent year as the bootstrap source
        weights=last_weights,
        n_paths=2000,
        horizon_days=252,
        rng=np.random.default_rng(2026),
    )
    # Each gate is a boolean; each probability is in [0, 1]
    assert isinstance(mc.gate_prob_loss_40pct_pass, bool)
    assert isinstance(mc.gate_prob_loss_60pct_pass, bool)
    assert isinstance(mc.gate_dd_95th_pass, bool)
    assert isinstance(mc.gate_rck_bound_pass, bool)
    assert 0.0 <= mc.prob_loss_40pct <= 1.0
    assert 0.0 <= mc.dd_p95 <= 1.0
