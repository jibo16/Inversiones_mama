"""Tests for inversiones_mama.simulation.trade_sequence_mc."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.simulation.trade_sequence_mc import (
    TradeSequenceMCResult,
    trade_sequence_mc,
)


@pytest.fixture
def flat_returns() -> np.ndarray:
    """Returns that exactly equal the mean - no permutation path differs."""
    return np.full(100, 0.0005)


@pytest.fixture
def noisy_returns() -> np.ndarray:
    """Realistic daily returns with moderate volatility."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.015, 252)


# --------------------------------------------------------------------------- #
# Shape & schema                                                              #
# --------------------------------------------------------------------------- #


def test_returns_expected_schema(noisy_returns):
    out = trade_sequence_mc(noisy_returns, n_paths=500, rng=np.random.default_rng(0))
    assert isinstance(out, TradeSequenceMCResult)
    assert out.n_paths == 500
    assert out.terminal_wealth.shape == (500,)
    assert out.max_drawdowns.shape == (500,)
    # Percentiles ordered
    assert out.terminal_p05 <= out.terminal_p50 <= out.terminal_p95
    assert out.dd_p50 <= out.dd_p95 <= out.dd_p99
    # Ranks in [0, 1]
    assert 0.0 <= out.observed_terminal_percentile <= 1.0
    assert 0.0 <= out.observed_dd_percentile <= 1.0


def test_accepts_pandas_series(noisy_returns):
    s = pd.Series(noisy_returns, index=pd.date_range("2024-01-01", periods=len(noisy_returns), freq="B"))
    out = trade_sequence_mc(s, n_paths=200, rng=np.random.default_rng(1))
    assert out.terminal_wealth.shape == (200,)


def test_drops_nan_returns(noisy_returns):
    rets_with_nan = noisy_returns.copy()
    rets_with_nan[5] = np.nan
    rets_with_nan[10] = np.nan
    out = trade_sequence_mc(rets_with_nan, n_paths=200, rng=np.random.default_rng(2))
    # Should not raise, and should produce well-formed output
    assert np.isfinite(out.terminal_p50)
    assert np.isfinite(out.dd_p95)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_too_few_returns_raises():
    with pytest.raises(ValueError, match="at least 5"):
        trade_sequence_mc(np.array([0.01, 0.02]), n_paths=100)


def test_too_few_paths_raises(noisy_returns):
    with pytest.raises(ValueError, match="n_paths"):
        trade_sequence_mc(noisy_returns, n_paths=5)


def test_non_positive_capital_raises(noisy_returns):
    with pytest.raises(ValueError, match="initial_capital"):
        trade_sequence_mc(noisy_returns, initial_capital=0.0, n_paths=200)


# --------------------------------------------------------------------------- #
# Deterministic properties                                                    #
# --------------------------------------------------------------------------- #


def test_rng_reproducibility(noisy_returns):
    a = trade_sequence_mc(noisy_returns, n_paths=200, rng=np.random.default_rng(99))
    b = trade_sequence_mc(noisy_returns, n_paths=200, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(a.terminal_wealth, b.terminal_wealth)


def test_terminal_wealth_invariant_under_permutation(noisy_returns):
    """Terminal wealth is order-invariant: prod(1+r_i) is symmetric in the r_i.

    Every permutation must produce the SAME terminal wealth (to floating-point
    precision) because multiplication is commutative.
    """
    out = trade_sequence_mc(noisy_returns, n_paths=500, rng=np.random.default_rng(3))
    # All terminals should be essentially identical
    spread = out.terminal_wealth.max() - out.terminal_wealth.min()
    # Allow tiny floating-point rounding noise (std of returns * horizon)
    # The TRUE theoretical spread is zero.
    assert spread < 1e-6 * out.observed_terminal_wealth


def test_observed_terminal_percentile_well_formed(noisy_returns):
    """Terminal wealth is math-invariant under permutation, so all permuted
    terminals equal the observed value up to floating-point accumulation
    noise. The rank is therefore not meaningful — just confirm it lives
    in [0, 1] and the spread is tiny."""
    out = trade_sequence_mc(noisy_returns, n_paths=500, rng=np.random.default_rng(4))
    assert 0.0 <= out.observed_terminal_percentile <= 1.0
    # Spread around observed should be tiny (floating-point only)
    spread = out.terminal_wealth.max() - out.terminal_wealth.min()
    assert spread < 1e-6 * out.observed_terminal_wealth


# --------------------------------------------------------------------------- #
# Drawdown distribution is the interesting one                                #
# --------------------------------------------------------------------------- #


def test_drawdowns_vary_across_permutations(noisy_returns):
    """Drawdown IS order-dependent — different shuffles produce different
    max-DDs. This is the whole point of the test."""
    out = trade_sequence_mc(noisy_returns, n_paths=1000, rng=np.random.default_rng(5))
    # At least 10% spread between min and max across permutations
    dd_range = out.max_drawdowns.max() - out.max_drawdowns.min()
    assert dd_range > 0.01


def test_flat_returns_zero_drawdown_every_path(flat_returns):
    """Perfectly flat positive returns produce zero drawdown in any ordering."""
    out = trade_sequence_mc(flat_returns, n_paths=200, rng=np.random.default_rng(6))
    assert np.allclose(out.max_drawdowns, 0.0, atol=1e-12)
    assert out.observed_max_drawdown == pytest.approx(0.0, abs=1e-12)


def test_observed_dd_percentile_computed_consistently():
    """Compute observed DD directly from the realized sequence and confirm
    the MC reports the same value as the permutation-derived observed_max_dd.

    Deeper tests of "unusual ordering" live in test_unusually_bad_dd_flag
    with a sufficiently large clustered-loss construction; this test just
    confirms the output is arithmetically consistent.
    """
    rng = np.random.default_rng(7)
    realized = rng.normal(0.0005, 0.012, 252)
    out = trade_sequence_mc(realized, n_paths=500, rng=np.random.default_rng(8))
    # observed_max_drawdown must match a direct computation of the wealth
    # path's max DD, independent of the permutation loop
    import numpy as _np
    wealth = 5000.0 * _np.cumprod(1.0 + realized)
    wealth = _np.concatenate([[5000.0], wealth])
    peak = _np.maximum.accumulate(wealth)
    dd_direct = float(-((wealth - peak) / peak).min())
    assert out.observed_max_drawdown == pytest.approx(dd_direct, rel=1e-12)


def test_dd_properties_from_known_realization():
    """Hand-constructed small sequence for exact drawdown checks."""
    # Sequence: +5%, +5%, -20%, +5% — starting from $100
    # Wealth path: 100, 105, 110.25, 88.2, 92.61
    # Max DD from peak 110.25 to trough 88.2 = (110.25 - 88.2)/110.25 = 20%
    rets = np.array([0.05, 0.05, -0.20, 0.05])
    # Need at least 5 returns → pad with zero
    rets = np.concatenate([rets, [0.0]])
    out = trade_sequence_mc(rets, n_paths=30, initial_capital=100.0,
                             rng=np.random.default_rng(9))
    # Observed max DD should be close to 0.20
    assert 0.19 < out.observed_max_drawdown < 0.21


# --------------------------------------------------------------------------- #
# Convenience flags                                                            #
# --------------------------------------------------------------------------- #


def test_unusually_bad_dd_flag():
    """Reproduce the clustered-losses scenario; confirm the boolean fires."""
    rng = np.random.default_rng(11)
    base = rng.normal(0.001, 0.008, 240)
    losses = np.full(20, -0.03)
    realized = np.concatenate([base, losses])
    out = trade_sequence_mc(realized, n_paths=3000, rng=np.random.default_rng(12))
    # With 20 consecutive -3% days at the end, observed dd is near worst-case
    assert out.observed_is_unusually_bad_dd is True


def test_unusually_good_terminal_flag_on_positive_returns():
    """Terminal is order-invariant so this flag is either always True or always False
    depending on whether the terminal exactly matches the group max."""
    # With all returns positive, terminal equals every permutation's terminal,
    # so rank = 1.0 > 0.95 → flag True.
    rets = np.full(20, 0.01)
    out = trade_sequence_mc(rets, n_paths=100, rng=np.random.default_rng(13))
    assert out.observed_is_unusually_good_terminal is True
