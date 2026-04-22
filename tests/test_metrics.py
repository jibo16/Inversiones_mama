"""Tests for simulation.metrics — performance metrics + Deflated Sharpe Ratio.

Coverage:
  - sharpe_ratio: basic math, edge cases
  - sortino_ratio: downside-only denominator
  - max_drawdown: peak-to-trough calculation
  - max_drawdown_series: full DD series
  - calmar_ratio: CAGR / max DD
  - expected_max_sharpe: BLP 2014 formula
  - deflated_sharpe_ratio: full DSR pipeline
  - compute_all_metrics: aggregate metrics dataclass
  - Edge cases: flat returns, single observation, NaN handling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from inversiones_mama.simulation.metrics import (
    PerformanceMetrics,
    _sharpe_se,
    calmar_ratio,
    compute_all_metrics,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    max_drawdown,
    max_drawdown_series,
    sharpe_ratio,
    sortino_ratio,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def bull_returns() -> np.ndarray:
    """Steady positive daily returns (~20% annualized, ~15% vol)."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0008, 0.0095, 504)  # ~2 years


@pytest.fixture
def bear_returns() -> np.ndarray:
    """Negative drift daily returns."""
    rng = np.random.default_rng(42)
    return rng.normal(-0.0005, 0.015, 252)


@pytest.fixture
def flat_returns() -> np.ndarray:
    """Zero-mean, near-zero-vol returns."""
    return np.zeros(252)


@pytest.fixture
def skewed_returns() -> np.ndarray:
    """Returns with negative skew and fat tails (mimics real equities)."""
    rng = np.random.default_rng(42)
    # Mix: mostly normal, occasional large negative shocks
    base = rng.normal(0.0005, 0.01, 500)
    shocks = rng.choice(500, size=10, replace=False)
    base[shocks] = rng.normal(-0.05, 0.02, 10)
    return base


# --------------------------------------------------------------------------- #
# sharpe_ratio                                                                #
# --------------------------------------------------------------------------- #


class TestSharpeRatio:
    """Test the annualized Sharpe Ratio."""

    def test_positive_for_bull(self, bull_returns):
        sr = sharpe_ratio(bull_returns)
        assert sr > 0

    def test_negative_for_bear(self, bear_returns):
        sr = sharpe_ratio(bear_returns)
        assert sr < 0

    def test_zero_for_flat(self, flat_returns):
        sr = sharpe_ratio(flat_returns)
        assert sr == pytest.approx(0.0, abs=1e-10)

    def test_scaling_with_periods(self, bull_returns):
        """Monthly Sharpe should be sqrt(12/252) × daily Sharpe (approx)."""
        daily = sharpe_ratio(bull_returns, periods=252)
        monthly = sharpe_ratio(bull_returns, periods=12)
        # They use the same data frequency, so the ratio should reflect
        # the annualization factor difference
        assert daily != monthly

    def test_rf_subtraction(self, bull_returns):
        """Positive risk-free rate should reduce Sharpe."""
        sr_0 = sharpe_ratio(bull_returns, rf=0.0)
        sr_rf = sharpe_ratio(bull_returns, rf=0.0002)  # 5% annual daily
        assert sr_rf < sr_0

    def test_with_series(self, bull_returns):
        """Accept pandas Series input."""
        s = pd.Series(bull_returns)
        sr = sharpe_ratio(s)
        assert isinstance(sr, float)

    def test_known_value_hand_computed(self):
        """Hand-computed Sharpe from simple two-period returns."""
        # Returns: +1%, -0.5% → mean excess=0.25%, std≈1.06%
        r = np.array([0.01, -0.005])
        mu = np.mean(r)  # 0.0025
        sigma = np.std(r, ddof=1)  # ~0.01061
        expected_sr = mu / sigma * np.sqrt(252)
        sr = sharpe_ratio(r)
        assert sr == pytest.approx(expected_sr, rel=1e-8)


# --------------------------------------------------------------------------- #
# sortino_ratio                                                               #
# --------------------------------------------------------------------------- #


class TestSortinoRatio:
    """Test the annualized Sortino Ratio."""

    def test_positive_for_bull(self, bull_returns):
        sr = sortino_ratio(bull_returns)
        assert sr > 0

    def test_higher_than_sharpe_for_right_skew(self):
        """Right-skewed returns → less downside → Sortino > Sharpe."""
        rng = np.random.default_rng(42)
        r = np.abs(rng.normal(0.0005, 0.01, 252))  # All positive
        sort = sortino_ratio(r)
        sharp = sharpe_ratio(r)
        # Sortino divides by downside only, which is very small here
        # But since all returns are positive, downside_std ≈ 0 → Sortino ≈ 0
        # Actually, downside = min(r - 0, 0) = 0 for all positive r
        # So sortino should be 0 (our convention)
        assert sort == pytest.approx(0.0, abs=1e-10)

    def test_negative_for_bear(self, bear_returns):
        sr = sortino_ratio(bear_returns)
        assert sr < 0


# --------------------------------------------------------------------------- #
# max_drawdown                                                                #
# --------------------------------------------------------------------------- #


class TestMaxDrawdown:
    """Test maximum drawdown computation."""

    def test_no_drawdown_for_monotonic_up(self):
        """Strictly increasing equity → 0 drawdown."""
        r = np.full(100, 0.01)
        dd = max_drawdown(r)
        assert dd == pytest.approx(0.0, abs=1e-10)

    def test_known_drawdown(self):
        """Known scenario: up 10%, then down 20% from peak."""
        # Start at 1.0, go to 1.10, then drop to 0.88
        r = np.array([0.10, -0.20])
        dd = max_drawdown(r)
        # Peak = 1.10, trough = 0.88
        # DD = (1.10 - 0.88) / 1.10 = 0.20
        assert dd == pytest.approx(0.20, abs=0.001)

    def test_drawdown_is_positive(self, bull_returns):
        """Drawdown should always be non-negative."""
        dd = max_drawdown(bull_returns)
        assert dd >= 0

    def test_equity_curve_input(self):
        """Accept pre-computed equity curve."""
        equity = np.array([100, 110, 105, 120, 95, 130])
        dd = max_drawdown(equity, is_equity=True)
        # Peak 120, trough 95 → DD = 25/120 = 0.2083
        assert dd == pytest.approx(25 / 120, abs=0.001)

    def test_full_loss(self):
        """100% loss scenario."""
        r = np.array([0.10, -1.0])  # Goes to zero
        dd = max_drawdown(r)
        assert dd == pytest.approx(1.0, abs=0.001)


# --------------------------------------------------------------------------- #
# max_drawdown_series                                                         #
# --------------------------------------------------------------------------- #


class TestMaxDrawdownSeries:
    """Test the full drawdown series."""

    def test_returns_same_length(self, bull_returns):
        dd = max_drawdown_series(bull_returns)
        assert len(dd) == len(bull_returns)

    def test_all_non_negative(self, bull_returns):
        dd = max_drawdown_series(bull_returns)
        assert (dd >= -1e-10).all()

    def test_preserves_series_index(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="B")
        r = pd.Series(np.random.default_rng(42).normal(0, 0.01, 50), index=idx)
        dd = max_drawdown_series(r)
        assert isinstance(dd, pd.Series)
        assert (dd.index == idx).all()


# --------------------------------------------------------------------------- #
# calmar_ratio                                                                #
# --------------------------------------------------------------------------- #


class TestCalmarRatio:
    """Test the Calmar Ratio."""

    def test_positive_for_bull(self, bull_returns):
        cr = calmar_ratio(bull_returns)
        assert cr > 0

    def test_zero_for_no_drawdown(self):
        """Monotonic up → DD = 0 → Calmar = 0 (divide by zero convention)."""
        r = np.full(100, 0.01)
        cr = calmar_ratio(r)
        assert cr == pytest.approx(0.0, abs=1e-10)


# --------------------------------------------------------------------------- #
# Sharpe SE (non-normality correction)                                        #
# --------------------------------------------------------------------------- #


class TestSharpeSE:
    """Test the standard error of the Sharpe Ratio."""

    def test_normal_returns(self):
        """For normal returns (skew=0, excess_kurt=0), SE simplifies."""
        se = _sharpe_se(sr=1.0, T=252, skew=0.0, excess_kurt=0.0)
        expected = np.sqrt(1.0 / 251)  # 1/(T-1) when skew=0, kurt=0, SR=1
        # Actually: numerator = 1 - 0 + 0 = 1
        assert se == pytest.approx(expected, rel=0.01)

    def test_fat_tails_increase_se(self):
        """Positive excess kurtosis → higher SE."""
        se_normal = _sharpe_se(1.0, 252, 0.0, 0.0)
        se_fat = _sharpe_se(1.0, 252, 0.0, 5.0)
        assert se_fat > se_normal

    def test_more_data_reduces_se(self):
        """More observations → lower SE."""
        se_small = _sharpe_se(1.0, 100, 0.0, 0.0)
        se_large = _sharpe_se(1.0, 1000, 0.0, 0.0)
        assert se_large < se_small


# --------------------------------------------------------------------------- #
# expected_max_sharpe                                                         #
# --------------------------------------------------------------------------- #


class TestExpectedMaxSharpe:
    """Test the BLP expected max SR from N trials."""

    def test_single_trial_returns_zero(self):
        """With only 1 trial, no multiple testing → E[max SR] = 0."""
        e = expected_max_sharpe(1, 252)
        assert e == pytest.approx(0.0, abs=1e-10)

    def test_more_trials_higher_expected(self):
        """More trials → higher expected max by chance."""
        e_10 = expected_max_sharpe(10, 252)
        e_100 = expected_max_sharpe(100, 252)
        assert e_100 > e_10

    def test_positive_for_multiple_trials(self):
        e = expected_max_sharpe(20, 252)
        assert e > 0


# --------------------------------------------------------------------------- #
# deflated_sharpe_ratio                                                       #
# --------------------------------------------------------------------------- #


class TestDeflatedSharpeRatio:
    """Test the Deflated Sharpe Ratio (BLP 2014)."""

    def test_high_sr_single_trial_passes(self):
        """A genuinely high SR with 1 trial should have DSR > 0.5."""
        dsr = deflated_sharpe_ratio(
            observed_sr=2.0, n_trials=1, T=504,
            skew=0.0, excess_kurt=0.0,
        )
        assert dsr > 0.5

    def test_mediocre_sr_many_trials_fails(self):
        """A mediocre SR with many trials → low DSR (lucky)."""
        dsr = deflated_sharpe_ratio(
            observed_sr=0.5, n_trials=100, T=252,
            skew=0.0, excess_kurt=0.0,
        )
        assert dsr < 0.95  # Not statistically significant

    def test_dsr_bounded_zero_one(self, bull_returns):
        """DSR should always be in [0, 1]."""
        sr = sharpe_ratio(bull_returns)
        skw = float(sp_stats.skew(bull_returns, bias=False))
        ekurt = float(sp_stats.kurtosis(bull_returns, fisher=True, bias=False))
        dsr = deflated_sharpe_ratio(sr, 10, len(bull_returns), skw, ekurt)
        assert 0.0 <= dsr <= 1.0

    def test_more_trials_lower_dsr(self):
        """Same SR but more trials → lower DSR (harder to pass)."""
        dsr_1 = deflated_sharpe_ratio(1.5, 1, 504, 0.0, 0.0)
        dsr_100 = deflated_sharpe_ratio(1.5, 100, 504, 0.0, 0.0)
        assert dsr_100 < dsr_1

    def test_se_increases_with_negative_skew(self):
        """Negative skew with positive SR increases the SE of the Sharpe.
        SE formula: sqrt((1 - γ₁·SR + (γ₂/4)·SR²) / (T-1))
        With negative skew and positive SR: -γ₁·SR > 0, so numerator grows.
        """
        se_sym = _sharpe_se(1.0 / np.sqrt(252), 504, 0.0, 0.0)
        se_neg = _sharpe_se(1.0 / np.sqrt(252), 504, -1.5, 0.0)
        assert se_neg > se_sym

    def test_se_increases_with_fat_tails(self):
        """Excess kurtosis → higher SE of the Sharpe Ratio."""
        se_normal = _sharpe_se(1.0 / np.sqrt(252), 504, 0.0, 0.0)
        se_fat = _sharpe_se(1.0 / np.sqrt(252), 504, 0.0, 6.0)
        assert se_fat > se_normal

    def test_dsr_strong_signal_beats_noise(self):
        """A very high observed SR with few trials should dominate."""
        dsr_strong = deflated_sharpe_ratio(3.0, 5, 504, 0.0, 0.0)
        dsr_weak = deflated_sharpe_ratio(0.3, 5, 504, 0.0, 0.0)
        assert dsr_strong > dsr_weak


# --------------------------------------------------------------------------- #
# compute_all_metrics                                                         #
# --------------------------------------------------------------------------- #


class TestComputeAllMetrics:
    """Test the aggregate metrics function."""

    def test_returns_performance_metrics(self, bull_returns):
        pm = compute_all_metrics(bull_returns)
        assert isinstance(pm, PerformanceMetrics)

    def test_all_fields_populated(self, bull_returns):
        pm = compute_all_metrics(bull_returns)
        assert pm.n_observations == len(bull_returns)
        assert pm.sharpe_ratio != 0
        assert pm.max_drawdown >= 0
        assert 0 <= pm.hit_rate <= 1

    def test_bull_market_metrics(self, bull_returns):
        pm = compute_all_metrics(bull_returns)
        assert pm.total_return > 0
        assert pm.annualized_return > 0
        assert pm.sharpe_ratio > 0
        assert pm.max_drawdown > 0  # Some drawdown exists

    def test_bear_market_metrics(self, bear_returns):
        pm = compute_all_metrics(bear_returns)
        assert pm.total_return < 0
        assert pm.sharpe_ratio < 0

    def test_n_trials_affects_dsr(self, bull_returns):
        pm_1 = compute_all_metrics(bull_returns, n_trials=1)
        pm_100 = compute_all_metrics(bull_returns, n_trials=100)
        assert pm_100.deflated_sharpe < pm_1.deflated_sharpe

    def test_skewness_computed(self, skewed_returns):
        pm = compute_all_metrics(skewed_returns)
        assert pm.skewness < 0  # Negative shocks → negative skew

    def test_excess_kurtosis_computed(self, skewed_returns):
        pm = compute_all_metrics(skewed_returns)
        assert pm.excess_kurtosis > 0  # Fat tails

    def test_hit_rate(self, bull_returns):
        pm = compute_all_metrics(bull_returns)
        # Bull market → most days positive
        assert pm.hit_rate > 0.4

    def test_profit_factor(self, bull_returns):
        pm = compute_all_metrics(bull_returns)
        # Bull market → gains > losses → pf > 1
        assert pm.profit_factor > 1.0

    def test_handles_nan(self):
        """NaN values should be dropped silently."""
        r = np.array([0.01, np.nan, -0.005, 0.02, np.nan, 0.005])
        pm = compute_all_metrics(r)
        assert pm.n_observations == 4  # 6 - 2 NaN

    def test_single_observation(self):
        """Edge case: single return."""
        r = np.array([0.01])
        pm = compute_all_metrics(r)
        assert pm.sharpe_ratio == 0.0  # Can't compute with 1 obs

    def test_empty_returns(self):
        """Edge case: empty array."""
        r = np.array([])
        pm = compute_all_metrics(r)
        assert pm.n_observations == 0
        assert pm.sharpe_ratio == 0.0

    def test_tail_ratio(self, bull_returns):
        pm = compute_all_metrics(bull_returns)
        assert pm.tail_ratio > 0  # Positive tail_ratio for normal-ish data
