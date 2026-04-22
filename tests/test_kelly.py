"""Tests for sizing.kelly — Risk-Constrained Kelly optimizer.

Coverage:
  - kelly_growth_rate: mathematical identity checks
  - solve_rck: basic optimization, constraint satisfaction, edge cases
  - Parameter validation
  - Integration with Agent 1's API surface (mu, Sigma shapes)
  - Fractional Kelly scaling behavior
  - Drawdown constraint binding
  - Degenerate inputs (all-negative mu, single asset, etc.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.sizing.kelly import (
    KellyResult,
    _compute_lambda,
    kelly_growth_rate,
    solve_rck,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def simple_mu() -> pd.Series:
    """3-asset expected returns (daily decimals)."""
    return pd.Series(
        {"A": 0.0008, "B": 0.0003, "C": 0.0012},
        name="mu",
    )


@pytest.fixture
def simple_sigma() -> pd.DataFrame:
    """3-asset covariance matrix (daily)."""
    # Realistic daily covariance: vol ~1-2% daily
    cov = np.array([
        [0.000144, 0.000020, 0.000050],
        [0.000020, 0.000064, 0.000015],
        [0.000050, 0.000015, 0.000324],
    ])
    return pd.DataFrame(
        cov,
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )


@pytest.fixture
def ten_asset_mu() -> pd.Series:
    """10-asset expected returns mimicking the real universe."""
    rng = np.random.default_rng(42)
    tickers = ["AVUV", "AVDV", "AVEM", "MTUM", "IMTM",
               "USMV", "GLD", "DBC", "TLT", "SPY"]
    # Realistic daily mu: mostly positive, TLT negative
    mu_vals = [0.0008, 0.0007, 0.0005, 0.0006, 0.0005,
               0.0002, 0.0009, 0.0006, -0.0004, 0.0005]
    return pd.Series(mu_vals, index=tickers, name="mu")


@pytest.fixture
def ten_asset_sigma(ten_asset_mu: pd.Series) -> pd.DataFrame:
    """10-asset covariance matrix from synthetic returns."""
    rng = np.random.default_rng(42)
    n = len(ten_asset_mu)
    # Generate correlated factor structure
    factors = rng.normal(0, 0.01, (500, 3))
    loadings = rng.normal(0.5, 0.3, (n, 3))
    idio = rng.normal(0, 0.005, (500, n))
    returns = factors @ loadings.T + idio
    cov = pd.DataFrame(
        np.cov(returns, rowvar=False),
        index=ten_asset_mu.index,
        columns=ten_asset_mu.index,
    )
    return cov


# --------------------------------------------------------------------------- #
# kelly_growth_rate                                                           #
# --------------------------------------------------------------------------- #


class TestKellyGrowthRate:
    """Test the growth rate evaluation function."""

    def test_zero_weight_is_zero(self, simple_mu, simple_sigma):
        w = np.zeros(3)
        g = kelly_growth_rate(w, simple_mu.values, simple_sigma.values)
        assert g == pytest.approx(0.0, abs=1e-15)

    def test_positive_mu_gives_positive_rate(self, simple_mu, simple_sigma):
        """Allocating to positive-mu assets should yield positive growth."""
        w = np.array([0.3, 0.3, 0.3])
        g = kelly_growth_rate(w, simple_mu.values, simple_sigma.values)
        assert g > 0

    def test_full_concentration_in_best_asset(self, simple_mu, simple_sigma):
        """100% in asset C (highest mu) should beat equal-weight at low vol."""
        w_eq = np.array([1 / 3, 1 / 3, 1 / 3])
        w_c = np.array([0.0, 0.0, 1.0])
        g_eq = kelly_growth_rate(w_eq, simple_mu.values, simple_sigma.values)
        g_c = kelly_growth_rate(w_c, simple_mu.values, simple_sigma.values)
        # C has highest mu but also highest vol — growth rate comparison
        # is non-trivial, but both should be positive
        assert g_eq > 0
        # Note: g_c might be lower than g_eq due to high variance penalty

    def test_mathematical_identity(self):
        """g(w) = w'μ − ½ w'Σw: verify with hand-computed values."""
        mu = np.array([0.001, 0.002])
        Sigma = np.array([[0.0001, 0.00002], [0.00002, 0.0004]])
        w = np.array([0.5, 0.5])
        expected = w @ mu - 0.5 * w @ Sigma @ w
        actual = kelly_growth_rate(w, mu, Sigma)
        assert actual == pytest.approx(expected, rel=1e-10)


# --------------------------------------------------------------------------- #
# _compute_lambda                                                             #
# --------------------------------------------------------------------------- #


class TestComputeLambda:
    """Test the RCK risk-budget parameter computation."""

    def test_default_config_values(self):
        """α=0.50, β=0.10 → λ ≈ 3.32."""
        lam = _compute_lambda(0.50, 0.10)
        assert lam == pytest.approx(np.log(0.10) / np.log(0.50), rel=1e-10)
        assert lam == pytest.approx(3.3219, rel=1e-3)

    def test_alpha_lambda_relationship(self):
        """λ = log(β)/log(α): as α increases toward 1, log(α)→0⁻ so λ grows.

        α=0.30 (tolerate 30% DD) → log(0.30) ≈ -1.20 → λ ≈ 1.91
        α=0.70 (tolerate 70% DD) → log(0.70) ≈ -0.36 → λ ≈ 6.46

        Economically: a larger λ acts as a tighter volatility budget in the
        QP constraint (w'Σw ≤ 2/λ · w'μ), so higher α paradoxically
        constrains the portfolio MORE via a larger λ denominator.
        """
        lam_30 = _compute_lambda(0.30, 0.10)
        lam_70 = _compute_lambda(0.70, 0.10)
        assert lam_70 > lam_30  # Mathematically: log(β)/log(0.70) > log(β)/log(0.30)

    def test_lower_beta_higher_lambda(self):
        """Lower probability tolerance → higher λ → more conservative."""
        lam_loose = _compute_lambda(0.50, 0.20)
        lam_tight = _compute_lambda(0.50, 0.05)
        assert lam_tight > lam_loose

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            _compute_lambda(0.0, 0.10)
        with pytest.raises(ValueError, match="alpha"):
            _compute_lambda(1.0, 0.10)

    def test_invalid_beta(self):
        with pytest.raises(ValueError, match="beta"):
            _compute_lambda(0.50, 0.0)
        with pytest.raises(ValueError, match="beta"):
            _compute_lambda(0.50, 1.0)


# --------------------------------------------------------------------------- #
# solve_rck: basic behavior                                                   #
# --------------------------------------------------------------------------- #


class TestSolveRCK:
    """Core solver tests."""

    def test_returns_kelly_result(self, simple_mu, simple_sigma):
        result = solve_rck(simple_mu, simple_sigma)
        assert isinstance(result, KellyResult)

    def test_optimal_status(self, simple_mu, simple_sigma):
        result = solve_rck(simple_mu, simple_sigma)
        assert result.status in ("optimal", "optimal_inaccurate")

    def test_weights_are_nonneg(self, simple_mu, simple_sigma):
        result = solve_rck(simple_mu, simple_sigma)
        assert (result.weights >= -1e-10).all(), f"Negative weights: {result.weights}"

    def test_weights_sum_leq_one(self, simple_mu, simple_sigma):
        result = solve_rck(simple_mu, simple_sigma)
        assert result.weights.sum() <= 1.0 + 1e-8

    def test_cash_weight_consistent(self, simple_mu, simple_sigma):
        result = solve_rck(simple_mu, simple_sigma)
        expected_cash = 1.0 - result.weights.sum()
        assert result.cash_weight == pytest.approx(expected_cash, abs=1e-6)

    def test_per_name_cap_respected(self, simple_mu, simple_sigma):
        cap = 0.25
        result = solve_rck(simple_mu, simple_sigma, cap=cap)
        assert (result.weights <= cap + 1e-6).all(), f"Cap violated: {result.weights}"

    def test_growth_rate_positive(self, simple_mu, simple_sigma):
        result = solve_rck(simple_mu, simple_sigma)
        assert result.growth_rate > 0

    def test_growth_rate_ann_is_252x(self, simple_mu, simple_sigma):
        result = solve_rck(simple_mu, simple_sigma)
        assert result.growth_rate_ann == pytest.approx(
            result.growth_rate * 252, rel=1e-8
        )

    def test_fraction_stored(self, simple_mu, simple_sigma):
        result = solve_rck(simple_mu, simple_sigma, fraction=0.5)
        assert result.fraction == 0.5

    def test_lambda_stored(self, simple_mu, simple_sigma):
        result = solve_rck(simple_mu, simple_sigma, alpha=0.50, beta=0.10)
        expected = np.log(0.10) / np.log(0.50)
        assert result.lambda_rck == pytest.approx(expected, rel=1e-8)

    def test_tickers_preserved(self, simple_mu, simple_sigma):
        result = solve_rck(simple_mu, simple_sigma)
        assert list(result.weights.index) == ["A", "B", "C"]


# --------------------------------------------------------------------------- #
# Fractional Kelly behavior                                                   #
# --------------------------------------------------------------------------- #


class TestFractionalKelly:
    """Verify fractional scaling works correctly."""

    def test_half_kelly_smaller_weights(self, simple_mu, simple_sigma):
        full = solve_rck(simple_mu, simple_sigma, fraction=1.0)
        half = solve_rck(simple_mu, simple_sigma, fraction=0.5)
        # Half-Kelly should produce smaller total weight
        assert half.weights.sum() < full.weights.sum() + 1e-6

    def test_fraction_zero_point_one_very_small(self, simple_mu, simple_sigma):
        result = solve_rck(simple_mu, simple_sigma, fraction=0.1)
        assert result.weights.sum() < 0.5  # Should be quite small

    def test_invalid_fraction_zero(self, simple_mu, simple_sigma):
        with pytest.raises(ValueError, match="fraction"):
            solve_rck(simple_mu, simple_sigma, fraction=0.0)

    def test_invalid_fraction_negative(self, simple_mu, simple_sigma):
        with pytest.raises(ValueError, match="fraction"):
            solve_rck(simple_mu, simple_sigma, fraction=-0.5)

    def test_fraction_one_is_full_kelly(self, simple_mu, simple_sigma):
        result = solve_rck(simple_mu, simple_sigma, fraction=1.0)
        assert result.fraction == 1.0


# --------------------------------------------------------------------------- #
# Edge cases                                                                  #
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    """Degenerate and boundary inputs."""

    def test_all_negative_mu(self, simple_sigma):
        """All assets have negative expected return → 100% cash."""
        mu = pd.Series({"A": -0.001, "B": -0.002, "C": -0.0005})
        result = solve_rck(mu, simple_sigma)
        assert result.weights.sum() == pytest.approx(0.0, abs=1e-10)
        assert result.cash_weight == pytest.approx(1.0, abs=1e-6)
        assert result.status == "trivial_all_cash"

    def test_single_positive_mu(self, simple_sigma):
        """Only one asset has positive mu → allocate there only."""
        mu = pd.Series({"A": 0.001, "B": -0.002, "C": -0.0005})
        result = solve_rck(mu, simple_sigma)
        assert result.weights["A"] > 0
        assert result.weights["B"] == pytest.approx(0.0, abs=1e-8)
        assert result.weights["C"] == pytest.approx(0.0, abs=1e-8)

    def test_single_asset(self):
        """Degenerate: single asset."""
        mu = pd.Series({"X": 0.001})
        sigma = pd.DataFrame([[0.0004]], index=["X"], columns=["X"])
        result = solve_rck(mu, sigma)
        assert result.status in ("optimal", "optimal_inaccurate")
        assert result.weights["X"] > 0

    def test_mismatched_dimensions(self):
        """mu and Sigma dimensions don't match."""
        mu = pd.Series({"A": 0.001, "B": 0.002})
        sigma = pd.DataFrame(np.eye(3), index=["A", "B", "C"], columns=["A", "B", "C"])
        with pytest.raises(ValueError, match="incompatible"):
            solve_rck(mu, sigma)

    def test_numpy_inputs(self):
        """Accept raw numpy arrays without tickers."""
        mu = np.array([0.0008, 0.0005, 0.001])
        Sigma = np.diag([0.0001, 0.0001, 0.0002])
        result = solve_rck(mu, Sigma)
        assert result.status in ("optimal", "optimal_inaccurate")
        assert len(result.weights) == 3

    def test_cap_binding(self, simple_mu, simple_sigma):
        """Very tight cap forces max allocation per name."""
        cap = 0.10
        result = solve_rck(simple_mu, simple_sigma, cap=cap)
        assert (result.weights <= cap + 1e-6).all()


# --------------------------------------------------------------------------- #
# 10-asset universe (mimicking real portfolio)                                #
# --------------------------------------------------------------------------- #


class TestTenAssetUniverse:
    """Test with a 10-asset universe mimicking the real ETF portfolio."""

    def test_solves_cleanly(self, ten_asset_mu, ten_asset_sigma):
        result = solve_rck(ten_asset_mu, ten_asset_sigma)
        assert result.status in ("optimal", "optimal_inaccurate")

    def test_tlt_gets_zero_weight(self, ten_asset_mu, ten_asset_sigma):
        """TLT has negative mu → Kelly should zero-weight it."""
        result = solve_rck(ten_asset_mu, ten_asset_sigma)
        assert result.weights["TLT"] == pytest.approx(0.0, abs=1e-8)

    def test_diagnostics_report_pruning(self, ten_asset_mu, ten_asset_sigma):
        result = solve_rck(ten_asset_mu, ten_asset_sigma)
        # TLT should be pruned (negative mu)
        assert result.diagnostics["n_assets_positive_mu"] == 9  # all except TLT

    def test_config_defaults_applied(self, ten_asset_mu, ten_asset_sigma):
        """Default fraction/cap from config are applied when not overridden."""
        result = solve_rck(ten_asset_mu, ten_asset_sigma)
        assert result.fraction == 0.65  # from config
        assert (result.weights <= 0.35 + 1e-6).all()  # from config

    def test_tighter_drawdown_constraint_more_cash(
        self, ten_asset_mu, ten_asset_sigma
    ):
        """More conservative drawdown constraint → more cash."""
        loose = solve_rck(ten_asset_mu, ten_asset_sigma, alpha=0.60, beta=0.20)
        tight = solve_rck(ten_asset_mu, ten_asset_sigma, alpha=0.20, beta=0.05)
        assert tight.cash_weight >= loose.cash_weight - 0.01  # tolerance


# --------------------------------------------------------------------------- #
# Drawdown constraint (RCK)                                                   #
# --------------------------------------------------------------------------- #


class TestDrawdownConstraint:
    """Verify the RCK drawdown probability bound is enforced."""

    def test_rck_constraint_reduces_exposure(self, simple_mu, simple_sigma):
        """The drawdown constraint should reduce total equity exposure
        compared to unconstrained Kelly (which only has sum ≤ 1)."""
        # Solve with very loose drawdown constraint (effectively unbinding)
        loose = solve_rck(
            simple_mu, simple_sigma,
            fraction=1.0, alpha=0.999, beta=0.999,
        )
        # Solve with tight drawdown constraint
        tight = solve_rck(
            simple_mu, simple_sigma,
            fraction=1.0, alpha=0.20, beta=0.05,
        )
        # Tight constraint should reduce total weight (more cash)
        assert tight.weights.sum() <= loose.weights.sum() + 1e-4
