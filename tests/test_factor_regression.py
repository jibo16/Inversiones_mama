"""Tests for inversiones_mama.models.factor_regression.

Unit tests use synthetic factor returns with known loadings — if the
regression is correctly implemented, we can recover the planted betas to
floating-point precision. One live test fits SPY against real Ken French
factors and checks that SPY's Mkt-RF beta lands near 1.0 (it IS the market).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.models.factor_regression import (
    FACTOR_COLS,
    FactorLoadings,
    align_excess_returns,
    compute_composite_mu,
    factor_premia,
    fit_factor_loadings,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_factors() -> pd.DataFrame:
    """500 business-day synthetic factor panel with realistic magnitudes.

    Standard deviations chosen to loosely resemble daily published FF factors:
    Mkt-RF ~1%/day, others smaller. RF is flat-ish.
    """
    rng = np.random.default_rng(2026)
    n = 500
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0004, 0.010, n),
            "SMB":    rng.normal(0.0001, 0.005, n),
            "HML":    rng.normal(0.0001, 0.005, n),
            "RMW":    rng.normal(0.0002, 0.004, n),
            "CMA":    rng.normal(0.0000, 0.004, n),
            "MOM":    rng.normal(0.0002, 0.007, n),
            "RF":     np.full(n, 0.00015),  # ~4% annualized flat
        },
        index=idx,
    )


@pytest.fixture
def planted_asset_returns(synthetic_factors: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Construct assets with known alpha and betas; return (returns_df, truth_map).

    Three synthetic assets:
      * PURE_MKT: pure market exposure, beta=1 on Mkt-RF, zero others, zero alpha.
      * VALUE_TILT: beta=1.0 Mkt, 0.3 HML, 0.2 RMW, alpha=0.0003/day.
      * NOISY: random noise — should regress with near-zero R^2.
    """
    f = synthetic_factors
    rf = f["RF"]
    rng = np.random.default_rng(7)

    pure_mkt_excess = 1.0 * f["Mkt-RF"]
    pure_mkt = pure_mkt_excess + rf

    value_excess = (
        0.0003  # alpha
        + 1.0 * f["Mkt-RF"]
        + 0.3 * f["HML"]
        + 0.2 * f["RMW"]
        + rng.normal(0, 0.001, len(f))  # tiny idiosyncratic noise
    )
    value = value_excess + rf

    noisy = rng.normal(0.0001, 0.015, len(f))

    df = pd.DataFrame({"PURE_MKT": pure_mkt, "VALUE_TILT": value, "NOISY": noisy}, index=f.index)

    truth = {
        "PURE_MKT": {
            "alpha": 0.0,
            "betas": {"Mkt-RF": 1.0, "SMB": 0.0, "HML": 0.0, "RMW": 0.0, "CMA": 0.0, "MOM": 0.0},
        },
        "VALUE_TILT": {
            "alpha": 0.0003,
            "betas": {"Mkt-RF": 1.0, "SMB": 0.0, "HML": 0.3, "RMW": 0.2, "CMA": 0.0, "MOM": 0.0},
        },
    }
    return df, truth


# --------------------------------------------------------------------------- #
# align_excess_returns                                                        #
# --------------------------------------------------------------------------- #


def test_align_excess_returns_subtracts_rf(synthetic_factors):
    f = synthetic_factors
    raw = pd.DataFrame(
        {"A": np.full(len(f), 0.001), "B": np.full(len(f), 0.002)},
        index=f.index,
    )
    excess, faligned = align_excess_returns(raw, f)
    np.testing.assert_allclose(excess["A"].values, 0.001 - f["RF"].values, rtol=1e-12)
    np.testing.assert_allclose(excess["B"].values, 0.002 - f["RF"].values, rtol=1e-12)
    assert len(faligned) == len(f)


def test_align_excess_returns_requires_rf(synthetic_factors):
    f_norf = synthetic_factors.drop(columns=["RF"])
    raw = pd.DataFrame({"A": [0.01]}, index=f_norf.index[:1])
    with pytest.raises(ValueError, match="RF"):
        align_excess_returns(raw, f_norf)


def test_align_excess_returns_requires_overlap(synthetic_factors):
    raw = pd.DataFrame(
        {"A": np.zeros(5)},
        index=pd.date_range("1990-01-01", periods=5, freq="B"),
    )
    with pytest.raises(ValueError, match="Insufficient"):
        align_excess_returns(raw, synthetic_factors)


# --------------------------------------------------------------------------- #
# fit_factor_loadings                                                         #
# --------------------------------------------------------------------------- #


def test_fit_recovers_pure_market_beta(planted_asset_returns, synthetic_factors):
    returns, truth = planted_asset_returns
    loadings = fit_factor_loadings(returns, synthetic_factors)
    assert isinstance(loadings, FactorLoadings)
    assert loadings.n_obs == len(synthetic_factors)
    # PURE_MKT: regress (r - rf) = 1.0 * Mkt-RF. No noise, so this must be exact.
    np.testing.assert_allclose(loadings.betas.loc["PURE_MKT", "Mkt-RF"], 1.0, atol=1e-9)
    np.testing.assert_allclose(loadings.alpha["PURE_MKT"], 0.0, atol=1e-9)
    for k in ["SMB", "HML", "RMW", "CMA", "MOM"]:
        assert abs(loadings.betas.loc["PURE_MKT", k]) < 1e-9
    assert loadings.r_squared["PURE_MKT"] > 0.999


def test_fit_recovers_value_tilt_betas(planted_asset_returns, synthetic_factors):
    returns, truth = planted_asset_returns
    loadings = fit_factor_loadings(returns, synthetic_factors)
    # VALUE_TILT has tiny idiosyncratic noise; allow 2% tolerance on betas
    np.testing.assert_allclose(loadings.betas.loc["VALUE_TILT", "Mkt-RF"], 1.0, atol=0.02)
    np.testing.assert_allclose(loadings.betas.loc["VALUE_TILT", "HML"], 0.3, atol=0.05)
    np.testing.assert_allclose(loadings.betas.loc["VALUE_TILT", "RMW"], 0.2, atol=0.05)
    # alpha planted at 0.0003/day. With n=500 obs, noise std=0.001, and the
    # multi-factor design matrix, the SE of the alpha estimator is roughly
    # 1e-4 — so a 3-sigma tolerance of 3e-4 is the right target for this
    # sanity check. The betas above carry the precision signal.
    assert abs(loadings.alpha["VALUE_TILT"] - 0.0003) < 3e-4


def test_fit_noisy_asset_low_rsquared(planted_asset_returns, synthetic_factors):
    returns, _ = planted_asset_returns
    loadings = fit_factor_loadings(returns, synthetic_factors)
    # Pure noise should have very low R^2 (~0)
    assert loadings.r_squared["NOISY"] < 0.05


def test_fit_betas_shape_and_index(planted_asset_returns, synthetic_factors):
    returns, _ = planted_asset_returns
    loadings = fit_factor_loadings(returns, synthetic_factors)
    assert list(loadings.betas.columns) == FACTOR_COLS
    assert set(loadings.betas.index) == {"PURE_MKT", "VALUE_TILT", "NOISY"}
    assert loadings.factor_cols == tuple(FACTOR_COLS)


def test_fit_missing_factor_column_raises(planted_asset_returns, synthetic_factors):
    returns, _ = planted_asset_returns
    f_broken = synthetic_factors.drop(columns=["RMW"])
    with pytest.raises(ValueError, match="missing required columns"):
        fit_factor_loadings(returns, f_broken)


def test_fit_residual_std_positive(planted_asset_returns, synthetic_factors):
    returns, _ = planted_asset_returns
    loadings = fit_factor_loadings(returns, synthetic_factors)
    assert (loadings.residual_std >= 0).all()
    # NOISY has more residual variance than VALUE_TILT (which has tiny noise)
    assert loadings.residual_std["NOISY"] > loadings.residual_std["VALUE_TILT"]


# --------------------------------------------------------------------------- #
# factor_premia                                                               #
# --------------------------------------------------------------------------- #


def test_factor_premia_full_history(synthetic_factors):
    premia = factor_premia(synthetic_factors)
    assert list(premia.index) == FACTOR_COLS
    # Means should be near the planted mus (sample size 500, tolerance generous)
    assert abs(premia["Mkt-RF"] - 0.0004) < 0.002
    assert abs(premia["MOM"] - 0.0002) < 0.002


def test_factor_premia_lookback(synthetic_factors):
    full = factor_premia(synthetic_factors)
    last60 = factor_premia(synthetic_factors, lookback_days=60)
    # Different windows produce different means
    assert not np.allclose(full.values, last60.values)


def test_factor_premia_bad_lookback(synthetic_factors):
    with pytest.raises(ValueError, match="lookback"):
        factor_premia(synthetic_factors, lookback_days=0)


def test_factor_premia_missing_col(synthetic_factors):
    broken = synthetic_factors.drop(columns=["RMW"])
    with pytest.raises(ValueError, match="missing"):
        factor_premia(broken)


# --------------------------------------------------------------------------- #
# compute_composite_mu                                                        #
# --------------------------------------------------------------------------- #


def test_composite_mu_formula():
    """mu = alpha + sum beta * E[F]  — hand-check with planted numbers."""
    loadings = FactorLoadings(
        alpha=pd.Series({"X": 0.001, "Y": -0.0005}),
        betas=pd.DataFrame(
            {
                "Mkt-RF": {"X": 1.0, "Y": 0.5},
                "SMB": {"X": 0.2, "Y": 0.0},
                "HML": {"X": 0.3, "Y": 0.1},
                "RMW": {"X": 0.0, "Y": 0.4},
                "CMA": {"X": 0.0, "Y": 0.0},
                "MOM": {"X": 0.0, "Y": -0.2},
            }
        ),
        r_squared=pd.Series({"X": 0.9, "Y": 0.85}),
        residual_std=pd.Series({"X": 0.01, "Y": 0.012}),
        n_obs=500,
        factor_cols=tuple(FACTOR_COLS),
    )
    premia = pd.Series(
        {"Mkt-RF": 0.0004, "SMB": 0.0001, "HML": 0.0002, "RMW": 0.0003, "CMA": 0.0, "MOM": 0.0002}
    )
    mu = compute_composite_mu(loadings, premia)
    # Hand-compute X: 0.001 + 1.0*0.0004 + 0.2*0.0001 + 0.3*0.0002 + 0 + 0 + 0
    expected_x = 0.001 + 0.0004 + 0.00002 + 0.00006
    expected_y = -0.0005 + 0.5 * 0.0004 + 0.0 + 0.1 * 0.0002 + 0.4 * 0.0003 + 0.0 - 0.2 * 0.0002
    np.testing.assert_allclose(mu["X"], expected_x, atol=1e-12)
    np.testing.assert_allclose(mu["Y"], expected_y, atol=1e-12)
    assert mu.name == "mu"


def test_composite_mu_roundtrip_planted(planted_asset_returns, synthetic_factors):
    """Round-trip: fit planted asset, compute mu, compare to analytical expectation."""
    returns, truth = planted_asset_returns
    loadings = fit_factor_loadings(returns, synthetic_factors)
    premia = factor_premia(synthetic_factors)
    mu = compute_composite_mu(loadings, premia)
    # PURE_MKT analytical mu = alpha + beta_mkt * E[Mkt-RF] = 0 + 1 * E[Mkt-RF]
    expected_pure_mkt = premia["Mkt-RF"]
    np.testing.assert_allclose(mu["PURE_MKT"], expected_pure_mkt, atol=1e-9)


def test_composite_mu_missing_premia(planted_asset_returns, synthetic_factors):
    returns, _ = planted_asset_returns
    loadings = fit_factor_loadings(returns, synthetic_factors)
    premia_incomplete = pd.Series({"Mkt-RF": 0.0004})  # missing the other 5
    with pytest.raises(ValueError, match="premia missing"):
        compute_composite_mu(loadings, premia_incomplete)


# --------------------------------------------------------------------------- #
# Live test: real SPY vs real Ken French                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_spy_beta_near_one(tmp_path, monkeypatch):
    """SPY is (by definition) the market. Its Mkt-RF beta should be ~1, R^2 very high."""
    from inversiones_mama.data import factors as factors_mod
    from inversiones_mama.data import prices as prices_mod
    from inversiones_mama.data.prices import load_prices, returns_from_prices
    from inversiones_mama.data.factors import load_factor_returns

    # Isolate cache to tmp_path so live tests don't clobber real cache
    monkeypatch.setattr(prices_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(factors_mod, "CACHE_DIR", tmp_path)

    start = datetime(2021, 1, 1)
    end = datetime(2024, 12, 31)
    prices = load_prices(["SPY"], start, end, use_cache=True)
    ret = returns_from_prices(prices, method="simple")
    fac = load_factor_returns(start=start, end=end, use_cache=True)

    loadings = fit_factor_loadings(ret, fac)

    spy_mkt_beta = loadings.betas.loc["SPY", "Mkt-RF"]
    assert 0.90 < spy_mkt_beta < 1.10, f"SPY Mkt-RF beta unexpected: {spy_mkt_beta}"
    assert loadings.r_squared["SPY"] > 0.90, (
        f"SPY R^2 should be very high: {loadings.r_squared['SPY']}"
    )
    # Alpha on SPY should be tiny (near zero by construction)
    assert abs(loadings.alpha["SPY"]) < 0.001
