"""Tests for macro regression + CCAR stress projection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.models.factor_regression import FactorLoadings
from inversiones_mama.models.macro_regression import (
    fit_factor_macro_regression,
    project_factor_returns,
    quarterly_factor_returns_from_daily,
)
from inversiones_mama.simulation.ccar_stress import (
    CCAR_SEVERELY_ADVERSE_2025,
    load_scenario,
    project_portfolio_stress,
    run_ccar_stress,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_daily_factors() -> pd.DataFrame:
    """15 years of synthetic daily factor returns (Mkt-RF, HML, MOM, RF)."""
    rng = np.random.default_rng(20260422)
    n = 15 * 252
    idx = pd.date_range("2010-01-04", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0003, 0.010, n),
            "HML":    rng.normal(0.0001, 0.005, n),
            "MOM":    rng.normal(0.0002, 0.007, n),
            "RF":     np.full(n, 0.00015),
        },
        index=idx,
    )


@pytest.fixture
def synthetic_macro_panel() -> pd.DataFrame:
    """15 years of synthetic quarterly macros covering 2010-2025."""
    rng = np.random.default_rng(2026)
    qidx = pd.date_range("2010-03-31", periods=60, freq="QE")
    # Unemployment walks around 5% with a 2009-style spike at the start
    unrate = 9.5 - 0.1 * np.arange(60) + rng.normal(0, 0.2, 60)
    unrate = np.clip(unrate, 3.0, 12.0)
    # VIX walks around 18
    vix = 18 + rng.normal(0, 3, 60)
    # BBB spread walks around 2.5
    baa = 2.5 + rng.normal(0, 0.3, 60)
    return pd.DataFrame({"unemployment": unrate, "vix": vix, "baa_10y_spread": baa}, index=qidx)


@pytest.fixture
def synthetic_loadings() -> FactorLoadings:
    """Three assets with planted factor exposures."""
    betas = pd.DataFrame(
        {"Mkt-RF": [1.0, 0.8, 0.5],
         "HML":    [0.3, 0.0, -0.2],
         "MOM":    [0.0, 0.3, 0.5]},
        index=["A", "B", "C"],
    )
    return FactorLoadings(
        alpha=pd.Series({"A": 0.0002, "B": 0.0001, "C": 0.0003}),
        betas=betas,
        r_squared=pd.Series({"A": 0.9, "B": 0.8, "C": 0.7}),
        residual_std=pd.Series({"A": 0.005, "B": 0.007, "C": 0.009}),
        n_obs=252,
        factor_cols=("Mkt-RF", "HML", "MOM"),
    )


# --------------------------------------------------------------------------- #
# Scenario sanity                                                             #
# --------------------------------------------------------------------------- #


def test_ccar_scenario_shape():
    s = load_scenario("severely_adverse_2025")
    assert len(s) == 9
    assert set(s.columns) >= {"unemployment", "vix", "baa_10y_spread", "sp500"}


def test_ccar_scenario_unknown_raises():
    with pytest.raises(ValueError, match="Unknown scenario"):
        load_scenario("bogus")


def test_ccar_scenario_matches_jorge_parameters():
    """Sanity-check the scenario encodes the Jorge-stated severity levels."""
    s = CCAR_SEVERELY_ADVERSE_2025
    # Unemployment peaks at 10
    assert s["unemployment"].max() == pytest.approx(10.0, abs=0.1)
    # S&P 500 drops >= 50% from peak
    assert s["sp500"].min() / s["sp500"].max() <= 0.55
    # Home prices -25% peak-to-trough
    assert s["home_price_index"].min() / s["home_price_index"].max() <= 0.80
    # VIX spikes to at least 70
    assert s["vix"].max() >= 70
    # Short rates near zero
    assert s["treasury_3m"].min() <= 0.5


# --------------------------------------------------------------------------- #
# Macro regression                                                            #
# --------------------------------------------------------------------------- #


def test_quarterly_compounding_correct():
    daily = pd.DataFrame(
        {"Mkt-RF": [0.01, 0.02, -0.01, 0.005]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    q = quarterly_factor_returns_from_daily(daily)
    expected = (1.01 * 1.02 * 0.99 * 1.005) - 1.0
    assert q.iloc[0, 0] == pytest.approx(expected, rel=1e-10)


def test_fit_factor_macro_regression_recovers_intercept_on_flat_data(synthetic_macro_panel):
    """If factor returns are constant and macros change, intercept ≈ factor value
    and betas ≈ 0."""
    qidx = synthetic_macro_panel.index
    flat_factor = pd.DataFrame({"Mkt-RF": [0.02] * len(qidx)}, index=qidx)
    reg = fit_factor_macro_regression(flat_factor, synthetic_macro_panel, factor_cols=["Mkt-RF"])
    # Intercept should be ~0.02
    assert reg.intercepts["Mkt-RF"] == pytest.approx(0.02, abs=0.001)
    # All betas small
    assert reg.betas.loc["Mkt-RF"].abs().max() < 0.001


def test_fit_factor_macro_regression_shape(synthetic_daily_factors, synthetic_macro_panel):
    fq = quarterly_factor_returns_from_daily(synthetic_daily_factors.drop(columns=["RF"]))
    reg = fit_factor_macro_regression(fq, synthetic_macro_panel)
    assert set(reg.factor_cols) == {"Mkt-RF", "HML", "MOM"}
    assert set(reg.macro_cols) == {"unemployment", "vix", "baa_10y_spread"}
    assert reg.betas.shape == (3, 3)
    assert reg.n_obs > 40


def test_fit_raises_when_too_few_overlapping_quarters():
    tiny_f = pd.DataFrame({"Mkt-RF": [0.01, 0.02]},
                          index=pd.to_datetime(["2024-03-31", "2024-06-30"]))
    tiny_m = pd.DataFrame({"unemployment": [4.0, 4.2]},
                          index=pd.to_datetime(["2024-03-31", "2024-06-30"]))
    with pytest.raises(ValueError, match="Only"):
        fit_factor_macro_regression(tiny_f, tiny_m, min_obs=20)


def test_project_factor_returns_shape(synthetic_daily_factors, synthetic_macro_panel):
    fq = quarterly_factor_returns_from_daily(synthetic_daily_factors.drop(columns=["RF"]))
    reg = fit_factor_macro_regression(fq, synthetic_macro_panel)
    # Build an 8-quarter delta path that includes the same macro columns
    deltas = pd.DataFrame(
        {m: np.linspace(-0.5, 0.5, 8) for m in reg.macro_cols},
        index=pd.date_range("2026-03-31", periods=8, freq="QE"),
    )
    projected = project_factor_returns(reg, deltas)
    assert projected.shape == (8, len(reg.factor_cols))


def test_project_factor_returns_requires_all_macros(synthetic_daily_factors, synthetic_macro_panel):
    fq = quarterly_factor_returns_from_daily(synthetic_daily_factors.drop(columns=["RF"]))
    reg = fit_factor_macro_regression(fq, synthetic_macro_panel)
    # Missing one macro column
    bad = pd.DataFrame(
        {"unemployment": [0.1] * 5},
        index=pd.date_range("2026-03-31", periods=5, freq="QE"),
    )
    with pytest.raises(ValueError, match="missing required columns"):
        project_factor_returns(reg, bad)


# --------------------------------------------------------------------------- #
# Portfolio stress                                                            #
# --------------------------------------------------------------------------- #


def test_project_portfolio_stress_shape(synthetic_daily_factors, synthetic_macro_panel, synthetic_loadings):
    fq = quarterly_factor_returns_from_daily(synthetic_daily_factors.drop(columns=["RF"]))
    reg = fit_factor_macro_regression(fq, synthetic_macro_panel)
    deltas = pd.DataFrame(
        {m: np.linspace(-1, 1, 8) for m in reg.macro_cols},
        index=pd.date_range("2026-03-31", periods=8, freq="QE"),
    )
    projected = project_factor_returns(reg, deltas)

    weights = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})
    result = project_portfolio_stress(
        current_weights=weights,
        asset_loadings=synthetic_loadings,
        projected_factor_returns=projected,
        initial_wealth=5000.0,
        scenario_name="test",
        regression=reg,
    )
    assert result.horizon_quarters == 8
    assert len(result.wealth_path) == 8
    assert result.wealth_path.iloc[-1] == pytest.approx(result.terminal_wealth)
    assert result.max_drawdown >= 0.0


def test_project_portfolio_stress_positive_factor_returns_grow_wealth(synthetic_loadings):
    """If every projected quarterly factor return is +5%, wealth should grow."""
    qidx = pd.date_range("2026-03-31", periods=4, freq="QE")
    projected = pd.DataFrame(
        {"Mkt-RF": [0.05] * 4, "HML": [0.05] * 4, "MOM": [0.05] * 4},
        index=qidx,
    )
    # Dummy regression only used for the summary; its R^2 is unused in compute
    reg = type(synthetic_loadings)(
        alpha=pd.Series(0.0, index=[]),
        betas=pd.DataFrame(),
        r_squared=pd.Series(0.0, index=["Mkt-RF", "HML", "MOM"]),
        residual_std=pd.Series(0.0, index=[]),
        n_obs=0,
        factor_cols=("Mkt-RF", "HML", "MOM"),
    ) if False else _FakeReg()

    weights = pd.Series({"A": 1.0, "B": 0.0, "C": 0.0})
    result = project_portfolio_stress(
        current_weights=weights,
        asset_loadings=synthetic_loadings,
        projected_factor_returns=projected,
        initial_wealth=1000.0,
        scenario_name="test",
        regression=reg,
    )
    # Asset A has beta=(1, 0.3, 0) and alpha=0.0002 → per-quarter return
    # ≈ 0.0002 + 1*0.05 + 0.3*0.05 + 0*0.05 = 0.0652
    # Compound over 4 quarters ≈ 1.0652^4 ≈ 1.2875
    assert result.terminal_wealth > 1000.0
    assert result.terminal_wealth < 1500.0


class _FakeReg:
    """Minimal regression duck-type for tests that don't exercise regression fields."""
    r_squared = pd.Series({"Mkt-RF": 0.8, "HML": 0.5, "MOM": 0.6})
    n_obs = 60


# --------------------------------------------------------------------------- #
# End-to-end                                                                  #
# --------------------------------------------------------------------------- #


def test_run_ccar_stress_end_to_end_synthetic(synthetic_daily_factors, synthetic_macro_panel, synthetic_loadings):
    """Full pipeline on synthetic data without touching FRED or yfinance."""
    weights = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})

    # Replace the canonical scenario's columns with ours so alignment works
    # Monkey-patch by injecting a custom scenario matching our macros
    from inversiones_mama.simulation import ccar_stress as stress_mod

    custom_scenario = pd.DataFrame(
        {"unemployment": [4.0, 4.5, 6.0, 8.0, 9.5, 10.0, 9.0, 7.0, 5.5],
         "vix":          [  18,   25,   55,   75,   60,   40,   30,   22,   18],
         "baa_10y_spread": [2.0, 3.0, 5.0, 6.5, 6.0, 4.5, 3.5, 2.5, 2.0]},
        index=pd.date_range("2026-03-31", periods=9, freq="QE"),
    )

    def fake_load(name):
        if name == "custom":
            return custom_scenario.copy()
        return stress_mod.CCAR_SEVERELY_ADVERSE_2025.copy()

    # Use the real load_scenario but pass our custom via the scenario arg
    old_load = stress_mod.load_scenario
    stress_mod.load_scenario = fake_load
    try:
        result = run_ccar_stress(
            factors_daily=synthetic_daily_factors,
            macro_panel_quarterly=synthetic_macro_panel,
            current_weights=weights,
            asset_loadings=synthetic_loadings,
            initial_wealth=5000.0,
            scenario_name="custom",
        )
    finally:
        stress_mod.load_scenario = old_load

    assert result.horizon_quarters == 8  # 9 levels -> 8 diffs
    assert result.terminal_wealth > 0
    assert 0.0 <= result.max_drawdown <= 1.0
    # R^2 of the fit is in [0, 1]
    assert (result.regression_r_squared >= 0).all()
    assert (result.regression_r_squared <= 1).all()
