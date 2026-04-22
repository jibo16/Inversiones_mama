"""Tests for inversiones_mama.validation.gates.

Three classes of tests:

1. Schema sanity on a synthetic end-to-end run (no network).
2. Gate-logic unit tests (pass/fail on planted metrics).
3. Live end-to-end: produces the real v1a verdict against the 10-ETF
   universe and asserts the report is well-formed. Marked ``@pytest.mark.live``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.backtest.engine import BacktestConfig
from inversiones_mama.validation.gates import (
    GateVerdict,
    ValidationReport,
    render_report,
    run_full_validation,
)


# --------------------------------------------------------------------------- #
# Reuse engine test fixture                                                   #
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_prices_and_factors() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copy of the fixture from test_engine.py. ~2.4 years, 5 assets."""
    rng = np.random.default_rng(20260422)
    n = 600
    dates = pd.date_range("2023-01-02", periods=n, freq="B")

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
    betas = {
        "A": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "B": [1.0, 0.5, 0.3, 0.1, 0.0, 0.0],
        "C": [1.0, 0.0, 0.0, 0.0, 0.0, 0.4],
        "D": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        "E": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    fmat = factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]].to_numpy()
    idio = rng.normal(0, 0.003, (n, len(betas)))
    asset_rets = {}
    for i, (t, b) in enumerate(betas.items()):
        asset_rets[t] = rf + fmat @ np.asarray(b) + idio[:, i]
    rdf = pd.DataFrame(asset_rets, index=dates)
    prices = (1.0 + rdf).cumprod() * 100.0
    first = pd.DataFrame(
        {t: [100.0] for t in asset_rets},
        index=[dates[0] - pd.tseries.offsets.BDay()],
    )
    prices = pd.concat([first, prices])
    first_factor = factors.iloc[[0]].copy()
    first_factor.index = [prices.index[0]]
    factors_full = pd.concat([first_factor, factors])
    return prices, factors_full


# --------------------------------------------------------------------------- #
# Schema                                                                      #
# --------------------------------------------------------------------------- #


def test_run_full_validation_synthetic_schema(synthetic_prices_and_factors):
    prices, factors = synthetic_prices_and_factors
    report = run_full_validation(
        prices, factors,
        mc_n_paths=500, mc_horizon_days=60,
        rng=np.random.default_rng(0),
    )
    assert isinstance(report, ValidationReport)
    assert report.initial_wealth == 5000.0
    assert report.n_rebalances >= 1
    # Performance metrics populated
    assert np.isfinite(report.metrics_full.sharpe_ratio)
    assert np.isfinite(report.metrics_full.deflated_sharpe)
    # Gates populated
    assert len(report.gates) >= 2  # at least turnover + OOS Sharpe
    for g in report.gates:
        assert isinstance(g, GateVerdict)
        assert isinstance(g.passed, bool)
        assert g.name and g.description


def test_run_full_validation_produces_ismoos_split(synthetic_prices_and_factors):
    """With >= 504 daily observations, the validator should produce IS/OOS splits."""
    prices, factors = synthetic_prices_and_factors
    report = run_full_validation(
        prices, factors, mc_n_paths=200, mc_horizon_days=40,
        rng=np.random.default_rng(1),
    )
    assert report.oos_split_date is not None
    assert report.metrics_is is not None
    assert report.metrics_oos is not None


def test_run_full_validation_mc_gates_present(synthetic_prices_and_factors):
    prices, factors = synthetic_prices_and_factors
    report = run_full_validation(
        prices, factors, mc_n_paths=200, mc_horizon_days=40,
        rng=np.random.default_rng(2),
    )
    gate_names = {g.name for g in report.gates}
    assert "annualized_turnover_cost" in gate_names
    assert "oos_sharpe_positive" in gate_names
    # MC gates should be there because MC ran
    assert "mc_prob_loss_40pct" in gate_names
    assert "mc_dd_95th_pct" in gate_names
    assert "mc_rck_bound_honored" in gate_names


def test_render_report_multiline(synthetic_prices_and_factors):
    prices, factors = synthetic_prices_and_factors
    report = run_full_validation(
        prices, factors, mc_n_paths=200, mc_horizon_days=40,
        rng=np.random.default_rng(3),
    )
    text = render_report(report)
    assert "INVERSIONES_MAMA v1a VALIDATION REPORT" in text
    assert "OVERALL v1a VERDICT" in text
    assert "PASS" in text or "FAIL" in text
    # At least one gate line
    assert "Gate verdicts" in text


def test_all_pass_property(synthetic_prices_and_factors):
    prices, factors = synthetic_prices_and_factors
    report = run_full_validation(
        prices, factors, mc_n_paths=200, mc_horizon_days=40,
        rng=np.random.default_rng(4),
    )
    # all_pass must match the conjunction over every gate
    expected = all(g.passed for g in report.gates)
    assert report.all_pass == expected


# --------------------------------------------------------------------------- #
# Config overrides                                                            #
# --------------------------------------------------------------------------- #


def test_run_full_validation_respects_custom_config(synthetic_prices_and_factors):
    prices, factors = synthetic_prices_and_factors
    # Use a tighter cap so target weights differ from defaults
    cfg = BacktestConfig(initial_capital=10_000.0, per_name_cap=0.25)
    report = run_full_validation(
        prices, factors, config=cfg,
        mc_n_paths=200, mc_horizon_days=40,
        rng=np.random.default_rng(5),
    )
    assert report.initial_wealth == 10_000.0
    assert report.config.per_name_cap == 0.25
    # Target weights must respect the tightened cap at every rebalance
    # (check via inspection: highest weight in most recent MC rebalance)
    if report.mc_result is not None:
        assert (report.mc_result.weights <= 0.25 + 1e-9).all()


# --------------------------------------------------------------------------- #
# Live verdict                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_v1a_verdict(tmp_path, monkeypatch, capsys):
    """Real v1a verdict on the 10-ETF universe.

    This test does NOT assert pass/fail — the verdict depends on the strategy's
    realized behavior, which is the whole question. It asserts only that the
    report is well-formed and prints the full renderer for human review.
    """
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

    report = run_full_validation(
        prices, factors,
        mc_n_paths=2_000,
        mc_horizon_days=252,
        rng=np.random.default_rng(20260422),
    )

    # Well-formedness checks
    assert report.n_rebalances >= 20
    assert 100 < report.final_wealth < 1_000_000
    assert np.isfinite(report.metrics_full.sharpe_ratio)
    assert np.isfinite(report.metrics_full.deflated_sharpe)
    assert 0.0 <= report.metrics_full.deflated_sharpe <= 1.0
    assert report.metrics_is is not None and report.metrics_oos is not None
    assert report.mc_result is not None
    assert all(isinstance(g.passed, bool) for g in report.gates)

    text = render_report(report)
    assert "OVERALL v1a VERDICT" in text
    # Print so the human-readable verdict shows up in live-test output
    print("\n" + text)
    captured = capsys.readouterr()
    assert "INVERSIONES_MAMA v1a" in captured.out
