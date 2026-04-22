"""Smoke tests for config constants — make sure nothing's malformed."""

from __future__ import annotations

from inversiones_mama import config


def test_universe_has_10_tickers():
    assert len(config.UNIVERSE) == 10


def test_universe_contains_benchmark():
    assert config.BENCHMARK in config.UNIVERSE


def test_kelly_fraction_bounded():
    assert 0.0 < config.KELLY_FRACTION <= 1.0


def test_max_weight_per_name_bounded():
    assert 0.0 < config.MAX_WEIGHT_PER_NAME <= 1.0


def test_lookback_days_reasonable():
    assert 60 <= config.LOOKBACK_DAYS <= 1260  # 3mo to 5yr sanity


def test_rck_parameters():
    assert 0.0 < config.RCK_MAX_DRAWDOWN_THRESHOLD < 1.0
    assert 0.0 < config.RCK_MAX_DRAWDOWN_PROBABILITY < 1.0


def test_ibkr_cost_constants_sane():
    assert config.IBKR_FIXED_PER_SHARE > 0
    assert config.IBKR_MIN_PER_ORDER > 0
    assert 0 < config.IBKR_MAX_PCT_TRADE < 1
    assert config.IBKR_SLIPPAGE_BPS >= 0


def test_gates_instance():
    g = config.GATES
    assert 0 < g.max_prob_loss_40pct < 1
    assert 0 < g.max_prob_loss_60pct <= g.max_prob_loss_40pct
    assert 0 < g.max_dd_95th_pct < 1
    assert g.min_oos_sharpe >= 0
    assert g.max_annual_turnover_cost > 0
