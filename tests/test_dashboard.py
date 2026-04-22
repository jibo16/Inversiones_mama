"""Tests for the dashboard helper layer.

The Streamlit app itself (``scripts/dashboard.py``) is not unit-tested
here — only the pure-Python helpers that produce Plotly figures and
load cached artifacts. The Streamlit UI is verified by launching it
(see the smoke-test step in the commit notes).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from inversiones_mama.dashboard import charts, data_sources


# --------------------------------------------------------------------------- #
# Artifact loaders                                                            #
# --------------------------------------------------------------------------- #


def test_available_artifacts_returns_bool_map():
    out = data_sources.available_artifacts()
    assert isinstance(out, dict)
    assert all(isinstance(v, bool) for v in out.values())


def test_load_paper_trade_log_missing(tmp_path):
    assert data_sources.load_paper_trade_log(tmp_path / "nope.json") is None


def test_load_paper_trade_log_malformed(tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text("not json", encoding="utf-8")
    assert data_sources.load_paper_trade_log(bad) is None


def test_load_paper_summary_missing(tmp_path):
    assert data_sources.load_paper_summary(tmp_path / "nope.json") is None


def test_load_paper_summary_roundtrip(tmp_path):
    path = tmp_path / "summary.json"
    payload = {"order_count": 3, "fill_rate": 1.0, "target_weights": {"AAPL": 0.5}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    out = data_sources.load_paper_summary(path)
    assert out == payload


def test_load_universe_stats_missing(tmp_path):
    assert data_sources.load_universe_stats(tmp_path / "nope.csv") is None


def test_load_universe_stats_roundtrip(tmp_path):
    df = pd.DataFrame({"ann_return": [0.1, 0.15]}, index=pd.Index(["A", "B"], name="ticker"))
    p = tmp_path / "universe_stats.csv"
    df.to_csv(p)
    out = data_sources.load_universe_stats(p)
    assert out is not None
    assert list(out.index) == ["A", "B"]


def test_tickers_matches_universe():
    tickers = data_sources.tickers()
    assert len(tickers) == 10
    assert "SPY" in tickers


# --------------------------------------------------------------------------- #
# Alpaca snapshot + breaker state helpers                                     #
# --------------------------------------------------------------------------- #


def test_load_alpaca_snapshot_returns_none_without_keys(monkeypatch):
    for name in ("ALPACA_API_KEY_ID", "ALPACA_API_KEY", "alpaca_key",
                 "ALPACA_API_SECRET_KEY", "ALPACA_SECRET_KEY", "alpaca_secret"):
        monkeypatch.delenv(name, raising=False)
    assert data_sources.load_alpaca_account_snapshot() is None


def test_compute_breaker_state_ok():
    s = data_sources.compute_breaker_state(
        current_wealth=5000.0, peak_wealth=5000.0, threshold_dd=0.5,
    )
    assert s["state"] == "ok"
    assert s["current_drawdown"] == 0.0


def test_compute_breaker_state_warn():
    s = data_sources.compute_breaker_state(
        current_wealth=3000.0, peak_wealth=5000.0, threshold_dd=0.5, warn_threshold_dd=0.3,
    )
    assert s["state"] == "warn"
    assert s["current_drawdown"] == pytest.approx(0.4)


def test_compute_breaker_state_tripped():
    s = data_sources.compute_breaker_state(
        current_wealth=2000.0, peak_wealth=5000.0, threshold_dd=0.5,
    )
    assert s["state"] == "tripped"
    assert s["current_drawdown"] == pytest.approx(0.6)


def test_compute_breaker_state_zero_peak():
    s = data_sources.compute_breaker_state(
        current_wealth=100.0, peak_wealth=0.0, threshold_dd=0.5,
    )
    assert s["current_drawdown"] == 0.0


# --------------------------------------------------------------------------- #
# Chart helpers                                                               #
# --------------------------------------------------------------------------- #


def _make_wealth_series(n: int = 200, start: float = 5000.0) -> pd.Series:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, n)
    return pd.Series(start * (1.0 + rets).cumprod(), index=pd.date_range("2024-01-01", periods=n, freq="B"))


def test_equity_curve_chart_returns_figure():
    wealth = _make_wealth_series()
    fig = charts.equity_curve_chart(wealth)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    # Wealth series trace should match our input length
    assert len(fig.data[0].x) == len(wealth)


def test_equity_curve_with_rebalance_markers():
    wealth = _make_wealth_series()
    rebal = [wealth.index[50], wealth.index[100], wealth.index[150]]
    fig = charts.equity_curve_chart(wealth, rebalance_dates=rebal)
    # Two traces: wealth line + rebalance markers
    assert len(fig.data) == 2


def test_drawdown_chart_shape_and_negativity():
    wealth = _make_wealth_series()
    fig = charts.drawdown_chart(wealth)
    assert isinstance(fig, go.Figure)
    # Drawdown values must be <= 0
    dd_values = fig.data[0].y
    assert max(dd_values) <= 1e-9


def test_mc_terminal_histogram_returns_figure():
    rng = np.random.default_rng(1)
    terminals = rng.normal(5500, 500, 1000)
    fig = charts.mc_terminal_histogram(terminals, initial_capital=5000.0)
    assert isinstance(fig, go.Figure)
    # Should have at least the histogram + 3 reference lines (drawn as shapes/annotations)
    assert len(fig.data) == 1
    # vlines are shapes, not data traces — layout.shapes should have 3
    assert len(fig.layout.shapes) == 3


def test_mc_drawdown_histogram_returns_figure():
    rng = np.random.default_rng(2)
    dds = rng.uniform(0.0, 0.3, 1000)
    fig = charts.mc_drawdown_histogram(dds)
    assert isinstance(fig, go.Figure)
    # p95 and p99 markers
    assert len(fig.layout.shapes) == 2


def test_mc_fan_chart_shape():
    rng = np.random.default_rng(3)
    paths = 5000.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, (500, 253)), axis=1)
    paths = np.concatenate([np.full((500, 1), 5000.0), paths], axis=1)
    fig = charts.mc_fan_chart(paths, initial_capital=5000.0)
    assert isinstance(fig, go.Figure)
    # Expect outer band + inner band + median = 3 traces
    assert len(fig.data) == 3


def test_weights_pie_includes_cash_slice():
    w = pd.Series({"A": 0.3, "B": 0.25, "C": 0.1})
    fig = charts.weights_pie(w)
    assert isinstance(fig, go.Figure)
    labels = list(fig.data[0].labels)
    assert "CASH" in labels  # 1 - 0.65 = 0.35 cash slice
    assert "A" in labels


def test_weights_pie_no_cash_when_fully_allocated():
    w = pd.Series({"A": 0.5, "B": 0.5})
    fig = charts.weights_pie(w)
    labels = list(fig.data[0].labels)
    assert "CASH" not in labels


def test_weights_pie_drops_near_zero_weights():
    w = pd.Series({"A": 0.5, "B": 0.0, "C": 1e-6})
    fig = charts.weights_pie(w)
    labels = list(fig.data[0].labels)
    assert "B" not in labels
    assert "C" not in labels


def test_correlation_heatmap_returns_figure():
    corr = pd.DataFrame(
        [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    fig = charts.correlation_heatmap(corr)
    assert isinstance(fig, go.Figure)


def test_factor_loadings_heatmap_returns_figure():
    betas = pd.DataFrame(
        {"Mkt-RF": [1.0, 0.9], "SMB": [0.2, 0.1], "MOM": [0.0, 0.4]},
        index=["AAPL", "MTUM"],
    )
    fig = charts.factor_loadings_heatmap(betas)
    assert isinstance(fig, go.Figure)


def test_composite_mu_bars_returns_figure():
    mu = pd.Series({"A": 0.08, "B": -0.03, "C": 0.12})
    fig = charts.composite_mu_bars(mu)
    assert isinstance(fig, go.Figure)
    # Sorted ascending
    x_values = list(fig.data[0].x)
    assert x_values == sorted(x_values)


def test_slippage_histogram_empty_returns_placeholder():
    fig = charts.slippage_histogram([])
    assert isinstance(fig, go.Figure)
    # No data traces — just an annotation
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) >= 1


def test_slippage_histogram_populated():
    fig = charts.slippage_histogram([5.0, 10.0, -3.0, 2.5])
    assert len(fig.data) == 1


def test_fill_rate_gauge_colors_by_threshold():
    # Green zone
    g = charts.fill_rate_gauge(0.98)
    assert g.data[0].gauge.bar.color == "#2ca02c"
    # Amber zone
    a = charts.fill_rate_gauge(0.85)
    assert a.data[0].gauge.bar.color == "#ff7f0e"
    # Red zone
    r = charts.fill_rate_gauge(0.50)
    assert r.data[0].gauge.bar.color == "#d62728"


def test_fill_rate_gauge_clamps_out_of_range():
    fig_low = charts.fill_rate_gauge(-0.5)
    fig_high = charts.fill_rate_gauge(1.5)
    # Values are clamped to [0, 100]
    assert fig_low.data[0].value == 0.0
    assert fig_high.data[0].value == 100.0
