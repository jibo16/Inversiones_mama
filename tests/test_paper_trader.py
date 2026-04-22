"""Tests for inversiones_mama.execution.paper_trader."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.execution.ibkr import OrderIntent
from inversiones_mama.execution.paper_trader import (
    DryRunClient,
    ExecutionClient,
    PaperRebalanceSummary,
    PaperTradingOrchestrator,
)
from inversiones_mama.execution.trade_log import TradeLog


# --------------------------------------------------------------------------- #
# Synthetic prices+factors for the orchestrator                                #
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_market():
    """Generate a ~1.5-year history for 5 factor ETFs plus the factor panel."""
    rng = np.random.default_rng(2026_04_22)
    n = 400
    dates = pd.date_range("2024-11-01", periods=n, freq="B")

    factors_raw = {
        "Mkt-RF": rng.normal(0.0004, 0.010, n),
        "SMB":    rng.normal(0.0001, 0.005, n),
        "HML":    rng.normal(0.0001, 0.005, n),
        "RMW":    rng.normal(0.0002, 0.004, n),
        "CMA":    rng.normal(0.0000, 0.004, n),
        "MOM":    rng.normal(0.0002, 0.007, n),
        "RF":     np.full(n, 0.00015),
    }
    factors = pd.DataFrame(factors_raw, index=dates)

    betas = {
        "A": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "B": [1.0, 0.5, 0.3, 0.1, 0.0, 0.0],
        "C": [1.0, 0.0, 0.0, 0.0, 0.0, 0.4],
        "D": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        "E": [0.8, 0.2, 0.1, 0.1, 0.0, 0.2],
    }
    fmat = factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]].to_numpy()
    idio = rng.normal(0, 0.003, (n, len(betas)))
    asset_rets = {}
    for i, (t, b) in enumerate(betas.items()):
        asset_rets[t] = factors["RF"].to_numpy() + fmat @ np.asarray(b) + idio[:, i]

    returns_df = pd.DataFrame(asset_rets, index=dates)
    prices = (1.0 + returns_df).cumprod() * 100.0
    return prices, factors


# --------------------------------------------------------------------------- #
# DryRunClient unit tests                                                     #
# --------------------------------------------------------------------------- #


def test_dryrun_client_satisfies_protocol():
    c = DryRunClient(starting_cash=1000.0)
    assert isinstance(c, ExecutionClient)


def test_dryrun_fill_at_expected_price():
    c = DryRunClient(starting_cash=1000.0, latest_prices={"AAPL": 100.0})
    fill = c.submit_order(OrderIntent(ticker="AAPL", shares=5))
    assert fill.status == "filled"
    assert fill.fill_price == 100.0
    assert fill.filled_quantity == 5
    assert c.get_positions() == {"AAPL": 5}
    assert c.get_cash() == 500.0


def test_dryrun_rejects_when_no_price():
    c = DryRunClient(starting_cash=1000.0)  # no prices set
    fill = c.submit_order(OrderIntent(ticker="AAPL", shares=5))
    assert fill.status == "rejected"
    assert fill.filled_quantity == 0
    assert c.get_cash() == 1000.0
    assert c.get_positions() == {}


def test_dryrun_rejects_when_insufficient_cash():
    c = DryRunClient(starting_cash=100.0, latest_prices={"X": 50.0})
    fill = c.submit_order(OrderIntent(ticker="X", shares=10))  # costs 500
    assert fill.status == "rejected"
    assert fill.context["reason"] == "insufficient_cash"
    assert c.get_cash() == 100.0


def test_dryrun_sell_increases_cash():
    c = DryRunClient(starting_cash=1000.0, latest_prices={"X": 50.0})
    c.submit_order(OrderIntent(ticker="X", shares=10))  # buy 10
    assert c.get_cash() == 500.0
    fill = c.submit_order(OrderIntent(ticker="X", shares=-5))  # sell 5
    assert fill.status == "filled"
    assert c.get_cash() == 750.0
    assert c.get_positions() == {"X": 5}


def test_dryrun_zeroing_position_removes_entry():
    c = DryRunClient(starting_cash=1000.0, latest_prices={"X": 50.0})
    c.submit_order(OrderIntent(ticker="X", shares=10))
    c.submit_order(OrderIntent(ticker="X", shares=-10))
    assert c.get_positions() == {}  # fully closed positions drop out


def test_dryrun_set_latest_price_updates_quote():
    c = DryRunClient()
    c.set_latest_price("X", 42.0)
    assert c.get_latest_price("X") == 42.0


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #


def test_orchestrator_rejects_empty_history():
    c = DryRunClient(starting_cash=5000.0)
    with pytest.raises(ValueError, match="prices_history"):
        PaperTradingOrchestrator(c, pd.DataFrame(), pd.DataFrame({"RF": [0.0]}))


def test_orchestrator_rejects_empty_factors(synthetic_market):
    prices, _ = synthetic_market
    c = DryRunClient(starting_cash=5000.0)
    with pytest.raises(ValueError, match="factors_history"):
        PaperTradingOrchestrator(c, prices, pd.DataFrame())


def test_orchestrator_raises_on_short_overlap(synthetic_market):
    prices, factors = synthetic_market
    # Reduce factors to a tiny slice so overlap < lookback_days
    factors_short = factors.tail(50)
    c = DryRunClient(starting_cash=5000.0)
    orch = PaperTradingOrchestrator(c, prices, factors_short, lookback_days=252)
    # Seed prices so the initial price fetch doesn't short-circuit
    for t in prices.columns:
        c.set_latest_price(t, float(prices[t].iloc[-1]))
    with pytest.raises(ValueError, match="aligned rows"):
        orch.rebalance()


def test_orchestrator_runs_one_rebalance(synthetic_market):
    prices, factors = synthetic_market
    c = DryRunClient(starting_cash=5000.0)
    # Set latest prices for each ticker based on last row
    for t in prices.columns:
        c.set_latest_price(t, float(prices[t].iloc[-1]))
    orch = PaperTradingOrchestrator(c, prices, factors, lookback_days=252)
    summary = orch.rebalance()
    assert isinstance(summary, PaperRebalanceSummary)
    assert summary.tickers == tuple(prices.columns)
    # Target weights are a Series over the same tickers, non-negative, capped
    assert set(summary.target_weights.index) == set(prices.columns)
    assert (summary.target_weights >= -1e-9).all()
    assert (summary.target_weights <= 0.35 + 1e-9).all()
    # We placed at least one buy (starting from zero positions -> deploy cash)
    assert summary.order_count >= 1
    # TradeLog logged every order
    assert len(summary.trade_log) == summary.order_count


def test_orchestrator_second_rebalance_respects_current_positions(synthetic_market):
    """After one rebalance, positions are non-zero; a second rebalance should
    produce *smaller* weight deltas than the first."""
    prices, factors = synthetic_market
    c = DryRunClient(starting_cash=5000.0)
    for t in prices.columns:
        c.set_latest_price(t, float(prices[t].iloc[-1]))
    orch = PaperTradingOrchestrator(c, prices, factors, lookback_days=252)
    s1 = orch.rebalance()
    s2 = orch.rebalance()

    # Current weights after s1 should be close to s1.target_weights (DryRun fills perfect)
    # So the weight delta at s2 should be nearly zero.
    deltas2 = (s2.target_weights - s2.current_weights_before).abs().sum()
    deltas1 = (s1.target_weights - s1.current_weights_before).abs().sum()
    assert deltas2 < deltas1 + 1e-9, "second rebalance should have smaller weight change than first"


def test_orchestrator_handles_missing_price_for_one_ticker(synthetic_market):
    prices, factors = synthetic_market
    c = DryRunClient(starting_cash=5000.0)
    # Set prices for all but one ticker
    tickers = list(prices.columns)
    for t in tickers[:-1]:
        c.set_latest_price(t, float(prices[t].iloc[-1]))
    # The last ticker has no price: it must be skipped (not cause an exception)
    orch = PaperTradingOrchestrator(c, prices, factors, lookback_days=252)
    summary = orch.rebalance()
    # All filled orders should be for tickers with prices
    for entry in summary.trade_log:
        assert entry.signal.ticker in tickers[:-1]


def test_orchestrator_trade_log_is_persistable(synthetic_market, tmp_path):
    prices, factors = synthetic_market
    c = DryRunClient(starting_cash=5000.0)
    for t in prices.columns:
        c.set_latest_price(t, float(prices[t].iloc[-1]))
    orch = PaperTradingOrchestrator(c, prices, factors, lookback_days=252)
    summary = orch.rebalance()
    path = tmp_path / "paper_trades.json"
    summary.trade_log.save(path)
    loaded = TradeLog.load(path)
    assert len(loaded) == len(summary.trade_log)


def test_orchestrator_signal_context_propagates(synthetic_market):
    prices, factors = synthetic_market
    c = DryRunClient(starting_cash=5000.0)
    for t in prices.columns:
        c.set_latest_price(t, float(prices[t].iloc[-1]))
    orch = PaperTradingOrchestrator(c, prices, factors, lookback_days=252)
    summary = orch.rebalance(signal_context={"run_id": "test-run-42"})
    for entry in summary.trade_log:
        assert entry.signal.context["run_id"] == "test-run-42"


# --------------------------------------------------------------------------- #
# Circuit-breaker + PDT gate integration                                      #
# --------------------------------------------------------------------------- #


def test_orchestrator_circuit_breaker_halts_on_trip(synthetic_market):
    """If the breaker is already tripped (wealth below threshold), rebalance halts."""
    from inversiones_mama.execution.circuit_breaker import CircuitBreaker

    prices, factors = synthetic_market
    c = DryRunClient(starting_cash=2000.0)  # start "below" an assumed 5000 peak
    for t in prices.columns:
        c.set_latest_price(t, float(prices[t].iloc[-1]))
    orch = PaperTradingOrchestrator(c, prices, factors, lookback_days=252)

    # Pretend the peak was 5000 and we're now at 2000 (60% drawdown).
    # Threshold 0.30 → this will trip on first update.
    breaker = CircuitBreaker(threshold=0.30, initial_wealth=5000.0)

    summary = orch.rebalance(circuit_breaker=breaker)
    assert summary.halted is True
    assert summary.order_count == 0
    assert "circuit_breaker_tripped" in (summary.halt_reason or "")
    assert summary.breaker_drawdown is not None
    assert summary.breaker_drawdown > 0.30


def test_orchestrator_circuit_breaker_passes_when_ok(synthetic_market):
    """When wealth hasn't dropped below threshold, rebalance proceeds normally."""
    from inversiones_mama.execution.circuit_breaker import CircuitBreaker

    prices, factors = synthetic_market
    c = DryRunClient(starting_cash=5000.0)
    for t in prices.columns:
        c.set_latest_price(t, float(prices[t].iloc[-1]))
    orch = PaperTradingOrchestrator(c, prices, factors, lookback_days=252)
    breaker = CircuitBreaker(threshold=0.50, initial_wealth=5000.0)
    summary = orch.rebalance(circuit_breaker=breaker)
    assert summary.halted is False
    assert summary.order_count >= 1
    assert summary.breaker_drawdown == 0.0


def test_orchestrator_exempt_pdt_tracker_never_blocks(synthetic_market):
    """PDT-exempt account (>= $25k) never has orders skipped regardless of log."""
    from inversiones_mama.execution.pdt import PDTTracker

    prices, factors = synthetic_market
    c = DryRunClient(starting_cash=30_000.0)
    for t in prices.columns:
        c.set_latest_price(t, float(prices[t].iloc[-1]))
    orch = PaperTradingOrchestrator(c, prices, factors, lookback_days=252)
    pdt = PDTTracker(account_equity=30_000.0)
    assert pdt.exempt is True

    summary = orch.rebalance(pdt_tracker=pdt)
    assert summary.halt_reason is None
    assert summary.order_count >= 1
