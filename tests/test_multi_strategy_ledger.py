"""Unit tests for the MultiStrategyLedger."""

from __future__ import annotations

import pytest

from inversiones_mama.execution.multi_strategy_ledger import (
    MultiStrategyLedger,
    ReconcileReport,
)


@pytest.fixture
def ledger(tmp_path):
    db = tmp_path / "ledger.db"
    led = MultiStrategyLedger(db_path=db)
    yield led
    led.close()


# --- Strategy creation -----------------------------------------------------


def test_create_strategy_records_deposit(ledger):
    ledger.create_strategy("s_test", "vol_targeting", "LIQUID_ETFS", 5000.0)
    assert ledger.cash("s_test") == pytest.approx(5000.0)
    assert ledger.positions("s_test") == {}
    summary = ledger.strategy_summary("s_test")
    assert summary["cash"] == pytest.approx(5000.0)
    assert summary["n_positions"] == 0


def test_create_duplicate_strategy_errors(ledger):
    ledger.create_strategy("dup", "a", "u", 1000.0)
    with pytest.raises(ValueError, match="already exists"):
        ledger.create_strategy("dup", "a", "u", 1000.0)


def test_create_duplicate_strategy_skip(ledger):
    ledger.create_strategy("dup", "a", "u", 1000.0)
    ledger.create_strategy("dup", "b", "x", 9999.0, if_exists="skip")
    # First values preserved
    assert ledger.cash("dup") == pytest.approx(1000.0)


def test_create_duplicate_strategy_replace(ledger):
    ledger.create_strategy("dup", "a", "u", 1000.0)
    ledger.record_fill("dup", "SPY", "buy", 1, 100.0)
    ledger.create_strategy("dup", "b", "x", 5000.0, if_exists="replace")
    # Old fills wiped, new starting_cash
    assert ledger.cash("dup") == pytest.approx(5000.0)
    assert ledger.positions("dup") == {}


def test_list_strategies(ledger):
    ledger.create_strategy("a", "x", "u1", 1000.0)
    ledger.create_strategy("b", "y", "u2", 2000.0)
    out = ledger.list_strategies()
    ids = [s["strategy_id"] for s in out]
    assert "a" in ids and "b" in ids


# --- Fill recording --------------------------------------------------------


def test_record_buy_decreases_cash_and_adds_position(ledger):
    ledger.create_strategy("s", "a", "u", 10000.0)
    ledger.record_fill("s", "SPY", "buy", 5, 200.0)
    assert ledger.cash("s") == pytest.approx(10000.0 - 1000.0)
    assert ledger.positions("s") == {"SPY": pytest.approx(5.0)}


def test_record_sell_increases_cash_and_reduces_position(ledger):
    ledger.create_strategy("s", "a", "u", 10000.0)
    ledger.record_fill("s", "SPY", "buy", 10, 100.0)
    ledger.record_fill("s", "SPY", "sell", 4, 110.0)
    assert ledger.positions("s") == {"SPY": pytest.approx(6.0)}
    # Cash: 10000 - (10*100) + (4*110) = 9000 + 440 = 9440
    assert ledger.cash("s") == pytest.approx(9440.0)


def test_fractional_shares_supported(ledger):
    ledger.create_strategy("s", "a", "u", 10000.0)
    ledger.record_fill("s", "SPY", "buy", 3.5, 200.0)
    ledger.record_fill("s", "SPY", "sell", 0.5, 210.0)
    assert ledger.positions("s") == {"SPY": pytest.approx(3.0)}
    # Cash: 10000 - (3.5*200) + (0.5*210) = 10000 - 700 + 105 = 9405
    assert ledger.cash("s") == pytest.approx(9405.0)


def test_zero_qty_rejected(ledger):
    ledger.create_strategy("s", "a", "u", 1000.0)
    with pytest.raises(ValueError, match="qty must be positive"):
        ledger.record_fill("s", "SPY", "buy", 0, 100.0)


def test_negative_qty_rejected(ledger):
    ledger.create_strategy("s", "a", "u", 1000.0)
    with pytest.raises(ValueError, match="qty must be positive"):
        ledger.record_fill("s", "SPY", "buy", -1, 100.0)


def test_invalid_side_rejected(ledger):
    ledger.create_strategy("s", "a", "u", 1000.0)
    with pytest.raises(ValueError, match="side must be"):
        ledger.record_fill("s", "SPY", "hold", 1, 100.0)


def test_fill_against_unknown_strategy_errors(ledger):
    with pytest.raises(KeyError, match="unknown strategy_id"):
        ledger.record_fill("ghost", "SPY", "buy", 1, 100.0)


def test_commission_reduces_cash(ledger):
    ledger.create_strategy("s", "a", "u", 10000.0)
    ledger.record_fill("s", "SPY", "buy", 5, 200.0, commission=0.35)
    # Cash: 10000 - 1000 - 0.35
    assert ledger.cash("s") == pytest.approx(8999.65)


# --- Avg cost --------------------------------------------------------------


def test_avg_cost_single_buy(ledger):
    ledger.create_strategy("s", "a", "u", 10000.0)
    ledger.record_fill("s", "SPY", "buy", 5, 200.0)
    assert ledger.avg_cost("s", "SPY") == pytest.approx(200.0)


def test_avg_cost_multiple_buys_weighted(ledger):
    ledger.create_strategy("s", "a", "u", 100000.0)
    ledger.record_fill("s", "SPY", "buy", 5, 200.0)   # cost=1000
    ledger.record_fill("s", "SPY", "buy", 5, 220.0)   # cost=1100, total=2100 / 10 = 210
    assert ledger.avg_cost("s", "SPY") == pytest.approx(210.0)


def test_avg_cost_after_partial_sell(ledger):
    """Proportional-reduction model: selling half the shares halves the basis."""
    ledger.create_strategy("s", "a", "u", 100000.0)
    ledger.record_fill("s", "SPY", "buy", 10, 200.0)  # basis = 2000, qty = 10
    ledger.record_fill("s", "SPY", "sell", 5, 250.0)  # half sold -> basis = 1000, qty = 5
    assert ledger.avg_cost("s", "SPY") == pytest.approx(200.0)


def test_avg_cost_returns_none_when_flat(ledger):
    ledger.create_strategy("s", "a", "u", 10000.0)
    ledger.record_fill("s", "SPY", "buy", 5, 200.0)
    ledger.record_fill("s", "SPY", "sell", 5, 210.0)
    assert ledger.avg_cost("s", "SPY") is None


def test_avg_cost_unknown_ticker_returns_none(ledger):
    ledger.create_strategy("s", "a", "u", 10000.0)
    assert ledger.avg_cost("s", "NOPE") is None


# --- Cross-strategy isolation ---------------------------------------------


def test_two_strategies_same_ticker_tracked_independently(ledger):
    ledger.create_strategy("a", "x", "u", 10000.0)
    ledger.create_strategy("b", "y", "u", 10000.0)
    ledger.record_fill("a", "SPY", "buy", 3, 100.0)
    ledger.record_fill("b", "SPY", "buy", 5, 101.0)
    assert ledger.positions("a") == {"SPY": pytest.approx(3.0)}
    assert ledger.positions("b") == {"SPY": pytest.approx(5.0)}
    assert ledger.total_positions() == {"SPY": pytest.approx(8.0)}
    assert ledger.cash("a") == pytest.approx(10000 - 300)
    assert ledger.cash("b") == pytest.approx(10000 - 505)


def test_cross_strategy_sells_dont_affect_each_other(ledger):
    ledger.create_strategy("a", "x", "u", 10000.0)
    ledger.create_strategy("b", "y", "u", 10000.0)
    ledger.record_fill("a", "SPY", "buy", 10, 100.0)
    ledger.record_fill("b", "SPY", "buy", 10, 100.0)
    ledger.record_fill("a", "SPY", "sell", 3, 110.0)  # Only strategy a sells
    assert ledger.positions("a") == {"SPY": pytest.approx(7.0)}
    assert ledger.positions("b") == {"SPY": pytest.approx(10.0)}
    assert ledger.total_positions() == {"SPY": pytest.approx(17.0)}


# --- Reconciliation --------------------------------------------------------


def test_reconcile_in_sync(ledger):
    ledger.create_strategy("a", "x", "u", 10000.0)
    ledger.create_strategy("b", "y", "u", 10000.0)
    ledger.record_fill("a", "SPY", "buy", 3, 100.0)
    ledger.record_fill("b", "SPY", "buy", 5, 101.0)
    ledger.record_fill("a", "GLD", "buy", 2, 180.0)
    report = ledger.reconcile_against_broker({"SPY": 8, "GLD": 2})
    assert report.in_sync is True
    assert report.drift == {}


def test_reconcile_detects_drift(ledger):
    ledger.create_strategy("s", "a", "u", 10000.0)
    ledger.record_fill("s", "SPY", "buy", 5, 100.0)
    # Broker says 7 SPY but ledger has 5 -> drift of +2
    report = ledger.reconcile_against_broker({"SPY": 7})
    assert report.in_sync is False
    assert report.drift == {"SPY": pytest.approx(2.0)}


def test_reconcile_detects_only_broker_ticker(ledger):
    ledger.create_strategy("s", "a", "u", 10000.0)
    ledger.record_fill("s", "SPY", "buy", 5, 100.0)
    # Broker has a position ledger doesn't know about
    report = ledger.reconcile_against_broker({"SPY": 5, "GHOST": 1})
    assert report.in_sync is False
    assert "GHOST" in report.tickers_only_in_broker
    assert report.drift.get("GHOST") == pytest.approx(1.0)


def test_reconcile_detects_only_ledger_ticker(ledger):
    ledger.create_strategy("s", "a", "u", 10000.0)
    ledger.record_fill("s", "SPY", "buy", 5, 100.0)
    # Broker has nothing
    report = ledger.reconcile_against_broker({})
    assert report.in_sync is False
    assert "SPY" in report.tickers_only_in_ledger
    assert report.drift.get("SPY") == pytest.approx(-5.0)


def test_reconcile_within_tolerance_is_ok(ledger):
    """Fractional-share rounding within 1e-4 shares is considered in-sync."""
    ledger.create_strategy("s", "a", "u", 10000.0)
    ledger.record_fill("s", "SPY", "buy", 5.00003, 100.0)
    report = ledger.reconcile_against_broker({"SPY": 5.0}, tolerance=1e-4)
    assert report.in_sync is True


def test_reconcile_report_text_format(ledger):
    ledger.create_strategy("s", "a", "u", 10000.0)
    ledger.record_fill("s", "SPY", "buy", 5, 100.0)
    report = ledger.reconcile_against_broker({"SPY": 7})
    text = report.to_text()
    assert "DRIFT DETECTED" in text
    assert "SPY" in text


# --- Summary ---------------------------------------------------------------


def test_strategy_summary_no_prices(ledger):
    ledger.create_strategy("s", "vol_targeting", "LIQUID_ETFS", 5000.0)
    ledger.record_fill("s", "SPY", "buy", 5, 200.0)
    sm = ledger.strategy_summary("s")
    assert sm["strategy_id"] == "s"
    assert sm["allocator"] == "vol_targeting"
    assert sm["starting_cash"] == pytest.approx(5000.0)
    assert sm["cash"] == pytest.approx(4000.0)
    assert sm["n_positions"] == 1
    assert sm["market_value"] == 0.0  # no prices provided
    assert sm["unrealized_pnl"] is None


def test_strategy_summary_with_prices_computes_pnl(ledger):
    ledger.create_strategy("s", "a", "u", 10000.0)
    ledger.record_fill("s", "SPY", "buy", 10, 200.0)   # basis 200
    # Current price $220 -> unrealized PnL = (220 - 200) * 10 = 200
    sm = ledger.strategy_summary("s", latest_prices={"SPY": 220.0})
    assert sm["market_value"] == pytest.approx(10 * 220)
    assert sm["unrealized_pnl"] == pytest.approx(200.0)
    # Equity = cash (8000) + market value (2200) = 10200
    assert sm["equity"] == pytest.approx(10200.0)
    assert sm["return_vs_start"] == pytest.approx(0.02)  # +2%


# --- Bulk seeding ----------------------------------------------------------


def test_bulk_record_fills_seeds_multiple(ledger):
    ledger.create_strategy("seed", "x", "u", 100000.0)
    fills = [
        {"ticker": "SPY", "side": "buy", "qty": 5, "fill_price": 200.0},
        {"ticker": "QQQ", "side": "buy", "qty": 3, "fill_price": 500.0},
        {"ticker": "GLD", "side": "buy", "qty": 2, "fill_price": 180.0},
    ]
    ids = ledger.bulk_record_fills("seed", fills)
    assert len(ids) == 3
    pos = ledger.positions("seed")
    assert pos["SPY"] == pytest.approx(5)
    assert pos["QQQ"] == pytest.approx(3)
    assert pos["GLD"] == pytest.approx(2)


# --- Persistence -----------------------------------------------------------


def test_ledger_persists_across_instances(tmp_path):
    db = tmp_path / "persist.db"
    # First instance: write
    led1 = MultiStrategyLedger(db_path=db)
    led1.create_strategy("s", "a", "u", 5000.0)
    led1.record_fill("s", "SPY", "buy", 3, 100.0)
    led1.close()
    # Second instance: read
    led2 = MultiStrategyLedger(db_path=db)
    assert led2.positions("s") == {"SPY": pytest.approx(3)}
    assert led2.cash("s") == pytest.approx(5000 - 300)
    led2.close()
