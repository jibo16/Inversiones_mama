"""Unit tests for the IBKR TWS adapter (ib_insync).

No live TWS is required — all tests inject a mock ``IB`` object.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from inversiones_mama.execution.ibkr import OrderIntent
from inversiones_mama.execution.ibkr_tws import (
    IBKRTWSClient,
    IBKRTWSConfig,
    IBKRTWSConnectError,
    IBKRTWSError,
)


# --- Helpers ----------------------------------------------------------------


def _make_contract(symbol: str) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, secType="STK", exchange="SMART", currency="USD")


def _make_mock_ib(
    *,
    positions: list | None = None,
    account_summary: list | None = None,
    quote: dict | None = None,
    placed_trade: MagicMock | None = None,
) -> MagicMock:
    """Build a mock ib_insync.IB that returns configurable responses."""
    ib = MagicMock(name="mock_ib")
    ib.isConnected.return_value = True
    ib.connect.return_value = None
    ib.disconnect.return_value = None

    def qualify(c):  # returns list of contracts (ib_insync contract)
        # ib_insync returns the same Stock objects; we just enrich with symbol
        c.symbol = c.symbol.upper()
        return [c]

    ib.qualifyContracts.side_effect = qualify
    ib.positions.return_value = positions or []
    ib.accountSummary.return_value = account_summary or []

    # Market data: return a SimpleNamespace with configurable bid/ask/last
    quote_obj = SimpleNamespace(
        bid=(quote or {}).get("bid"),
        ask=(quote or {}).get("ask"),
        last=(quote or {}).get("last"),
    )
    ib.reqMktData.return_value = quote_obj
    ib.cancelMktData.return_value = None
    ib.sleep.return_value = None
    if placed_trade is not None:
        ib.placeOrder.return_value = placed_trade
    return ib


# --- Config -----------------------------------------------------------------


def test_config_from_env_defaults(monkeypatch):
    for v in ("IBKR_TWS_HOST", "IBKR_TWS_PORT", "IBKR_TWS_CLIENT_ID"):
        monkeypatch.delenv(v, raising=False)
    cfg = IBKRTWSConfig.from_env()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 7497
    assert cfg.client_id == 1
    assert cfg.is_paper is True


def test_config_from_env_live_port(monkeypatch):
    monkeypatch.setenv("IBKR_TWS_PORT", "7496")
    cfg = IBKRTWSConfig.from_env()
    assert cfg.port == 7496
    assert cfg.is_paper is False


# --- Connection -------------------------------------------------------------


def test_connect_is_idempotent():
    ib = _make_mock_ib()
    # Disconnected on first check, connected after we call IB.connect()
    ib.isConnected.side_effect = [False, True, True]
    client = IBKRTWSClient(ib=ib)
    client.connect()
    client.connect()
    # Only the first call reaches through to IB.connect; subsequent short-circuit
    assert ib.connect.call_count == 1


def test_connect_wraps_failure_in_named_exception():
    ib = MagicMock()
    ib.isConnected.return_value = False
    ib.connect.side_effect = ConnectionRefusedError("no gateway running")

    client = IBKRTWSClient(ib=ib)
    with pytest.raises(IBKRTWSConnectError, match="Could not connect to TWS"):
        client.connect()


def test_context_manager_connects_and_disconnects():
    ib = _make_mock_ib()
    with IBKRTWSClient(ib=ib) as client:
        assert client._ib is ib
    assert ib.disconnect.called


# --- Positions & cash -------------------------------------------------------


def test_get_positions_parses_ib_positions():
    positions = [
        SimpleNamespace(contract=_make_contract("AVDV"), position=11),
        SimpleNamespace(contract=_make_contract("GLD"), position=3),
        SimpleNamespace(contract=_make_contract("ZERO"), position=0),  # should drop
    ]
    ib = _make_mock_ib(positions=positions)
    client = IBKRTWSClient(ib=ib)
    out = client.get_positions()
    assert out == {"AVDV": 11, "GLD": 3}


def test_get_cash_prefers_total_cash_value():
    summary = [
        SimpleNamespace(tag="TotalCashValue", currency="USD", value="12345.67"),
        SimpleNamespace(tag="CashBalance", currency="USD", value="99999.99"),
    ]
    ib = _make_mock_ib(account_summary=summary)
    client = IBKRTWSClient(ib=ib)
    assert client.get_cash() == pytest.approx(12345.67)


def test_get_cash_falls_back_to_cash_balance():
    summary = [
        SimpleNamespace(tag="CashBalance", currency="USD", value="5000"),
    ]
    ib = _make_mock_ib(account_summary=summary)
    client = IBKRTWSClient(ib=ib)
    assert client.get_cash() == pytest.approx(5000.0)


def test_get_cash_raises_when_no_usd_field():
    summary = [
        SimpleNamespace(tag="NetLiquidation", currency="USD", value="100000"),
    ]
    ib = _make_mock_ib(account_summary=summary)
    client = IBKRTWSClient(ib=ib)
    with pytest.raises(IBKRTWSError, match="could not read USD cash"):
        client.get_cash()


# --- Quote lookup -----------------------------------------------------------


def test_get_latest_price_uses_midpoint_when_bid_ask_present():
    ib = _make_mock_ib(quote={"bid": 100.0, "ask": 100.20, "last": 100.10})
    client = IBKRTWSClient(ib=ib)
    px = client.get_latest_price("SPY")
    assert px == pytest.approx(100.10)


def test_get_latest_price_falls_back_to_last_when_nobook():
    ib = _make_mock_ib(quote={"bid": None, "ask": None, "last": 99.5})
    client = IBKRTWSClient(ib=ib)
    assert client.get_latest_price("XYZ") == pytest.approx(99.5)


def test_get_latest_price_returns_none_when_all_missing():
    ib = _make_mock_ib(quote={"bid": None, "ask": None, "last": None})
    client = IBKRTWSClient(ib=ib)
    assert client.get_latest_price("XYZ") is None


# --- Order submission -------------------------------------------------------


def _make_filled_trade(qty: int, avg_price: float, order_id: int = 42) -> MagicMock:
    trade = MagicMock()
    trade.orderStatus.status = "Filled"
    trade.orderStatus.filled = qty
    trade.orderStatus.avgFillPrice = avg_price
    trade.isDone.return_value = True
    trade.order.orderId = order_id
    return trade


def _make_rejected_trade(reason: str = "Cancelled", order_id: int = 43) -> MagicMock:
    trade = MagicMock()
    trade.orderStatus.status = reason
    trade.orderStatus.filled = 0
    trade.orderStatus.avgFillPrice = 0.0
    trade.isDone.return_value = True
    trade.order.orderId = order_id
    return trade


def test_submit_market_order_filled():
    trade = _make_filled_trade(qty=10, avg_price=105.25)
    ib = _make_mock_ib(placed_trade=trade)
    client = IBKRTWSClient(ib=ib)
    fill = client.submit_order(OrderIntent(ticker="AVDV", shares=10, order_type="MKT"))
    assert fill.status == "filled"
    assert fill.filled_quantity == 10
    assert fill.fill_price == pytest.approx(105.25)
    assert fill.broker_order_id == "42"
    assert ib.placeOrder.called


def test_submit_zero_shares_rejected_locally():
    ib = _make_mock_ib()
    client = IBKRTWSClient(ib=ib)
    fill = client.submit_order(OrderIntent(ticker="AVDV", shares=0))
    assert fill.status == "rejected"
    assert not ib.placeOrder.called


def test_submit_unknown_order_type_rejected():
    ib = _make_mock_ib()
    client = IBKRTWSClient(ib=ib)
    fill = client.submit_order(OrderIntent(ticker="AVDV", shares=5, order_type="WEIRD"))
    assert fill.status == "rejected"
    assert fill.context["reason"] == "unsupported_order_type"
    assert not ib.placeOrder.called


def test_submit_limit_order_without_price_rejected():
    ib = _make_mock_ib()
    client = IBKRTWSClient(ib=ib)
    fill = client.submit_order(OrderIntent(ticker="AVDV", shares=5, order_type="LMT"))
    assert fill.status == "rejected"
    assert fill.context["reason"] == "limit_order_missing_price"
    assert not ib.placeOrder.called


def test_submit_rejection_from_tws():
    trade = _make_rejected_trade(reason="Cancelled")
    ib = _make_mock_ib(placed_trade=trade)
    client = IBKRTWSClient(ib=ib)
    fill = client.submit_order(OrderIntent(ticker="XYZ", shares=10))
    assert fill.status == "rejected"
    assert fill.filled_quantity == 0


def test_submit_negative_shares_sends_sell_side():
    trade = _make_filled_trade(qty=5, avg_price=50.0)
    ib = _make_mock_ib(placed_trade=trade)
    client = IBKRTWSClient(ib=ib)
    fill = client.submit_order(OrderIntent(ticker="AVDV", shares=-5, order_type="MKT"))
    assert fill.status == "filled"
    # Verify the order passed to placeOrder had action="SELL"
    placed_order = ib.placeOrder.call_args[0][1]
    assert placed_order.action == "SELL"
    assert placed_order.totalQuantity == 5
