"""Tests for inversiones_mama.execution.alpaca — mocked HTTP only."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from inversiones_mama.execution.alpaca import (
    AlpacaAPIError,
    AlpacaAuthError,
    AlpacaClient,
    AlpacaConfig,
)
from inversiones_mama.execution.ibkr import OrderIntent
from inversiones_mama.execution.paper_trader import ExecutionClient


# --------------------------------------------------------------------------- #
# Fake session                                                                #
# --------------------------------------------------------------------------- #


def _fake_response(status_code: int, json_payload: object, text: str = "") -> MagicMock:
    r = MagicMock(spec=requests.Response)
    r.status_code = status_code
    r.text = text or "<mocked>"
    if 200 <= status_code < 300:
        r.raise_for_status.return_value = None
    else:
        r.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status_code}")
    r.json.return_value = json_payload
    return r


def _client_with_routes(routes: dict[tuple[str, str], MagicMock]) -> AlpacaClient:
    """Build a client whose session routes (method, url) to canned responses.

    Unrecognized URLs raise AssertionError to catch test bugs.
    """
    config = AlpacaConfig(api_key="KEY", api_secret="SECRET",
                          poll_interval_seconds=0.0, poll_max_wait_seconds=0.2)
    client = AlpacaClient(config=config)

    def dispatch(method, url, **kwargs):
        for (m, u), resp in routes.items():
            if m.upper() == method.upper() and u in url:
                return resp
        raise AssertionError(f"Unrouted {method} {url}")

    client.session.get = MagicMock(side_effect=lambda url, **kw: dispatch("GET", url, **kw))
    client.session.post = MagicMock(side_effect=lambda url, **kw: dispatch("POST", url, **kw))
    return client


# --------------------------------------------------------------------------- #
# Config / construction                                                       #
# --------------------------------------------------------------------------- #


def test_config_from_env_reads_keys(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key-abc")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret-xyz")
    c = AlpacaConfig.from_env()
    assert c.api_key == "key-abc"
    assert c.api_secret == "secret-xyz"
    assert c.is_paper is True  # default paper


def test_config_from_env_raises_without_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    with pytest.raises(AlpacaAuthError, match="ALPACA_API_KEY_ID"):
        AlpacaConfig.from_env()


def test_config_respects_custom_base_url(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets/")  # trailing slash
    c = AlpacaConfig.from_env()
    assert c.base_url == "https://api.alpaca.markets"  # trimmed
    assert c.is_paper is False


def test_client_satisfies_execution_client_protocol():
    config = AlpacaConfig(api_key="k", api_secret="s")
    client = AlpacaClient(config=config)
    assert isinstance(client, ExecutionClient)


def test_auth_headers_set():
    config = AlpacaConfig(api_key="MY-KEY", api_secret="MY-SECRET")
    client = AlpacaClient(config=config)
    assert client.session.headers.get("APCA-API-KEY-ID") == "MY-KEY"
    assert client.session.headers.get("APCA-API-SECRET-KEY") == "MY-SECRET"


# --------------------------------------------------------------------------- #
# get_cash / get_positions                                                    #
# --------------------------------------------------------------------------- #


def test_get_cash():
    client = _client_with_routes({
        ("GET", "/v2/account"): _fake_response(200, {"cash": "5000.00", "buying_power": "10000"}),
    })
    assert client.get_cash() == 5000.0


def test_get_cash_missing_field_raises():
    client = _client_with_routes({
        ("GET", "/v2/account"): _fake_response(200, {"buying_power": "10000"}),
    })
    with pytest.raises(AlpacaAPIError, match="'cash' field"):
        client.get_cash()


def test_get_positions_empty():
    client = _client_with_routes({
        ("GET", "/v2/positions"): _fake_response(200, []),
    })
    assert client.get_positions() == {}


def test_get_positions_parsed():
    client = _client_with_routes({
        ("GET", "/v2/positions"): _fake_response(200, [
            {"symbol": "AAPL", "qty": "10"},
            {"symbol": "msft", "qty": "5"},  # lower-case should be normalized
            {"symbol": "NVDA", "qty": "-3"},  # short position
        ]),
    })
    positions = client.get_positions()
    assert positions == {"AAPL": 10, "MSFT": 5, "NVDA": -3}


def test_get_positions_skips_malformed():
    client = _client_with_routes({
        ("GET", "/v2/positions"): _fake_response(200, [
            {"symbol": "AAPL", "qty": "10"},
            {"symbol": "", "qty": "5"},           # empty symbol
            {"symbol": "BOGUS", "qty": None},     # no quantity
        ]),
    })
    positions = client.get_positions()
    assert positions == {"AAPL": 10}


# --------------------------------------------------------------------------- #
# get_latest_price                                                            #
# --------------------------------------------------------------------------- #


def test_get_latest_price_midpoint():
    client = _client_with_routes({
        ("GET", "/v2/stocks/AAPL/quotes/latest"):
            _fake_response(200, {"quote": {"ap": 100.0, "bp": 99.0}}),
    })
    assert client.get_latest_price("AAPL") == 99.5


def test_get_latest_price_ask_only_fallback():
    client = _client_with_routes({
        ("GET", "/v2/stocks/AAPL/quotes/latest"):
            _fake_response(200, {"quote": {"ap": 100.0, "bp": 0.0}}),
    })
    assert client.get_latest_price("AAPL") == 100.0


def test_get_latest_price_empty_quote_returns_none():
    client = _client_with_routes({
        ("GET", "/v2/stocks/AAPL/quotes/latest"): _fake_response(200, {"quote": {}}),
    })
    assert client.get_latest_price("AAPL") is None


def test_get_latest_price_http_error_returns_none():
    client = _client_with_routes({
        ("GET", "/v2/stocks/BOGUS/quotes/latest"): _fake_response(404, {}),
    })
    assert client.get_latest_price("BOGUS") is None


def test_get_latest_price_empty_ticker_returns_none():
    client = _client_with_routes({})
    assert client.get_latest_price("") is None


# --------------------------------------------------------------------------- #
# submit_order                                                                #
# --------------------------------------------------------------------------- #


def test_submit_order_zero_shares_rejected():
    client = _client_with_routes({})
    fill = client.submit_order(OrderIntent(ticker="AAPL", shares=0))
    assert fill.status == "rejected"
    assert fill.context["reason"] == "zero_shares"


def test_submit_order_filled_immediately():
    client = _client_with_routes({
        ("POST", "/v2/orders"): _fake_response(201, {"id": "order-123", "status": "accepted"}),
        ("GET", "/v2/orders/order-123"): _fake_response(200, {
            "id": "order-123",
            "status": "filled",
            "filled_avg_price": "150.25",
            "filled_qty": "10",
            "filled_at": "2026-04-22T14:30:00Z",
        }),
    })
    fill = client.submit_order(OrderIntent(ticker="AAPL", shares=10))
    assert fill.status == "filled"
    assert fill.fill_price == 150.25
    assert fill.filled_quantity == 10
    assert fill.broker_order_id == "order-123"


def test_submit_order_short_sell():
    """Negative shares → side=sell."""
    captured: dict = {}

    def capture_post(url, json=None, **kw):
        captured["url"] = url
        captured["body"] = json
        return _fake_response(201, {"id": "order-456", "status": "accepted"})

    config = AlpacaConfig(api_key="k", api_secret="s",
                          poll_interval_seconds=0.0, poll_max_wait_seconds=0.0)
    client = AlpacaClient(config=config)
    client.session.post = MagicMock(side_effect=capture_post)
    client.session.get = MagicMock(side_effect=lambda url, **kw: _fake_response(200, {
        "id": "order-456", "status": "submitted",
    }))

    fill = client.submit_order(OrderIntent(ticker="TSLA", shares=-5))
    assert captured["body"]["side"] == "sell"
    assert captured["body"]["qty"] == "5"
    assert captured["body"]["symbol"] == "TSLA"
    # Poll timed out immediately → "submitted" status
    assert fill.status == "submitted"


def test_submit_order_rejected_by_alpaca():
    client = _client_with_routes({
        ("POST", "/v2/orders"): _fake_response(201, {"id": "ord", "status": "accepted"}),
        ("GET", "/v2/orders/ord"): _fake_response(200, {
            "id": "ord", "status": "rejected",
        }),
    })
    fill = client.submit_order(OrderIntent(ticker="AAPL", shares=10))
    assert fill.status == "rejected"
    assert fill.filled_quantity == 0


def test_submit_order_api_error_returns_rejected_fill():
    client = _client_with_routes({
        ("POST", "/v2/orders"): _fake_response(422, {"message": "bad qty"}),
    })
    fill = client.submit_order(OrderIntent(ticker="AAPL", shares=10))
    assert fill.status == "rejected"
    assert fill.context["reason"] == "api_error"


def test_submit_order_no_order_id_in_response():
    client = _client_with_routes({
        ("POST", "/v2/orders"): _fake_response(201, {"status": "accepted"}),  # missing id
    })
    fill = client.submit_order(OrderIntent(ticker="AAPL", shares=10))
    assert fill.status == "rejected"
    assert fill.context["reason"] == "no_order_id"


# --------------------------------------------------------------------------- #
# Auth errors                                                                 #
# --------------------------------------------------------------------------- #


def test_auth_error_surfaces_on_401():
    client = _client_with_routes({
        ("GET", "/v2/account"): _fake_response(401, {"message": "forbidden"}),
    })
    with pytest.raises(AlpacaAuthError):
        client.get_cash()


def test_auth_error_surfaces_on_403():
    client = _client_with_routes({
        ("GET", "/v2/account"): _fake_response(403, {}),
    })
    with pytest.raises(AlpacaAuthError):
        client.get_cash()


# --------------------------------------------------------------------------- #
# check_auth                                                                  #
# --------------------------------------------------------------------------- #


def test_check_auth_returns_account_payload():
    payload = {"cash": "5000", "portfolio_value": "5100", "status": "ACTIVE"}
    client = _client_with_routes({
        ("GET", "/v2/account"): _fake_response(200, payload),
    })
    out = client.check_auth()
    assert out == payload
