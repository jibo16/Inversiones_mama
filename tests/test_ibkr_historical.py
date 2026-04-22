"""Tests for inversiones_mama.data.ibkr_historical.

All tests are offline — the IBKR Client Portal Gateway is mocked via a
fake session that returns canned JSON. Live tests (if added later)
should be marked ``@pytest.mark.live`` and explicitly require a running,
authenticated Gateway.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.data.ibkr_historical import (
    IBKRHistoricalLoader,
    IBKRHistoricalRequest,
    _payload_to_frame,
)
from inversiones_mama.execution.ibkr import (
    IBKRClientPortalClient,
    IBKRClientPortalConfig,
    IBKRConnectionError,
    IBKRDataError,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _canned_history_payload(n_bars: int = 10, start_ms: int = 1697040000000, price_factor: int = 1) -> dict:
    """Build a realistic IBKR history response for testing."""
    bars = []
    t = start_ms
    px = 100.0
    for _ in range(n_bars):
        px_scaled = px * price_factor
        bars.append({
            "t": t,
            "o": px_scaled,
            "h": px_scaled * 1.01 * price_factor / max(price_factor, 1),
            "l": px_scaled * 0.99,
            "c": px_scaled,
            "v": 1_000_000,
        })
        t += 86_400_000  # +1 day
        px *= 1.001
    return {
        "symbol": "SPY",
        "text": "SPY ETF",
        "priceFactor": price_factor,
        "data": bars,
        "points": n_bars,
    }


def _make_loader_with_mocked_session(
    history_payload: dict | None = None,
    conid_payload: list | None = None,
    auth_payload: dict | None = None,
    raise_on_history: Exception | None = None,
) -> IBKRHistoricalLoader:
    """Build a loader whose underlying IBKRClientPortalClient has a mocked session."""
    config = IBKRClientPortalConfig(
        base_url="https://localhost:5000/v1/api",
        ws_url="wss://localhost:5000/v1/api/ws",
        verify_ssl=False,
        timeout_seconds=5.0,
    )
    client = IBKRClientPortalClient(config=config)

    # Mock HTTP session
    def fake_request(method, url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        # Route by URL path
        if "iserver/auth/status" in url:
            resp.json.return_value = auth_payload or {"authenticated": True, "connected": True}
            return resp
        if "iserver/secdef/search" in url:
            default = [{"conid": 756733, "symbol": "SPY"}]
            resp.json.return_value = conid_payload if conid_payload is not None else default
            return resp
        raise AssertionError(f"Unexpected URL: {url}")

    def fake_get(url, **kwargs):
        if "iserver/marketdata/history" in url:
            if raise_on_history is not None:
                raise raise_on_history
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.json.return_value = history_payload or _canned_history_payload()
            return resp
        raise AssertionError(f"Unexpected GET: {url}")

    client.session.request = MagicMock(side_effect=fake_request)
    client.session.get = MagicMock(side_effect=fake_get)

    loader = IBKRHistoricalLoader(client=client, inter_request_seconds=0.0)
    return loader


# --------------------------------------------------------------------------- #
# Payload parsing                                                             #
# --------------------------------------------------------------------------- #


def test_payload_to_frame_basic_shape():
    df = _payload_to_frame(_canned_history_payload(n_bars=5))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 5
    assert df.index.name == "date"
    assert df.index.dtype.kind == "M"
    # Index must be strictly increasing
    assert df.index.is_monotonic_increasing


def test_payload_to_frame_applies_price_factor():
    df = _payload_to_frame(_canned_history_payload(n_bars=3, price_factor=100))
    # The canned payload builds scaled prices (px * 100); after divide-by-factor the close should be ~100
    assert 99.0 < df["close"].iloc[0] < 101.0


def test_payload_to_frame_handles_empty_payload():
    assert _payload_to_frame({}).empty
    assert _payload_to_frame({"data": []}).empty
    assert _payload_to_frame({"data": None}).empty


def test_payload_to_frame_tolerates_bad_bar_entries():
    payload = {
        "data": [
            {"t": 1697040000000, "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000},
            None,  # bad entry
            {"t": None, "o": 100, "c": 100},  # missing timestamp
            {"t": 1697126400000, "o": 100, "h": 101, "l": 99, "c": 101, "v": 1000},
        ],
        "priceFactor": 1,
    }
    df = _payload_to_frame(payload)
    assert len(df) == 2  # only the two valid bars


def test_payload_to_frame_drops_rows_with_missing_close():
    payload = {
        "data": [
            {"t": 1697040000000, "o": 100, "h": 101, "l": 99, "c": None, "v": 1000},
            {"t": 1697126400000, "o": 100, "h": 101, "l": 99, "c": 101, "v": 1000},
        ],
        "priceFactor": 1,
    }
    df = _payload_to_frame(payload)
    assert len(df) == 1


def test_payload_to_frame_zero_price_factor_falls_back_to_one():
    payload = _canned_history_payload(n_bars=2)
    payload["priceFactor"] = 0
    df = _payload_to_frame(payload)
    assert not df.empty


# --------------------------------------------------------------------------- #
# Request validation                                                          #
# --------------------------------------------------------------------------- #


def test_request_validates_period():
    bad = IBKRHistoricalRequest(ticker="SPY", period="bogus", bar="1d")
    with pytest.raises(ValueError, match="unknown period"):
        bad.validate()


# --------------------------------------------------------------------------- #
# Authentication gate                                                         #
# --------------------------------------------------------------------------- #


def test_ensure_authenticated_passes_when_status_true():
    loader = _make_loader_with_mocked_session()
    loader.ensure_authenticated()  # should not raise


def test_ensure_authenticated_raises_when_status_false():
    loader = _make_loader_with_mocked_session(auth_payload={"authenticated": False})
    with pytest.raises(IBKRConnectionError, match="2FA"):
        loader.ensure_authenticated()


def test_ensure_authenticated_wraps_generic_errors():
    loader = _make_loader_with_mocked_session()
    loader.client.session.request = MagicMock(side_effect=RuntimeError("boom"))
    with pytest.raises(IBKRConnectionError):
        loader.ensure_authenticated()


# --------------------------------------------------------------------------- #
# Conid resolution                                                            #
# --------------------------------------------------------------------------- #


def test_resolve_conid_happy_path():
    loader = _make_loader_with_mocked_session()
    conid = loader.resolve_conid("SPY")
    assert conid == 756733


def test_resolve_conid_cached():
    loader = _make_loader_with_mocked_session()
    conid1 = loader.resolve_conid("SPY")
    conid2 = loader.resolve_conid("spy")  # case-insensitive, cache hit
    assert conid1 == conid2
    # search endpoint called only once
    search_calls = [c for c in loader.client.session.request.call_args_list
                    if "secdef/search" in c.args[1]]
    assert len(search_calls) == 1


def test_resolve_conid_raises_when_no_match():
    loader = _make_loader_with_mocked_session(conid_payload=[])
    # Either our own "could not resolve" message or the sibling's
    # "did not return a contract id" both satisfy the contract (IBKRDataError mentioning BOGUS)
    with pytest.raises(IBKRDataError, match="BOGUS"):
        loader.resolve_conid("BOGUS")


# --------------------------------------------------------------------------- #
# fetch_bars                                                                  #
# --------------------------------------------------------------------------- #


def test_fetch_bars_returns_dataframe():
    loader = _make_loader_with_mocked_session()
    df = loader.fetch_bars("SPY", period="1y", bar="1d")
    assert not df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing


def test_fetch_bars_bad_period_raises():
    loader = _make_loader_with_mocked_session()
    with pytest.raises(ValueError, match="unknown period"):
        loader.fetch_bars("SPY", period="bogus")


def test_fetch_bars_wraps_http_error():
    loader = _make_loader_with_mocked_session(raise_on_history=RuntimeError("HTTP 503"))
    with pytest.raises(IBKRConnectionError, match="history request failed"):
        loader.fetch_bars("SPY")


# --------------------------------------------------------------------------- #
# fetch_many                                                                  #
# --------------------------------------------------------------------------- #


def test_fetch_many_happy_path():
    loader = _make_loader_with_mocked_session()
    df = loader.fetch_many(["SPY", "QQQ", "TLT"], period="1y")
    assert not df.empty
    assert set(df.columns) == {"SPY", "QQQ", "TLT"}
    assert df.index.name == "date"


def test_fetch_many_empty_tickers_raises():
    loader = _make_loader_with_mocked_session()
    with pytest.raises(ValueError, match="non-empty"):
        loader.fetch_many([])


def test_fetch_many_bad_on_error_raises():
    loader = _make_loader_with_mocked_session()
    with pytest.raises(ValueError, match="on_error"):
        loader.fetch_many(["SPY"], on_error="panic")


def test_fetch_many_skip_mode_tolerates_per_ticker_failure():
    """One ticker fails conid lookup; loader keeps going."""
    config = IBKRClientPortalConfig(base_url="https://localhost:5000/v1/api",
                                     ws_url="wss://localhost:5000/v1/api/ws",
                                     verify_ssl=False, timeout_seconds=5.0)
    client = IBKRClientPortalClient(config=config)

    def fake_request(method, url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        if "auth/status" in url:
            resp.json.return_value = {"authenticated": True}
            return resp
        if "secdef/search" in url:
            sym = kwargs["params"]["symbol"]
            # BOGUS returns empty list -> IBKRDataError
            resp.json.return_value = [] if sym == "BOGUS" else [{"conid": 1, "symbol": sym}]
            return resp
        raise AssertionError(f"Unexpected URL: {url}")

    def fake_get(url, **kwargs):
        if "marketdata/history" in url:
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.json.return_value = _canned_history_payload(n_bars=3)
            return resp
        raise AssertionError(f"Unexpected GET: {url}")

    client.session.request = MagicMock(side_effect=fake_request)
    client.session.get = MagicMock(side_effect=fake_get)
    loader = IBKRHistoricalLoader(client=client, inter_request_seconds=0.0)

    df = loader.fetch_many(["SPY", "BOGUS", "QQQ"], period="1y", on_error="skip")
    assert set(df.columns) == {"SPY", "QQQ"}


def test_fetch_many_raise_mode_propagates_error():
    loader = _make_loader_with_mocked_session(conid_payload=[])
    with pytest.raises(IBKRDataError):
        loader.fetch_many(["BOGUS"], period="1y", on_error="raise")


def test_fetch_many_all_fail_raises_aggregate():
    """If every ticker fails, fetch_many raises with a summary."""
    loader = _make_loader_with_mocked_session(conid_payload=[])
    with pytest.raises(IBKRDataError, match="All IBKR fetches failed"):
        loader.fetch_many(["BAD1", "BAD2"], period="1y", on_error="skip")


def test_fetch_many_normalizes_tickers():
    loader = _make_loader_with_mocked_session()
    df = loader.fetch_many(["spy", "  QQQ  ", "TLT"], period="1y")
    assert set(df.columns) == {"SPY", "QQQ", "TLT"}


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def test_from_env_constructs_loader(monkeypatch):
    monkeypatch.setenv("IBKR_CP_BASE_URL", "https://localhost:5000/v1/api")
    loader = IBKRHistoricalLoader.from_env()
    assert isinstance(loader, IBKRHistoricalLoader)
    assert loader.client.config.base_url == "https://localhost:5000/v1/api"
