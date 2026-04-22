"""Tests for inversiones_mama.data.fundamentals.

Only the Protocol and vendor stubs exist today. These tests verify the
contract: stubs raise ``NotImplementedError`` with a useful message so
callers cannot silently get empty data.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from inversiones_mama.data.fundamentals import (
    CANONICAL_FEATURES,
    AlphaVantageFundamentalsLoader,
    FinnhubFundamentalsLoader,
    FundamentalsLoader,
)


def test_canonical_features_nontrivial():
    assert len(CANONICAL_FEATURES) >= 8
    # Must include the three standard pillars
    expected_subset = {"pe_trailing", "roe", "revenue_growth_yoy", "market_cap_usd"}
    assert expected_subset.issubset(set(CANONICAL_FEATURES))


def test_finnhub_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key-123")
    loader = FinnhubFundamentalsLoader()
    assert loader.api_key == "test-key-123"


def test_finnhub_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "env-key")
    loader = FinnhubFundamentalsLoader(api_key="explicit-key")
    assert loader.api_key == "explicit-key"


def test_alphavantage_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "av-test")
    loader = AlphaVantageFundamentalsLoader()
    assert loader.api_key == "av-test"


def test_finnhub_stub_load_as_of_raises():
    loader = FinnhubFundamentalsLoader(api_key="x")
    with pytest.raises(NotImplementedError, match="Phase 1"):
        loader.load_as_of("AAPL", pd.Timestamp("2020-01-01"))


def test_finnhub_stub_load_panel_raises():
    loader = FinnhubFundamentalsLoader(api_key="x")
    with pytest.raises(NotImplementedError):
        loader.load_panel(["AAPL"], datetime(2020, 1, 1), datetime(2020, 12, 31))


def test_alphavantage_stub_load_as_of_raises():
    loader = AlphaVantageFundamentalsLoader(api_key="x")
    with pytest.raises(NotImplementedError, match="Phase 1"):
        loader.load_as_of("AAPL", pd.Timestamp("2020-01-01"))


def test_alphavantage_stub_load_panel_raises():
    loader = AlphaVantageFundamentalsLoader(api_key="x")
    with pytest.raises(NotImplementedError):
        loader.load_panel(["AAPL"], datetime(2020, 1, 1), datetime(2020, 12, 31))


def test_finnhub_satisfies_protocol():
    loader = FinnhubFundamentalsLoader(api_key="x")
    assert isinstance(loader, FundamentalsLoader)


def test_alphavantage_satisfies_protocol():
    loader = AlphaVantageFundamentalsLoader(api_key="x")
    assert isinstance(loader, FundamentalsLoader)
