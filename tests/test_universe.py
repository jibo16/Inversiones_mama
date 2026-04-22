"""Tests for inversiones_mama.data.universe.

Focused on ``StaticUniverse`` (the only concrete implementation today)
and the ``PointInTimeUniverse`` Protocol. Vendor-backed implementations
(Sharadar, Finnhub premium constituents) are deferred to v2 and their
stubs here only verify the NotImplementedError contract.
"""

from __future__ import annotations

import pandas as pd
import pytest

from inversiones_mama.data.universe import (
    PointInTimeUniverse,
    SharadarPointInTimeUniverse,
    StaticUniverse,
)


# --------------------------------------------------------------------------- #
# StaticUniverse                                                              #
# --------------------------------------------------------------------------- #


def test_static_universe_dedup_and_sort():
    u = StaticUniverse(["TSLA", "aapl", "MSFT", "aapl", "  nvda  "])
    assert u.tickers == ("AAPL", "MSFT", "NVDA", "TSLA")


def test_static_universe_rejects_empty():
    with pytest.raises(ValueError, match="at least one ticker"):
        StaticUniverse([])


def test_static_universe_members_always_same():
    u = StaticUniverse(["SPY", "QQQ"])
    m1 = u.members_on(pd.Timestamp("2020-01-01"))
    m2 = u.members_on(pd.Timestamp("2025-12-31"))
    assert m1 == m2 == ["QQQ", "SPY"]


def test_static_universe_active_range_known():
    u = StaticUniverse(["SPY"])
    r = u.active_range("SPY")
    assert r is not None
    assert r[0] <= r[1]


def test_static_universe_active_range_unknown():
    u = StaticUniverse(["SPY"])
    assert u.active_range("NVDA") is None


def test_static_universe_contains():
    u = StaticUniverse(["SPY", "QQQ"])
    assert "SPY" in u
    assert "spy" in u  # case-insensitive
    assert 42 not in u  # non-string
    assert "QQQQ" not in u


def test_static_universe_all_tickers_equals_members():
    u = StaticUniverse(["A", "B", "C"])
    assert set(u.all_tickers()) == set(u.members_on(pd.Timestamp.today()))


def test_static_universe_len_and_repr():
    u = StaticUniverse(["A", "B", "C", "D", "E", "F", "G"])
    assert len(u) == 7
    r = repr(u)
    assert "StaticUniverse" in r
    assert "+2" in r  # preview shows first 5 then "+2"


# --------------------------------------------------------------------------- #
# Protocol adherence                                                          #
# --------------------------------------------------------------------------- #


def test_static_universe_satisfies_protocol():
    u = StaticUniverse(["AAPL"])
    assert isinstance(u, PointInTimeUniverse)


def test_sharadar_stub_satisfies_protocol():
    u = SharadarPointInTimeUniverse(api_key="dummy")
    assert isinstance(u, PointInTimeUniverse)


# --------------------------------------------------------------------------- #
# Stubs raise NotImplementedError                                             #
# --------------------------------------------------------------------------- #


def test_sharadar_stub_members_on_raises():
    u = SharadarPointInTimeUniverse()
    with pytest.raises(NotImplementedError, match="Phase 1"):
        u.members_on(pd.Timestamp("2020-01-01"))


def test_sharadar_stub_active_range_raises():
    u = SharadarPointInTimeUniverse()
    with pytest.raises(NotImplementedError):
        u.active_range("AAPL")


def test_sharadar_stub_all_tickers_raises():
    u = SharadarPointInTimeUniverse()
    with pytest.raises(NotImplementedError):
        u.all_tickers()
