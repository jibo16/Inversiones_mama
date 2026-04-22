"""Tests for inversiones_mama.data.liquid_universe."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from inversiones_mama.data.liquid_universe import (
    LIQUID_ETFS,
    NASDAQ100_CORE,
    SP100_CORE,
    all_curated_tickers,
    build_liquid_universe,
    top_k_by_volume,
)
from inversiones_mama.data.universe import StaticUniverse


# --------------------------------------------------------------------------- #
# Curated list sanity                                                         #
# --------------------------------------------------------------------------- #


def test_sp100_size_reasonable():
    assert 90 <= len(SP100_CORE) <= 110


def test_nasdaq100_size_reasonable():
    assert 80 <= len(NASDAQ100_CORE) <= 110


def test_liquid_etfs_includes_core_broad_exposure():
    etfs = set(LIQUID_ETFS)
    assert "SPY" in etfs
    assert "QQQ" in etfs
    assert "TLT" in etfs
    assert "GLD" in etfs
    assert "AVUV" in etfs  # v1a universe member


def test_curated_tickers_all_uppercase_and_unique():
    for label, lst in [("sp100", SP100_CORE), ("nasdaq100", NASDAQ100_CORE), ("etfs", LIQUID_ETFS)]:
        assert len(lst) == len(set(lst)), f"{label} has duplicates"
        for t in lst:
            assert t == t.upper(), f"{label} has non-upper ticker: {t}"


def test_all_curated_tickers_deduped():
    flat = all_curated_tickers()
    assert len(flat) == len(set(flat))
    # The union is larger than each individual list
    assert len(flat) >= max(len(SP100_CORE), len(NASDAQ100_CORE), len(LIQUID_ETFS))


# --------------------------------------------------------------------------- #
# build_liquid_universe                                                       #
# --------------------------------------------------------------------------- #


def test_build_universe_sp100():
    u = build_liquid_universe("sp100")
    assert isinstance(u, StaticUniverse)
    assert "AAPL" in u
    assert "SPY" not in u  # SPY lives in etfs, not sp100


def test_build_universe_etfs():
    u = build_liquid_universe("etfs")
    assert "SPY" in u
    assert "AAPL" not in u  # individual stocks not in etfs set


def test_build_universe_all_is_union():
    sp = set(build_liquid_universe("sp100").all_tickers())
    nd = set(build_liquid_universe("nasdaq100").all_tickers())
    et = set(build_liquid_universe("etfs").all_tickers())
    all_u = set(build_liquid_universe("all").all_tickers())
    assert all_u == sp | nd | et


def test_build_universe_respects_limit():
    u = build_liquid_universe("sp100", limit=20)
    assert len(u) == 20


def test_build_universe_bad_limit_raises():
    with pytest.raises(ValueError, match="limit"):
        build_liquid_universe("sp100", limit=0)


def test_build_universe_bad_kind_raises():
    with pytest.raises(ValueError, match="Unknown kind"):
        build_liquid_universe("made_up_kind")


def test_build_universe_case_insensitive_kind():
    assert len(build_liquid_universe("SP100")) == len(build_liquid_universe("sp100"))


# --------------------------------------------------------------------------- #
# top_k_by_volume                                                             #
# --------------------------------------------------------------------------- #


def test_top_k_by_volume_rejects_bad_k():
    with pytest.raises(ValueError, match="k must be positive"):
        top_k_by_volume(["SPY"], k=0)


def test_top_k_by_volume_empty_tickers():
    assert top_k_by_volume([], k=5) == []


@pytest.mark.live
def test_top_k_by_volume_live():
    """Pull real prices for a small set and verify we get a k-subset back."""
    tickers = ["SPY", "QQQ", "TLT", "GLD", "AAPL"]
    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=30)
    top3 = top_k_by_volume(tickers, k=3, start=start, end=end)
    assert len(top3) == 3
    assert set(top3).issubset(set(tickers))
