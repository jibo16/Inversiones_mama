"""Tests for inversiones_mama.data.prices.

Split into offline (using monkeypatched yfinance) and live tests marked
``@pytest.mark.live``. Run live with ``pytest -m live``; skip with ``-m 'not live'``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.data import prices as prices_mod
from inversiones_mama.data.prices import (
    fetch_prices_yfinance,
    load_prices,
    returns_from_prices,
)


# -------------------------- Offline: pure logic ------------------------------


def test_fetch_prices_empty_tickers_raises():
    with pytest.raises(ValueError):
        fetch_prices_yfinance([], datetime(2024, 1, 1), datetime(2024, 2, 1))


def test_returns_from_prices_simple(sample_prices_df):
    rets = returns_from_prices(sample_prices_df, method="simple")
    # Drops the first row
    assert len(rets) == len(sample_prices_df) - 1
    # First return matches manual calc
    expected = sample_prices_df.iloc[1] / sample_prices_df.iloc[0] - 1
    np.testing.assert_allclose(rets.iloc[0].values, expected.values, rtol=1e-12)


def test_returns_from_prices_log(sample_prices_df):
    rets = returns_from_prices(sample_prices_df, method="log")
    expected_log = np.log(sample_prices_df.iloc[1] / sample_prices_df.iloc[0])
    np.testing.assert_allclose(rets.iloc[0].values, expected_log.values, rtol=1e-10)


def test_returns_from_prices_unknown_method(sample_prices_df):
    with pytest.raises(ValueError):
        returns_from_prices(sample_prices_df, method="bogus")


# -------------------------- Offline: monkeypatched fetch ---------------------


def test_load_prices_uses_cache(tmp_path, monkeypatch, sample_prices_df):
    """Second call to load_prices must hit cache, not refetch."""
    call_count = {"n": 0}

    # Produce a yfinance-shaped DataFrame: DatetimeIndex, freq=None (like real data)
    synthetic = sample_prices_df[["A"]].rename(columns={"A": "SPY"}).copy()
    synthetic.index = pd.DatetimeIndex(synthetic.index.values, name="date")  # drop freq

    def fake_fetch(tickers, start, end, auto_adjust=True):
        call_count["n"] += 1
        return synthetic

    monkeypatch.setattr(prices_mod, "fetch_prices_yfinance", fake_fetch)
    monkeypatch.setattr(prices_mod, "CACHE_DIR", tmp_path)

    start = datetime(2024, 1, 1)
    end = datetime(2024, 3, 1)

    df1 = load_prices(["SPY"], start, end, cache_max_age_hours=24)
    df2 = load_prices(["SPY"], start, end, cache_max_age_hours=24)

    assert call_count["n"] == 1, "second call should have hit the cache"
    # check_freq=False because parquet round-trip does not preserve index freq metadata,
    # and real yfinance data has freq=None anyway (trading days skip weekends/holidays).
    pd.testing.assert_frame_equal(df1, df2, check_freq=False)


def test_load_prices_bypasses_cache_when_disabled(tmp_path, monkeypatch, sample_prices_df):
    call_count = {"n": 0}

    def fake_fetch(tickers, start, end, auto_adjust=True):
        call_count["n"] += 1
        return sample_prices_df[["A"]].rename(columns={"A": tickers[0]})

    monkeypatch.setattr(prices_mod, "fetch_prices_yfinance", fake_fetch)
    monkeypatch.setattr(prices_mod, "CACHE_DIR", tmp_path)

    start = datetime(2024, 1, 1)
    end = datetime(2024, 3, 1)

    load_prices(["SPY"], start, end, use_cache=False)
    load_prices(["SPY"], start, end, use_cache=False)
    assert call_count["n"] == 2


def test_load_prices_unknown_source_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(prices_mod, "CACHE_DIR", tmp_path)
    with pytest.raises(ValueError, match="Unknown source"):
        load_prices(["SPY"], datetime(2024, 1, 1), datetime(2024, 2, 1), source="bogus")


def test_load_prices_ibkr_source_raises_not_implemented(tmp_path, monkeypatch):
    """Until the IBKR account is wired, source='ibkr' must surface clearly."""
    monkeypatch.setattr(prices_mod, "CACHE_DIR", tmp_path)
    with pytest.raises(NotImplementedError):
        load_prices(["SPY"], datetime(2024, 1, 1), datetime(2024, 2, 1), source="ibkr", use_cache=False)


def test_load_prices_empty_fetch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(prices_mod, "CACHE_DIR", tmp_path)

    def fake_fetch(tickers, start, end, auto_adjust=True):
        return pd.DataFrame()

    monkeypatch.setattr(prices_mod, "fetch_prices_yfinance", fake_fetch)
    with pytest.raises(RuntimeError, match="Empty price DataFrame"):
        load_prices(["DOESNOTEXIST"], datetime(2024, 1, 1), datetime(2024, 2, 1), use_cache=False)


# -------------------------- Live: real yfinance ------------------------------


@pytest.fixture(scope="module")
def live_date_range():
    end = datetime.today() - timedelta(days=2)
    start = end - timedelta(days=120)  # ~4 trading months
    return start, end


@pytest.mark.live
def test_live_fetch_single_ticker(live_date_range):
    start, end = live_date_range
    df = fetch_prices_yfinance(["SPY"], start, end)
    assert not df.empty
    assert "SPY" in df.columns
    assert df.index.name == "date"
    assert (df["SPY"] > 0).all()
    # Trading days in ~120 calendar days is ~80+
    assert len(df) >= 50


@pytest.mark.live
def test_live_fetch_multi_ticker(live_date_range):
    start, end = live_date_range
    df = fetch_prices_yfinance(["SPY", "GLD", "TLT"], start, end)
    assert set(df.columns) <= {"SPY", "GLD", "TLT"}
    assert len(df) >= 50
    # All columns should have >0 prices
    assert (df > 0).all().all()


@pytest.mark.live
def test_live_load_prices_roundtrip(tmp_path, monkeypatch, live_date_range):
    monkeypatch.setattr(prices_mod, "CACHE_DIR", tmp_path)
    start, end = live_date_range
    df = load_prices(["SPY", "GLD"], start, end, use_cache=True)
    assert not df.empty
    # Second call should be cached (no new HTTP)
    df2 = load_prices(["SPY", "GLD"], start, end, use_cache=True)
    pd.testing.assert_frame_equal(df, df2)
