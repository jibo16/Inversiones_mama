"""Tests for inversiones_mama.data.volume.

Offline-only: the live-yfinance paths are best exercised via the
integration test in test_engine.py, which already pulls real ADV
during the full 5-year SP500 verdict. Here we confirm the module's
cache + parser plumbing via monkey-patching.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from inversiones_mama.data import volume as volume_mod
from inversiones_mama.data.volume import load_adv_shares, load_adv_dollars


def _make_panel(tickers: list[str], volumes: dict[str, list[int]], closes: dict[str, list[float]]) -> pd.DataFrame:
    """Build a MultiIndex-column DataFrame mimicking our internal panel shape."""
    idx = pd.date_range("2026-03-22", periods=len(next(iter(volumes.values()))), freq="B")
    vol_df = pd.DataFrame(volumes, index=idx)
    close_df = pd.DataFrame(closes, index=idx)
    vol_df.columns = pd.MultiIndex.from_product([["volume"], tickers])
    close_df.columns = pd.MultiIndex.from_product([["close"], tickers])
    merged = pd.concat([vol_df, close_df], axis=1)
    merged.index.name = "date"
    return merged


def test_load_adv_shares_empty_tickers_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(volume_mod, "CACHE_DIR", tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        load_adv_shares([], end=datetime(2026, 4, 22))


def test_load_adv_shares_uses_mean(monkeypatch, tmp_path):
    """Mock the panel fetch and verify load_adv_shares computes the mean correctly."""
    monkeypatch.setattr(volume_mod, "CACHE_DIR", tmp_path)

    panel = _make_panel(
        tickers=["AAPL", "MSFT"],
        volumes={"AAPL": [100_000_000, 90_000_000, 110_000_000, 100_000_000, 100_000_000],
                 "MSFT": [50_000_000, 52_000_000, 48_000_000, 50_000_000, 50_000_000]},
        closes={"AAPL": [150.0, 150.5, 149.5, 150.0, 150.0],
                "MSFT": [400.0, 401.0, 399.0, 400.0, 400.0]},
    )
    monkeypatch.setattr(volume_mod, "_fetch_volume_panel",
                        lambda tickers, end, window_days: panel)

    advs = load_adv_shares(["AAPL", "MSFT"], end=datetime(2026, 4, 22), use_cache=False)
    assert advs["AAPL"] == pytest.approx(100_000_000.0, rel=1e-6)
    assert advs["MSFT"] == pytest.approx(50_000_000.0, rel=1e-6)


def test_load_adv_dollars_multiplies_price(monkeypatch, tmp_path):
    monkeypatch.setattr(volume_mod, "CACHE_DIR", tmp_path)
    panel = _make_panel(
        tickers=["SPY"],
        volumes={"SPY": [80_000_000] * 5},
        closes={"SPY": [500.0] * 5},
    )
    monkeypatch.setattr(volume_mod, "_fetch_volume_panel",
                        lambda tickers, end, window_days: panel)

    dv = load_adv_dollars(["SPY"], end=datetime(2026, 4, 22), use_cache=False)
    # 80M shares * $500 = $40B daily dollar volume
    assert dv["SPY"] == pytest.approx(80_000_000.0 * 500.0, rel=1e-9)


def test_load_adv_cache_roundtrip(monkeypatch, tmp_path):
    """Second load with use_cache=True shouldn't re-call the fetcher."""
    monkeypatch.setattr(volume_mod, "CACHE_DIR", tmp_path)

    call_counter = {"n": 0}

    def counting_fetch(tickers, end, window_days):
        call_counter["n"] += 1
        return _make_panel(
            tickers=list(tickers),
            volumes={t: [1_000_000] * 5 for t in tickers},
            closes={t: [100.0] * 5 for t in tickers},
        )

    monkeypatch.setattr(volume_mod, "_fetch_volume_panel", counting_fetch)

    load_adv_shares(["AAPL"], end=datetime(2026, 4, 22), use_cache=True)
    load_adv_shares(["AAPL"], end=datetime(2026, 4, 22), use_cache=True)
    # Second call is satisfied from the parquet cache
    assert call_counter["n"] == 1
