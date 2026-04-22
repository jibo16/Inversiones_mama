"""Tests for inversiones_mama.data.macro.

Offline tests monkey-patch the HTTP layer. One live test (marked
``@pytest.mark.live``) actually hits FRED to verify the integration.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from inversiones_mama.data import macro as macro_mod
from inversiones_mama.data.macro import (
    FRED_MACRO_SERIES,
    fetch_fred_series,
    load_macro_panel,
)


_SAMPLE_FRED_CSV = """observation_date,UNRATE
2024-01-01,3.7
2024-02-01,3.9
2024-03-01,3.8
2024-04-01,.
2024-05-01,4.0
"""


def _fake_resp(text: str, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.text = text
    r.raise_for_status.return_value = None
    return r


def test_fetch_fred_parses_csv(monkeypatch):
    monkeypatch.setattr(macro_mod.requests, "get",
                         lambda *a, **kw: _fake_resp(_SAMPLE_FRED_CSV))
    s = fetch_fred_series("UNRATE")
    assert s.name == "UNRATE"
    # "." placeholder is dropped by dropna()
    assert len(s) == 4
    assert s.iloc[0] == pytest.approx(3.7)
    assert s.iloc[-1] == pytest.approx(4.0)


def test_fetch_fred_http_failure_raises(monkeypatch):
    def _boom(*a, **kw):
        raise ConnectionError("no network")
    monkeypatch.setattr(macro_mod.requests, "get", _boom)
    with pytest.raises(RuntimeError, match="FRED fetch failed"):
        fetch_fred_series("UNRATE")


def test_fetch_fred_date_column_aliases(monkeypatch):
    """FRED historically used 'DATE'; newer dumps use 'observation_date'."""
    alt = "DATE,FAKESERIES\n2024-01-01,1.0\n2024-02-01,2.0\n"
    monkeypatch.setattr(macro_mod.requests, "get", lambda *a, **kw: _fake_resp(alt))
    s = fetch_fred_series("FAKESERIES")
    assert len(s) == 2
    assert s.iloc[-1] == 2.0


def test_load_macro_panel_quarterly_resample(monkeypatch, tmp_path):
    """load_macro_panel should resample monthly/daily series to quarter-end."""
    monkeypatch.setattr(macro_mod, "CACHE_DIR", tmp_path)

    def fake_fetch(series_id, start=None, end=None, timeout=30.0):
        # Monthly series spanning two quarters
        idx = pd.to_datetime(
            ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"]
        )
        if series_id == "UNRATE":
            return pd.Series([3.7, 3.9, 3.8, 4.0, 4.1, 4.0], index=idx, name="UNRATE")
        if series_id == "CPIAUCSL":
            return pd.Series([300, 301, 302, 303, 304, 305], index=idx, name="CPIAUCSL")
        raise AssertionError(f"unexpected series {series_id}")

    monkeypatch.setattr(macro_mod, "fetch_fred_series", fake_fetch)

    panel = load_macro_panel(
        series_map={"unemployment": "UNRATE", "cpi": "CPIAUCSL"},
        freq="QE",
        use_cache=False,
    )
    assert list(panel.columns) == ["unemployment", "cpi"]
    # We have two quarter-ends (March, June)
    assert len(panel) == 2


def test_load_macro_panel_cache_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(macro_mod, "CACHE_DIR", tmp_path)

    call_count = {"n": 0}

    def counting_fetch(series_id, start=None, end=None, timeout=30.0):
        call_count["n"] += 1
        idx = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
        return pd.Series([1.0, 2.0, 3.0], index=idx, name=series_id)

    monkeypatch.setattr(macro_mod, "fetch_fred_series", counting_fetch)

    load_macro_panel(series_map={"a": "A"}, freq="ME", use_cache=True)
    load_macro_panel(series_map={"a": "A"}, freq="ME", use_cache=True)
    # Second call is cache hit; fetch is called only once (the first run)
    assert call_count["n"] == 1


def test_load_macro_panel_all_fetches_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(macro_mod, "CACHE_DIR", tmp_path)

    def fail_fetch(series_id, start=None, end=None, timeout=30.0):
        raise RuntimeError(f"network down: {series_id}")

    monkeypatch.setattr(macro_mod, "fetch_fred_series", fail_fetch)

    with pytest.raises(RuntimeError, match="all FRED fetches failed"):
        load_macro_panel(series_map={"a": "A"}, use_cache=False)


def test_canonical_series_keys_are_stable():
    expected = {
        "real_gdp", "unemployment", "cpi",
        "treasury_3m", "treasury_10y", "baa_10y_spread",
        "vix", "home_price_index", "sp500", "dxy",
    }
    assert set(FRED_MACRO_SERIES) == expected


@pytest.mark.live
def test_live_fetch_unrate():
    s = fetch_fred_series("UNRATE", start=datetime(2023, 1, 1), end=datetime(2024, 1, 1))
    assert not s.empty
    # Unemployment rate should plausibly live in [2, 15] %
    assert s.min() > 2.0
    assert s.max() < 15.0


@pytest.mark.live
def test_live_load_macro_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(macro_mod, "CACHE_DIR", tmp_path)
    panel = load_macro_panel(
        series_map={"unemployment": "UNRATE", "cpi": "CPIAUCSL"},
        start=datetime(2020, 1, 1), end=datetime(2024, 1, 1),
        freq="QE", use_cache=True,
    )
    assert not panel.empty
    assert "unemployment" in panel.columns
