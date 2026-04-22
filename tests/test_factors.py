"""Tests for inversiones_mama.data.factors.

Offline tests exercise the parser against a synthetic Ken-French-format CSV.
Live tests (``-m live``) download the real zips from Dartmouth.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.data import factors as factors_mod
from inversiones_mama.data.factors import (
    fetch_ff5f_daily,
    fetch_momentum_daily,
    load_factor_returns,
    parse_ken_french_csv,
)


# --- Synthetic Ken French fixtures -----------------------------------------


SYNTHETIC_FF5_CSV = """This file was created by CMPT_ME_BEME_OP_INV_RETS using the 202504 CRSP database. Test fixture only.
The 1-month TBill return is from Ibbotson Associates.

,Mkt-RF,SMB,HML,RMW,CMA,RF
20240102,  0.50, -0.10,  0.20,  0.30, -0.05,  0.02
20240103, -0.30,  0.05, -0.10, -0.20,  0.10,  0.02
20240104,  0.80,  0.15,  0.25,  0.10, -0.15,  0.02
20240105,  0.10,  0.00,  0.05,  0.00,  0.00,  0.02

Annual Factors: January-December

,Mkt-RF,SMB,HML,RMW,CMA,RF
2024, 24.00, 3.00, 5.00, 4.00, -2.00, 0.50
"""


SYNTHETIC_MOM_CSV = """Momentum factor — test fixture. The 1-month TBill return is from Ibbotson.

,Mom
20240102,  0.40
20240103, -0.15
20240104,  0.60
20240105,  0.05

Annual Factors: January-December

,Mom
2024, 12.00
"""


def _zip_csv(name: str, text: str) -> bytes:
    """Helper: wrap a CSV string in a zip with a single entry."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, text)
    return buf.getvalue()


# --- Parser tests (offline, no network) ------------------------------------


def test_parser_ff5_shape():
    df = parse_ken_french_csv(SYNTHETIC_FF5_CSV)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(df) == 4
    assert df.index.name == "date"
    assert df.index.dtype.kind == "M"  # datetime


def test_parser_percent_to_decimal_conversion():
    df = parse_ken_french_csv(SYNTHETIC_FF5_CSV)
    # Published 0.50% -> 0.005 decimal
    np.testing.assert_allclose(df.iloc[0]["Mkt-RF"], 0.005, atol=1e-12)
    np.testing.assert_allclose(df.iloc[2]["HML"], 0.0025, atol=1e-12)
    # RF is the same small value across rows
    assert (df["RF"] == 0.0002).all()


def test_parser_stops_at_blank_line():
    """Annual section must not leak into the daily DataFrame."""
    df = parse_ken_french_csv(SYNTHETIC_FF5_CSV)
    # If we had leaked the annual row, 2024 at index 0 would be present as a date
    assert pd.Timestamp("2024-01-01") not in df.index
    assert df.index.max() == pd.Timestamp("2024-01-05")


def test_parser_mom_column_kept_raw():
    df = parse_ken_french_csv(SYNTHETIC_MOM_CSV)
    # Parser preserves original column name; fetch_momentum_daily normalizes it
    assert list(df.columns) == ["Mom"]
    assert len(df) == 4


def test_parser_rejects_csv_without_header():
    with pytest.raises(RuntimeError, match="header"):
        parse_ken_french_csv("no header here\njust text\n")


def test_parser_rejects_csv_without_data_rows():
    empty = """Header description

,Mkt-RF,SMB

Annual Factors:
,Mkt-RF,SMB
2024,1.0,2.0
"""
    with pytest.raises(RuntimeError, match="No data rows"):
        parse_ken_french_csv(empty)


# --- Fetch functions with mocked downloads ---------------------------------


def test_fetch_ff5f_with_mocked_download(monkeypatch):
    monkeypatch.setattr(
        factors_mod, "_download_zip", lambda url, timeout=30.0: _zip_csv("FF5.csv", SYNTHETIC_FF5_CSV)
    )
    df = fetch_ff5f_daily()
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(df) == 4


def test_fetch_mom_normalizes_column_name(monkeypatch):
    monkeypatch.setattr(
        factors_mod, "_download_zip", lambda url, timeout=30.0: _zip_csv("MOM.csv", SYNTHETIC_MOM_CSV)
    )
    df = fetch_momentum_daily()
    assert list(df.columns) == ["MOM"], "column should be normalized from 'Mom' to 'MOM'"
    assert len(df) == 4


def test_load_factor_returns_merged(tmp_path, monkeypatch):
    """End-to-end: both downloads mocked; result merged, cached, filtered."""
    def fake_download(url, timeout=30.0):
        if "5_Factors" in url:
            return _zip_csv("FF5.csv", SYNTHETIC_FF5_CSV)
        if "Momentum" in url:
            return _zip_csv("MOM.csv", SYNTHETIC_MOM_CSV)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(factors_mod, "_download_zip", fake_download)
    monkeypatch.setattr(factors_mod, "CACHE_DIR", tmp_path)

    df = load_factor_returns(use_cache=False)
    assert set(df.columns) == {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "MOM"}
    assert len(df) == 4  # inner join on identical date ranges


def test_load_factor_returns_date_filter(tmp_path, monkeypatch):
    def fake_download(url, timeout=30.0):
        if "5_Factors" in url:
            return _zip_csv("FF5.csv", SYNTHETIC_FF5_CSV)
        return _zip_csv("MOM.csv", SYNTHETIC_MOM_CSV)

    monkeypatch.setattr(factors_mod, "_download_zip", fake_download)
    monkeypatch.setattr(factors_mod, "CACHE_DIR", tmp_path)

    df = load_factor_returns(
        start=datetime(2024, 1, 3),
        end=datetime(2024, 1, 4),
        use_cache=False,
    )
    assert len(df) == 2
    assert df.index.min() == pd.Timestamp("2024-01-03")
    assert df.index.max() == pd.Timestamp("2024-01-04")


def test_load_factor_returns_caches(tmp_path, monkeypatch):
    """Second call hits parquet cache instead of re-downloading."""
    call_count = {"n": 0}

    def fake_download(url, timeout=30.0):
        call_count["n"] += 1
        if "5_Factors" in url:
            return _zip_csv("FF5.csv", SYNTHETIC_FF5_CSV)
        return _zip_csv("MOM.csv", SYNTHETIC_MOM_CSV)

    monkeypatch.setattr(factors_mod, "_download_zip", fake_download)
    monkeypatch.setattr(factors_mod, "CACHE_DIR", tmp_path)

    df1 = load_factor_returns(use_cache=True, cache_max_age_hours=24 * 7)
    df2 = load_factor_returns(use_cache=True, cache_max_age_hours=24 * 7)
    assert call_count["n"] == 2, "first call fetches FF5 + MOM (2 downloads)"
    pd.testing.assert_frame_equal(df1, df2, check_freq=False)


# --- Live tests (real Ken French download) ---------------------------------


@pytest.mark.live
def test_live_fetch_ff5f():
    df = fetch_ff5f_daily()
    assert not df.empty
    assert set(df.columns) >= {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"}
    # Factor returns should all be within plausible daily bounds
    assert df.abs().max().max() < 0.50
    # History should go back at least to the 1960s
    assert df.index.min() < pd.Timestamp("1970-01-01")


@pytest.mark.live
def test_live_fetch_momentum():
    df = fetch_momentum_daily()
    assert not df.empty
    assert "MOM" in df.columns
    assert df["MOM"].abs().max() < 0.30


@pytest.mark.live
def test_live_load_factor_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(factors_mod, "CACHE_DIR", tmp_path)
    df = load_factor_returns(
        start=datetime(2020, 1, 1),
        end=datetime(2024, 12, 31),
        use_cache=True,
    )
    assert not df.empty
    assert set(df.columns) >= {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"}
    # ~5 years of daily data ≈ 1250 rows
    assert 1000 < len(df) < 1400
