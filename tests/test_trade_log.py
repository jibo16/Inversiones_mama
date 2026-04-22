"""Tests for inversiones_mama.execution.trade_log."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from inversiones_mama.execution.trade_log import (
    FillRecord,
    SignalRecord,
    TradeLog,
    TradeLogEntry,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _sig(ticker: str, t: datetime, px: float, size: int) -> SignalRecord:
    return SignalRecord(
        ticker=ticker, signal_time=t, expected_price=px, expected_size=size,
    )


def _fill(t_order: datetime, t_fill: datetime | None, px: float | None, qty: int, status: str = "filled") -> FillRecord:
    return FillRecord(
        order_time=t_order, fill_time=t_fill, fill_price=px, filled_quantity=qty, status=status,
    )


# --------------------------------------------------------------------------- #
# Derived metrics                                                             #
# --------------------------------------------------------------------------- #


def test_slippage_positive_on_buy_above_expected():
    t = _now_utc()
    e = TradeLogEntry(signal=_sig("AAPL", t, 100.0, 10), fill=_fill(t, t, 100.5, 10))
    assert e.slippage == pytest.approx(0.5)
    assert e.slippage_bps == pytest.approx(50.0)


def test_slippage_none_until_filled():
    t = _now_utc()
    e = TradeLogEntry(signal=_sig("AAPL", t, 100.0, 10), fill=_fill(t, None, None, 0, "submitted"))
    assert e.slippage is None
    assert e.slippage_bps is None


def test_execution_delay_in_ms():
    t0 = _now_utc()
    t1 = t0 + timedelta(milliseconds=750)
    e = TradeLogEntry(signal=_sig("X", t0, 50.0, 1), fill=_fill(t0, t1, 50.0, 1))
    assert 700 <= e.execution_delay_ms <= 800


def test_fill_ratio_partial():
    t = _now_utc()
    e = TradeLogEntry(signal=_sig("X", t, 10.0, 100), fill=_fill(t, t, 10.0, 40, "partial"))
    assert e.fill_ratio == pytest.approx(0.4)


def test_fill_ratio_clipped_at_1():
    t = _now_utc()
    # Over-fill (shouldn't happen but defensively clip)
    e = TradeLogEntry(signal=_sig("X", t, 10.0, 100), fill=_fill(t, t, 10.0, 120))
    assert e.fill_ratio == 1.0


def test_fill_ratio_rejected_zero():
    t = _now_utc()
    e = TradeLogEntry(signal=_sig("X", t, 10.0, 100), fill=_fill(t, None, None, 0, "rejected"))
    assert e.fill_ratio == 0.0


def test_slippage_bps_zero_expected_price_is_none():
    t = _now_utc()
    # Pathological: expected_price = 0
    e = TradeLogEntry(signal=_sig("X", t, 0.0, 1), fill=_fill(t, t, 1.0, 1))
    assert e.slippage_bps is None


# --------------------------------------------------------------------------- #
# TradeLog behavior                                                           #
# --------------------------------------------------------------------------- #


def test_tradelog_record_and_iterate():
    t = _now_utc()
    log = TradeLog()
    log.record(_sig("A", t, 10.0, 1), _fill(t, t, 10.1, 1))
    log.record(_sig("B", t, 20.0, 2), _fill(t, t, 19.9, 2))
    assert len(log) == 2
    tickers = [e.signal.ticker for e in log]
    assert tickers == ["A", "B"]


def test_tradelog_to_frame_empty_has_columns():
    log = TradeLog()
    df = log.to_frame()
    assert "slippage" in df.columns
    assert "fill_ratio" in df.columns
    assert df.empty


def test_tradelog_to_frame_populated():
    t = _now_utc()
    log = TradeLog()
    log.record(_sig("A", t, 10.0, 1), _fill(t, t, 10.1, 1))
    log.record(_sig("B", t, 20.0, 2), _fill(t, t, 19.9, 2, "partial"))
    df = log.to_frame()
    assert len(df) == 2
    assert set(df["ticker"]) == {"A", "B"}
    assert pd.api.types.is_numeric_dtype(df["slippage"])


def test_tradelog_summary_stats():
    t = _now_utc()
    log = TradeLog()
    log.record(_sig("A", t, 100.0, 10), _fill(t, t, 100.5, 10))   # +50bps
    log.record(_sig("B", t, 200.0, 5), _fill(t, t, 199.6, 5))     # -20bps
    log.record(_sig("C", t, 50.0, 1), _fill(t, None, None, 0, "rejected"))
    s = log.summary()
    assert s["n_signals"] == 3
    assert s["n_filled"] == 2
    assert s["n_rejected"] == 1
    assert s["fill_rate"] == pytest.approx(2 / 3)
    assert s["mean_slippage_bps"] == pytest.approx((50.0 + -20.0) / 2)
    assert s["abs_max_slippage_bps"] == pytest.approx(50.0)


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #


def test_tradelog_save_load_roundtrip(tmp_path):
    t = _now_utc()
    log = TradeLog()
    log.record(
        _sig("AAPL", t, 100.0, 10),
        _fill(t, t + timedelta(milliseconds=500), 100.25, 10),
    )
    log.record(_sig("MSFT", t, 300.0, -5), _fill(t, None, None, 0, "submitted"))
    path = tmp_path / "trades.json"
    log.save(path)
    assert path.exists()

    loaded = TradeLog.load(path)
    assert len(loaded) == 2
    entries = loaded.entries
    assert entries[0].signal.ticker == "AAPL"
    assert entries[0].slippage_bps == pytest.approx(25.0)
    assert entries[1].fill.status == "submitted"
    assert entries[1].slippage is None


def test_tradelog_to_json_is_valid_json():
    t = _now_utc()
    log = TradeLog()
    log.record(_sig("X", t, 1.0, 1), _fill(t, t, 1.01, 1))
    payload = log.to_json()
    parsed = json.loads(payload)
    assert isinstance(parsed, list)
    assert parsed[0]["ticker"] == "X"
    # Derived fields are serialized too
    assert parsed[0]["slippage"] == pytest.approx(0.01)


def test_tradelog_save_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "sub" / "trades.json"
    TradeLog().save(path)
    assert path.exists()


def test_tradelog_record_returns_entry():
    t = _now_utc()
    log = TradeLog()
    entry = log.record(_sig("X", t, 5.0, 1), _fill(t, t, 5.01, 1))
    assert isinstance(entry, TradeLogEntry)
    assert entry.signal.ticker == "X"
