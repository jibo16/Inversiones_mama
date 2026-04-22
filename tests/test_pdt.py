"""Tests for inversiones_mama.execution.pdt."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from inversiones_mama.execution.pdt import (
    PDT_EQUITY_THRESHOLD,
    PDT_MAX_DAY_TRADES,
    DayTradeEvent,
    PDTTracker,
)
from inversiones_mama.execution.trade_log import (
    FillRecord,
    SignalRecord,
    TradeLog,
    TradeLogEntry,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _entry(ticker: str, size: int, fill_time: datetime, status: str = "filled", price: float = 100.0) -> TradeLogEntry:
    return TradeLogEntry(
        signal=SignalRecord(
            ticker=ticker,
            signal_time=fill_time - timedelta(milliseconds=50),
            expected_price=price,
            expected_size=size,
        ),
        fill=FillRecord(
            order_time=fill_time - timedelta(milliseconds=10),
            fill_time=fill_time,
            fill_price=price,
            filled_quantity=abs(size),
            status=status,
        ),
    )


def _t(y: int, m: int, d: int, hh: int = 10) -> datetime:
    return datetime(y, m, d, hh, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Exempt accounts                                                             #
# --------------------------------------------------------------------------- #


def test_exempt_when_equity_at_threshold():
    tracker = PDTTracker(account_equity=25_000.0)
    assert tracker.exempt is True
    assert tracker.is_restricted(TradeLog()) is False


def test_exempt_when_equity_above_threshold():
    tracker = PDTTracker(account_equity=100_000.0)
    assert tracker.exempt is True


def test_not_exempt_below_threshold():
    tracker = PDTTracker(account_equity=5_000.0)
    assert tracker.exempt is False


def test_rejects_negative_equity():
    with pytest.raises(ValueError, match="non-negative"):
        PDTTracker(account_equity=-100.0)


# --------------------------------------------------------------------------- #
# Day-trade detection                                                         #
# --------------------------------------------------------------------------- #


def test_no_day_trade_on_buy_only():
    log = TradeLog()
    log.append(_entry("AAPL", 10, _t(2026, 4, 22)))
    tracker = PDTTracker(account_equity=5000.0)
    assert tracker.day_trades_from_log(log) == []


def test_no_day_trade_on_different_days():
    log = TradeLog()
    log.append(_entry("AAPL", 10, _t(2026, 4, 22)))
    log.append(_entry("AAPL", -10, _t(2026, 4, 23)))
    tracker = PDTTracker(account_equity=5000.0)
    assert tracker.day_trades_from_log(log) == []


def test_day_trade_detected_same_day_same_ticker():
    log = TradeLog()
    log.append(_entry("AAPL", 10, _t(2026, 4, 22, 10)))
    log.append(_entry("AAPL", -10, _t(2026, 4, 22, 14)))
    tracker = PDTTracker(account_equity=5000.0)
    events = tracker.day_trades_from_log(log)
    assert len(events) == 1
    assert events[0].ticker == "AAPL"
    assert events[0].trade_date == date(2026, 4, 22)
    assert events[0].round_trips == 1


def test_day_trade_different_tickers_separate_events():
    log = TradeLog()
    log.append(_entry("AAPL", 10, _t(2026, 4, 22, 10)))
    log.append(_entry("AAPL", -10, _t(2026, 4, 22, 11)))
    log.append(_entry("MSFT", 5, _t(2026, 4, 22, 12)))
    log.append(_entry("MSFT", -5, _t(2026, 4, 22, 13)))
    tracker = PDTTracker(account_equity=5000.0)
    events = tracker.day_trades_from_log(log)
    assert len(events) == 2
    tickers = {e.ticker for e in events}
    assert tickers == {"AAPL", "MSFT"}


def test_day_trade_multiple_round_trips_same_ticker():
    """4 fills (buy, sell, buy, sell) on same day/ticker = 2 round-trips."""
    log = TradeLog()
    log.append(_entry("AAPL", 10, _t(2026, 4, 22, 9)))
    log.append(_entry("AAPL", -10, _t(2026, 4, 22, 10)))
    log.append(_entry("AAPL", 5, _t(2026, 4, 22, 11)))
    log.append(_entry("AAPL", -5, _t(2026, 4, 22, 12)))
    tracker = PDTTracker(account_equity=5000.0)
    events = tracker.day_trades_from_log(log)
    assert len(events) == 1
    assert events[0].round_trips == 2


def test_unfilled_orders_ignored():
    """Submitted/rejected orders don't count as day trades."""
    log = TradeLog()
    log.append(_entry("AAPL", 10, _t(2026, 4, 22, 10)))
    log.append(TradeLogEntry(
        signal=SignalRecord(
            ticker="AAPL", signal_time=_t(2026, 4, 22, 14),
            expected_price=100.0, expected_size=-10,
        ),
        fill=FillRecord(
            order_time=_t(2026, 4, 22, 14),
            fill_time=None, fill_price=None, filled_quantity=0, status="rejected",
        ),
    ))
    tracker = PDTTracker(account_equity=5000.0)
    assert tracker.day_trades_from_log(log) == []


# --------------------------------------------------------------------------- #
# Rolling-window counting                                                     #
# --------------------------------------------------------------------------- #


def test_count_in_window_empty_log():
    tracker = PDTTracker(account_equity=5000.0)
    assert tracker.count_in_window(TradeLog(), as_of=date(2026, 4, 22)) == 0


def test_count_in_window_inside_window():
    log = TradeLog()
    # 3 separate day trades on 3 consecutive days
    for d in (20, 21, 22):
        log.append(_entry("AAPL", 1, _t(2026, 4, d, 10)))
        log.append(_entry("AAPL", -1, _t(2026, 4, d, 11)))
    tracker = PDTTracker(account_equity=5000.0)
    # All three are within a 5-day window ending 2026-04-22
    assert tracker.count_in_window(log, as_of=date(2026, 4, 22)) == 3


def test_count_in_window_older_events_excluded():
    log = TradeLog()
    # Day trade on April 1 (outside window)
    log.append(_entry("AAPL", 1, _t(2026, 4, 1, 10)))
    log.append(_entry("AAPL", -1, _t(2026, 4, 1, 11)))
    # Day trade on April 22 (inside window)
    log.append(_entry("AAPL", 1, _t(2026, 4, 22, 10)))
    log.append(_entry("AAPL", -1, _t(2026, 4, 22, 11)))
    tracker = PDTTracker(account_equity=5000.0)
    assert tracker.count_in_window(log, as_of=date(2026, 4, 22)) == 1


def test_default_as_of_uses_latest_trade_date():
    log = TradeLog()
    log.append(_entry("AAPL", 1, _t(2026, 3, 1, 10)))
    log.append(_entry("AAPL", -1, _t(2026, 3, 1, 11)))
    tracker = PDTTracker(account_equity=5000.0)
    # Default as_of = 2026-03-01 → 1 day trade in window
    assert tracker.count_in_window(log) == 1


# --------------------------------------------------------------------------- #
# Restriction logic                                                           #
# --------------------------------------------------------------------------- #


def test_is_restricted_below_threshold_with_4_day_trades():
    log = TradeLog()
    # 4 day trades on consecutive days
    for day in (19, 20, 21, 22):
        log.append(_entry("AAPL", 1, _t(2026, 4, day, 10)))
        log.append(_entry("AAPL", -1, _t(2026, 4, day, 11)))
    tracker = PDTTracker(account_equity=5000.0)
    assert tracker.is_restricted(log, as_of=date(2026, 4, 22)) is True


def test_is_not_restricted_with_3_day_trades():
    log = TradeLog()
    for day in (20, 21, 22):
        log.append(_entry("AAPL", 1, _t(2026, 4, day, 10)))
        log.append(_entry("AAPL", -1, _t(2026, 4, day, 11)))
    tracker = PDTTracker(account_equity=5000.0)
    assert tracker.is_restricted(log, as_of=date(2026, 4, 22)) is False


def test_exempt_account_never_restricted():
    log = TradeLog()
    # 10 day trades — would absolutely break PDT if not exempt
    for day in range(13, 23):
        log.append(_entry("AAPL", 1, _t(2026, 4, day, 10)))
        log.append(_entry("AAPL", -1, _t(2026, 4, day, 11)))
    tracker = PDTTracker(account_equity=100_000.0)
    assert tracker.is_restricted(log, as_of=date(2026, 4, 22)) is False


# --------------------------------------------------------------------------- #
# Forward-check gate                                                          #
# --------------------------------------------------------------------------- #


def test_can_execute_new_day_trade_with_zero_used():
    tracker = PDTTracker(account_equity=5000.0)
    assert tracker.can_execute_new_day_trade(TradeLog()) is True


def test_can_execute_new_day_trade_with_3_used():
    log = TradeLog()
    for day in (20, 21, 22):
        log.append(_entry("AAPL", 1, _t(2026, 4, day, 10)))
        log.append(_entry("AAPL", -1, _t(2026, 4, day, 11)))
    tracker = PDTTracker(account_equity=5000.0)
    # Already at max (3). One more would trigger.
    assert tracker.can_execute_new_day_trade(log, as_of=date(2026, 4, 22)) is False


def test_can_execute_exempt_always_true():
    log = TradeLog()
    for day in (20, 21, 22):
        log.append(_entry("AAPL", 1, _t(2026, 4, day, 10)))
        log.append(_entry("AAPL", -1, _t(2026, 4, day, 11)))
    tracker = PDTTracker(account_equity=50_000.0)
    assert tracker.can_execute_new_day_trade(log, as_of=date(2026, 4, 22)) is True


def test_remaining_day_trades_counts_down():
    tracker = PDTTracker(account_equity=5000.0)
    log = TradeLog()
    assert tracker.remaining_day_trades(log, as_of=date(2026, 4, 22)) == 3
    log.append(_entry("AAPL", 1, _t(2026, 4, 22, 10)))
    log.append(_entry("AAPL", -1, _t(2026, 4, 22, 11)))
    assert tracker.remaining_day_trades(log, as_of=date(2026, 4, 22)) == 2


def test_remaining_day_trades_exempt_sentinel():
    tracker = PDTTracker(account_equity=50_000.0)
    assert tracker.remaining_day_trades(TradeLog()) > 1000


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #


def test_constants_match_finra():
    assert PDT_EQUITY_THRESHOLD == 25_000.0
    assert PDT_MAX_DAY_TRADES == 3
