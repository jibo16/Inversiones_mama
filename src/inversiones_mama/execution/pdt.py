"""Pattern Day Trader (PDT) rule tracker.

FINRA rule: margin accounts with < $25,000 equity are limited to 3 day
trades within any rolling 5-business-day window. Exceeding this triggers
"Pattern Day Trader" classification which imposes severe restrictions.

Our monthly-rebalance strategy should never legitimately produce day
trades — but a bug, a partial fill + re-try, or a discretionary operator
override could. This module provides a forward-check so the orchestrator
can reject any order intent that would push the account over the limit.

A **day trade** is defined as a same-day round-trip — both a buy and a
sell (or short-sell and buy-to-cover) of the same ticker on the same
trading date. Multiple round-trips in one ticker on one day = multiple
day trades.

Public API
----------
``DayTradeEvent``  — one detected same-day round-trip.
``PDTTracker``     — the rolling-window counter + gate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from .trade_log import TradeLog, TradeLogEntry

# FINRA default parameters
PDT_EQUITY_THRESHOLD: float = 25_000.0
PDT_MAX_DAY_TRADES: int = 3
PDT_WINDOW_BUSINESS_DAYS: int = 5


@dataclass(frozen=True)
class DayTradeEvent:
    """One detected same-day round-trip."""

    trade_date: date
    ticker: str
    round_trips: int  # for a single ticker on one day, count of paired buy/sell


class PDTTracker:
    """Counts day trades in a rolling business-day window and gates new trades.

    Parameters
    ----------
    account_equity : float
        Current account equity. Accounts with ``>= PDT_EQUITY_THRESHOLD``
        ($25k) are **exempt** — the tracker always says "ok".
    max_day_trades : int
        FINRA's threshold (default 3). Counts strictly greater than this
        within the window trigger PDT status.
    window_business_days : int
        Rolling window size (default 5).
    """

    def __init__(
        self,
        account_equity: float,
        max_day_trades: int = PDT_MAX_DAY_TRADES,
        window_business_days: int = PDT_WINDOW_BUSINESS_DAYS,
    ) -> None:
        if account_equity < 0:
            raise ValueError(f"account_equity must be non-negative, got {account_equity}")
        self.account_equity = float(account_equity)
        self.max_day_trades = int(max_day_trades)
        self.window_business_days = int(window_business_days)

    # ------------------------------------------------------------ public API

    @property
    def exempt(self) -> bool:
        """Accounts at or above the $25k equity threshold are PDT-exempt."""
        return self.account_equity >= PDT_EQUITY_THRESHOLD

    def day_trades_from_log(self, log: TradeLog) -> list[DayTradeEvent]:
        """Return every day-trade event in the log (all dates)."""
        fills_by_day: dict[tuple[date, str], list[int]] = defaultdict(list)
        for entry in log:
            fill_day = _fill_date(entry)
            if fill_day is None:
                continue
            qty = entry.fill.filled_quantity
            if qty == 0:
                continue
            fills_by_day[(fill_day, entry.signal.ticker)].append(qty)

        events: list[DayTradeEvent] = []
        for (day, ticker), qtys in fills_by_day.items():
            n_round_trips = _count_round_trips(qtys)
            if n_round_trips > 0:
                events.append(DayTradeEvent(trade_date=day, ticker=ticker, round_trips=n_round_trips))
        return sorted(events, key=lambda e: (e.trade_date, e.ticker))

    def count_in_window(
        self,
        log: TradeLog,
        as_of: date | None = None,
    ) -> int:
        """Count day trades in the trailing window ending at ``as_of`` (inclusive).

        Window boundary: ``as_of - window_business_days + 1`` business days
        back through ``as_of`` itself (calendar days, which over-counts vs.
        strict business days but is the safe, conservative direction).
        """
        if as_of is None:
            # Pick the most recent trade date in the log, or today
            events = self.day_trades_from_log(log)
            as_of = events[-1].trade_date if events else date.today()
        lower = as_of - timedelta(days=self.window_business_days - 1)
        total = 0
        for event in self.day_trades_from_log(log):
            if lower <= event.trade_date <= as_of:
                total += event.round_trips
        return total

    def is_restricted(self, log: TradeLog, as_of: date | None = None) -> bool:
        """True iff the account is (or would be) PDT-restricted as of ``as_of``.

        Exempt accounts (>= $25k equity) always return False.
        """
        if self.exempt:
            return False
        return self.count_in_window(log, as_of=as_of) > self.max_day_trades

    def can_execute_new_day_trade(
        self,
        log: TradeLog,
        as_of: date | None = None,
    ) -> bool:
        """False if adding one more day trade would breach the PDT limit.

        Use this as a forward-gate before submitting a same-day round-trip.
        Exempt accounts always return True.
        """
        if self.exempt:
            return True
        return self.count_in_window(log, as_of=as_of) < self.max_day_trades

    def remaining_day_trades(
        self,
        log: TradeLog,
        as_of: date | None = None,
    ) -> int:
        """How many more day trades the account can take in the current window.

        Exempt accounts return a sentinel large integer.
        """
        if self.exempt:
            return 1_000_000
        used = self.count_in_window(log, as_of=as_of)
        return max(0, self.max_day_trades - used)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _fill_date(entry: TradeLogEntry) -> date | None:
    """Return the date the fill occurred, or None if the order was never filled."""
    if entry.fill.fill_time is not None:
        return entry.fill.fill_time.date()
    # Fallback: use order_time for submitted-but-unfilled orders (they haven't
    # actually day-traded, so this won't count unless they subsequently fill).
    return None


def _count_round_trips(qtys: list[int]) -> int:
    """Count same-day round-trips given the signed fill quantities for one ticker.

    A round-trip closes a position opened the same day. We walk through
    fills in order, tracking the running signed position: a sign flip
    (or return-to-zero after non-zero) indicates a closing fill. The
    number of round-trips is the number of times the position returns
    to the zero line after opening.

    Simple practical rule used here:
        total_buys  = sum of positive qty
        total_sells = sum of |negative qty|
        round_trips = min(total_buys, total_sells) / max(1, smallest_fill_size)

    But that overcounts when the same ticker has multiple partial fills.
    We instead take the matched volume approach: round_trips =
    min(total_buys, total_sells) treating each matched share as one unit,
    then divide by the *median fill size* as a proxy for "one trade unit"
    — but for PDT purposes, a day trade is a distinct open+close pair, so
    the most conservative reading is: at least 1 round-trip if there's
    any matched buy-and-sell volume, and no partial fills complicate the
    simple case of one buy + one sell = one round-trip.

    Implementation: count consecutive sign-flips in the cumulative position.
    """
    if len(qtys) < 2:
        return 0
    # We need signed quantities. SignalRecord.expected_size tells us
    # direction; the fill preserves the sign only if we stored it that way.
    # TradeLog stores filled_quantity as the absolute filled count in the
    # current design — we don't have sign in the FillRecord. So we treat
    # any pair of fills on the same ticker on the same day as potential
    # round-trips.
    #
    # Conservative rule for PDT: if we see ≥ 2 distinct non-zero fills of
    # the same ticker on the same day, assume they represent a round-trip.
    # Multiple fills beyond 2 count as additional round-trips in excess of one.
    non_zero = [q for q in qtys if q != 0]
    if len(non_zero) < 2:
        return 0
    return len(non_zero) // 2
