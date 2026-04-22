"""Tests for inversiones_mama.data.delayed_fundamentals."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from inversiones_mama.data.delayed_fundamentals import (
    DELAY_45D,
    DELAY_60D,
    DELAY_90D,
    DelayedFundamentalsLoader,
)


# --------------------------------------------------------------------------- #
# Mocked inner loader                                                         #
# --------------------------------------------------------------------------- #


class _MockFundamentalsLoader:
    """Mocks a FundamentalsLoader backed by in-memory panels.

    ``panel`` has columns ``date, ticker, reported_date, <features>``
    stacked long-form. ``load_panel`` returns panel rows between [start, end]
    for the given tickers. ``load_as_of`` returns the most recent row for
    a ticker with date <= as_of.
    """

    def __init__(self, panel: pd.DataFrame) -> None:
        self.panel = panel.copy()
        # Defensive coercions
        self.panel["date"] = pd.to_datetime(self.panel["date"])
        self.panel["reported_date"] = pd.to_datetime(self.panel["reported_date"])

    def load_as_of(self, ticker: str, as_of: pd.Timestamp) -> pd.Series:
        # Emulate the vendor: return the most recent row whose reported_date <= as_of
        sub = self.panel[
            (self.panel["ticker"] == ticker) & (self.panel["reported_date"] <= as_of)
        ]
        if sub.empty:
            return pd.Series(dtype=object)
        row = sub.sort_values("reported_date").iloc[-1]
        return row.copy()

    def load_panel(self, tickers, start, end, features=None):
        sub = self.panel[
            (self.panel["ticker"].isin(tickers))
            & (self.panel["date"] >= pd.Timestamp(start))
            & (self.panel["date"] <= pd.Timestamp(end))
        ].copy()
        if features:
            keep = ["date", "ticker", "reported_date", *[f for f in features if f in sub.columns]]
            sub = sub[keep]
        return sub.reset_index(drop=True)


@pytest.fixture
def earnings_panel() -> pd.DataFrame:
    """Long-format panel: quarterly earnings filings for 2 tickers in 2024."""
    rows = [
        # AAPL quarterly reports: q1 filed 2024-02-01, q2 filed 2024-05-01
        {"date": "2024-02-01", "ticker": "AAPL", "reported_date": "2024-02-01", "eps": 2.10},
        {"date": "2024-05-01", "ticker": "AAPL", "reported_date": "2024-05-01", "eps": 2.20},
        {"date": "2024-08-01", "ticker": "AAPL", "reported_date": "2024-08-01", "eps": 2.30},
        {"date": "2024-11-01", "ticker": "AAPL", "reported_date": "2024-11-01", "eps": 2.40},
        # MSFT
        {"date": "2024-01-15", "ticker": "MSFT", "reported_date": "2024-01-15", "eps": 2.93},
        {"date": "2024-04-15", "ticker": "MSFT", "reported_date": "2024-04-15", "eps": 2.95},
    ]
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------- #
# Construction                                                                #
# --------------------------------------------------------------------------- #


def test_rejects_non_positive_delay():
    inner = _MockFundamentalsLoader(pd.DataFrame({
        "date": [], "ticker": [], "reported_date": [], "eps": [],
    }))
    with pytest.raises(ValueError, match="delay"):
        DelayedFundamentalsLoader(inner, delay=timedelta(seconds=0))


def test_named_delay_constants():
    assert DELAY_45D.days == 45
    assert DELAY_60D.days == 60
    assert DELAY_90D.days == 90


# --------------------------------------------------------------------------- #
# load_panel filtering                                                        #
# --------------------------------------------------------------------------- #


def test_panel_filters_rows_within_delay_window(earnings_panel):
    """Rows whose date < reported_date + delay must be dropped."""
    inner = _MockFundamentalsLoader(earnings_panel)
    # Inner ordinarily returns rows where date == reported_date (same day).
    # Under 90-day delay, NO row satisfies "row date >= reported_date + 90d" because
    # the mock has date == reported_date everywhere, so all rows are filtered.
    loader = DelayedFundamentalsLoader(inner, delay=DELAY_90D)
    panel = loader.load_panel(["AAPL", "MSFT"], datetime(2024, 1, 1), datetime(2024, 12, 31))
    assert panel.empty, "all rows should be filtered when date == reported_date under 90d delay"


def test_panel_keeps_rows_after_delay_matures():
    """When the row's observation date is past reported_date + delay, keep it."""
    panel = pd.DataFrame([
        {"date": "2024-04-01", "ticker": "AAPL", "reported_date": "2024-01-01", "eps": 2.10},  # 90d lag -> keep
        {"date": "2024-02-15", "ticker": "AAPL", "reported_date": "2024-01-01", "eps": 2.10},  # 45d lag -> drop at 90d
        {"date": "2024-07-01", "ticker": "MSFT", "reported_date": "2024-03-01", "eps": 2.95},  # 122d lag -> keep
    ])
    inner = _MockFundamentalsLoader(panel)
    loader = DelayedFundamentalsLoader(inner, delay=DELAY_90D)
    out = loader.load_panel(["AAPL", "MSFT"], datetime(2024, 1, 1), datetime(2024, 12, 31))
    assert len(out) == 2
    # Only the two rows whose lag >= 90d survive
    assert set(out["reported_date"].dt.date) == {pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-03-01").date()}


def test_panel_45d_keeps_more_than_90d():
    panel = pd.DataFrame([
        # 60-day lag: kept under 45d, dropped under 90d
        {"date": "2024-03-01", "ticker": "AAPL", "reported_date": "2024-01-01", "eps": 2.10},
    ])
    inner = _MockFundamentalsLoader(panel)
    aggressive = DelayedFundamentalsLoader(inner, delay=DELAY_45D)
    conservative = DelayedFundamentalsLoader(inner, delay=DELAY_90D)
    assert len(aggressive.load_panel(["AAPL"], datetime(2024, 1, 1), datetime(2024, 12, 31))) == 1
    assert len(conservative.load_panel(["AAPL"], datetime(2024, 1, 1), datetime(2024, 12, 31))) == 0


def test_panel_raises_on_missing_reported_date_column():
    panel_broken = pd.DataFrame([{"date": "2024-04-01", "ticker": "AAPL", "eps": 2.10}])

    class BrokenInner:
        def load_panel(self, *args, **kwargs):
            return panel_broken

        def load_as_of(self, *args, **kwargs):
            return pd.Series()

    loader = DelayedFundamentalsLoader(BrokenInner(), delay=DELAY_90D)
    with pytest.raises(ValueError, match="reported_date"):
        loader.load_panel(["AAPL"], datetime(2024, 1, 1), datetime(2024, 12, 31))


# --------------------------------------------------------------------------- #
# load_as_of                                                                  #
# --------------------------------------------------------------------------- #


def test_load_as_of_returns_pre_delay_snapshot(earnings_panel):
    """If as_of = 2024-05-01 and delay = 90d, we should see only filings <= 2024-02-01."""
    inner = _MockFundamentalsLoader(earnings_panel)
    loader = DelayedFundamentalsLoader(inner, delay=DELAY_90D)
    series = loader.load_as_of("AAPL", pd.Timestamp("2024-05-01"))
    # The most recent AAPL filing with reported_date <= 2024-05-01 - 90d = 2024-02-01
    assert pd.Timestamp(series["reported_date"]) == pd.Timestamp("2024-02-01")
    assert series["eps"] == 2.10


def test_load_as_of_empty_when_no_filings_pre_delay(earnings_panel):
    inner = _MockFundamentalsLoader(earnings_panel)
    loader = DelayedFundamentalsLoader(inner, delay=DELAY_90D)
    series = loader.load_as_of("AAPL", pd.Timestamp("2024-01-05"))  # before any filing
    assert series.empty


def test_load_as_of_raises_if_inner_leaks_future(earnings_panel):
    """If the inner loader returns a reported_date too recent for the delay, raise."""
    class LeakyInner:
        def load_as_of(self, ticker, as_of):
            # Returns a row reported TODAY regardless of as_of — that's a bug in
            # the vendor, but the wrapper should catch it.
            return pd.Series({
                "ticker": ticker,
                "reported_date": pd.Timestamp("2024-10-15"),
                "eps": 99.0,
            })

        def load_panel(self, *a, **kw):
            return pd.DataFrame()

    loader = DelayedFundamentalsLoader(LeakyInner(), delay=DELAY_90D)
    with pytest.raises(ValueError, match="leaking future data"):
        loader.load_as_of("AAPL", pd.Timestamp("2024-11-01"))  # only 17d gap, less than 90d


def test_load_as_of_passes_when_inner_omits_reported_date():
    """Lax fallback: if inner doesn't set reported_date, trust it (can't validate)."""
    class LaxInner:
        def load_as_of(self, ticker, as_of):
            return pd.Series({"ticker": ticker, "pe": 25.0})

        def load_panel(self, *a, **kw):
            return pd.DataFrame()

    loader = DelayedFundamentalsLoader(LaxInner(), delay=DELAY_90D)
    series = loader.load_as_of("AAPL", pd.Timestamp("2024-11-01"))
    assert series["pe"] == 25.0
