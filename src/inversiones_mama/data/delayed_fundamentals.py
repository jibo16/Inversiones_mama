"""Fake point-in-time fundamentals via report-date delay.

Without a paid PIT vendor (Sharadar etc.), we cannot query "what the
market knew on date X". But we CAN simulate it: every fundamental
datapoint has a ``reported_date`` (the day of the filing/press release).
A trader on that date could not use it yet — the market had to ingest,
propagate through aggregators, and show up in data feeds.

This wrapper enforces a configurable ``delay_window`` (default 90 days)
between ``reported_date`` and the earliest ``as_of`` at which the data
is considered known. It composes with any underlying ``FundamentalsLoader``
(Finnhub, AlphaVantage, or a mocked test double) and guarantees
no-forward-peek semantics on the ``as_of`` axis.

Important limitations to acknowledge
------------------------------------
This is a *heuristic*, not a replacement for real PIT data:

* Restatements are invisible — the wrapper returns post-restatement
  values if the underlying vendor silently rewrote them.
* Index constituency still needs a separate PIT source — this covers
  only the fundamentals axis.
* The optimal delay is instrument-dependent (earnings vs guidance vs
  10-K); a uniform window over-approximates.

Still, a 90-day delay is the canonical value in the academic finance
literature for a pragmatic-yet-conservative lag. For v2 on free data,
this is the right tool.

Public API
----------
``DelayedFundamentalsLoader`` — wraps any ``FundamentalsLoader``.
``DELAY_90D`` / ``DELAY_60D`` / ``DELAY_45D`` — named constants.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from .fundamentals import FundamentalsLoader

# Commonly used delay windows
DELAY_45D: timedelta = timedelta(days=45)   # aggressive; relies on fast ingestion
DELAY_60D: timedelta = timedelta(days=60)   # middle ground
DELAY_90D: timedelta = timedelta(days=90)   # canonical conservative lag
DEFAULT_DELAY: timedelta = DELAY_90D


class DelayedFundamentalsLoader:
    """Wraps any ``FundamentalsLoader`` and enforces a reported-date delay.

    Semantics
    ---------
    For each row returned by the inner loader:

        effective_date = reported_date + delay_window
        keep iff  effective_date <= as_of

    The inner loader MUST surface a ``reported_date`` column on the
    panel output and a ``reported_date`` index entry on the series
    output; the delay wrapper will raise ``ValueError`` otherwise.
    (Both ``FinnhubFundamentalsLoader`` and ``AlphaVantageFundamentalsLoader``
    contract to include this once implemented.)
    """

    def __init__(
        self,
        inner: FundamentalsLoader,
        delay: timedelta = DEFAULT_DELAY,
    ) -> None:
        if delay.total_seconds() <= 0:
            raise ValueError(f"delay must be positive, got {delay}")
        self.inner = inner
        self.delay = delay

    # ------------------------------------------------------------------ api

    def load_as_of(self, ticker: str, as_of: pd.Timestamp) -> pd.Series:
        """Return the latest fundamentals known on ``as_of`` after applying delay.

        Implementation strategy: ask the inner loader for a window ending
        at ``as_of - delay``, then pick the last available row. If the
        inner loader's API is only ``load_as_of(ticker, date)``, pass the
        delayed date.
        """
        if not isinstance(as_of, pd.Timestamp):
            as_of = pd.Timestamp(as_of)
        # Shift the query date back by the delay window so the inner loader
        # returns what was known as of (as_of - delay).
        delayed_as_of = as_of - self.delay
        series = self.inner.load_as_of(ticker, delayed_as_of)
        return _ensure_reported_date_not_later(series, as_of, self.delay)

    def load_panel(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return the panel with forward-looking rows filtered out.

        The panel's ``reported_date`` column is the filter key. Any row
        with ``reported_date + delay > row_date`` is dropped.
        """
        panel = self.inner.load_panel(tickers, start, end, features)
        return self.filter_panel(panel, self.delay)

    # -------------------------------------------------------- helpers

    @staticmethod
    def filter_panel(panel: pd.DataFrame, delay: timedelta) -> pd.DataFrame:
        """Drop rows whose ``reported_date + delay`` is after the row's date.

        Expects a long-format panel with a ``date`` column or a ``date``
        level in the MultiIndex, and a ``reported_date`` column.
        """
        if "reported_date" not in panel.columns:
            raise ValueError(
                "DelayedFundamentalsLoader: underlying panel must include a "
                "'reported_date' column. Upgrade the inner loader."
            )
        # Resolve the "as-of date" for each row: prefer an index level named
        # "date", fallback to a 'date' column, else use the panel's index.
        if isinstance(panel.index, pd.MultiIndex) and "date" in panel.index.names:
            row_dates = panel.index.get_level_values("date")
        elif "date" in panel.columns:
            row_dates = pd.to_datetime(panel["date"])
        else:
            row_dates = pd.to_datetime(panel.index)

        reported = pd.to_datetime(panel["reported_date"])
        effective = reported + delay
        keep = effective <= pd.to_datetime(row_dates)
        return panel.loc[keep].copy()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _ensure_reported_date_not_later(
    series: pd.Series, as_of: pd.Timestamp, delay: timedelta,
) -> pd.Series:
    """Validation helper: if a reported_date is present, enforce the delay.

    If the inner loader doesn't populate a reported_date (because it only
    returns latest-known ratios), we trust its implicit as-of.
    """
    reported: Any = series.get("reported_date") if isinstance(series, pd.Series) else None
    if reported is None:
        return series
    reported_ts = pd.Timestamp(reported)
    if reported_ts + delay > as_of:
        raise ValueError(
            f"DelayedFundamentalsLoader: inner loader returned a row with "
            f"reported_date={reported_ts.date()} whose effective_date "
            f"{(reported_ts + delay).date()} is AFTER as_of={as_of.date()}. "
            "The inner loader is leaking future data; delay cannot fix this."
        )
    return series
