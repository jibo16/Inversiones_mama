"""Fundamentals data loader for v2 equities-scale alpha generation.

Phase 3 of v2 (Bayesian alpha) and Phase 4 (Skew-t fit on fundamentals-
conditioned returns) both require point-in-time financial-statement data
for the stock universe. This module defines the loader contract and
stubs out the two primary vendors.

Design notes
------------
* Every loader MUST respect point-in-time semantics: if you query
  ``load_as_of(ticker, 2020-06-30)`` you must receive only the fundamentals
  that would have been known to a trader on that date (filings up through
  that day, pre-restatement values when applicable).
* Cache to parquet via the existing ``data.cache.ParquetCache`` pattern.
* Rate-limit aware: Finnhub free tier is 60 calls/min, Alpha Vantage free
  is 25 calls/day. Implementations must batch and cache aggressively.

Public API
----------
``FundamentalsLoader`` — Protocol all vendor loaders must satisfy.
``FinnhubFundamentalsLoader`` — Finnhub implementation (stub).
``AlphaVantageFundamentalsLoader`` — AlphaVantage implementation (stub).
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd

# Standard canonical fundamental-feature set — every loader must surface
# at least these when data is available. Missing fields are NaN, never
# backfilled from the future.
CANONICAL_FEATURES: tuple[str, ...] = (
    # Valuation
    "pe_trailing",
    "pb",
    "ev_ebitda",
    "fcf_yield",
    # Quality
    "roe",
    "roic",
    "debt_to_equity",
    "interest_coverage",
    # Growth
    "revenue_growth_yoy",
    "earnings_growth_yoy",
    # Size
    "market_cap_usd",
)


@runtime_checkable
class FundamentalsLoader(Protocol):
    """Vendor-agnostic fundamentals interface."""

    def load_as_of(self, ticker: str, as_of: pd.Timestamp) -> pd.Series:
        """Return one Series of canonical features as-known on ``as_of``.

        Must only use filings dated on or before ``as_of``.
        """
        ...

    def load_panel(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return a long-format panel DataFrame.

        Index: MultiIndex[(date, ticker)]. Columns: features.
        Each row is the as-of fundamentals snapshot for that (date, ticker).
        """
        ...


# --------------------------------------------------------------------------- #
# Vendor stubs                                                                #
# --------------------------------------------------------------------------- #


class FinnhubFundamentalsLoader:
    """Stub — Finnhub fundamentals loader.

    Free tier: 60 calls/min. Real-time quotes + 30 years of fundamentals
    require Finnhub Pro (~$49/mo/market module). API key expected in
    ``FINNHUB_API_KEY`` env var.

    When implemented in a later session:
    * Use ``/api/v1/stock/metric`` for key ratios.
    * Use ``/api/v1/stock/financials-reported`` for full financial statements.
    * Respect the rate limit with a semaphore; cache every pull.
    """

    def __init__(self, api_key: str | None = None) -> None:
        import os

        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY")

    def load_as_of(self, ticker: str, as_of: pd.Timestamp) -> pd.Series:  # noqa: ARG002
        raise NotImplementedError("FinnhubFundamentalsLoader not yet implemented — see ARCHITECTURE_V2.md Phase 1")

    def load_panel(
        self,
        tickers: list[str],  # noqa: ARG002
        start: datetime,  # noqa: ARG002
        end: datetime,  # noqa: ARG002
        features: list[str] | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        raise NotImplementedError("FinnhubFundamentalsLoader not yet implemented")


class AlphaVantageFundamentalsLoader:
    """Stub — AlphaVantage fundamentals loader.

    Free tier: 25 calls/day — insufficient for v2 scale. Premium plans
    from $29.99 to $249.99/month. API key in ``ALPHAVANTAGE_API_KEY``.

    When implemented:
    * ``OVERVIEW`` endpoint for key ratios.
    * ``INCOME_STATEMENT``, ``BALANCE_SHEET``, ``CASH_FLOW`` for raw filings.
    * Aggressive parquet cache — each pull costs real quota.
    """

    def __init__(self, api_key: str | None = None) -> None:
        import os

        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")

    def load_as_of(self, ticker: str, as_of: pd.Timestamp) -> pd.Series:  # noqa: ARG002
        raise NotImplementedError("AlphaVantageFundamentalsLoader not yet implemented — see ARCHITECTURE_V2.md Phase 1")

    def load_panel(
        self,
        tickers: list[str],  # noqa: ARG002
        start: datetime,  # noqa: ARG002
        end: datetime,  # noqa: ARG002
        features: list[str] | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        raise NotImplementedError("AlphaVantageFundamentalsLoader not yet implemented")
