"""Point-in-time constituent universe — survivorship-bias defense for v2.

A ``PointInTimeUniverse`` answers "which tickers were index members on a
given historical date?" *using only information knowable at that date*.
This is the single most important defense against survivorship bias when
scaling to thousands of equities (Jorge's v2 mandate Phase 1). Without it,
naively downloading today's Russell 3000 constituents' history bakes in
the upward bias of every company that survived, while silently discarding
bankruptcies, delistings, and acquisitions.

This module provides:

* ``PointInTimeUniverse`` — abstract Protocol that vendor-specific
  implementations must satisfy.
* ``StaticUniverse`` — trivial implementation backed by a hardcoded
  ticker list. Useful for v1a and for tests; does **not** provide
  survivorship-bias protection by itself (the ticker list is assumed
  valid for the whole date range).

Vendor implementations to write in later sessions (see
``docs/ARCHITECTURE_V2.md``):

* ``SharadarPointInTimeUniverse`` — Sharadar Core US Fundamentals.
* ``FinnhubPointInTimeUniverse`` — Finnhub Premium historical constituents.
* ``ReconstructedPointInTimeUniverse`` — Russell/S&P reconstitution press-release backed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PointInTimeUniverse(Protocol):
    """Read-only view of the investable universe at arbitrary historical dates.

    All implementations must satisfy strict no-forward-peek semantics:
    ``members_on(as_of)`` must return the *same* list it would have
    returned if the query were issued on date ``as_of``, regardless of
    any future information the implementation happens to have ingested.
    """

    def members_on(self, as_of: pd.Timestamp) -> list[str]:
        """Return ticker symbols that were members of the universe on ``as_of``."""
        ...

    def active_range(self, ticker: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Inclusive [first_day, last_day] the ticker was a member, or None."""
        ...

    def all_tickers(self) -> list[str]:
        """Every ticker that has EVER been a member across the full history.

        This is the superset needed to fetch price/fundamentals data — it
        includes currently delisted names. Do NOT confuse this with
        ``members_on(today)``.
        """
        ...


# --------------------------------------------------------------------------- #
# Concrete implementations                                                    #
# --------------------------------------------------------------------------- #


class StaticUniverse:
    """Trivial PointInTimeUniverse backed by a fixed ticker list.

    The list is assumed valid for the entire history — i.e. every ticker
    is reported as a member on every date. This is appropriate for the
    v1a 10-ETF universe (where all tickers lived the whole window) but
    is **not** a substitute for a true survivorship-safe vendor feed on
    v2's thousand-stock universe.

    Attributes
    ----------
    tickers : tuple[str, ...]
        Sorted, de-duplicated ticker symbols.
    """

    def __init__(self, tickers: list[str] | tuple[str, ...]) -> None:
        if not tickers:
            raise ValueError("StaticUniverse requires at least one ticker")
        self.tickers = tuple(sorted({t.upper().strip() for t in tickers if t.strip()}))

    def members_on(self, as_of: pd.Timestamp) -> list[str]:  # noqa: ARG002 - PIT-compatible signature
        return list(self.tickers)

    def active_range(self, ticker: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if ticker.upper().strip() not in self.tickers:
            return None
        # The static universe has no history bounds; report a permissive range.
        return (pd.Timestamp.min, pd.Timestamp.max)

    def all_tickers(self) -> list[str]:
        return list(self.tickers)

    def __len__(self) -> int:
        return len(self.tickers)

    def __contains__(self, ticker: object) -> bool:
        if not isinstance(ticker, str):
            return False
        return ticker.upper().strip() in self.tickers

    def __repr__(self) -> str:
        preview = ", ".join(self.tickers[:5])
        if len(self.tickers) > 5:
            preview += f", ...+{len(self.tickers) - 5}"
        return f"StaticUniverse([{preview}])"


# --------------------------------------------------------------------------- #
# Vendor stubs (to be implemented in later sessions)                          #
# --------------------------------------------------------------------------- #


class SharadarPointInTimeUniverse:
    """Stub — Sharadar Core US Fundamentals point-in-time constituent feed.

    Implement in a future session with a Sharadar subscription. Until
    then, every method raises ``NotImplementedError`` so callers fail
    fast if they wire this up prematurely.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def members_on(self, as_of: pd.Timestamp) -> list[str]:  # noqa: ARG002
        raise NotImplementedError("SharadarPointInTimeUniverse not yet implemented — see ARCHITECTURE_V2.md Phase 1")

    def active_range(self, ticker: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:  # noqa: ARG002
        raise NotImplementedError("SharadarPointInTimeUniverse not yet implemented")

    def all_tickers(self) -> list[str]:
        raise NotImplementedError("SharadarPointInTimeUniverse not yet implemented")
