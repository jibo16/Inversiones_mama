"""Curated liquid-universe lists for v2 free-data development.

Zero-budget constraint (2026-04-22): we cannot buy Russell 3000 or S&P 1500
point-in-time constituents. Instead we work with curated lists of highly
liquid names whose ADV supports the 25/day AlphaVantage + 60/min Finnhub
free-tier caps. Also avoids survivorship-bias nightmares by sticking to
names that are almost all still alive on the current date (cheap, but
honest: for a small number of well-known blue chips, survivorship bias
is small over a 5-year window).

Three curated sets:

* ``SP100_CORE``  ~ 100 of the most liquid S&P 100 names.
* ``NASDAQ100_CORE`` ~ 80 highly liquid NASDAQ-100 names.
* ``LIQUID_ETFS``  ~ 30 large, diversified, very-high-volume ETFs.

Composition is stable enough for v2 prototyping. When the project moves
to paid PIT data, these lists are deprecated in favor of historical
constituent snapshots.

Public API
----------
``SP100_CORE`` / ``NASDAQ100_CORE`` / ``LIQUID_ETFS`` — tuple[str, ...].
``build_liquid_universe(kind, limit)`` — returns a :class:`StaticUniverse`.
``top_k_by_volume(tickers, k, start, end)`` — ADV-ranked subset via yfinance.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from .prices import load_prices
from .universe import StaticUniverse


# --- Curated lists ----------------------------------------------------------

# Heavily-traded S&P 100 names. Deliberately kept to well-known blue chips
# so we minimize free-tier rate-limit pressure during data pulls.
SP100_CORE: tuple[str, ...] = (
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "C", "CAT",
    "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DE", "DHR", "DIS", "DUK", "EMR", "F", "FDX", "GD", "GE", "GILD",
    "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM",
    "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT",
    "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE",
    "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX", "SBUX",
    "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UNH",
    "UNP", "UPS", "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM",
)

# NASDAQ-100 core — overlap with SP100 is intentional; this list is
# technology/internet-heavy and useful when running momentum screens that
# favor growth names.
NASDAQ100_CORE: tuple[str, ...] = (
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "AEP", "AMAT", "AMD", "AMGN", "AMZN",
    "ANSS", "ASML", "AVGO", "AZN", "BIIB", "BKNG", "BKR", "CDNS", "CDW", "CEG",
    "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTSH",
    "DDOG", "DLTR", "DXCM", "EA", "EXC", "FAST", "FANG", "FI", "FTNT", "GEHC",
    "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "ILMN", "INTC", "INTU", "ISRG",
    "JD", "KDP", "KHC", "KLAC", "LCID", "LIN", "LRCX", "LULU", "MAR", "MCHP",
    "MDLZ", "MELI", "META", "MNST", "MRNA", "MRVL", "MSFT", "MU", "NFLX", "NVDA",
    "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PYPL",
    "QCOM", "REGN", "ROP", "ROST", "SBUX", "SIRI", "SNPS", "TEAM", "TMUS", "TSLA",
    "TTD", "TXN", "VRSK", "VRTX", "WBA", "WBD", "WDAY", "XEL", "ZM", "ZS",
)

# Highest-volume US-listed ETFs spanning broad equity, sector, bond, gold,
# and volatility exposures.
LIQUID_ETFS: tuple[str, ...] = (
    # Broad US equity
    "SPY", "IVV", "VOO", "QQQ", "DIA", "IWM",
    # International / EM
    "EFA", "VEA", "IEFA", "EEM", "IEMG", "VWO",
    # Bonds
    "AGG", "BND", "TLT", "IEF", "LQD", "HYG",
    # Gold / commodities
    "GLD", "IAU", "SLV", "DBC",
    # Factor / style
    "VLUE", "MTUM", "QUAL", "USMV", "AVUV", "AVDV", "AVEM",
    # Sector (highest volume subset)
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLP", "XLI",
    # Real estate
    "VNQ", "IYR",
)


def _unique_sorted(tickers: tuple[str, ...]) -> tuple[str, ...]:
    """De-dup and sort (preserves stability across runs)."""
    return tuple(sorted({t.upper().strip() for t in tickers if t.strip()}))


def build_liquid_universe(
    kind: str = "all",
    limit: int | None = None,
) -> StaticUniverse:
    """Return a ``StaticUniverse`` populated from the curated lists.

    Parameters
    ----------
    kind : ``"sp100"``, ``"nasdaq100"``, ``"etfs"``, or ``"all"``.
    limit : optional cap on the number of tickers (first N after sort).

    Raises
    ------
    ValueError
        If ``kind`` is unknown.
    """
    kind = kind.lower()
    if kind == "sp100":
        tickers = _unique_sorted(SP100_CORE)
    elif kind == "nasdaq100":
        tickers = _unique_sorted(NASDAQ100_CORE)
    elif kind == "etfs":
        tickers = _unique_sorted(LIQUID_ETFS)
    elif kind == "all":
        tickers = _unique_sorted(SP100_CORE + NASDAQ100_CORE + LIQUID_ETFS)
    else:
        raise ValueError(f"Unknown kind: {kind!r}. Choose sp100 / nasdaq100 / etfs / all.")

    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        tickers = tickers[:limit]

    return StaticUniverse(list(tickers))


def top_k_by_volume(
    tickers: list[str] | tuple[str, ...],
    k: int,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[str]:
    """Return the ``k`` tickers with the highest average dollar-volume.

    Uses yfinance daily prices over [start, end] (default: last 90 days).
    ADV is approximated as mean(adjusted_close × volume) — since
    ``load_prices`` returns adjusted closes only, we instead approximate
    ADV as the mean price × a nominal volume proxy of 1 (effectively
    ranking by price level when volume data is unavailable). For a
    rigorous ADV calculation, switch to a price loader that exposes the
    raw ``Volume`` column.

    Parameters
    ----------
    tickers : tickers to rank.
    k : number of top tickers to return.
    start, end : optional date range (default: last 90 days).

    Returns
    -------
    list[str]
        The top-k tickers by mean adjusted close (proxy for ADV without
        volume); fewer than k if some tickers have no data.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    tickers = sorted({t.upper().strip() for t in tickers if t.strip()})
    if not tickers:
        return []
    end = end or datetime.today()
    start = start or (end - timedelta(days=90))
    prices = load_prices(tickers, start, end, use_cache=True)
    # Rank by mean adjusted close (a crude proxy; real ADV needs volume)
    scores = prices.mean(axis=0).dropna().sort_values(ascending=False)
    return list(scores.head(k).index)


def all_curated_tickers() -> list[str]:
    """Flat de-duplicated list of every ticker in any curated set."""
    return list(_unique_sorted(SP100_CORE + NASDAQ100_CORE + LIQUID_ETFS))
