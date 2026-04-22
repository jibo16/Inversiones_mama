"""Price data loader — yfinance primary, IBKR stub fallback.

Public API:
    fetch_prices_yfinance(tickers, start, end, auto_adjust=True) -> DataFrame
    load_prices(tickers, start, end, use_cache=True, source="yfinance") -> DataFrame
    returns_from_prices(prices) -> DataFrame

DataFrame shape: index=date (pandas DatetimeIndex, name="date"), columns=ticker,
values=adjusted close. Missing rows are left as NaN; callers decide how to fill.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime

import pandas as pd
import yfinance as yf

from ..config import CACHE_DIR
from .cache import ParquetCache

log = logging.getLogger(__name__)


def fetch_prices_yfinance(
    tickers: Iterable[str],
    start: datetime,
    end: datetime,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Pull adjusted close from yfinance.

    Returns a wide DataFrame: index=date, columns=ticker. Empty DataFrame
    (no columns) if all tickers fail; partial DataFrame with missing columns
    if some fail.
    """
    tickers_list = list(tickers)
    if not tickers_list:
        raise ValueError("tickers must be non-empty")

    data = yf.download(
        tickers_list,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if data is None or len(data) == 0:
        return pd.DataFrame()

    # yfinance returns either a plain columned DF (single ticker) or a
    # MultiIndex column DF (multi-ticker). Normalize to wide (ticker cols).
    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(data.columns.get_level_values(0))
        closes = {t: data[t]["Close"] for t in tickers_list if t in level0}
        wide = pd.DataFrame(closes)
    else:
        # Single ticker returns flat columns: Open/High/Low/Close/Volume
        if "Close" not in data.columns:
            return pd.DataFrame()
        wide = pd.DataFrame({tickers_list[0]: data["Close"]})

    # Normalize index dtype to datetime64[ns] so fresh fetch matches parquet round-trip
    wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index).values.astype("datetime64[ns]"))
    wide.index.name = "date"
    return wide.sort_index().dropna(how="all")


def load_prices(
    tickers: Iterable[str],
    start: datetime,
    end: datetime | None = None,
    use_cache: bool = True,
    cache_max_age_hours: float = 24.0,
    source: str = "yfinance",
) -> pd.DataFrame:
    """Load prices with parquet caching.

    Parameters
    ----------
    tickers : list of ticker strings
    start, end : datetime range (end defaults to today)
    use_cache : if True, serve from cache when fresh; otherwise refetch
    cache_max_age_hours : cache entries older than this refetch
    source : "yfinance" (default) or "ibkr" (raises until Step X live wiring)
    """
    tickers_sorted = sorted(tickers)
    end = end or datetime.today()
    key = f"prices_{source}_{'-'.join(tickers_sorted)}_{start.date()}_{end.date()}"

    cache = ParquetCache(CACHE_DIR)
    if use_cache and cache.is_fresh(key, cache_max_age_hours):
        log.debug("cache hit: %s", key)
        return cache.get(key)

    if source == "yfinance":
        df = fetch_prices_yfinance(tickers_sorted, start, end)
    elif source == "ibkr":
        # Lazy import — IBKR historical needs a running, authenticated Gateway.
        from .ibkr_historical import IBKRHistoricalLoader

        loader = IBKRHistoricalLoader.from_env()
        loader.ensure_authenticated()
        period = _derive_ibkr_period(start, end)
        df = loader.fetch_many(tickers_sorted, period=period)
        # Trim to exact [start, end] since IBKR's period is coarse ("1y"/"5y"/etc.)
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end)
        df = df[(df.index >= lo) & (df.index <= hi)]
    else:
        raise ValueError(f"Unknown source: {source}")

    if df.empty:
        raise RuntimeError(f"Empty price DataFrame for {tickers_sorted} from {source}")

    if use_cache:
        cache.put(key, df)
        # Return via cache to guarantee identical dtype/shape on subsequent calls
        return cache.get(key)
    return df


def _derive_ibkr_period(start: datetime, end: datetime) -> str:
    """Pick the smallest IBKR period string that covers [start, end].

    IBKR's ``/iserver/marketdata/history`` takes coarse period strings
    (``"1y"``, ``"5y"``, etc.). We pick the first alias whose span
    covers the requested range and trim the excess client-side.
    """
    days = (end - start).days
    if days <= 365:
        return "1y"
    if days <= 365 * 2:
        return "2y"
    if days <= 365 * 3:
        return "3y"
    if days <= 365 * 5:
        return "5y"
    return "10y"


def returns_from_prices(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """Compute daily returns from a prices DataFrame.

    method : "simple" -> (p_t / p_{t-1}) - 1
             "log"    -> ln(p_t / p_{t-1})
    Drops the first row (NaN from diff).
    """
    if method == "simple":
        r = prices.pct_change()
    elif method == "log":
        import numpy as np

        r = (prices / prices.shift(1)).apply(np.log)
    else:
        raise ValueError(f"Unknown return method: {method}")
    return r.iloc[1:]
