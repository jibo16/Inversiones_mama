"""Average-daily-volume (ADV) loader for market-impact modeling.

The LOB-walk slippage model (``backtest/costs.py::estimate_slippage``)
takes a per-ticker ADV to compute the participation rate of each trade
and apply non-linear market impact penalties above 1% of ADV. This
module produces that ADV — pulled from yfinance daily ``Volume`` columns
and cached as parquet.

Public API
----------
``load_adv_shares(tickers, end, window_days=30, use_cache=True) -> dict[str, float]``
    Per-ticker mean daily share volume over the trailing window.

``load_adv_dollars(tickers, end, window_days=30, use_cache=True) -> dict[str, float]``
    Per-ticker mean daily dollar volume (close price × share volume).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from ..config import CACHE_DIR
from .cache import ParquetCache

log = logging.getLogger(__name__)


def _cache_key(kind: str, tickers: list[str], end: datetime, window_days: int) -> str:
    return f"adv_{kind}_{'-'.join(tickers)}_{end.date()}_{window_days}d"


def _fetch_volume_panel(
    tickers: list[str],
    end: datetime,
    window_days: int,
) -> pd.DataFrame:
    """Pull daily OHLCV for the trailing window, return a wide DataFrame
    with columns = tickers and cells = Volume (shares)."""
    # Pull extra history so weekend/holiday gaps don't shrink the window
    start = end - timedelta(days=window_days * 2 + 10)
    data = yf.download(
        tickers, start=start, end=end,
        auto_adjust=False,  # raw volumes; adjusted close not needed here
        progress=False, group_by="ticker", threads=True,
    )
    if data is None or len(data) == 0:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(data.columns.get_level_values(0))
        vols = {t: data[t]["Volume"] for t in tickers if t in level0}
        closes = {t: data[t]["Close"] for t in tickers if t in level0}
    else:
        # Single ticker: columns are plain Open/High/Low/Close/Volume
        vols = {tickers[0]: data["Volume"]}
        closes = {tickers[0]: data["Close"]}
    vol_df = pd.DataFrame(vols).tail(window_days).dropna(how="all")
    close_df = pd.DataFrame(closes).tail(window_days).dropna(how="all")
    # Store as a two-level frame so one parquet holds both
    vol_df.columns = pd.MultiIndex.from_product([["volume"], vol_df.columns])
    close_df.columns = pd.MultiIndex.from_product([["close"], close_df.columns])
    merged = pd.concat([vol_df, close_df], axis=1)
    merged.index = pd.DatetimeIndex(pd.to_datetime(merged.index).values.astype("datetime64[ns]"))
    merged.index.name = "date"
    return merged


def _load_panel_cached(
    tickers: list[str],
    end: datetime,
    window_days: int,
    use_cache: bool,
    cache_max_age_hours: float,
) -> pd.DataFrame:
    tickers = sorted({t.upper().strip() for t in tickers if t.strip()})
    if not tickers:
        raise ValueError("tickers must be non-empty")
    key = _cache_key("panel", tickers, end, window_days)
    cache = ParquetCache(CACHE_DIR)
    if use_cache and cache.is_fresh(key, cache_max_age_hours):
        log.debug("ADV panel cache hit: %s", key)
        panel = cache.get(key)
        # Rebuild MultiIndex if parquet flattened it
        if not isinstance(panel.columns, pd.MultiIndex):
            return _unflatten_panel(panel)
        return panel
    panel = _fetch_volume_panel(tickers, end, window_days)
    if panel.empty:
        raise RuntimeError(f"Empty ADV panel for {tickers}")
    if use_cache:
        # Parquet can't preserve MultiIndex columns directly; flatten for storage
        cache.put(key, _flatten_panel(panel))
    return panel


def _flatten_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Flatten (level, ticker) MultiIndex cols to 'level__ticker' strings for parquet."""
    flat = panel.copy()
    flat.columns = [f"{lvl}__{t}" for lvl, t in panel.columns]
    return flat


def _unflatten_panel(flat: pd.DataFrame) -> pd.DataFrame:
    out = flat.copy()
    out.columns = pd.MultiIndex.from_tuples(
        [tuple(c.split("__", 1)) for c in flat.columns]
    )
    return out


def load_adv_shares(
    tickers: list[str],
    end: datetime | None = None,
    window_days: int = 30,
    use_cache: bool = True,
    cache_max_age_hours: float = 24.0,
) -> dict[str, float]:
    """Return ``{ticker: mean_daily_share_volume}`` over the trailing window.

    yfinance-sourced, parquet-cached. Tickers with insufficient volume
    data return ``NaN``.
    """
    end = end or datetime.today()
    panel = _load_panel_cached(tickers, end, window_days, use_cache, cache_max_age_hours)
    vols = panel["volume"] if ("volume" in panel.columns.get_level_values(0)) else pd.DataFrame()
    out: dict[str, float] = {}
    for t in panel["volume"].columns:
        s = vols[t].dropna()
        out[str(t).upper()] = float(s.mean()) if len(s) >= 5 else float("nan")
    return out


def load_adv_dollars(
    tickers: list[str],
    end: datetime | None = None,
    window_days: int = 30,
    use_cache: bool = True,
    cache_max_age_hours: float = 24.0,
) -> dict[str, float]:
    """Return ``{ticker: mean_daily_dollar_volume}`` over the trailing window."""
    end = end or datetime.today()
    panel = _load_panel_cached(tickers, end, window_days, use_cache, cache_max_age_hours)
    vols = panel["volume"]
    closes = panel["close"]
    out: dict[str, float] = {}
    for t in vols.columns:
        v = vols[t]
        c = closes[t] if t in closes.columns else None
        if c is None:
            out[str(t).upper()] = float("nan")
            continue
        dollar_vol = (v * c).dropna()
        out[str(t).upper()] = float(dollar_vol.mean()) if len(dollar_vol) >= 5 else float("nan")
    return out
