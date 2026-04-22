"""Macro-economic time series loader (FRED-sourced, zero budget).

FRED publishes free CSVs at a stable URL per series; no API key needed
for the standard macroeconomic indicators we need to regress our 6
factors against. We pull a curated set of variables that cover the
dimensions that define the Fed's CCAR Severely Adverse scenario:
growth, labor, inflation, rate curve, credit, and risk.

Public API
----------
``FRED_MACRO_SERIES`` — canonical series id map used by the CCAR stress.
``fetch_fred_series(series_id, start, end) -> pd.Series``
``load_macro_panel(series_ids, start, end, freq='QE') -> pd.DataFrame``
    Quarterly-resampled panel. Most series are monthly or daily; we
    resample-to-last-observation so each quarter ends with the most
    recent known value.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

from ..config import CACHE_DIR
from .cache import ParquetCache

log = logging.getLogger(__name__)

# Canonical FRED series ids for the CCAR Severely Adverse regression.
# The set is a deliberate subset of the Fed's 28-variable scenario —
# the ones that map cleanly onto the 6 factors in our model.
#
# "sp500" (FRED series id SP500) is INTENTIONALLY EXCLUDED:
# 1. FRED's public SP500 daily series starts in 2015, which made the
#    regression window collapse to 38 quarters post-NaN-drop — dropping
#    the entire 2008 crisis out of the fit. The stress-test betas were
#    therefore artificially tame.
# 2. The series is a near-dual of the Mkt-RF factor we're regressing
#    onto. Keeping it would induce data leakage (trivial self-explanation
#    of equity returns by contemporaneous equity levels).
# The CCAR scenario's sp500 path is still used for reporting/display;
# the regression projects Mkt-RF via unemployment + credit spreads + VIX
# instead.
FRED_MACRO_SERIES: dict[str, str] = {
    "real_gdp":          "GDPC1",        # real GDP (quarterly, $B chained)  since 1947
    "unemployment":      "UNRATE",       # civilian unemployment (monthly, %) since 1948
    "cpi":               "CPIAUCSL",     # CPI-U all urban (monthly, index)   since 1947
    "treasury_3m":       "DGS3MO",       # 3-month Treasury (daily, %)        since 1981
    "treasury_10y":      "DGS10",        # 10-year Treasury (daily, %)        since 1962
    "baa_10y_spread":    "BAA10Y",       # Moody's BAA - 10y (daily, %)       since 1986
    "vix":               "VIXCLS",       # CBOE VIX (daily, level)            since 1990
    "home_price_index":  "CSUSHPINSA",   # Case-Shiller US HPI (monthly)      since 1987
    "dxy":               "DTWEXBGS",     # trade-weighted USD (daily, index)  since 2006
}


# --------------------------------------------------------------------------- #
# Fetch                                                                       #
# --------------------------------------------------------------------------- #


_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(
    series_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
    timeout: float = 30.0,
) -> pd.Series:
    """Download a single FRED series as a pandas Series.

    Uses the public fredgraph.csv endpoint — no API key needed for the
    macroeconomic series we care about. Returns the series indexed by
    observation date, with non-numeric placeholders (``.``) converted to
    NaN.
    """
    params: dict[str, str] = {"id": series_id}
    if start is not None:
        params["cosd"] = start.date().isoformat()
    if end is not None:
        params["coed"] = end.date().isoformat()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; inversiones-mama/0.1; research)"}
    try:
        resp = requests.get(_FRED_CSV_URL, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"FRED fetch failed for {series_id}: {exc}") from exc

    df = pd.read_csv(io.StringIO(resp.text))
    # FRED CSV columns: "observation_date" (or "DATE" in older dumps), <SERIES_ID>
    date_col = next(
        (c for c in df.columns if c.lower() in ("observation_date", "date")),
        df.columns[0],
    )
    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        raise RuntimeError(f"FRED response has no value column for {series_id}")
    value_col = value_cols[0]

    df[date_col] = pd.to_datetime(df[date_col])
    # FRED uses "." for missing values
    s = pd.to_numeric(df[value_col].replace(".", pd.NA), errors="coerce")
    out = pd.Series(s.values, index=df[date_col].values, name=series_id)
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index).values.astype("datetime64[ns]"))
    out.index.name = "date"
    return out.sort_index().dropna()


def load_macro_panel(
    series_map: dict[str, str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    freq: str = "QE",
    use_cache: bool = True,
    cache_max_age_hours: float = 24.0 * 7,  # macros update slowly; weekly cache is fine
) -> pd.DataFrame:
    """Load a quarterly-resampled macro panel.

    Parameters
    ----------
    series_map : optional override of {column_name: fred_series_id}.
        Defaults to ``FRED_MACRO_SERIES``.
    start, end : optional date range.
    freq : pandas resample frequency. ``"QE"`` = quarter-end (default).
        ``"ME"`` = month-end, ``"A"`` = annual. Daily series are
        resampled to last-observation-per-period.
    use_cache / cache_max_age_hours : parquet cache controls.
    """
    series_map = series_map or FRED_MACRO_SERIES
    cache = ParquetCache(CACHE_DIR)

    # Cache key bakes in the series ids + date window + freq
    sorted_keys = sorted(series_map.items())
    key = (
        f"fred_macro_{freq}_"
        f"{'-'.join(f'{k}={v}' for k, v in sorted_keys)}_"
        f"{start.date() if start else 'nostart'}_{end.date() if end else 'noend'}"
    )

    if use_cache and cache.is_fresh(key, cache_max_age_hours):
        log.debug("macro panel cache hit: %s", key)
        return cache.get(key)

    columns: dict[str, pd.Series] = {}
    for col_name, fred_id in series_map.items():
        log.info("fetching FRED series %s -> %s", fred_id, col_name)
        try:
            s = fetch_fred_series(fred_id, start=start, end=end)
        except RuntimeError as exc:
            log.warning("skipping %s (%s)", col_name, exc)
            continue
        columns[col_name] = s

    if not columns:
        raise RuntimeError("all FRED fetches failed; check network")

    raw = pd.DataFrame(columns)
    # Resample to requested frequency — last observation per period so we
    # use the most recent known value at quarter-end.
    resampled = raw.resample(freq).last().dropna(how="all")
    resampled.index.name = "date"

    if use_cache:
        cache.put(key, resampled)
        # Reload through the cache so subsequent calls yield identical dtype
        return cache.get(key)
    return resampled
