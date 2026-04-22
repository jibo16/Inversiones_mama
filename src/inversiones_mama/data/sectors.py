"""Ticker -> sector mapping for the sector-cap constraint in RCK.

The RCK optimizer pre-shrinkage will happily stack 6-7 highest-mu names
into a single sector (observed: SP500 backtest's latest rebalance was
100% in semis/hardware — CIEN, LITE, MU, SATS, STX, WDC, FIX). To
prevent that, :func:`inversiones_mama.sizing.kelly.solve_rck` takes an
optional ``sector_map`` argument with a per-sector weight cap. This
module produces that mapping.

Two data sources:
* **ETFs:** hardcoded ``ETF_SECTORS`` table. Covers every ticker in
  ``config.UNIVERSE`` and ``data.liquid_universe.LIQUID_ETFS``. ETF
  sectors are coarser than GICS (we use buckets like "US Equity Broad"
  or "Broad Commodities") because a 30% bucket cap on something as
  generic as "Equities" would be meaningless.
* **Individual equities:** scraped from Wikipedia's S&P 500 table, same
  source as :func:`fetch_sp500_tickers`. GICS Sector column is used.

Public API
----------
``ETF_SECTORS`` — static ticker -> sector dict covering known ETFs.
``fetch_sp500_sectors()`` — runtime scrape returning ticker -> GICS sector.
``build_sector_map(tickers)`` — combine ETF + SP500 sources; unmapped
    tickers get a conservative "Unknown" bucket (which still participates
    in the sector cap — they're grouped together rather than unconstrained).
"""

from __future__ import annotations

import logging
from typing import Iterable

log = logging.getLogger(__name__)


# --- ETF sector taxonomy ----------------------------------------------------
# Coarser than GICS; organized so that the 30% sector cap creates meaningful
# spread between structurally uncorrelated exposures (equity vs. rates vs.
# commodities, developed vs. EM, credit vs. govt).
ETF_SECTORS: dict[str, str] = {
    # US broad equity
    "SPY": "US Equity Broad", "IVV": "US Equity Broad", "VOO": "US Equity Broad",
    "VTI": "US Equity Broad", "ITOT": "US Equity Broad", "QQQ": "US Equity Broad",
    "DIA": "US Equity Broad", "IWM": "US Equity Broad",
    # US equity factor/style
    "VLUE": "US Equity Factor", "MTUM": "US Equity Factor", "QUAL": "US Equity Factor",
    "USMV": "US Equity Factor", "SPLV": "US Equity Factor",
    "AVUV": "US Equity Factor", "DFAC": "US Equity Factor", "DFAS": "US Equity Factor",
    "VBR": "US Equity Factor", "IJS": "US Equity Factor", "IWN": "US Equity Factor",
    "IWF": "US Equity Factor", "IWD": "US Equity Factor",
    # International developed
    "EFA": "Intl Developed", "VEA": "Intl Developed", "IEFA": "Intl Developed",
    "SCHF": "Intl Developed", "IEUR": "Intl Developed", "EWJ": "Intl Developed",
    "EWG": "Intl Developed", "EWU": "Intl Developed",
    "AVDV": "Intl Developed", "IMTM": "Intl Developed",
    # Emerging markets
    "EEM": "Emerging Markets", "IEMG": "Emerging Markets", "VWO": "Emerging Markets",
    "SCHE": "Emerging Markets", "MCHI": "Emerging Markets", "INDA": "Emerging Markets",
    "EWZ": "Emerging Markets", "AVEM": "Emerging Markets",
    # US Treasury (by duration bucket, but all "govt rates")
    "SHY": "US Treasury", "SHV": "US Treasury", "BIL": "US Treasury",
    "IEF": "US Treasury", "IEI": "US Treasury", "TLT": "US Treasury",
    "GOVT": "US Treasury", "TIP": "US Treasury", "VTIP": "US Treasury",
    # US credit
    "AGG": "US Credit", "BND": "US Credit", "LQD": "US Credit",
    "VCIT": "US Credit", "VCSH": "US Credit", "HYG": "US Credit",
    "JNK": "US Credit", "USHY": "US Credit", "FLOT": "US Credit",
    "MUB": "US Credit", "SUB": "US Credit",
    # International bonds
    "BNDX": "Intl Bonds", "IAGG": "Intl Bonds", "EMB": "Intl Bonds", "PCY": "Intl Bonds",
    # Precious metals
    "GLD": "Precious Metals", "IAU": "Precious Metals", "GLDM": "Precious Metals",
    "SLV": "Precious Metals", "PPLT": "Precious Metals",
    # Commodities
    "DBC": "Broad Commodities", "PDBC": "Broad Commodities", "GSG": "Broad Commodities",
    "USO": "Energy Commodities", "UNG": "Energy Commodities", "BNO": "Energy Commodities",
    "DBA": "Ag Commodities", "CORN": "Ag Commodities",
    # Sector SPDRs
    "XLF": "US Sector Financials", "XLK": "US Sector Technology",
    "XLE": "US Sector Energy", "XLV": "US Sector Healthcare",
    "XLY": "US Sector Cons Discretionary", "XLP": "US Sector Cons Staples",
    "XLI": "US Sector Industrials", "XLB": "US Sector Materials",
    "XLU": "US Sector Utilities", "XLRE": "US Sector Real Estate",
    "XLC": "US Sector Communications",
    # Real estate
    "VNQ": "US Sector Real Estate", "IYR": "US Sector Real Estate",
    "REM": "US Sector Real Estate",
    # Volatility (direct vol exposure)
    "VXX": "Volatility", "SVXY": "Volatility",
}


# --- Wikipedia SP500 sector scrape ------------------------------------------

_SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

_SP500_SECTOR_CACHE: dict[str, str] | None = None  # process-level cache


def fetch_sp500_sectors(timeout: float = 15.0) -> dict[str, str]:
    """Scrape ticker -> GICS sector from Wikipedia's S&P 500 table.

    Uses ``requests`` with a User-Agent header (Wikipedia returns HTTP 403
    to the default urllib UA that pandas uses), then parses the HTML
    string with :func:`pandas.read_html`.

    Process-level cached — subsequent calls in the same process return
    the cached dict without a network round-trip.
    """
    global _SP500_SECTOR_CACHE  # noqa: PLW0603
    if _SP500_SECTOR_CACHE is not None:
        return dict(_SP500_SECTOR_CACHE)

    import io  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415
    import requests  # noqa: PLC0415

    headers = {"User-Agent": "Mozilla/5.0 (compatible; inversiones-mama/0.1; research)"}
    try:
        resp = requests.get(_SP500_WIKIPEDIA_URL, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        log.warning("SP500 sector fetch (requests) failed: %s", exc)
        _SP500_SECTOR_CACHE = {}
        return {}

    try:
        tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})
    except Exception as exc:  # noqa: BLE001
        log.warning("SP500 sector fetch (read_html) failed: %s", exc)
        _SP500_SECTOR_CACHE = {}
        return {}

    if not tables:
        _SP500_SECTOR_CACHE = {}
        return {}

    df = tables[0]
    sym_col = next((c for c in df.columns if str(c).lower() in ("symbol", "ticker")), None)
    sec_col = next(
        (c for c in df.columns if "gics sector" in str(c).lower() or c == "GICS Sector"),
        None,
    )
    if sym_col is None or sec_col is None:
        log.warning("SP500 Wikipedia table is missing symbol/sector columns: %s", list(df.columns))
        _SP500_SECTOR_CACHE = {}
        return {}

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        sym = str(row[sym_col]).replace(".", "-").upper().strip()
        sec = str(row[sec_col]).strip()
        if sym and sec and sec.lower() != "nan":
            mapping[sym] = sec
    _SP500_SECTOR_CACHE = dict(mapping)
    log.info("fetched %d SP500 sector mappings from Wikipedia", len(mapping))
    return dict(mapping)


# --- Combined builder -------------------------------------------------------


def build_sector_map(
    tickers: Iterable[str],
    *,
    include_sp500: bool = True,
) -> dict[str, str]:
    """Return ticker -> sector for every ticker in ``tickers``.

    Lookup order:
      1. :data:`ETF_SECTORS` (hardcoded).
      2. SP500 Wikipedia scrape (if ``include_sp500`` and the ticker isn't
         already resolved).

    **Unmapped tickers are OMITTED from the output** (not bucketed under
    "Unknown"). The downstream kelly solver treats missing entries as
    "no sector constraint for that ticker" — just the per-name cap. This
    avoids the pathological case where a failed Wikipedia fetch causes
    every SP500 ticker to cluster under one bucket and the 30% cap
    collapses the investable universe to 30% of capital.
    """
    out: dict[str, str] = {}
    tickers_list = [str(t).upper().strip() for t in tickers if str(t).strip()]
    needs_sp500 = False
    for t in tickers_list:
        if t in ETF_SECTORS:
            out[t] = ETF_SECTORS[t]
        else:
            needs_sp500 = True

    if needs_sp500 and include_sp500:
        sp500_map = fetch_sp500_sectors()
        for t in tickers_list:
            if t in out:
                continue
            if t in sp500_map:
                out[t] = sp500_map[t]

    return out
