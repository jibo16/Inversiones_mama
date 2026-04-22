"""Kenneth French data library loader — Fama-French 5 factors + Momentum.

Pulls daily factor returns directly from Dartmouth's data library,
parses the idiosyncratic Ken French CSV format (description header,
daily data section, optional annual-averages tail), converts percentages
to decimals, and caches the merged 6-factor panel as parquet.

We avoid pandas-datareader because Python 3.13 removed ``distutils`` and
pandas-datareader breaks on import as of 2026-04.

Public API
----------
fetch_ff5f_daily() -> DataFrame
    Daily returns for Mkt-RF, SMB, HML, RMW, CMA, RF (decimals).
fetch_momentum_daily() -> DataFrame
    Daily returns for MOM (a.k.a. UMD) (decimals).
load_factor_returns(start, end, use_cache) -> DataFrame
    Merged 6-factor panel (Mkt-RF, SMB, HML, RMW, CMA, MOM, RF).

Source
------
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from ..config import CACHE_DIR
from .cache import ParquetCache

log = logging.getLogger(__name__)

# Dartmouth Ken French library — daily factor zips
FF5F_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)

# Ken French's server returns 403 to clients without a User-Agent. Any plausible
# UA works; we identify this project so admins can trace traffic if needed.
_UA = "Mozilla/5.0 (compatible; inversiones-mama/0.1.0; research)"


def _download_zip(url: str, timeout: float = 30.0) -> bytes:
    """GET a zip from the Ken French library, returning raw bytes."""
    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _extract_csv_text(zip_bytes: bytes) -> str:
    """Read the single CSV out of a Ken French zip and decode it."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("No CSV inside Ken French zip")
        with zf.open(csv_names[0]) as f:
            # Ken French uses latin-1 for occasional special chars in descriptions
            return f.read().decode("latin-1")


def parse_ken_french_csv(text: str) -> pd.DataFrame:
    """Parse Ken French's daily CSV into a DataFrame (decimals).

    Format:
        <description lines...>
        <blank>
        ,<Factor1>,<Factor2>,...
        <YYYYMMDD>,<val1>,<val2>,...
        <YYYYMMDD>,<val1>,<val2>,...
        ...
        <blank>        <- terminates the daily section
        <annual averages or other sections — we ignore>

    Values are published in percent (e.g., 0.50 means 0.5%); we divide by 100.
    """
    lines = text.splitlines()

    # Find the header row: first non-empty line starting with "," and containing letters
    header_idx: int | None = None
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(",") and any(ch.isalpha() for ch in line):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not locate Ken French CSV header row")

    columns = [c.strip() for c in lines[header_idx].split(",")]
    columns[0] = "date"  # first column is the empty date placeholder

    rows: list[list[str]] = []
    for raw in lines[header_idx + 1 :]:
        line = raw.strip()
        if not line:
            break  # blank line ends the daily section; ignore subsequent sections
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(columns):
            continue
        if not parts[0].isdigit():
            continue  # skip section headers that might slip through
        rows.append(parts)

    if not rows:
        raise RuntimeError("No data rows parsed from Ken French CSV")

    df = pd.DataFrame(rows, columns=columns)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float) / 100.0  # pct -> decimal
    df.index.name = "date"
    return df


def fetch_ff5f_daily() -> pd.DataFrame:
    """Download and parse the daily FF 5-factor file.

    Columns: Mkt-RF, SMB, HML, RMW, CMA, RF. Decimals.
    """
    log.info("downloading FF5F daily from Ken French library...")
    return parse_ken_french_csv(_extract_csv_text(_download_zip(FF5F_URL)))


def fetch_momentum_daily() -> pd.DataFrame:
    """Download and parse the daily momentum factor file.

    The published column is "Mom"; normalized to "MOM" for downstream code.
    """
    log.info("downloading momentum daily from Ken French library...")
    df = parse_ken_french_csv(_extract_csv_text(_download_zip(MOM_URL)))
    df = df.rename(columns={c: "MOM" for c in df.columns if c.lower() == "mom"})
    return df


def load_factor_returns(
    start: datetime | None = None,
    end: datetime | None = None,
    use_cache: bool = True,
    cache_max_age_hours: float = 24.0 * 7,  # factors update rarely; week cache OK
) -> pd.DataFrame:
    """Load the merged 6-factor panel (FF5F + MOM) with caching.

    Returns a DataFrame indexed by date with columns:
        Mkt-RF, SMB, HML, RMW, CMA, MOM, RF

    Parameters
    ----------
    start, end : optional datetime filters. If both None, full history is returned.
    use_cache : serve from parquet cache if fresh.
    cache_max_age_hours : refresh if cache older than this.
    """
    key = "ff6f_daily_latest"
    cache = ParquetCache(CACHE_DIR)

    if use_cache and cache.is_fresh(key, cache_max_age_hours):
        df = cache.get(key)
    else:
        ff5 = fetch_ff5f_daily()
        mom = fetch_momentum_daily()
        # MOM history is typically shorter than FF5F; inner join keeps only common dates
        df = ff5.join(mom, how="inner")
        if use_cache:
            cache.put(key, df)
            df = cache.get(key)  # consistency with prices.py pattern

    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df
