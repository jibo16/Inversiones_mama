"""Parquet-backed cache for market data downloads.

Rationale: yfinance and Ken French downloads are slow and rate-limited. We
persist results as parquet keyed by (source, tickers, date range) and
invalidate by mtime. This keeps development iterations fast and avoids
hammering free APIs.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


_SAFE_KEY = re.compile(r"[^A-Za-z0-9._-]+")


class ParquetCache:
    """Simple disk cache for pandas DataFrames, keyed by user-supplied string.

    Not thread-safe. Fine for single-process dev/backtest use.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        """Filesystem-safe path for this cache key."""
        safe = _SAFE_KEY.sub("_", key)[:180]  # cap length for Windows path limits
        return self.root / f"{safe}.parquet"

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def is_fresh(self, key: str, max_age_hours: float = 24.0) -> bool:
        """True iff the cache entry exists and was written within `max_age_hours`."""
        p = self.path(key)
        if not p.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
        return age < timedelta(hours=max_age_hours)

    def get(self, key: str) -> pd.DataFrame:
        """Load a cached DataFrame. Raises FileNotFoundError if absent."""
        return pd.read_parquet(self.path(key))

    def put(self, key: str, df: pd.DataFrame) -> Path:
        """Write `df` to the cache under `key`. Returns the path."""
        p = self.path(key)
        # parquet does not like unnamed or duplicate indexes; coerce to known form
        if df.index.name is None:
            df = df.copy()
            df.index.name = "index"
        df.to_parquet(p)
        return p

    def invalidate(self, key: str) -> bool:
        """Delete the entry for `key`. Returns True if a file was removed."""
        p = self.path(key)
        if p.exists():
            p.unlink()
            return True
        return False
