"""Tests for inversiones_mama.data.cache — no network."""

from __future__ import annotations

import os
import time

import pandas as pd
import pytest

from inversiones_mama.data.cache import ParquetCache


def test_cache_put_get_roundtrip(tmp_path):
    cache = ParquetCache(tmp_path)
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    cache.put("foo", df)
    assert cache.exists("foo")
    loaded = cache.get("foo")
    pd.testing.assert_frame_equal(df, loaded, check_names=False)


def test_cache_unsafe_key_sanitized(tmp_path):
    """Keys with slashes or special chars should produce a valid filename."""
    cache = ParquetCache(tmp_path)
    df = pd.DataFrame({"a": [1.0]})
    path = cache.put("weird/key:with*chars", df)
    assert path.exists()
    # The file name must not contain problematic chars
    assert not any(c in path.name for c in ['/', ':', '*', '\\'])


def test_cache_freshness_fresh(tmp_path):
    cache = ParquetCache(tmp_path)
    cache.put("bar", pd.DataFrame({"a": [1.0]}))
    assert cache.is_fresh("bar", max_age_hours=1)


def test_cache_freshness_stale(tmp_path):
    cache = ParquetCache(tmp_path)
    cache.put("bar", pd.DataFrame({"a": [1.0]}))
    # Age the file by rewriting mtime to 2h ago
    p = cache.path("bar")
    old = time.time() - 7200
    os.utime(p, (old, old))
    assert not cache.is_fresh("bar", max_age_hours=1)


def test_cache_freshness_missing(tmp_path):
    cache = ParquetCache(tmp_path)
    assert not cache.is_fresh("never-put", max_age_hours=1)


def test_cache_invalidate(tmp_path):
    cache = ParquetCache(tmp_path)
    cache.put("baz", pd.DataFrame({"a": [1.0]}))
    assert cache.exists("baz")
    assert cache.invalidate("baz")
    assert not cache.exists("baz")
    # Invalidating a missing key returns False, not an exception
    assert cache.invalidate("baz") is False


def test_cache_preserves_index_name(tmp_path):
    cache = ParquetCache(tmp_path)
    idx = pd.date_range("2024-01-01", periods=5)
    idx.name = "date"
    df = pd.DataFrame({"SPY": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    cache.put("dated", df)
    loaded = cache.get("dated")
    assert loaded.index.name == "date"


def test_cache_coerces_unnamed_index(tmp_path):
    """Parquet dislikes unnamed indexes — cache should handle gracefully."""
    cache = ParquetCache(tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3]})  # default RangeIndex, no name
    cache.put("unnamed", df)
    loaded = cache.get("unnamed")
    # Values round-trip even if index is renamed
    pd.testing.assert_series_equal(df["a"].reset_index(drop=True), loaded["a"].reset_index(drop=True))
