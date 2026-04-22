"""Shared pytest fixtures for all Inversiones_mama tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_returns_df() -> pd.DataFrame:
    """Deterministic sample returns: 3 assets, 60 business days."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    return pd.DataFrame(
        {
            "A": rng.normal(0.0005, 0.012, 60),
            "B": rng.normal(0.0003, 0.008, 60),
            "C": rng.normal(0.0008, 0.018, 60),
        },
        index=idx,
    )


@pytest.fixture
def sample_prices_df(sample_returns_df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic sample prices derived from sample_returns_df."""
    prices = (1.0 + sample_returns_df).cumprod() * 100.0
    prices.index.name = "date"
    return prices
