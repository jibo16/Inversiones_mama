"""Tests for inversiones_mama.simulation.bootstrap.

Two classes of tests:

1. **Shape / membership / RNG-determinism tests** (fast, deterministic).
2. **Statistical property tests** (seeded RNG; tolerances sized for 10k samples).

The statistical tests are the substantive ones — they are what distinguishes
a block bootstrap from an IID bootstrap: preservation of autocorrelation,
volatility clustering, and cross-asset correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.simulation.bootstrap import (
    bootstrap_iter,
    circular_block_bootstrap,
    moving_block_bootstrap,
    stationary_bootstrap,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def ar1_series() -> np.ndarray:
    """AR(1) series with strong autocorrelation phi=0.5 — for autocorr tests."""
    rng = np.random.default_rng(123)
    n = 1000
    phi = 0.5
    x = np.zeros(n)
    eps = rng.normal(0, 1, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


@pytest.fixture
def multi_asset_df() -> pd.DataFrame:
    """Two-asset DataFrame with strong cross-correlation."""
    rng = np.random.default_rng(456)
    n = 800
    a = rng.normal(0, 1, n)
    b = 0.8 * a + rng.normal(0, 0.6, n)  # planted correlation ~0.8
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": a, "B": b}, index=idx)


# --------------------------------------------------------------------------- #
# Shape, membership, determinism                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fn", [moving_block_bootstrap, stationary_bootstrap, circular_block_bootstrap])
def test_shape_1d(fn, ar1_series):
    # Positional arg so the same call works for MBB/CBB (block_length) and stationary (mean_block_length)
    out = fn(ar1_series, 5, 60, 10, rng=np.random.default_rng(0))
    assert out.shape == (5, 60)


@pytest.mark.parametrize("fn", [moving_block_bootstrap, stationary_bootstrap, circular_block_bootstrap])
def test_shape_2d(fn, multi_asset_df):
    out = fn(multi_asset_df, 3, 40, 8, rng=np.random.default_rng(0))
    assert out.shape == (3, 40, 2)


@pytest.mark.parametrize("fn", [moving_block_bootstrap, stationary_bootstrap, circular_block_bootstrap])
def test_shape_from_series(fn, multi_asset_df):
    s = multi_asset_df["A"]
    out = fn(s, 2, 30, 5, rng=np.random.default_rng(0))
    assert out.shape == (2, 30)


@pytest.mark.parametrize("fn", [moving_block_bootstrap, stationary_bootstrap, circular_block_bootstrap])
def test_membership_values_in_original(fn, ar1_series):
    """Every bootstrapped value must come from the original series."""
    out = fn(ar1_series, 3, 50, 7, rng=np.random.default_rng(1))
    original_set = set(np.round(ar1_series, 12).tolist())
    resampled_set = set(np.round(out.ravel(), 12).tolist())
    assert resampled_set.issubset(original_set)


@pytest.mark.parametrize("fn", [moving_block_bootstrap, stationary_bootstrap, circular_block_bootstrap])
def test_rng_reproducibility(fn, ar1_series):
    a = fn(ar1_series, 4, 30, 5, rng=np.random.default_rng(42))
    b = fn(ar1_series, 4, 30, 5, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# Error paths                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fn", [moving_block_bootstrap, stationary_bootstrap, circular_block_bootstrap])
def test_bad_horizon(fn, ar1_series):
    with pytest.raises(ValueError, match="horizon"):
        fn(ar1_series, 2, 0, 5)


@pytest.mark.parametrize("fn", [moving_block_bootstrap, stationary_bootstrap, circular_block_bootstrap])
def test_bad_block_length(fn, ar1_series):
    # Both MBB/CBB (block_length) and stationary (mean_block_length) reject <= 0
    with pytest.raises(ValueError, match="block_length"):
        fn(ar1_series, 2, 20, 0)


def test_mbb_block_too_long():
    x = np.arange(10)
    with pytest.raises(ValueError, match="cannot exceed"):
        moving_block_bootstrap(x, n_samples=2, horizon=5, block_length=20)


def test_bad_ndim():
    from inversiones_mama.simulation.bootstrap import _as_2d

    with pytest.raises(ValueError, match="1-D or 2-D"):
        _as_2d(np.zeros((2, 2, 2)))


def test_insufficient_obs():
    with pytest.raises(ValueError, match="at least 2"):
        stationary_bootstrap(np.array([0.01]), n_samples=1, horizon=5)


# --------------------------------------------------------------------------- #
# Circular wrap test                                                          #
# --------------------------------------------------------------------------- #


def test_cbb_wraps_around():
    """Circular bootstrap must produce indices that wrap modulo n_periods."""
    x = np.arange(10, dtype=float)
    # Large horizon relative to series forces wrap
    out = circular_block_bootstrap(x, n_samples=1, horizon=25, block_length=5,
                                    rng=np.random.default_rng(0))
    # All values must be from [0, 9]
    assert set(out.ravel().tolist()).issubset(set(range(10)))


# --------------------------------------------------------------------------- #
# Statistical property tests — the whole point of block bootstrap             #
# --------------------------------------------------------------------------- #


def _autocorr_lag1(x: np.ndarray) -> float:
    """Estimate lag-1 autocorrelation of a 1-D series."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    num = (x[:-1] * x[1:]).sum()
    den = (x * x).sum()
    return float(num / den)


def test_stationary_bootstrap_preserves_mean(ar1_series):
    out = stationary_bootstrap(ar1_series, n_samples=2000, horizon=500,
                                mean_block_length=20, rng=np.random.default_rng(7))
    boot_means = out.mean(axis=1)
    # Mean of bootstrap means should be close to sample mean
    assert abs(boot_means.mean() - ar1_series.mean()) < 0.05


def test_stationary_bootstrap_preserves_volatility(ar1_series):
    out = stationary_bootstrap(ar1_series, n_samples=1000, horizon=500,
                                mean_block_length=20, rng=np.random.default_rng(8))
    boot_stds = out.std(axis=1)
    original_std = ar1_series.std()
    # Mean bootstrap std should be within ~5% of sample std
    assert abs(boot_stds.mean() - original_std) / original_std < 0.05


def test_block_bootstrap_preserves_autocorrelation(ar1_series):
    """The crucial test: block bootstrap must preserve lag-1 autocorr;
    IID bootstrap would destroy it."""
    original_ac = _autocorr_lag1(ar1_series)  # ~0.5
    assert original_ac > 0.4  # sanity: fixture delivers AR(1) as intended

    # Stationary bootstrap with sensible block length
    out = stationary_bootstrap(ar1_series, n_samples=500, horizon=200,
                                mean_block_length=20, rng=np.random.default_rng(9))
    boot_ac = np.array([_autocorr_lag1(path) for path in out])
    # Expect mean bootstrap ac to recover ~60-100% of original ac (block bootstraps
    # lose some dependence at block boundaries). IID bootstrap would give ~0.
    assert boot_ac.mean() > 0.6 * original_ac


def test_iid_bootstrap_destroys_autocorrelation(ar1_series):
    """Contrast test: block_length=1 == IID bootstrap, should kill autocorr."""
    out = moving_block_bootstrap(ar1_series, n_samples=200, horizon=500,
                                  block_length=1, rng=np.random.default_rng(11))
    boot_ac = np.array([_autocorr_lag1(path) for path in out])
    # IID bootstrap: autocorr should center on 0
    assert abs(boot_ac.mean()) < 0.1


def test_stationary_bootstrap_preserves_cross_correlation(multi_asset_df):
    """Multi-asset: cross-asset correlation must survive resampling."""
    orig_corr = multi_asset_df["A"].corr(multi_asset_df["B"])
    assert orig_corr > 0.6  # sanity

    out = stationary_bootstrap(multi_asset_df, n_samples=300, horizon=300,
                                mean_block_length=15, rng=np.random.default_rng(13))
    # For each sample, compute corr between cols 0 and 1
    boot_corrs = np.array([np.corrcoef(p[:, 0], p[:, 1])[0, 1] for p in out])
    # Should be close to original (block bootstrap preserves contemporaneous structure)
    assert abs(boot_corrs.mean() - orig_corr) < 0.05


def test_mbb_block_length_concentrates_autocorrelation():
    """Larger block size -> closer recovery of lag-1 autocorr."""
    rng = np.random.default_rng(17)
    n = 2000
    phi = 0.7
    x = np.zeros(n)
    eps = rng.normal(0, 1, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]

    ac_orig = _autocorr_lag1(x)
    ac_b5 = np.array([
        _autocorr_lag1(p)
        for p in moving_block_bootstrap(x, 200, 500, block_length=5, rng=np.random.default_rng(0))
    ]).mean()
    ac_b50 = np.array([
        _autocorr_lag1(p)
        for p in moving_block_bootstrap(x, 200, 500, block_length=50, rng=np.random.default_rng(0))
    ]).mean()
    # Longer blocks preserve more autocorr
    assert ac_b50 > ac_b5
    # b50 should be close to original
    assert abs(ac_b50 - ac_orig) < 0.10


# --------------------------------------------------------------------------- #
# Generator variant                                                           #
# --------------------------------------------------------------------------- #


def test_bootstrap_iter_shapes(ar1_series):
    rng = np.random.default_rng(0)
    it = bootstrap_iter(ar1_series, horizon=40, block_length=10, rng=rng, method="stationary")
    s1 = next(it)
    s2 = next(it)
    assert s1.shape == (40,)
    assert s2.shape == (40,)
    # Successive draws must differ (probabilistic; overwhelmingly likely)
    assert not np.array_equal(s1, s2)


def test_bootstrap_iter_2d(multi_asset_df):
    rng = np.random.default_rng(0)
    it = bootstrap_iter(multi_asset_df, horizon=30, block_length=5, rng=rng, method="moving")
    sample = next(it)
    assert sample.shape == (30, 2)


def test_bootstrap_iter_unknown_method(ar1_series):
    with pytest.raises(ValueError, match="unknown method"):
        next(bootstrap_iter(ar1_series, horizon=10, method="bogus"))
