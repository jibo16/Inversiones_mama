"""Unit tests for Hierarchical Risk Parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.sizing.hrp import (
    _correlation_distance,
    _inverse_variance_portfolio,
    hrp_cluster_order,
    hrp_weights,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20260422)


def _gen_block_correlated_returns(rng, n_assets_per_block: int = 5,
                                    n_blocks: int = 4, n_obs: int = 1000,
                                    intra_rho: float = 0.7) -> pd.DataFrame:
    """T x N returns where assets cluster into `n_blocks` of size
    `n_assets_per_block` with intra-block correlation ``intra_rho``.
    """
    n = n_blocks * n_assets_per_block
    cov = np.eye(n) * 0.01
    for b in range(n_blocks):
        lo = b * n_assets_per_block
        hi = lo + n_assets_per_block
        for i in range(lo, hi):
            for j in range(lo, hi):
                if i != j:
                    cov[i, j] = intra_rho * 0.01
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(n))
    z = rng.standard_normal((n_obs, n))
    rets = z @ L.T
    cols = [f"B{b}_A{a}" for b in range(n_blocks) for a in range(n_assets_per_block)]
    return pd.DataFrame(rets, columns=cols)


# --- _correlation_distance -------------------------------------------------


def test_correlation_distance_zero_on_diagonal():
    corr = pd.DataFrame(np.eye(3), columns=list("ABC"), index=list("ABC"))
    d = _correlation_distance(corr)
    np.testing.assert_allclose(np.diag(d.values), 0.0, atol=1e-12)


def test_correlation_distance_symmetric():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.standard_normal((200, 4)), columns=list("ABCD"))
    corr = r.corr()
    d = _correlation_distance(corr)
    np.testing.assert_allclose(d.values, d.values.T, atol=1e-12)


def test_correlation_distance_bounded_on_perfect_correlation():
    """rho=1 -> d=0, rho=-1 -> d=1."""
    corr = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]],
                        columns=list("AB"), index=list("AB"))
    d = _correlation_distance(corr)
    np.testing.assert_allclose(d.values, 0.0, atol=1e-12)

    corr_neg = pd.DataFrame([[1.0, -1.0], [-1.0, 1.0]],
                            columns=list("AB"), index=list("AB"))
    d2 = _correlation_distance(corr_neg)
    assert d2.loc["A", "B"] == pytest.approx(1.0)
    assert d2.loc["B", "A"] == pytest.approx(1.0)


# --- _inverse_variance_portfolio ------------------------------------------


def test_ivp_equal_weight_on_equal_variance():
    cov = pd.DataFrame(0.01 * np.eye(4),
                       columns=list("ABCD"), index=list("ABCD"))
    w = _inverse_variance_portfolio(cov)
    np.testing.assert_allclose(w.values, [0.25, 0.25, 0.25, 0.25])


def test_ivp_downweights_high_variance_asset():
    cov = pd.DataFrame(np.diag([0.01, 0.04]),  # B 4x more volatile
                       columns=list("AB"), index=list("AB"))
    w = _inverse_variance_portfolio(cov)
    assert w["A"] > w["B"]
    # Inverse-variance ratio 4:1 -> weights 0.8 / 0.2
    assert w["A"] == pytest.approx(0.8)
    assert w["B"] == pytest.approx(0.2)


# --- hrp_cluster_order -----------------------------------------------------


def test_cluster_order_returns_permutation(rng):
    rets = _gen_block_correlated_returns(rng)
    corr = rets.corr()
    order = hrp_cluster_order(corr)
    assert sorted(order) == list(range(len(corr)))


def test_cluster_order_keeps_blocks_contiguous(rng):
    """On deliberately-blocked correlation matrix, same-block assets
    should be contiguous in the quasi-diagonal order."""
    rets = _gen_block_correlated_returns(
        rng, n_assets_per_block=5, n_blocks=3, intra_rho=0.9,
    )
    corr = rets.corr()
    order = hrp_cluster_order(corr)
    tickers = [corr.index[i] for i in order]
    block_ids = [t.split("_")[0] for t in tickers]
    # Each block's 5 members should form a single contiguous run
    run_lengths = []
    prev = None
    current_len = 0
    for b in block_ids:
        if b != prev:
            if current_len > 0:
                run_lengths.append(current_len)
            current_len = 1
            prev = b
        else:
            current_len += 1
    run_lengths.append(current_len)
    # We expect exactly 3 runs, each of length 5
    assert sorted(run_lengths, reverse=True)[:3] == [5, 5, 5]


# --- hrp_weights -----------------------------------------------------------


def test_hrp_weights_sum_to_one(rng):
    rets = _gen_block_correlated_returns(rng)
    w = hrp_weights(rets)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_hrp_weights_nonnegative_and_bounded(rng):
    rets = _gen_block_correlated_returns(rng)
    w = hrp_weights(rets)
    assert (w >= 0).all()
    assert (w <= 1).all()


def test_hrp_weights_index_matches_input_columns(rng):
    rets = _gen_block_correlated_returns(rng)
    w = hrp_weights(rets)
    assert list(w.index) == list(rets.columns)


def test_hrp_weights_reproducible(rng):
    rets = _gen_block_correlated_returns(rng)
    w1 = hrp_weights(rets)
    w2 = hrp_weights(rets)
    np.testing.assert_allclose(w1.values, w2.values, atol=1e-12)


def test_hrp_weights_equal_variance_roughly_equal_weights(rng):
    """On 4 uncorrelated assets with identical variance, HRP should
    produce weights close to 0.25 each."""
    rets = pd.DataFrame(
        rng.standard_normal((500, 4)) * 0.02,
        columns=list("ABCD"),
    )
    w = hrp_weights(rets)
    # Allow reasonable slop: HRP isn't exactly equal-weight due to sample
    # correlations, but should be within 0.15 of 0.25 on each asset.
    for t in w.index:
        assert abs(w[t] - 0.25) < 0.15, f"{t} weight = {w[t]:.4f}"


def test_hrp_rejects_single_asset():
    rets = pd.DataFrame({"A": np.random.randn(100)})
    with pytest.raises(ValueError, match="needs >= 2 assets"):
        hrp_weights(rets)


def test_hrp_handles_constant_asset(rng):
    """An asset with zero variance should get weight 0 without crashing."""
    rets = _gen_block_correlated_returns(rng)
    rets["ZERO"] = 0.0
    w = hrp_weights(rets)
    assert w["ZERO"] == pytest.approx(0.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_hrp_downweights_high_vol_cluster(rng):
    """Build two clusters, one with 4x the variance. The low-vol
    cluster should receive a larger weight allocation in aggregate."""
    n = 600
    low_vol = rng.standard_normal((n, 4)) * 0.01
    high_vol = rng.standard_normal((n, 4)) * 0.02
    # Add intra-cluster correlation by mixing a shared factor
    factor_low = rng.standard_normal((n, 1)) * 0.005
    factor_high = rng.standard_normal((n, 1)) * 0.01
    low_vol = low_vol + factor_low
    high_vol = high_vol + factor_high
    cols = [f"L{i}" for i in range(4)] + [f"H{i}" for i in range(4)]
    rets = pd.DataFrame(np.hstack([low_vol, high_vol]), columns=cols)
    w = hrp_weights(rets)
    low_w = w.filter(like="L").sum()
    high_w = w.filter(like="H").sum()
    assert low_w > high_w, f"low_w={low_w:.3f} vs high_w={high_w:.3f}"
