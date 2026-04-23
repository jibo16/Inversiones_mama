"""Hierarchical Risk Parity (López de Prado 2016).

Traditional Markowitz mean-variance optimization requires inverting the
covariance matrix, which becomes numerically unstable when the asset
count approaches or exceeds the observation count (the curse of
dimensionality + Markowitz's error-maximization). On the 1,512-ticker
super-wide universe, mean-variance collapses.

HRP sidesteps the covariance inversion entirely via three steps:

1. **Tree clustering.** Map the correlation matrix to a distance
   matrix ``d_{i,j} = sqrt(0.5 * (1 - rho_{i,j}))`` and run
   agglomerative hierarchical clustering.
2. **Quasi-diagonalization.** Reorder the covariance matrix so highly
   correlated assets sit near the diagonal (traversing the linkage
   tree in dendrogram order).
3. **Recursive bisection.** Top-down, at each split allocate capital
   inversely proportional to the variance of each sub-cluster. No
   matrix inversion; the result is robust even when ``Sigma`` is
   singular.

Reference:
    López de Prado, M. (2016). Building Diversified Portfolios that
    Outperform Out-of-Sample. Journal of Portfolio Management 42(4).

Public API
----------
``hrp_weights(returns, linkage_method='single') -> pd.Series``
    Main entry point. Takes a (T x N) returns DataFrame, returns an
    (N,) weight Series summing to 1, indexed by ticker.

``hrp_cluster_order(corr) -> list[int]``
    Quasi-diagonal ordering of the correlation matrix (for inspection
    / diagnostics).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """d_{i,j} = sqrt(0.5 * (1 - rho_{i,j})).  Zero on the diagonal."""
    # Clip to [-1, 1] to guard against floating-point drift.
    c = np.clip(corr.values, -1.0, 1.0)
    d = np.sqrt(0.5 * (1.0 - c))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def _quasi_diag(link_matrix: np.ndarray) -> list[int]:
    """Return the leaf order produced by traversing the linkage tree.

    Mirrors López de Prado Snippet 16.2.
    """
    link = link_matrix.astype(int)
    n = link.shape[0] + 1  # number of original observations
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])

    while sort_ix.max() >= n:
        sort_ix.index = range(0, 2 * len(sort_ix), 2)  # make room
        df0 = sort_ix[sort_ix >= n]                    # find cluster labels
        i = df0.index
        j = df0.values - n
        sort_ix[i] = link[j, 0]                        # replace label with left child
        df1 = pd.Series(link[j, 1], index=i + 1)       # and right child one slot over
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _inverse_variance_portfolio(cov_slice: pd.DataFrame) -> pd.Series:
    """Inverse-variance weights for one sub-cluster (Snippet 16.4 helper)."""
    ivp = 1.0 / np.diag(cov_slice.values)
    ivp /= ivp.sum()
    return pd.Series(ivp, index=cov_slice.index)


def _cluster_variance(cov: pd.DataFrame, cluster_items: list) -> float:
    """Variance of an inverse-variance weighted sub-cluster."""
    cov_slice = cov.loc[cluster_items, cluster_items]
    w = _inverse_variance_portfolio(cov_slice).values.reshape(-1, 1)
    return float((w.T @ cov_slice.values @ w)[0, 0])


def _recursive_bisection(cov: pd.DataFrame, sort_ix: list) -> pd.Series:
    """Compute HRP weights via recursive bisection (Snippet 16.4)."""
    weights = pd.Series(1.0, index=sort_ix)
    clusters: list[list] = [sort_ix]
    while clusters:
        # Bi-section each cluster. Stop when a cluster has only 1 asset.
        new_clusters: list[list] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            half = len(cluster) // 2
            c_left = cluster[:half]
            c_right = cluster[half:]
            var_left = _cluster_variance(cov, c_left)
            var_right = _cluster_variance(cov, c_right)
            # Inverse-variance split: larger-variance cluster gets less weight
            alpha = 1.0 - var_left / (var_left + var_right)
            weights.loc[c_left] *= alpha
            weights.loc[c_right] *= (1.0 - alpha)
            new_clusters.append(c_left)
            new_clusters.append(c_right)
        clusters = new_clusters
    return weights


def hrp_cluster_order(corr: pd.DataFrame, linkage_method: str = "single") -> list[int]:
    """Return the integer positions of assets in quasi-diagonal order.

    Useful for visualising the clustered structure or reordering the
    covariance matrix for diagnostics.
    """
    dist = _correlation_distance(corr)
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=linkage_method)
    return _quasi_diag(Z)


def hrp_weights(
    returns: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """Compute HRP portfolio weights from a returns DataFrame.

    Parameters
    ----------
    returns : DataFrame (T x N)
        Asset returns (columns = tickers). NaN-safe: columns with fewer
        than 2 non-null observations are dropped.
    linkage_method : str
        Scipy linkage method. López de Prado uses ``"single"``; other
        common choices are ``"ward"``, ``"average"``, ``"complete"``.

    Returns
    -------
    pd.Series
        Weights in [0, 1], indexed by ticker, summing to 1.
    """
    if returns.shape[1] < 2:
        raise ValueError(f"HRP needs >= 2 assets, got {returns.shape[1]}")

    # Drop assets with no variance (constant returns / all zeros) to keep
    # the correlation matrix well-defined.
    variances = returns.var()
    live = variances[variances > 0].index.tolist()
    if len(live) < 2:
        raise ValueError(f"HRP needs >= 2 assets with nonzero variance, got {len(live)}")
    rets = returns[live].dropna(how="any")
    if len(rets) < 2:
        raise ValueError(
            f"HRP needs >= 2 aligned rows after dropping NaN, got {len(rets)}"
        )

    corr = rets.corr()
    cov = rets.cov()

    sort_ix_int = hrp_cluster_order(corr, linkage_method=linkage_method)
    sort_tickers = [corr.index[i] for i in sort_ix_int]

    weights = _recursive_bisection(cov, sort_tickers)

    # Re-index to the caller's original column order with zero for any
    # asset we dropped for variance/NaN reasons.
    out = pd.Series(0.0, index=returns.columns)
    out.loc[weights.index] = weights.values
    # Normalise to sum-to-1 (numerical safety).
    total = out.sum()
    if total > 0:
        out = out / total
    return out
