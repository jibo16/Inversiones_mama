"""Covariance estimators — sample + Ledoit-Wolf shrinkage.

The v1a 10-ETF universe has 252 observations on 10 assets — sample
covariance is fine there. As we scale toward the v2 mandate (up to
thousands of stocks) the sample covariance becomes ill-conditioned:
its smallest eigenvalues under-estimate true values, meaning Kelly
(which uses Σ⁻¹) aggressively allocates to the noisiest, most
unreliable dimensions. This is Markowitz's curse.

Ledoit-Wolf (2003) shrinks the sample covariance toward a structured
target, picking the shrinkage intensity that minimizes expected
Frobenius distance to the true covariance. No hyperparameter tuning;
no cross-validation; closed-form answer.

This module provides two targets:

* **Diagonal target** — shrink toward ``(trace(S)/p) · I``. Simplest,
  very robust, produces an always-PSD estimator. Use when nothing is
  known about factor structure.

* **Constant-correlation target** — shrink toward a matrix with the
  sample variances on the diagonal and the average sample correlation
  off-diagonal. Better when assets cluster around a single latent
  market factor (typical for liquid equities).

Public API
----------
``sample_covariance(returns)``
``ledoit_wolf_diagonal(returns) -> (Sigma_shrunk, delta)``
``ledoit_wolf_constant_correlation(returns) -> (Sigma_shrunk, delta)``
``estimate_covariance(returns, method) -> DataFrame``   — dispatcher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _as_centered_array(returns: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    arr = np.asarray(returns, dtype=np.float64)
    # Drop rows with NaNs (common when some tickers have shorter history)
    mask = ~np.isnan(arr).any(axis=1)
    arr = arr[mask]
    n, p = arr.shape
    if n < 5 or p < 1:
        raise ValueError(f"returns too small to estimate covariance: n={n}, p={p}")
    centered = arr - arr.mean(axis=0, keepdims=True)
    return centered, n, p


def _clip_delta(delta: float) -> float:
    """Keep shrinkage intensity in [0, 1]; guard against numerical overflow."""
    if not np.isfinite(delta):
        return 0.0
    return float(max(0.0, min(1.0, delta)))


# --------------------------------------------------------------------------- #
# Estimators                                                                  #
# --------------------------------------------------------------------------- #


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Standard sample covariance. Equivalent to ``returns.cov()`` (after dropna)."""
    Xc, n, _ = _as_centered_array(returns)
    S = (Xc.T @ Xc) / (n - 1)
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


def ledoit_wolf_diagonal(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf shrinkage toward the constant-mean-variance diagonal target.

    Target F = (trace(S)/p) · I  — a diagonal matrix with equal variances.
    Optimal shrinkage (Ledoit & Wolf, 2004, "A well-conditioned estimator
    for large-dimensional covariance matrices"):

        delta* = min(1, b² / d²)

    where
        d² = ||S - F||_F²   (distance from sample to target)
        b² = (1/n²) · Σ_t ||x_t x_tᵀ - S||_F²,  clipped to ≤ d².

    Returns
    -------
    (Sigma_shrunk, delta) where ``Sigma_shrunk = delta * F + (1 - delta) * S``.
    """
    Xc, n, p = _as_centered_array(returns)
    S = (Xc.T @ Xc) / (n - 1)
    # Diagonal target
    target_var = float(np.trace(S)) / p
    F = target_var * np.eye(p)

    # d² = ||S - F||_F²
    d2 = float(np.linalg.norm(S - F, ord="fro") ** 2)
    if d2 <= 0:
        return pd.DataFrame(S, index=returns.columns, columns=returns.columns), 0.0

    # b² = (1/n²) * sum_t ||x_t x_tᵀ - S||_F²
    # Vectorized: for each row t, compute outer product x_t x_tᵀ and squared Frobenius distance to S.
    # Efficient form: b² = (1/n²) * sum_t (||x_t||⁴ - 2 x_tᵀ S x_t + ||S||_F²) expanded properly.
    # We use the direct sum but avoid materializing p×p per row.
    # ||x_t x_tᵀ - S||_F² = ||x_t||⁴ - 2 x_tᵀ S x_t + ||S||_F²
    sq_norms = (Xc * Xc).sum(axis=1)  # shape (n,)
    x_S_x = np.einsum("ti,ij,tj->t", Xc, S, Xc)  # shape (n,)
    S_fro2 = float(np.linalg.norm(S, ord="fro") ** 2)
    per_row = sq_norms**2 - 2.0 * x_S_x + S_fro2
    b2 = float(per_row.sum()) / (n**2)
    b2 = min(b2, d2)

    delta = _clip_delta(b2 / d2)
    Sigma = delta * F + (1.0 - delta) * S
    return (
        pd.DataFrame(Sigma, index=returns.columns, columns=returns.columns),
        delta,
    )


def ledoit_wolf_constant_correlation(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf shrinkage toward a constant-correlation target.

    Target F has S_ii on its diagonal and (r_bar · sqrt(S_ii S_jj)) off-diagonal,
    where r_bar is the average off-diagonal sample correlation. Usually
    better for equity portfolios than the pure diagonal target because it
    preserves the dominant market-factor structure.

    Uses the same delta* = min(1, b² / d²) rule with the same b² surrogate.
    """
    Xc, n, p = _as_centered_array(returns)
    S = (Xc.T @ Xc) / (n - 1)
    variances = np.diag(S).copy()
    std = np.sqrt(np.maximum(variances, 1e-30))
    corr = S / np.outer(std, std)
    # Average off-diagonal correlation
    if p > 1:
        off_diag_mask = ~np.eye(p, dtype=bool)
        r_bar = float(corr[off_diag_mask].mean())
    else:
        r_bar = 0.0
    F = r_bar * np.outer(std, std)
    np.fill_diagonal(F, variances)

    d2 = float(np.linalg.norm(S - F, ord="fro") ** 2)
    if d2 <= 0:
        return pd.DataFrame(S, index=returns.columns, columns=returns.columns), 0.0

    sq_norms = (Xc * Xc).sum(axis=1)
    x_S_x = np.einsum("ti,ij,tj->t", Xc, S, Xc)
    S_fro2 = float(np.linalg.norm(S, ord="fro") ** 2)
    b2 = float((sq_norms**2 - 2.0 * x_S_x + S_fro2).sum()) / (n**2)
    b2 = min(b2, d2)

    delta = _clip_delta(b2 / d2)
    Sigma = delta * F + (1.0 - delta) * S
    return (
        pd.DataFrame(Sigma, index=returns.columns, columns=returns.columns),
        delta,
    )


# --------------------------------------------------------------------------- #
# Dispatcher                                                                  #
# --------------------------------------------------------------------------- #


COVARIANCE_METHODS: tuple[str, ...] = (
    "sample",
    "lw_diagonal",
    "lw_constant_correlation",
)


def estimate_covariance(returns: pd.DataFrame, method: str = "sample") -> pd.DataFrame:
    """Return an estimated covariance matrix of ``returns`` by ``method``.

    method ∈ {'sample', 'lw_diagonal', 'lw_constant_correlation'}.
    """
    if method == "sample":
        return sample_covariance(returns)
    if method == "lw_diagonal":
        return ledoit_wolf_diagonal(returns)[0]
    if method == "lw_constant_correlation":
        return ledoit_wolf_constant_correlation(returns)[0]
    raise ValueError(
        f"Unknown covariance method: {method!r}. Choose one of {COVARIANCE_METHODS}."
    )
