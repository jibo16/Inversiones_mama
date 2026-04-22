"""Risk-Constrained Kelly (RCK) portfolio optimizer.

Solves the convex program that maximizes the Kelly growth rate
(expected log-wealth growth) subject to:

  1. **Fractional Kelly**: weights are scaled by a fraction (e.g. 0.65)
     to reduce variance of the growth rate estimator.
  2. **Per-name cap**: no single position exceeds ``cap`` (e.g. 35%).
  3. **Drawdown probability bound**: via the Grossman–Zhou ruin
     constraint, ``P(DD ≥ α) ≤ β``  ⟹  ``λ = log(β)/log(α)`` acts as
     an additional leverage / risk-budget limit.
  4. **No leverage**: weights sum to ≤ 1, all weights ≥ 0 (long-only).
  5. **No short selling**: enforced by ``w ≥ 0``.

The Kelly growth rate for log-normal returns is:

    g(w) = w'μ − ½ w'Σw

which is concave in w, so we solve a **convex maximization** (or
equivalently, minimize the negative). This is a QP with linear
constraints — CVXPY handles it natively with OSQP/ECOS/SCS.

Public API
----------
solve_rck(mu, Sigma, ...) -> KellyResult
    Core solver. Takes μ (daily), Σ (daily), returns optimal weights.

kelly_growth_rate(w, mu, Sigma) -> float
    Evaluate g(w) = w'μ − ½ w'Σw at a given weight vector.

Notes
-----
* All inputs/outputs are in **daily decimals** (not annualized).
  Annualization is the caller's job (metrics module).
* The solver intentionally does NOT annualize internally to prevent
  double-scaling bugs between the regression and metrics layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from inversiones_mama.config import (
    KELLY_FRACTION,
    MAX_WEIGHT_PER_NAME,
    RCK_MAX_DRAWDOWN_PROBABILITY,
    RCK_MAX_DRAWDOWN_THRESHOLD,
)


# --------------------------------------------------------------------------- #
# Result container                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KellyResult:
    """Output of :func:`solve_rck`.

    Attributes
    ----------
    weights : pd.Series
        Optimal portfolio weights (index = ticker).
    growth_rate : float
        Kelly growth rate g(w*) = w*'μ − ½ w*'Σw* (daily).
    growth_rate_ann : float
        g(w*) × 252 (annualized).
    status : str
        CVXPY solver status (``optimal``, ``optimal_inaccurate``, etc.).
    solver_stats : dict
        Raw CVXPY solve stats (iterations, solve time, etc.).
    fraction : float
        The Kelly fraction applied.
    lambda_rck : float
        Risk-budget parameter λ = log(β)/log(α).
    cash_weight : float
        1 − sum(w*), i.e. the implicit cash allocation.
    """

    weights: pd.Series
    growth_rate: float
    growth_rate_ann: float
    status: str
    solver_stats: dict[str, Any]
    fraction: float
    lambda_rck: float
    cash_weight: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Core math                                                                   #
# --------------------------------------------------------------------------- #


def kelly_growth_rate(
    w: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
) -> float:
    """Evaluate the Kelly growth rate g(w) = w'μ − ½ w'Σw.

    Parameters
    ----------
    w : (n,) array of weights.
    mu : (n,) array of expected daily returns.
    Sigma : (n, n) covariance matrix of daily returns.

    Returns
    -------
    float
        Growth rate in daily decimals.
    """
    w = np.asarray(w, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    return float(w @ mu - 0.5 * w @ Sigma @ w)


def _compute_lambda(alpha: float, beta: float) -> float:
    """Compute the RCK risk-budget parameter.

    λ = log(β) / log(α)

    where α = max drawdown threshold, β = max probability of hitting it.

    Example: α=0.50, β=0.10 → λ = log(0.10)/log(0.50) ≈ 3.32
    Interpretation: the portfolio's volatility budget is constrained so
    that the probability of a 50% drawdown stays below 10%.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha (DD threshold) must be in (0,1), got {alpha}")
    if not (0 < beta < 1):
        raise ValueError(f"beta (DD probability) must be in (0,1), got {beta}")
    return np.log(beta) / np.log(alpha)


# --------------------------------------------------------------------------- #
# Solver                                                                      #
# --------------------------------------------------------------------------- #


def solve_rck(
    mu: pd.Series | np.ndarray,
    Sigma: pd.DataFrame | np.ndarray,
    fraction: float = KELLY_FRACTION,
    cap: float = MAX_WEIGHT_PER_NAME,
    alpha: float = RCK_MAX_DRAWDOWN_THRESHOLD,
    beta: float = RCK_MAX_DRAWDOWN_PROBABILITY,
    min_weight: float = 0.0,
    solver: str | None = None,
    verbose: bool = False,
) -> KellyResult:
    """Solve the Risk-Constrained Kelly optimization.

    Maximizes  g(w) = w'μ − ½ w'Σw
    subject to:
        w ≥ min_weight  (default 0 = long-only)
        w_i ≤ cap       ∀i
        1'w ≤ 1         (no leverage; remainder = cash)
        w'Σw ≤ (2/λ) · w'μ   (drawdown probability bound)

    Then scales the optimal full-Kelly weights by ``fraction``.

    Parameters
    ----------
    mu : Series or array (n,)
        Daily expected returns per asset (from composite μ).
    Sigma : DataFrame or array (n, n)
        Daily return covariance matrix.
    fraction : float
        Kelly fraction (0 < f ≤ 1). Default from config.
    cap : float
        Per-name weight cap (0 < c ≤ 1). Default from config.
    alpha : float
        Maximum drawdown threshold (0 < α < 1). Default 0.50.
    beta : float
        Maximum probability of hitting α (0 < β < 1). Default 0.10.
    min_weight : float
        Minimum per-name weight. Default 0 (long-only).
    solver : str or None
        CVXPY solver name override (default: let CVXPY pick).
    verbose : bool
        Print CVXPY solver output.

    Returns
    -------
    KellyResult

    Raises
    ------
    ValueError
        If inputs are misshapen, non-PSD, or constraints are infeasible.
    """
    # --- Input normalization ------------------------------------------------
    if isinstance(mu, pd.Series):
        tickers = mu.index.tolist()
        mu_arr = mu.values.astype(np.float64)
    else:
        mu_arr = np.asarray(mu, dtype=np.float64)
        tickers = [f"asset_{i}" for i in range(len(mu_arr))]

    if isinstance(Sigma, pd.DataFrame):
        Sigma_arr = Sigma.values.astype(np.float64)
    else:
        Sigma_arr = np.asarray(Sigma, dtype=np.float64)

    n = len(mu_arr)
    if Sigma_arr.shape != (n, n):
        raise ValueError(
            f"Sigma shape {Sigma_arr.shape} incompatible with mu length {n}"
        )

    # --- Parameter validation -----------------------------------------------
    if not (0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if not (0 < cap <= 1.0):
        raise ValueError(f"cap must be in (0, 1], got {cap}")

    lambda_rck = _compute_lambda(alpha, beta)

    # --- Make Sigma PSD via regularization ----------------------------------
    # Small nudge to ensure strict positive-definiteness for QP solvers.
    eps = 1e-8 * np.eye(n)
    Sigma_reg = 0.5 * (Sigma_arr + Sigma_arr.T) + eps

    # --- Prune assets with non-positive expected return ----------------------
    # Kelly never allocates to μ ≤ 0 assets. Pruning them shrinks the QP
    # and avoids numerical noise from near-zero weights.
    positive_mask = mu_arr > 0
    n_positive = positive_mask.sum()

    if n_positive == 0:
        # All assets have non-positive expected returns → 100% cash
        weights_full = pd.Series(0.0, index=tickers, name="weight")
        return KellyResult(
            weights=weights_full,
            growth_rate=0.0,
            growth_rate_ann=0.0,
            status="trivial_all_cash",
            solver_stats={},
            fraction=fraction,
            lambda_rck=lambda_rck,
            cash_weight=1.0,
            diagnostics={"pruned_assets": tickers, "reason": "all mu <= 0"},
        )

    mu_pos = mu_arr[positive_mask]
    Sigma_pos = Sigma_reg[np.ix_(positive_mask, positive_mask)]
    n_pos = len(mu_pos)

    # --- CVXPY problem formulation ------------------------------------------
    w = cp.Variable(n_pos, nonneg=True)

    # Wrap Sigma_pos in cp.psd_wrap so CVXPY skips its internal ARPACK
    # eigenvalue check. We already PSD-regularized above and the caller is
    # expected to pass a PSD-clipped covariance for large universes. Without
    # this, N > ~200 causes "ArpackNoConvergence" failures on otherwise-valid
    # problems (observed on SP500 scale).
    Sigma_psd = cp.psd_wrap(Sigma_pos)

    # Objective: maximize g(w) = w'μ − ½ w'Σw
    # CVXPY convention: minimize -(w'μ) + ½ w'Σw
    objective = cp.Minimize(-mu_pos @ w + 0.5 * cp.quad_form(w, Sigma_psd))

    constraints = [
        w >= min_weight,      # lower bound
        w <= cap,             # per-name cap
        cp.sum(w) <= 1.0,    # no leverage (long-only, remainder = cash)
    ]

    # --- RCK drawdown probability constraint --------------------------------
    # From Grossman–Zhou: P(DD ≥ α) ≤ β  iff  w'Σw ≤ (2/λ) · w'μ
    # This is a second-order cone constraint when reformulated, but since
    # we're already maximizing g(w) = w'μ − ½ w'Σw, we can express it as:
    #     w'Σw ≤ (2/λ) * w'μ
    # which is convex when w'μ ≥ 0 (guaranteed by mu pruning + nonneg w).
    constraints.append(
        cp.quad_form(w, Sigma_psd) <= (2.0 / lambda_rck) * (mu_pos @ w)
    )

    prob = cp.Problem(objective, constraints)

    # --- Solve ---------------------------------------------------------------
    solve_kwargs: dict[str, Any] = {"verbose": verbose}
    if solver is not None:
        solve_kwargs["solver"] = solver

    try:
        prob.solve(**solve_kwargs)
    except cp.error.SolverError as e:
        # Fallback: try SCS which is more robust to ill-conditioning
        try:
            prob.solve(solver=cp.SCS, verbose=verbose)
        except cp.error.SolverError:
            raise ValueError(
                f"CVXPY failed with all solvers. Original error: {e}"
            ) from e

    # --- Extract results ----------------------------------------------------
    status = prob.status
    if status not in ("optimal", "optimal_inaccurate"):
        # Build full weight vector as zeros for infeasible/unbounded
        weights_full = pd.Series(0.0, index=tickers, name="weight")
        return KellyResult(
            weights=weights_full,
            growth_rate=0.0,
            growth_rate_ann=0.0,
            status=status,
            solver_stats=_extract_solver_stats(prob),
            fraction=fraction,
            lambda_rck=lambda_rck,
            cash_weight=1.0,
            diagnostics={"error": f"Solver status: {status}"},
        )

    w_full_kelly = np.maximum(w.value, 0.0)  # clip numerical noise

    # Apply fractional Kelly scaling
    w_fractional = w_full_kelly * fraction

    # Re-normalize if sum exceeds 1 after fraction scaling (shouldn't, but safe)
    total = w_fractional.sum()
    if total > 1.0:
        w_fractional = w_fractional / total

    # Map back to full ticker vector
    weights_arr = np.zeros(n)
    weights_arr[positive_mask] = w_fractional

    weights_series = pd.Series(weights_arr, index=tickers, name="weight")
    cash = 1.0 - weights_series.sum()

    # Evaluate growth rate at the fractional weights (not full-Kelly)
    g = kelly_growth_rate(weights_arr, mu_arr, Sigma_reg)

    return KellyResult(
        weights=weights_series,
        growth_rate=g,
        growth_rate_ann=g * 252,
        status=status,
        solver_stats=_extract_solver_stats(prob),
        fraction=fraction,
        lambda_rck=lambda_rck,
        cash_weight=float(cash),
        diagnostics={
            "n_assets_total": n,
            "n_assets_positive_mu": int(n_positive),
            "n_assets_active": int((weights_series > 1e-6).sum()),
            "full_kelly_sum": float(w_full_kelly.sum()),
            "fractional_kelly_sum": float(w_fractional.sum()),
        },
    )


def _extract_solver_stats(prob: cp.Problem) -> dict[str, Any]:
    """Safely extract solver statistics from a solved CVXPY problem."""
    stats: dict[str, Any] = {}
    try:
        if prob.solver_stats is not None:
            stats["solver_name"] = prob.solver_stats.solver_name
            stats["solve_time"] = prob.solver_stats.solve_time
            stats["num_iters"] = prob.solver_stats.num_iters
    except Exception:
        pass
    stats["optimal_value"] = prob.value
    return stats
