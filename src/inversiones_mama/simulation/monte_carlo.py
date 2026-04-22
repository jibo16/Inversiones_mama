"""Monte Carlo validation of Risk-Constrained Kelly drawdown bounds.

The RCK solver imposes a *theoretical* bound `P(max_drawdown >= alpha) <= beta`
derived from the Grossman-Zhou log-wealth analysis. This module empirically
tests that promise by:

  1. Solving RCK once on observed (mu, Sigma).
  2. Bootstrapping thousands of ``horizon_days``-long return paths from
     observed asset returns (block bootstrap, preserves autocorrelation).
  3. Applying the RCK weights to each path to simulate the portfolio's
     wealth trajectory.
  4. Measuring the empirical distribution of terminal wealth and maximum
     drawdown across paths.
  5. Reporting pass/fail versus Jorge's sanity gates and the RCK's own
     theoretical drawdown bound.

This is Phase 4 of the blueprint. It is the primary Monte Carlo gate for
v1a: if the empirical drawdown probability exceeds beta, the strategy's
risk budget is mis-specified and the rebalance should not go live.

Assumptions & scope
-------------------
* **Constant weights over horizon.** The MC does not re-solve RCK at each
  bootstrap step. It validates the risk budget of the CURRENT rebalance
  over a full ``horizon_days`` window. Adding mid-path re-solves is a v1b
  refinement.
* **No transaction cost in the MC.** We're validating the risk bound of
  the weights, not transaction-cost drag. Costs are modeled separately in
  the walk-forward engine.

Public API
----------
``MCValidationResult`` — bundles all output distributions and gate verdicts.
``run_mc_rck_validation(returns, mu, Sigma, ...) -> MCValidationResult``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    GATES,
    KELLY_FRACTION,
    MAX_WEIGHT_PER_NAME,
    RCK_MAX_DRAWDOWN_PROBABILITY,
    RCK_MAX_DRAWDOWN_THRESHOLD,
)
from ..sizing.kelly import solve_rck
from .bootstrap import moving_block_bootstrap, stationary_bootstrap


# --------------------------------------------------------------------------- #
# Result type                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MCValidationResult:
    """Output of :func:`run_mc_rck_validation`.

    All wealth values are in the same currency units as ``initial_capital``;
    ``*_pct_of_initial`` values are dimensionless ratios.
    """

    # Inputs / setup
    n_paths: int
    horizon_days: int
    initial_capital: float
    weights: pd.Series                 # RCK weights used
    rck_alpha: float                   # theoretical drawdown threshold
    rck_beta: float                    # theoretical drawdown probability cap

    # Distributions
    terminal_wealth: np.ndarray        # shape (n_paths,)
    max_drawdowns: np.ndarray          # shape (n_paths,), positive = loss

    # Aggregated statistics
    terminal_mean: float
    terminal_median: float
    terminal_p05: float                # 5th percentile
    terminal_p25: float
    terminal_p75: float
    terminal_p95: float
    dd_median: float
    dd_p75: float
    dd_p95: float
    dd_p99: float

    # Empirical probabilities (vs Jorge's gates)
    prob_loss_40pct: float             # P(final < 0.60 * initial)
    prob_loss_60pct: float             # P(final < 0.40 * initial)
    prob_dd_exceeds_rck_alpha: float   # the empirical RCK bound

    # Gate verdicts (see config.GATES)
    gate_prob_loss_40pct_pass: bool
    gate_prob_loss_60pct_pass: bool
    gate_dd_95th_pass: bool
    gate_rck_bound_pass: bool          # empirical P(DD>=alpha) <= beta


# --------------------------------------------------------------------------- #
# Core validator                                                              #
# --------------------------------------------------------------------------- #


def run_mc_rck_validation(
    returns: pd.DataFrame,
    mu: pd.Series | None = None,
    Sigma: pd.DataFrame | None = None,
    *,
    n_paths: int = 10_000,
    horizon_days: int = 252,
    initial_capital: float = 5_000.0,
    kelly_fraction: float = KELLY_FRACTION,
    cap: float = MAX_WEIGHT_PER_NAME,
    alpha: float = RCK_MAX_DRAWDOWN_THRESHOLD,
    beta: float = RCK_MAX_DRAWDOWN_PROBABILITY,
    bootstrap_method: str = "stationary",
    mean_block_length: int = 20,
    rng: np.random.Generator | None = None,
    weights: pd.Series | None = None,
) -> MCValidationResult:
    """Bootstrap-based empirical validation of the RCK drawdown bound.

    Parameters
    ----------
    returns : DataFrame
        Observed daily asset returns (tickers = columns). Used both to
        (optionally) fit mu/Sigma and as the bootstrap source.
    mu : Series, optional
        Daily expected returns. If None, falls back to ``returns.mean()``.
    Sigma : DataFrame, optional
        Daily covariance. If None, falls back to ``returns.cov()``.
    weights : Series, optional
        If provided, use these weights directly instead of solving RCK. Use
        this when you want to validate a specific rebalance's weights that
        were already produced by the engine.
    n_paths, horizon_days, initial_capital : MC sizing.
    kelly_fraction, cap, alpha, beta : RCK knobs (ignored if ``weights`` given).
    bootstrap_method : "stationary" (default) or "moving".
    mean_block_length : block-length parameter for the bootstrap.
    rng : numpy Generator for reproducibility.

    Returns
    -------
    MCValidationResult
    """
    if returns.empty:
        raise ValueError("returns DataFrame is empty")
    if n_paths < 10:
        raise ValueError(f"n_paths must be >= 10 for stable statistics; got {n_paths}")
    if horizon_days < 5:
        raise ValueError(f"horizon_days must be >= 5; got {horizon_days}")

    tickers = list(returns.columns)
    rng = rng if rng is not None else np.random.default_rng()

    # Resolve weights: either provided or RCK-solved
    if weights is None:
        mu_vec = mu if mu is not None else returns.mean()
        Sigma_mat = Sigma if Sigma is not None else returns.cov()
        result = solve_rck(
            mu_vec.reindex(tickers).fillna(0.0),
            Sigma_mat.reindex(index=tickers, columns=tickers),
            fraction=kelly_fraction,
            cap=cap,
            alpha=alpha,
            beta=beta,
        )
        weights_vec = result.weights
    else:
        weights_vec = weights

    w = weights_vec.reindex(tickers).fillna(0.0).to_numpy(dtype=np.float64)

    # Degenerate all-cash case: no simulation needed
    if np.all(w == 0):
        terminal = np.full(n_paths, initial_capital)
        max_dd = np.zeros(n_paths)
        return _assemble_result(
            weights_vec, w, tickers, n_paths, horizon_days, initial_capital,
            alpha, beta, terminal, max_dd,
        )

    # Bootstrap return paths: shape (n_paths, horizon_days, n_assets)
    if bootstrap_method == "stationary":
        paths = stationary_bootstrap(returns, n_paths, horizon_days, mean_block_length, rng=rng)
    elif bootstrap_method == "moving":
        paths = moving_block_bootstrap(returns, n_paths, horizon_days, int(mean_block_length), rng=rng)
    else:
        raise ValueError(f"bootstrap_method must be 'stationary' or 'moving', got {bootstrap_method}")

    # Portfolio returns per path: dot product along asset axis -> (n_paths, horizon_days)
    port_returns = paths @ w

    # Wealth path including initial (for drawdown): (n_paths, horizon_days + 1)
    wealth_path = np.concatenate(
        [np.full((n_paths, 1), initial_capital), initial_capital * np.cumprod(1.0 + port_returns, axis=1)],
        axis=1,
    )

    terminal = wealth_path[:, -1]

    # Max drawdown per path (positive magnitude)
    running_max = np.maximum.accumulate(wealth_path, axis=1)
    drawdowns = (wealth_path - running_max) / running_max
    max_dd = -drawdowns.min(axis=1)

    return _assemble_result(
        weights_vec, w, tickers, n_paths, horizon_days, initial_capital,
        alpha, beta, terminal, max_dd,
    )


def _assemble_result(
    weights_vec: pd.Series,
    w: np.ndarray,
    tickers: list[str],
    n_paths: int,
    horizon_days: int,
    initial_capital: float,
    alpha: float,
    beta: float,
    terminal: np.ndarray,
    max_dd: np.ndarray,
) -> MCValidationResult:
    threshold_40 = 0.60 * initial_capital  # 40% loss
    threshold_60 = 0.40 * initial_capital  # 60% loss

    prob_loss_40 = float((terminal < threshold_40).mean())
    prob_loss_60 = float((terminal < threshold_60).mean())
    prob_dd_alpha = float((max_dd >= alpha).mean())

    dd_p95 = float(np.percentile(max_dd, 95))
    dd_p99 = float(np.percentile(max_dd, 99))

    return MCValidationResult(
        n_paths=n_paths,
        horizon_days=horizon_days,
        initial_capital=initial_capital,
        weights=pd.Series(w, index=tickers, name="weights"),
        rck_alpha=alpha,
        rck_beta=beta,
        terminal_wealth=terminal,
        max_drawdowns=max_dd,
        terminal_mean=float(terminal.mean()),
        terminal_median=float(np.median(terminal)),
        terminal_p05=float(np.percentile(terminal, 5)),
        terminal_p25=float(np.percentile(terminal, 25)),
        terminal_p75=float(np.percentile(terminal, 75)),
        terminal_p95=float(np.percentile(terminal, 95)),
        dd_median=float(np.median(max_dd)),
        dd_p75=float(np.percentile(max_dd, 75)),
        dd_p95=dd_p95,
        dd_p99=dd_p99,
        prob_loss_40pct=prob_loss_40,
        prob_loss_60pct=prob_loss_60,
        prob_dd_exceeds_rck_alpha=prob_dd_alpha,
        gate_prob_loss_40pct_pass=prob_loss_40 < GATES.max_prob_loss_40pct,
        gate_prob_loss_60pct_pass=prob_loss_60 < GATES.max_prob_loss_60pct,
        gate_dd_95th_pass=dd_p95 < GATES.max_dd_95th_pct,
        gate_rck_bound_pass=prob_dd_alpha <= beta + 0.02,  # 2pp tolerance on empirical estimate
    )
