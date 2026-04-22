"""Trade-sequence permutation Monte Carlo.

Complementary stress test to the bootstrap-based MC in
``simulation.monte_carlo``. Where that module re-samples returns with
replacement to explore what *could have happened*, this module takes the
**exact realized returns** and reshuffles their order to answer: "was
the observed drawdown a property of the distribution, or a lucky/unlucky
ordering?"

This is the Portfolio-Maestro / López de Prado "Multiple Randomized
Backtests" idea applied to a single sequence. It preserves:

* mean return (exact)
* volatility (exact)
* skewness (exact)
* kurtosis (exact)

and destroys only:

* autocorrelation / volatility clustering
* calendar timing

That combination isolates path dependence. If the historical max
drawdown sits near the MEDIAN of the permutation distribution, the
observed drawdown is "typical" for this return distribution. If it sits
above the 99th percentile, the historical experience was unusually bad
(we got unlucky); if at the 1st percentile, unusually good (we got
lucky — be worried).

Public API
----------
``TradeSequenceMCResult`` — distributions from permuted paths.
``trade_sequence_mc(returns, n_paths=10_000, initial_capital=5000, rng=None)``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeSequenceMCResult:
    """Distributions produced by reshuffling the order of realized returns."""

    n_paths: int
    initial_capital: float
    # Observed / historical
    observed_terminal_wealth: float
    observed_max_drawdown: float
    # Permutation distributions
    terminal_wealth: np.ndarray           # (n_paths,)
    max_drawdowns: np.ndarray             # (n_paths,), positive magnitudes
    # Summary percentiles
    terminal_p05: float
    terminal_p50: float
    terminal_p95: float
    dd_p50: float
    dd_p95: float
    dd_p99: float
    # Percentile rank of the OBSERVED values in the permutation distribution.
    # A high observed_dd_percentile (e.g., 0.99) means the realized path
    # was unusually bad for this return distribution.
    observed_terminal_percentile: float
    observed_dd_percentile: float

    @property
    def observed_is_unusually_bad_dd(self) -> bool:
        """Realized max DD was worse than 95% of permuted paths."""
        return self.observed_dd_percentile >= 0.95

    @property
    def observed_is_unusually_good_terminal(self) -> bool:
        """Realized terminal beat 95% of permuted paths."""
        return self.observed_terminal_percentile >= 0.95


def trade_sequence_mc(
    returns: pd.Series | np.ndarray,
    n_paths: int = 10_000,
    initial_capital: float = 5_000.0,
    rng: np.random.Generator | None = None,
) -> TradeSequenceMCResult:
    """Run a permutation MC on the realized return sequence.

    Parameters
    ----------
    returns : Series or 1-D array of periodic (usually daily) decimal returns.
        Drawn from a completed backtest's realized path — these are what
        DID happen, reshuffled in order.
    n_paths : number of permutations to generate.
    initial_capital : start value for every permutation path.
    rng : optional numpy Generator for reproducibility.
    """
    rets = _as_1d_array(returns)
    if rets.size < 5:
        raise ValueError(f"need at least 5 returns to permute; got {rets.size}")
    if n_paths < 10:
        raise ValueError(f"n_paths must be >= 10 for stable percentiles; got {n_paths}")
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be positive; got {initial_capital}")

    rng = rng if rng is not None else np.random.default_rng()

    # Observed path
    observed_terminal = float(initial_capital * np.prod(1.0 + rets))
    observed_dd = float(_max_drawdown(initial_capital, rets))

    # Build permutations in a single vectorized pass where possible
    # For memory, fall back to a loop when n_paths * horizon gets large.
    horizon = rets.size
    approx_mb = 8 * n_paths * (horizon + 1) / 1e6
    wealth_paths = np.empty((n_paths, horizon + 1), dtype=np.float64)
    wealth_paths[:, 0] = initial_capital

    # Vectorized permutation: generate (n_paths, horizon) random indices
    # via np.argsort(rng.random(...)) — slightly slower than rng.permutation
    # in a loop but memory-adequate up to ~250k paths on a laptop.
    if approx_mb < 500:
        rand = rng.random((n_paths, horizon))
        perm_idx = np.argsort(rand, axis=1)
        permuted = rets[perm_idx]
        wealth_paths[:, 1:] = initial_capital * np.cumprod(1.0 + permuted, axis=1)
    else:
        # Memory-conservative loop
        for i in range(n_paths):
            wealth_paths[i, 1:] = initial_capital * np.cumprod(1.0 + rng.permutation(rets))

    terminal = wealth_paths[:, -1]
    # Max drawdown per path: running peak vs wealth
    running_max = np.maximum.accumulate(wealth_paths, axis=1)
    drawdowns = (wealth_paths - running_max) / running_max
    max_dd = -drawdowns.min(axis=1)

    obs_term_rank = float((terminal <= observed_terminal).mean())
    obs_dd_rank = float((max_dd <= observed_dd).mean())

    return TradeSequenceMCResult(
        n_paths=n_paths,
        initial_capital=initial_capital,
        observed_terminal_wealth=observed_terminal,
        observed_max_drawdown=observed_dd,
        terminal_wealth=terminal,
        max_drawdowns=max_dd,
        terminal_p05=float(np.percentile(terminal, 5)),
        terminal_p50=float(np.percentile(terminal, 50)),
        terminal_p95=float(np.percentile(terminal, 95)),
        dd_p50=float(np.percentile(max_dd, 50)),
        dd_p95=float(np.percentile(max_dd, 95)),
        dd_p99=float(np.percentile(max_dd, 99)),
        observed_terminal_percentile=obs_term_rank,
        observed_dd_percentile=obs_dd_rank,
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _as_1d_array(returns: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    return arr


def _max_drawdown(initial: float, rets: np.ndarray) -> float:
    """Max drawdown of the wealth path defined by ``initial * prod(1 + rets)``."""
    wealth = initial * np.cumprod(1.0 + rets)
    wealth = np.concatenate([[initial], wealth])
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    return float(-dd.min())
