"""Block bootstrap for generating synthetic return paths.

Preserves the temporal dependence structure of financial returns —
specifically volatility clustering and autocorrelation — that plain IID
bootstrap destroys. Used by :mod:`simulation.monte_carlo` to stress-test
Risk-Constrained Kelly over thousands of resampled market paths.

Three variants are provided:

* **Moving Block Bootstrap (MBB)** — fixed-length contiguous blocks
  sampled uniformly from the series.
* **Stationary Bootstrap** (Politis & Romano, 1994) — block lengths drawn
  from a geometric distribution with mean ``mean_block_length``. Produces
  a strictly stationary pseudo-sample; the preferred choice for financial
  return data per López de Prado.
* **Circular Block Bootstrap (CBB)** — MBB variant that treats the series
  as circular so every observation has equal selection probability.

All functions operate on either a 1-D array / Series (single asset) or a
2-D array / DataFrame (multi-asset, rows = time). For multi-asset inputs
the *same* random block indices are applied across all assets, so the
cross-asset correlation structure is preserved.

Public API
----------
``moving_block_bootstrap(returns, n_samples, horizon, block_length, rng)``
``stationary_bootstrap(returns, n_samples, horizon, mean_block_length, rng)``
``circular_block_bootstrap(returns, n_samples, horizon, block_length, rng)``
``bootstrap_iter(returns, horizon, block_length_or_mean, rng, method)``

All bulk functions return an ``ndarray`` of shape ``(n_samples, horizon)``
for 1-D input, or ``(n_samples, horizon, n_assets)`` for 2-D input.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

# Default block length for stationary bootstrap on daily equity returns.
# Rule of thumb (Politis & White 2004) for moderate series lengths.
DEFAULT_MEAN_BLOCK_LENGTH: int = 20


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _as_2d(returns: pd.DataFrame | pd.Series | np.ndarray) -> tuple[np.ndarray, bool, tuple[str, ...] | None]:
    """Normalize input to a 2-D ndarray; track whether the caller passed 1-D.

    Returns
    -------
    arr : (n_periods, n_assets) float64
    was_1d : True if original input was 1-D
    columns : tuple of column names (for DataFrame), else None
    """
    columns: tuple[str, ...] | None = None
    if isinstance(returns, pd.DataFrame):
        columns = tuple(str(c) for c in returns.columns)
        arr = returns.to_numpy(dtype=np.float64, copy=False)
        return arr, False, columns
    if isinstance(returns, pd.Series):
        arr = returns.to_numpy(dtype=np.float64, copy=False)[:, None]
        return arr, True, None
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None], True, None
    if arr.ndim == 2:
        return arr, False, None
    raise ValueError(f"returns must be 1-D or 2-D; got {arr.ndim}-D")


def _coerce_output(samples: np.ndarray, was_1d: bool) -> np.ndarray:
    """Strip the asset dimension if the caller passed 1-D input."""
    if was_1d:
        # (n_samples, horizon, 1) -> (n_samples, horizon)
        return samples[..., 0]
    return samples


def _validate_params(n_periods: int, horizon: int, block_length: int | float) -> None:
    if n_periods < 2:
        raise ValueError(f"need at least 2 observations; got {n_periods}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    if block_length <= 0:
        raise ValueError(f"block_length / mean_block_length must be positive; got {block_length}")


# --------------------------------------------------------------------------- #
# Moving Block Bootstrap                                                      #
# --------------------------------------------------------------------------- #


def moving_block_bootstrap(
    returns: pd.DataFrame | pd.Series | np.ndarray,
    n_samples: int,
    horizon: int,
    block_length: int = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample ``n_samples`` paths of length ``horizon`` via the MBB.

    Each path is formed by concatenating uniformly sampled blocks of
    exactly ``block_length`` contiguous observations. Any path whose
    length exceeds ``horizon`` after block concatenation is truncated to
    ``horizon``.
    """
    arr, was_1d, _ = _as_2d(returns)
    n_periods = arr.shape[0]
    block_length = int(block_length)
    _validate_params(n_periods, horizon, block_length)
    if block_length > n_periods:
        raise ValueError(
            f"block_length ({block_length}) cannot exceed n_periods ({n_periods})"
        )
    rng = rng if rng is not None else np.random.default_rng()

    n_blocks_per_path = int(np.ceil(horizon / block_length))
    max_start = n_periods - block_length + 1
    # (n_samples, n_blocks_per_path) random block start indices
    starts = rng.integers(0, max_start, size=(n_samples, n_blocks_per_path))

    # Expand each start to a block of indices, concatenate per sample, then truncate
    block_offsets = np.arange(block_length)
    # shape: (n_samples, n_blocks_per_path, block_length)
    idx = starts[..., None] + block_offsets[None, None, :]
    idx = idx.reshape(n_samples, n_blocks_per_path * block_length)
    idx = idx[:, :horizon]  # truncate to exact horizon

    # Fancy-index the returns array with shape (n_samples, horizon, n_assets)
    samples = arr[idx]
    return _coerce_output(samples, was_1d)


# --------------------------------------------------------------------------- #
# Stationary Bootstrap (Politis-Romano)                                       #
# --------------------------------------------------------------------------- #


def stationary_bootstrap(
    returns: pd.DataFrame | pd.Series | np.ndarray,
    n_samples: int,
    horizon: int,
    mean_block_length: float = DEFAULT_MEAN_BLOCK_LENGTH,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Stationary bootstrap with geometric block lengths.

    Each observation in the output path either continues the current block
    (with probability ``1 - p``) or starts a new block at a uniformly
    random position (with probability ``p = 1 / mean_block_length``). The
    series is treated as circular so all observations have equal selection
    probability.

    Produces a strictly stationary pseudo-sample — this is the preferred
    bootstrap for financial returns in the quantitative-finance literature.
    """
    arr, was_1d, _ = _as_2d(returns)
    n_periods = arr.shape[0]
    mean_block_length = float(mean_block_length)
    _validate_params(n_periods, horizon, mean_block_length)
    rng = rng if rng is not None else np.random.default_rng()

    p = 1.0 / mean_block_length  # probability of block restart at each step

    # Build the path of indices:
    # idx[t] = rand_start[t] if restart[t] else (idx[t-1] + 1) % n_periods
    restarts = rng.random((n_samples, horizon)) < p
    restarts[:, 0] = True  # always restart at t=0
    random_starts = rng.integers(0, n_periods, size=(n_samples, horizon))

    # Vectorize with a running pointer using cumulative restarts:
    # For each sample, accumulate increments from the last restart point.
    # This is equivalent to a per-row loop but we want to keep it fast.
    # Strategy: compute (n_samples, horizon) "since-restart" offsets, then
    # look up random_starts at the most recent restart.
    #
    # restart_idx[i, t] = t if restarts[i, t] else restart_idx[i, t-1]
    # offset[i, t]      = t - restart_idx[i, t]
    # idx[i, t]         = (random_starts[i, restart_idx[i, t]] + offset[i, t]) % n_periods

    # Efficient cumulative-max over restarts: use np.maximum.accumulate on
    # t * restarts, with zeros replaced by a "restart_indices" array of last seen t.
    t_grid = np.broadcast_to(np.arange(horizon), (n_samples, horizon))
    restart_indices = np.where(restarts, t_grid, 0)
    restart_idx = np.maximum.accumulate(restart_indices, axis=1)
    offset = t_grid - restart_idx

    # Take random start at the restart index per (sample, t)
    rows = np.arange(n_samples)[:, None]
    starts_at_restart = random_starts[rows, restart_idx]
    idx = (starts_at_restart + offset) % n_periods

    samples = arr[idx]
    return _coerce_output(samples, was_1d)


# --------------------------------------------------------------------------- #
# Circular Block Bootstrap                                                    #
# --------------------------------------------------------------------------- #


def circular_block_bootstrap(
    returns: pd.DataFrame | pd.Series | np.ndarray,
    n_samples: int,
    horizon: int,
    block_length: int = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """MBB that wraps circularly so every observation has equal probability."""
    arr, was_1d, _ = _as_2d(returns)
    n_periods = arr.shape[0]
    block_length = int(block_length)
    _validate_params(n_periods, horizon, block_length)
    rng = rng if rng is not None else np.random.default_rng()

    n_blocks_per_path = int(np.ceil(horizon / block_length))
    starts = rng.integers(0, n_periods, size=(n_samples, n_blocks_per_path))
    block_offsets = np.arange(block_length)
    idx = starts[..., None] + block_offsets[None, None, :]
    idx = idx.reshape(n_samples, n_blocks_per_path * block_length)
    idx = idx[:, :horizon] % n_periods  # circular wrap

    samples = arr[idx]
    return _coerce_output(samples, was_1d)


# --------------------------------------------------------------------------- #
# Generator variant (memory-efficient for large n_samples)                    #
# --------------------------------------------------------------------------- #


def bootstrap_iter(
    returns: pd.DataFrame | pd.Series | np.ndarray,
    horizon: int,
    block_length: int | float = DEFAULT_MEAN_BLOCK_LENGTH,
    rng: np.random.Generator | None = None,
    method: str = "stationary",
) -> Iterator[np.ndarray]:
    """Yield bootstrap samples one at a time.

    Shape of each yielded sample matches the bulk functions with
    ``n_samples=1`` (i.e. squeezed on axis 0): (horizon,) for 1-D input,
    (horizon, n_assets) for 2-D input. Use with ``itertools.islice`` when
    you want exactly N samples without allocating them all.
    """
    if method not in {"stationary", "moving", "circular"}:
        raise ValueError(f"unknown method: {method}")

    rng = rng if rng is not None else np.random.default_rng()
    while True:
        if method == "stationary":
            sample = stationary_bootstrap(returns, 1, horizon, block_length, rng=rng)
        elif method == "moving":
            sample = moving_block_bootstrap(returns, 1, horizon, int(block_length), rng=rng)
        else:
            sample = circular_block_bootstrap(returns, 1, horizon, int(block_length), rng=rng)
        yield sample[0]  # drop the n_samples axis
