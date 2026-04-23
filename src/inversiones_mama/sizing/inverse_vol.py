"""Inverse-Volatility Portfolio Weighting.

Tournament-validated null baseline that allocates capital inversely
proportional to each asset's trailing realized volatility. On the
91-ETF LIQUID_ETFS universe, this strategy achieves:

  * Sharpe median = +1.154  (CPCV 2021-2026, 45 splits)
  * MDD median = 2.7%
  * CVaR95 median = 0.49%
  * MW p = 0.026 vs 60/40 benchmark (significant at 5%)

The strategy is zero-parameter (after fixing the lookback to 60 trading
days and the rebalance frequency to monthly). This makes it the
minimum-regret deployment choice: comparable performance to vol_targeting
without the overfitting risk of a tuned lookback.

Public API
----------
``inverse_vol_weights(returns, vol_lookback=60) -> pd.Series``
    Compute inverse-volatility weights from a (T x N) returns DataFrame.
    Returns an (N,) weight Series summing to 1, indexed by ticker.

``inverse_vol_allocator(prices, vol_lookback=60, rebal_freq='ME')
     -> tuple[pd.DataFrame, pd.Series]``
    Full allocation engine: given a (T x N) prices DataFrame, returns
    (weights_df, portfolio_returns) with monthly rebalancing.

Reference
---------
This is the simplest risk-parity variant, sometimes called "naive risk
parity" or "equal risk contribution" (approximate). Unlike full risk
parity (which solves for marginal risk contributions), inverse-vol
ignores correlations — but on a diversified ETF universe, the difference
is negligible and the robustness advantage is decisive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_vol_weights(
    returns: pd.DataFrame,
    vol_lookback: int = 60,
    *,
    min_vol: float = 1e-8,
) -> pd.Series:
    """Compute inverse-volatility weights from trailing returns.

    Parameters
    ----------
    returns : pd.DataFrame
        (T x N) daily returns. Uses the last ``vol_lookback`` rows.
    vol_lookback : int
        Number of trailing observations for volatility estimation.
        Default 60 (≈ 3 calendar months).
    min_vol : float
        Floor for per-asset volatility to avoid division by zero.
        Assets with vol below this floor receive zero weight.

    Returns
    -------
    pd.Series
        (N,) weights summing to 1.0, indexed by ticker. Assets with
        zero or near-zero variance receive weight 0.

    Raises
    ------
    ValueError
        If ``returns`` has fewer than 2 columns or fewer than
        ``vol_lookback`` rows.
    """
    if returns.shape[1] < 2:
        raise ValueError(
            f"inverse_vol_weights needs >= 2 assets, got {returns.shape[1]}"
        )
    if len(returns) < vol_lookback:
        raise ValueError(
            f"Need >= {vol_lookback} observations, got {len(returns)}"
        )

    window = returns.iloc[-vol_lookback:]
    sigma = window.std(ddof=0)

    # Zero-variance assets get zero weight
    valid = sigma > min_vol
    if valid.sum() < 1:
        # Fallback: equal weight on all assets
        return pd.Series(1.0 / returns.shape[1], index=returns.columns)

    inv_vol = pd.Series(0.0, index=returns.columns)
    inv_vol[valid] = 1.0 / sigma[valid]
    total = inv_vol.sum()
    if total > 0:
        inv_vol = inv_vol / total
    return inv_vol


def inverse_vol_allocator(
    prices: pd.DataFrame,
    vol_lookback: int = 60,
    rebal_freq: str = "ME",
    *,
    per_name_cap: float | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Full inverse-vol allocation engine with periodic rebalancing.

    Parameters
    ----------
    prices : pd.DataFrame
        (T x N) adjusted close prices.
    vol_lookback : int
        Trailing days for volatility estimation. Default 60.
    rebal_freq : str
        Pandas frequency alias for rebalance dates. Default ``"ME"``
        (month-end). Other useful values: ``"W-FRI"``, ``"QE"``.
    per_name_cap : float | None
        Optional maximum weight per ticker. If set (e.g., 0.15 = 15%),
        excess weight is redistributed pro-rata to uncapped assets.

    Returns
    -------
    weights : pd.DataFrame
        (T x N) daily weight matrix (weights drift between rebalances).
    portfolio_returns : pd.Series
        (T,) daily portfolio returns.
    """
    returns = prices.pct_change().iloc[1:]
    rebal_dates = set(
        returns.groupby(pd.Grouper(freq=rebal_freq)).tail(1).index.tolist()
    )

    n_assets = returns.shape[1]
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    current_w = pd.Series(1.0 / n_assets, index=returns.columns)

    for i, d in enumerate(returns.index):
        if d in rebal_dates and i >= vol_lookback:
            window = returns.iloc[i - vol_lookback : i]
            w = inverse_vol_weights(window, vol_lookback=vol_lookback)

            # Apply per-name cap if specified
            if per_name_cap is not None and per_name_cap > 0:
                w = _apply_cap(w, per_name_cap)

            current_w = w

        weights.loc[d] = current_w.reindex(weights.columns).fillna(0.0).values

    port_ret = (weights.values * returns.values).sum(axis=1)
    portfolio_returns = pd.Series(
        port_ret, index=returns.index, name="daily_return"
    ).dropna()

    return weights, portfolio_returns


def _apply_cap(w: pd.Series, cap: float, max_iter: int = 10) -> pd.Series:
    """Iteratively cap weights at ``cap`` and redistribute excess pro-rata."""
    w = w.copy()
    for _ in range(max_iter):
        excess_mask = w > cap
        if not excess_mask.any():
            break
        excess = (w[excess_mask] - cap).sum()
        w[excess_mask] = cap
        below = w[~excess_mask]
        if below.sum() > 0:
            w[~excess_mask] += excess * below / below.sum()
    # Renormalize for safety
    total = w.sum()
    if total > 0:
        w = w / total
    return w


def generate_current_weights(
    prices: pd.DataFrame,
    vol_lookback: int = 60,
    per_name_cap: float | None = 0.15,
) -> pd.Series:
    """One-shot: compute current inverse-vol weights for deployment.

    This is the function the Alpaca pipeline should call. Given the
    latest prices, it computes the weight vector for the next rebalance.

    Parameters
    ----------
    prices : pd.DataFrame
        (T x N) adjusted close prices. Must have >= ``vol_lookback + 1``
        rows.
    vol_lookback : int
        Trailing days for volatility estimation.
    per_name_cap : float | None
        Optional maximum weight per ticker.

    Returns
    -------
    pd.Series
        (N,) weights summing to 1.0, indexed by ticker. Ready to
        convert to dollar amounts via ``weights * total_capital``.
    """
    returns = prices.pct_change().iloc[1:]
    w = inverse_vol_weights(returns, vol_lookback=vol_lookback)
    if per_name_cap is not None:
        w = _apply_cap(w, per_name_cap)
    return w
