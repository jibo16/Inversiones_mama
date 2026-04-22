"""Volatility Targeting strategy.

Category: Volatility (5.3)

Logic:
  1. Compute trailing N-day realized volatility for each ETF
  2. Scale each position weight inversely to its volatility
  3. Target a constant portfolio-level volatility (e.g., 15% annualized)
  4. If total weight > 1.0, cap at 1.0 (no leverage)
  5. Monthly rebalance

Parameters:
  - vol_lookback (int): trailing window for vol estimation (default: 60)
  - target_vol (float): target annualized portfolio vol (default: 0.15)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from inversiones_mama.exploration.base import Strategy, StrategyMeta


class VolatilityTargeting(Strategy):
    """Inverse-volatility weighting with a portfolio vol target."""

    def __init__(
        self,
        vol_lookback: int = 60,
        target_vol: float = 0.15,
    ) -> None:
        self._vol_lookback = vol_lookback
        self._target_vol = target_vol
        super().__init__(StrategyMeta(
            name=f"VolTarget_L{vol_lookback}_TV{int(target_vol*100)}",
            category="volatility",
            parameters={"vol_lookback": vol_lookback, "target_vol": target_vol},
            description=(
                f"Volatility targeting: inverse-vol weight each ETF, "
                f"target {target_vol:.0%} annualized portfolio vol, "
                f"{vol_lookback}-day lookback, monthly rebal."
            ),
        ))

    def generate_signals(
        self,
        prices: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate inverse-volatility weighted signals."""
        returns = prices.pct_change()

        # Rolling annualized volatility per ETF
        rolling_vol = returns.rolling(
            window=self._vol_lookback, min_periods=self._vol_lookback
        ).std() * np.sqrt(252)

        # Monthly rebalance dates
        monthly_dates = prices.resample("BME").last().index

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        current_weights = pd.Series(0.0, index=prices.columns)

        for date in prices.index:
            if date in monthly_dates:
                vol = rolling_vol.loc[date].dropna()
                vol = vol[vol > 0]  # skip zero-vol

                if len(vol) == 0:
                    current_weights = pd.Series(0.0, index=prices.columns)
                else:
                    # Inverse-vol weights (raw)
                    inv_vol = 1.0 / vol
                    raw_weights = inv_vol / inv_vol.sum()

                    # Estimate portfolio vol under these weights
                    # Simplified: assume uncorrelated for speed
                    port_vol = np.sqrt(
                        (raw_weights ** 2 * vol ** 2).sum()
                    )

                    # Scale factor to hit target vol
                    if port_vol > 0:
                        scale = self._target_vol / port_vol
                    else:
                        scale = 0.0

                    adjusted = raw_weights * scale

                    # Cap total weight at 1.0 (no leverage)
                    total = adjusted.sum()
                    if total > 1.0:
                        adjusted = adjusted / total

                    current_weights = pd.Series(0.0, index=prices.columns)
                    for ticker in adjusted.index:
                        current_weights[ticker] = max(adjusted[ticker], 0.0)

            weights.loc[date] = current_weights

        # Drop warmup
        weights = weights.iloc[self._vol_lookback:]
        return weights
