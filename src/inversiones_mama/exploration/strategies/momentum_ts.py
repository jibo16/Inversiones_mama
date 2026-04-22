"""Time-Series Momentum (Trend Following) strategy.

Category: Momentum (5.1)

Logic:
  1. For each ETF: compute trailing N-day return
  2. If return > 0 → hold (positive trend)
  3. If return ≤ 0 → go to cash (no position)
  4. Equal-weight across all assets with positive trend
  5. Rebalance monthly

Parameters:
  - lookback (int): trailing return window in trading days (default: 120)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from inversiones_mama.exploration.base import Strategy, StrategyMeta


class TimeSeriesMomentum(Strategy):
    """Hold ETFs with positive trailing returns; equal-weight."""

    def __init__(
        self,
        lookback: int = 120,
    ) -> None:
        self._lookback = lookback
        super().__init__(StrategyMeta(
            name=f"TSMomentum_L{lookback}",
            category="momentum",
            parameters={"lookback": lookback},
            description=(
                f"Time-series momentum: hold any ETF with positive "
                f"{lookback}-day trailing return, equal-weight, monthly rebal."
            ),
        ))

    def generate_signals(
        self,
        prices: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate trend-following signals."""
        trailing_ret = prices.pct_change(periods=self._lookback)

        # Monthly rebalance dates
        monthly_dates = prices.resample("BME").last().index

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        current_weights = pd.Series(0.0, index=prices.columns)

        for date in prices.index:
            if date in monthly_dates:
                rets = trailing_ret.loc[date].dropna()
                positive_trend = rets[rets > 0].index.tolist()

                current_weights = pd.Series(0.0, index=prices.columns)
                if positive_trend:
                    w = 1.0 / len(positive_trend)
                    for ticker in positive_trend:
                        current_weights[ticker] = w

            weights.loc[date] = current_weights

        # Drop warmup
        weights = weights.iloc[self._lookback:]
        return weights
