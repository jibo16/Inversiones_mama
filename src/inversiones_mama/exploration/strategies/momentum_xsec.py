"""Cross-Sectional Momentum strategy.

Category: Momentum (5.1)

Logic:
  1. Rank all ETFs by trailing N-day total return
  2. Select the top-K performers
  3. Equal-weight across the top-K positions
  4. Rebalance monthly

Parameters:
  - lookback (int): trailing return window in trading days (default: 120)
  - top_k (int): number of top performers to hold (default: 3)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from inversiones_mama.exploration.base import Strategy, StrategyMeta


class CrossSectionalMomentum(Strategy):
    """Long the top-K ETFs ranked by trailing return."""

    def __init__(
        self,
        lookback: int = 120,
        top_k: int = 3,
    ) -> None:
        self._lookback = lookback
        self._top_k = top_k
        super().__init__(StrategyMeta(
            name=f"XSMomentum_L{lookback}_K{top_k}",
            category="momentum",
            parameters={"lookback": lookback, "top_k": top_k},
            description=(
                f"Cross-sectional momentum: long top-{top_k} ETFs by "
                f"{lookback}-day trailing return, equal-weight, monthly rebal."
            ),
        ))

    def generate_signals(
        self,
        prices: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate monthly-rebalanced momentum signals."""
        # Trailing return over lookback window
        trailing_ret = prices.pct_change(periods=self._lookback)

        # Monthly rebalance dates (last trading day of each month)
        monthly_dates = prices.resample("BME").last().index

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        current_weights = pd.Series(0.0, index=prices.columns)

        for date in prices.index:
            if date in monthly_dates:
                # Rank and select top-K
                rets = trailing_ret.loc[date].dropna()
                if len(rets) < self._top_k:
                    current_weights = pd.Series(0.0, index=prices.columns)
                else:
                    top = rets.nlargest(self._top_k)
                    # Equal-weight across top-K
                    w = 1.0 / self._top_k
                    current_weights = pd.Series(0.0, index=prices.columns)
                    for ticker in top.index:
                        current_weights[ticker] = w

            weights.loc[date] = current_weights

        # Drop the lookback warmup period
        weights = weights.iloc[self._lookback:]

        return weights
