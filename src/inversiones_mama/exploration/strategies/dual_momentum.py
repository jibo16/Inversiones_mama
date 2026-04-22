"""Dual Momentum (Hybrid) strategy.

Category: Hybrid (5.5)

Logic:
  1. Rank ETFs by trailing N-day return (cross-sectional component)
  2. Filter: only hold assets with positive absolute return (time-series filter)
  3. If any pass → equal-weight top-K among those with positive return
  4. If NONE pass → 100% TLT (risk-off asset)
  5. Monthly rebalance

This combines cross-sectional momentum (relative ranking) with
absolute momentum (trend filter) — the Gary Antonacci dual momentum approach.

Parameters:
  - lookback (int): trailing return window in trading days (default: 120)
  - top_k (int): max number of top performers to hold (default: 3)
  - risk_off_asset (str): asset to hold when nothing passes (default: "TLT")
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from inversiones_mama.exploration.base import Strategy, StrategyMeta


class DualMomentum(Strategy):
    """Rank by relative momentum, filter by absolute momentum."""

    def __init__(
        self,
        lookback: int = 120,
        top_k: int = 3,
        risk_off_asset: str = "TLT",
    ) -> None:
        self._lookback = lookback
        self._top_k = top_k
        self._risk_off = risk_off_asset
        super().__init__(StrategyMeta(
            name=f"DualMom_L{lookback}_K{top_k}",
            category="hybrid",
            parameters={
                "lookback": lookback,
                "top_k": top_k,
                "risk_off_asset": risk_off_asset,
            },
            description=(
                f"Dual momentum: rank by {lookback}-day return, "
                f"filter positive absolute return, top-{top_k} equal-weight, "
                f"risk-off={risk_off_asset}."
            ),
        ))

    def generate_signals(
        self,
        prices: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate dual momentum signals."""
        trailing_ret = prices.pct_change(periods=self._lookback)

        # Monthly rebalance dates
        monthly_dates = prices.resample("BME").last().index

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        current_weights = pd.Series(0.0, index=prices.columns)

        for date in prices.index:
            if date in monthly_dates:
                rets = trailing_ret.loc[date].dropna()

                # Step 1: filter for positive absolute returns only
                positive = rets[rets > 0]

                current_weights = pd.Series(0.0, index=prices.columns)

                if len(positive) == 0:
                    # Risk-off: 100% TLT (or other safe asset)
                    if self._risk_off in prices.columns:
                        current_weights[self._risk_off] = 1.0
                else:
                    # Step 2: rank by return, select top-K
                    n_select = min(self._top_k, len(positive))
                    top = positive.nlargest(n_select)
                    w = 1.0 / n_select
                    for ticker in top.index:
                        current_weights[ticker] = w

            weights.loc[date] = current_weights

        # Drop warmup
        weights = weights.iloc[self._lookback:]
        return weights
