"""Dual Momentum (Antonacci) strategy.

Category: Hybrid (5.5)

Logic (fixed 2026-04-23 after forensic audit §5.1):
  1. Compute the trailing N-day return per asset.
  2. Apply a MARKET-WIDE absolute-momentum filter: if the median of the
     universe's trailing returns is negative (broad bear market), go
     risk-off (hold ``risk_off_asset`` at 100%, else stay in cash).
  3. Otherwise (market is risk-on): equal-weight the top-K assets by
     trailing return.
  4. Monthly rebalance.

This matches Gary Antonacci's original Dual Momentum formulation more
faithfully: absolute momentum is evaluated AT THE MARKET LEVEL, not
as a per-asset filter that's trivially redundant on wide universes.

The previous implementation applied the absolute-momentum filter
per-asset (``positive = rets[rets > 0]``) before picking top-K. On a
1,512-ticker universe the top 3 by cross-sectional return are always
positive, so the filter never bit -- making the strategy's signal
identical to ``momentum_xsec``. The forensic audit measured a max
absolute-Sharpe difference of 0.002 between the two strategies.

Parameters:
  - lookback (int): trailing return window in trading days (default: 120)
  - top_k (int): max number of top performers to hold (default: 3)
  - risk_off_asset (str): asset to hold in bear markets (default: "TLT";
      if not present in the universe, the strategy holds cash)
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
        """Generate dual momentum signals with market-wide absolute filter."""
        trailing_ret = prices.pct_change(periods=self._lookback)

        # Monthly rebalance dates
        monthly_dates = prices.resample("BME").last().index

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        current_weights = pd.Series(0.0, index=prices.columns)

        for date in prices.index:
            if date in monthly_dates:
                rets = trailing_ret.loc[date].dropna()
                current_weights = pd.Series(0.0, index=prices.columns)

                if len(rets) == 0:
                    # No data -> hold cash
                    pass
                elif float(rets.median()) <= 0.0:
                    # Market-wide absolute-momentum filter: the median of
                    # trailing returns is <= 0, i.e. broad bear market.
                    # Go risk-off: hold the configured safe asset if it
                    # is in the universe, else stay in cash.
                    if self._risk_off in prices.columns:
                        current_weights[self._risk_off] = 1.0
                else:
                    # Risk-on: equal-weight top-K by trailing return.
                    n_select = min(self._top_k, len(rets))
                    top = rets.nlargest(n_select)
                    w = 1.0 / n_select
                    for ticker in top.index:
                        current_weights[ticker] = w

            weights.loc[date] = current_weights

        # Drop warmup
        weights = weights.iloc[self._lookback:]
        return weights
