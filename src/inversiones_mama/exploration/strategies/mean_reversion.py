"""RSI Mean Reversion strategy.

Category: Mean Reversion (5.2)

Logic:
  1. Compute N-period RSI for each ETF
  2. BUY signal when RSI < oversold_threshold (default: 30)
  3. SELL signal when RSI > overbought_threshold (default: 70)
  4. Hold between signals
  5. Equal-weight across triggered positions
  6. Daily signal evaluation (captures short-term reversals)

Parameters:
  - rsi_period (int): RSI lookback in trading days (default: 14)
  - oversold (float): RSI level to enter (default: 30)
  - overbought (float): RSI level to exit (default: 70)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from inversiones_mama.exploration.base import Strategy, StrategyMeta


def _compute_rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute the Relative Strength Index for all columns."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothed moving average
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


class RSIMeanReversion(Strategy):
    """Buy oversold, sell overbought — RSI-based mean reversion."""

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        self._rsi_period = rsi_period
        self._oversold = oversold
        self._overbought = overbought
        super().__init__(StrategyMeta(
            name=f"RSIMeanRev_P{rsi_period}_OS{int(oversold)}_OB{int(overbought)}",
            category="mean_reversion",
            parameters={
                "rsi_period": rsi_period,
                "oversold": oversold,
                "overbought": overbought,
            },
            description=(
                f"RSI mean reversion: buy below RSI {oversold}, "
                f"sell above RSI {overbought}, period={rsi_period}."
            ),
        ))

    def generate_signals(
        self,
        prices: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate RSI-based mean reversion signals."""
        rsi = _compute_rsi(prices, self._rsi_period)

        # Track positions (state machine per asset)
        in_position = pd.DataFrame(False, index=prices.index, columns=prices.columns)

        for ticker in prices.columns:
            holding = False
            for i, date in enumerate(prices.index):
                if pd.isna(rsi.loc[date, ticker]):
                    continue

                if not holding and rsi.loc[date, ticker] < self._oversold:
                    holding = True
                elif holding and rsi.loc[date, ticker] > self._overbought:
                    holding = False

                in_position.loc[date, ticker] = holding

        # Convert to equal-weight across active positions
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        n_active = in_position.sum(axis=1)

        for date in prices.index:
            n = n_active[date]
            if n > 0:
                for ticker in prices.columns:
                    if in_position.loc[date, ticker]:
                        weights.loc[date, ticker] = 1.0 / n

        # Drop RSI warmup period
        warmup = self._rsi_period + 5
        weights = weights.iloc[warmup:]
        return weights
