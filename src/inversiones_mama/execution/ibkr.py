"""IBKR adapter — STUB.

Real implementation deferred until Jorge's IBKR account is live (2026-04-21
status: he is setting it up). This module defines the interface contract so
that:

  * `data/prices.py` can offer `source="ibkr"` as a future fallback;
  * the backtest engine and signal generator code against a stable API;
  * Agent 2 (Risk & Execution) has a predictable surface to integrate against.

When the account is live, replace each `NotImplementedError` with an
`ib_insync.IB` session. The method signatures here are the contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class OrderIntent:
    """A broker-agnostic order descriptor used by the rebalance layer."""

    ticker: str
    shares: int  # positive=buy, negative=sell
    order_type: str = "MKT"  # "MKT" | "LMT"
    limit_price: float | None = None
    tif: str = "DAY"  # time in force


class IBKRAdapter:
    """Facade over ib_insync. All methods are stubs until live wiring.

    Call sites should handle ``NotImplementedError`` by either (a) using
    yfinance for data, or (b) logging the intended order and skipping
    execution (paper mode).
    """

    # ------------------------------------------------------------------ connect

    @classmethod
    def connect(
        cls,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
    ) -> None:
        """Open TWS/Gateway connection. Not yet wired."""
        raise NotImplementedError(
            "IBKR not yet connected — waiting for Jorge's account setup. "
            "Use source='yfinance' for price data until then."
        )

    # -------------------------------------------------------------- market data

    @classmethod
    def fetch_prices(
        cls,
        tickers: Iterable[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Historical daily OHLC via reqHistoricalData. Not yet wired.

        Expected return: wide DataFrame (index=date, columns=ticker, values=close).
        """
        raise NotImplementedError(
            "IBKR historical data requires market-data subscriptions. "
            "Use data.prices.load_prices(source='yfinance') in v1a."
        )

    # ----------------------------------------------------------------- orders

    @classmethod
    def place_order(cls, intent: OrderIntent) -> str:
        """Submit an order, returning an IBKR order ID. Not yet wired."""
        raise NotImplementedError("IBKR live execution not yet wired.")

    @classmethod
    def cancel_order(cls, order_id: str) -> None:
        raise NotImplementedError("IBKR live execution not yet wired.")

    # -------------------------------------------------------------- account

    @classmethod
    def get_positions(cls) -> pd.DataFrame:
        """Current positions. Not yet wired.

        Expected return: DataFrame with columns [ticker, shares, avg_cost,
        market_value, unrealized_pnl].
        """
        raise NotImplementedError("IBKR not yet connected.")

    @classmethod
    def get_account_summary(cls) -> dict[str, float]:
        """Net liquidation value, buying power, etc. Not yet wired."""
        raise NotImplementedError("IBKR not yet connected.")
