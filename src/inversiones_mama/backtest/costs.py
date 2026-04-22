"""IBKR Tiered commission model + slippage estimator.

Implements the exact Interactive Brokers Tiered pricing schedule for
US stocks/ETFs, plus a configurable slippage model.

IBKR Tiered US Stock/ETF Schedule (as of 2026):
    - $0.0035 per share
    - Minimum $0.35 per order
    - Maximum 1.0% of trade value
    - Regulatory fees (SEC, FINRA TAF) are approximated as a flat add-on

The slippage model uses a square-root market impact model:
    slippage = base_bps + impact_bps * sqrt(shares / ADV)

For liquid ETFs (SPY, GLD, etc.) with tight spreads and deep books,
5 bps base slippage is conservative. The ADV-scaled impact is a
standard Almgren–Chriss simplification.

Public API
----------
ibkr_commission(shares, price) -> float
    Compute exact IBKR Tiered commission for a single order.

estimate_slippage(shares, price, adv=None, base_bps=5.0) -> float
    Estimate execution slippage for a single order.

total_trade_cost(shares, price, adv=None, ...) -> TradeCost
    Combined commission + slippage for a single trade.

portfolio_rebalance_cost(current_weights, target_weights, portfolio_value,
                         prices, adv=None) -> RebalanceCost
    Full rebalance cost estimation across all positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from inversiones_mama.config import (
    IBKR_FIXED_PER_SHARE,
    IBKR_MAX_PCT_TRADE,
    IBKR_MIN_PER_ORDER,
    IBKR_SLIPPAGE_BPS,
)


# --------------------------------------------------------------------------- #
# Result containers                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TradeCost:
    """Cost breakdown for a single trade."""

    ticker: str
    shares: int
    price: float
    trade_value: float
    commission: float
    slippage: float
    total_cost: float
    cost_bps: float  # total cost in basis points of trade value


@dataclass(frozen=True)
class RebalanceCost:
    """Aggregate cost of a full portfolio rebalance."""

    total_commission: float
    total_slippage: float
    total_cost: float
    cost_pct_portfolio: float  # total cost as % of portfolio value
    cost_pct_turnover: float   # total cost as % of turnover value
    turnover_value: float      # total $ traded (abs sum of all trades)
    turnover_pct: float        # turnover as % of portfolio value
    n_trades: int
    trade_details: list[TradeCost]


# --------------------------------------------------------------------------- #
# Commission model                                                            #
# --------------------------------------------------------------------------- #


def ibkr_commission(
    shares: int,
    price: float,
    per_share: float = IBKR_FIXED_PER_SHARE,
    min_per_order: float = IBKR_MIN_PER_ORDER,
    max_pct_trade: float = IBKR_MAX_PCT_TRADE,
) -> float:
    """Compute the exact IBKR Tiered commission for a US stock/ETF order.

    Parameters
    ----------
    shares : int
        Absolute number of shares traded (sign-agnostic; use abs value).
    price : float
        Execution price per share.
    per_share : float
        Per-share fee (default: $0.0035).
    min_per_order : float
        Minimum commission per order (default: $0.35).
    max_pct_trade : float
        Maximum commission as fraction of trade value (default: 1%).

    Returns
    -------
    float
        Commission in USD.
    """
    abs_shares = abs(shares)
    if abs_shares == 0:
        return 0.0
    if price <= 0:
        raise ValueError(f"Price must be positive, got {price}")

    trade_value = abs_shares * price
    raw_commission = abs_shares * per_share
    # Clamp: floor at min_per_order, cap at max_pct_trade * trade_value
    commission = max(raw_commission, min_per_order)
    commission = min(commission, max_pct_trade * trade_value)

    return round(commission, 4)


# --------------------------------------------------------------------------- #
# Slippage model                                                              #
# --------------------------------------------------------------------------- #


def estimate_slippage(
    shares: int,
    price: float,
    adv: float | None = None,
    base_bps: float = IBKR_SLIPPAGE_BPS,
    impact_coefficient: float = 0.1,
) -> float:
    """Estimate execution slippage for a single trade.

    Uses a square-root market impact model (Almgren–Chriss simplified):

        total_slippage_bps = base_bps + impact_coeff * sqrt(shares / ADV) * 10000

    If ADV is not provided, uses base_bps only (conservative for liquid ETFs).

    Parameters
    ----------
    shares : int
        Absolute number of shares.
    price : float
        Price per share.
    adv : float or None
        Average daily volume in shares. If None, only base slippage is used.
    base_bps : float
        Base slippage in basis points (default: 5 bps).
    impact_coefficient : float
        Scaling factor for the market impact term (default: 0.1).

    Returns
    -------
    float
        Slippage cost in USD.
    """
    abs_shares = abs(shares)
    if abs_shares == 0:
        return 0.0
    if price <= 0:
        raise ValueError(f"Price must be positive, got {price}")

    trade_value = abs_shares * price

    # Base spread cost
    slippage_bps = base_bps

    # Market impact (if ADV is available)
    if adv is not None and adv > 0:
        participation_rate = abs_shares / adv
        # Square-root impact model
        impact_bps = impact_coefficient * np.sqrt(participation_rate) * 10_000
        slippage_bps += impact_bps

    slippage_usd = trade_value * slippage_bps / 10_000
    return round(slippage_usd, 4)


# --------------------------------------------------------------------------- #
# Combined cost                                                               #
# --------------------------------------------------------------------------- #


def total_trade_cost(
    ticker: str,
    shares: int,
    price: float,
    adv: float | None = None,
    base_bps: float = IBKR_SLIPPAGE_BPS,
    per_share: float = IBKR_FIXED_PER_SHARE,
    min_per_order: float = IBKR_MIN_PER_ORDER,
    max_pct_trade: float = IBKR_MAX_PCT_TRADE,
    impact_coefficient: float = 0.1,
) -> TradeCost:
    """Compute combined commission + slippage for a single trade.

    Parameters
    ----------
    ticker : str
        Asset ticker (for labeling).
    shares : int
        Number of shares (positive=buy, negative=sell; abs is used).
    price : float
        Execution price per share.
    adv : float or None
        Average daily volume.
    base_bps : float
        Base slippage in bps.
    per_share, min_per_order, max_pct_trade :
        IBKR commission parameters.
    impact_coefficient : float
        Market impact scaling.

    Returns
    -------
    TradeCost
    """
    abs_shares = abs(shares)
    if abs_shares == 0:
        return TradeCost(
            ticker=ticker, shares=0, price=price, trade_value=0.0,
            commission=0.0, slippage=0.0, total_cost=0.0, cost_bps=0.0,
        )

    trade_value = abs_shares * price

    comm = ibkr_commission(
        abs_shares, price,
        per_share=per_share, min_per_order=min_per_order,
        max_pct_trade=max_pct_trade,
    )
    slip = estimate_slippage(
        abs_shares, price, adv=adv,
        base_bps=base_bps, impact_coefficient=impact_coefficient,
    )
    total = comm + slip
    cost_bps = (total / trade_value) * 10_000 if trade_value > 0 else 0.0

    return TradeCost(
        ticker=ticker,
        shares=abs_shares,
        price=price,
        trade_value=round(trade_value, 2),
        commission=comm,
        slippage=slip,
        total_cost=round(total, 4),
        cost_bps=round(cost_bps, 2),
    )


# --------------------------------------------------------------------------- #
# Portfolio-level rebalance cost                                              #
# --------------------------------------------------------------------------- #


def portfolio_rebalance_cost(
    current_weights: pd.Series,
    target_weights: pd.Series,
    portfolio_value: float,
    prices: pd.Series,
    adv: pd.Series | None = None,
    base_bps: float = IBKR_SLIPPAGE_BPS,
    per_share: float = IBKR_FIXED_PER_SHARE,
    min_per_order: float = IBKR_MIN_PER_ORDER,
    max_pct_trade: float = IBKR_MAX_PCT_TRADE,
    impact_coefficient: float = 0.1,
    min_trade_value: float = 10.0,
) -> RebalanceCost:
    """Estimate the total cost of rebalancing from current to target weights.

    Parameters
    ----------
    current_weights : Series
        Current portfolio weights (index=ticker).
    target_weights : Series
        Target portfolio weights (index=ticker).
    portfolio_value : float
        Total portfolio value in USD.
    prices : Series
        Current prices per share (index=ticker).
    adv : Series or None
        Average daily volume per ticker (index=ticker).
    base_bps : float
        Base slippage bps.
    per_share, min_per_order, max_pct_trade :
        IBKR commission parameters.
    impact_coefficient : float
        Market impact scaling.
    min_trade_value : float
        Minimum trade value to execute (skip tiny rebalances).

    Returns
    -------
    RebalanceCost
    """
    # Align all tickers
    all_tickers = sorted(
        set(current_weights.index) | set(target_weights.index) | set(prices.index)
    )
    curr = current_weights.reindex(all_tickers, fill_value=0.0)
    tgt = target_weights.reindex(all_tickers, fill_value=0.0)
    px = prices.reindex(all_tickers)

    delta_weights = tgt - curr
    trade_details: list[TradeCost] = []
    total_turnover = 0.0

    for ticker in all_tickers:
        dw = delta_weights[ticker]
        trade_value = abs(dw) * portfolio_value

        if trade_value < min_trade_value:
            continue
        if pd.isna(px[ticker]) or px[ticker] <= 0:
            continue

        shares = int(round(trade_value / px[ticker]))
        if shares == 0:
            continue

        ticker_adv = None
        if adv is not None and ticker in adv.index and not pd.isna(adv[ticker]):
            ticker_adv = float(adv[ticker])

        tc = total_trade_cost(
            ticker=ticker,
            shares=shares if dw > 0 else -shares,
            price=float(px[ticker]),
            adv=ticker_adv,
            base_bps=base_bps,
            per_share=per_share,
            min_per_order=min_per_order,
            max_pct_trade=max_pct_trade,
            impact_coefficient=impact_coefficient,
        )
        trade_details.append(tc)
        total_turnover += tc.trade_value

    total_comm = sum(t.commission for t in trade_details)
    total_slip = sum(t.slippage for t in trade_details)
    total_cost = total_comm + total_slip

    return RebalanceCost(
        total_commission=round(total_comm, 4),
        total_slippage=round(total_slip, 4),
        total_cost=round(total_cost, 4),
        cost_pct_portfolio=round(total_cost / portfolio_value * 100, 4) if portfolio_value > 0 else 0.0,
        cost_pct_turnover=round(total_cost / total_turnover * 100, 4) if total_turnover > 0 else 0.0,
        turnover_value=round(total_turnover, 2),
        turnover_pct=round(total_turnover / portfolio_value * 100, 2) if portfolio_value > 0 else 0.0,
        n_trades=len(trade_details),
        trade_details=trade_details,
    )
