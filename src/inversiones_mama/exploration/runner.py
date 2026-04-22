"""Exploration backtest runner — the engine that evaluates strategies.

Handles:
  1. Chronological train/test split (NO random shuffling)
  2. Signal generation via Strategy.generate_signals()
  3. Portfolio return calculation with weight changes
  4. Transaction cost deduction via backtest.costs
  5. Metric computation via simulation.metrics (including DSR)
  6. Automatic rejection gate (DSR < 0.1)
  7. Aggregation of n_trials across multiple strategy runs

This module DOES NOT import from sizing, execution, or validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from inversiones_mama.backtest.costs import ibkr_commission, estimate_slippage
from inversiones_mama.simulation.metrics import compute_all_metrics, PerformanceMetrics
from inversiones_mama.exploration.base import Strategy

log = logging.getLogger(__name__)

# DSR rejection threshold per mandate Section 6.4
DSR_REJECTION_THRESHOLD: float = 0.10


@dataclass(frozen=True)
class TradeRecord:
    """Single trade record for the trade log."""

    date: str
    ticker: str
    direction: str      # "buy" | "sell" | "rebalance"
    weight_before: float
    weight_after: float
    dollar_amount: float
    shares: int
    commission: float
    slippage: float


@dataclass
class StrategyResult:
    """Complete result of a strategy backtest."""

    # --- Identity ---
    strategy_name: str
    category: str
    parameters: dict[str, Any]
    # --- Status ---
    status: str  # "candidate" | "rejected"
    rejection_reason: str | None = None
    # --- Metrics (out-of-sample / test period) ---
    metrics_oos: PerformanceMetrics | None = None
    # --- Metrics (in-sample / train period) ---
    metrics_is: PerformanceMetrics | None = None
    # --- Equity curve ---
    equity_curve: pd.Series | None = None
    # --- Trade log ---
    trade_log: list[TradeRecord] = field(default_factory=list)
    # --- Split info ---
    train_start: str = ""
    train_end: str = ""
    test_start: str = ""
    test_end: str = ""
    n_trials: int = 1
    # --- Timing ---
    run_timestamp: str = ""

    def to_summary_dict(self) -> dict[str, Any]:
        """Generate the mandate-required JSON summary."""
        m = self.metrics_oos
        return {
            "strategy_name": self.strategy_name,
            "category": self.category,
            "parameters": self.parameters,
            "cagr": round(m.annualized_return, 6) if m else None,
            "sharpe": round(m.sharpe_ratio, 4) if m else None,
            "sortino": round(m.sortino_ratio, 4) if m else None,
            "max_drawdown": round(m.max_drawdown, 4) if m else None,
            "dsr": round(m.deflated_sharpe, 4) if m else None,
            "hit_rate": round(m.hit_rate, 4) if m else None,
            "profit_factor": round(m.profit_factor, 4) if m else None,
            "n_observations_oos": m.n_observations if m else None,
            "n_trials": self.n_trials,
            "train_period": f"{self.train_start} → {self.train_end}",
            "test_period": f"{self.test_start} → {self.test_end}",
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "run_timestamp": self.run_timestamp,
        }


def _compute_portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """Compute daily portfolio returns from weight and return matrices.

    weights are assumed to be the beginning-of-period weights.
    Portfolio return = sum(w_i * r_i) for each day.
    """
    # Align: only use dates and tickers present in both
    common_dates = weights.index.intersection(returns.index)
    common_tickers = weights.columns.intersection(returns.columns)

    if len(common_dates) == 0 or len(common_tickers) == 0:
        return pd.Series(dtype=np.float64)

    w = weights.loc[common_dates, common_tickers].fillna(0.0)
    r = returns.loc[common_dates, common_tickers].fillna(0.0)

    port_ret = (w * r).sum(axis=1)
    port_ret.name = "portfolio_return"
    return port_ret


def _estimate_rebalance_cost_bps(
    old_weights: pd.Series,
    new_weights: pd.Series,
    base_bps: float = 5.0,
    commission_bps: float = 3.5,
) -> float:
    """Quick-and-dirty rebalance cost in bps of portfolio value.

    turnover = sum(|w_new - w_old|) / 2
    cost_bps = turnover * (spread_bps + commission_bps)

    This is a simplified model for exploration speed. The production
    backtest engine uses the full IBKR Tiered model.
    """
    delta = (new_weights - old_weights).abs()
    turnover = delta.sum() / 2.0  # one-way turnover
    cost = turnover * (base_bps + commission_bps) / 10_000
    return cost


def _build_trade_log(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    portfolio_value: float = 5000.0,
) -> list[TradeRecord]:
    """Build a simplified trade log from weight changes."""
    records: list[TradeRecord] = []
    prev_w = pd.Series(0.0, index=weights.columns)

    for date in weights.index:
        curr_w = weights.loc[date]
        for ticker in weights.columns:
            old = prev_w.get(ticker, 0.0)
            new = curr_w[ticker]
            delta = new - old

            if abs(delta) < 0.001:
                continue

            dollar = abs(delta) * portfolio_value
            price = prices.loc[date, ticker] if date in prices.index and ticker in prices.columns else 0.0

            if price > 0:
                shares = int(round(dollar / price))
            else:
                shares = 0

            comm = ibkr_commission(shares, price) if shares > 0 and price > 0 else 0.0
            slip = estimate_slippage(shares, price) if shares > 0 and price > 0 else 0.0

            direction = "buy" if delta > 0 else "sell"

            records.append(TradeRecord(
                date=str(date.date()) if hasattr(date, 'date') else str(date),
                ticker=ticker,
                direction=direction,
                weight_before=round(old, 4),
                weight_after=round(new, 4),
                dollar_amount=round(dollar, 2),
                shares=shares,
                commission=round(comm, 4),
                slippage=round(slip, 4),
            ))

        prev_w = curr_w.copy()

    return records


def run_strategy(
    strategy: Strategy,
    prices: pd.DataFrame,
    train_frac: float = 0.6,
    n_trials: int = 1,
    portfolio_value: float = 5000.0,
    cost_deduction: bool = True,
) -> StrategyResult:
    """Run a single strategy through the exploration pipeline.

    Parameters
    ----------
    strategy : Strategy
        The strategy instance to evaluate.
    prices : DataFrame
        Adjusted close prices; index=date, columns=ticker.
    train_frac : float
        Fraction of data for training (default: 60% train, 40% test).
    n_trials : int
        Total number of strategies tested so far (for DSR).
    portfolio_value : float
        Notional portfolio value (default: $5,000).
    cost_deduction : bool
        Whether to deduct transaction costs from returns.

    Returns
    -------
    StrategyResult
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Chronological split (NO random shuffling) ---
    n = len(prices)
    split_idx = int(n * train_frac)

    if split_idx < 60 or (n - split_idx) < 30:
        return StrategyResult(
            strategy_name=strategy.name,
            category=strategy.category,
            parameters=strategy.parameters,
            status="rejected",
            rejection_reason="Insufficient data for train/test split",
            run_timestamp=timestamp,
            n_trials=n_trials,
        )

    prices_train = prices.iloc[:split_idx]
    prices_test = prices.iloc[split_idx:]

    train_start = str(prices_train.index[0].date())
    train_end = str(prices_train.index[-1].date())
    test_start = str(prices_test.index[0].date())
    test_end = str(prices_test.index[-1].date())

    log.info(
        "%s: train=%s→%s (%d days), test=%s→%s (%d days)",
        strategy.name, train_start, train_end, len(prices_train),
        test_start, test_end, len(prices_test),
    )

    # --- Generate signals on FULL data, then slice ---
    # Strategy sees train data for calibration, but we evaluate on test.
    # The strategy's generate_signals gets the full price history, but the
    # runner only evaluates OOS performance on the test period.
    try:
        weights_full = strategy.generate_signals(prices)
    except Exception as e:
        return StrategyResult(
            strategy_name=strategy.name,
            category=strategy.category,
            parameters=strategy.parameters,
            status="rejected",
            rejection_reason=f"Signal generation failed: {e}",
            run_timestamp=timestamp,
            n_trials=n_trials,
        )

    # Validate weights
    if weights_full.empty or len(weights_full) == 0:
        return StrategyResult(
            strategy_name=strategy.name,
            category=strategy.category,
            parameters=strategy.parameters,
            status="rejected",
            rejection_reason="Strategy produced empty signals",
            run_timestamp=timestamp,
            n_trials=n_trials,
        )

    # --- Compute returns ---
    returns = prices.pct_change().iloc[1:]

    # In-sample (train period)
    weights_is = weights_full.loc[weights_full.index.intersection(prices_train.index)]
    returns_is = returns.loc[returns.index.intersection(prices_train.index)]
    port_ret_is = _compute_portfolio_returns(weights_is, returns_is)

    # Out-of-sample (test period)
    weights_oos = weights_full.loc[weights_full.index.intersection(prices_test.index)]
    returns_oos = returns.loc[returns.index.intersection(prices_test.index)]
    port_ret_oos = _compute_portfolio_returns(weights_oos, returns_oos)

    if len(port_ret_oos) < 10:
        return StrategyResult(
            strategy_name=strategy.name,
            category=strategy.category,
            parameters=strategy.parameters,
            status="rejected",
            rejection_reason=f"OOS period too short ({len(port_ret_oos)} days)",
            run_timestamp=timestamp,
            n_trials=n_trials,
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
        )

    # --- Cost deduction (simplified for exploration speed) ---
    if cost_deduction and len(weights_oos) > 1:
        cost_series = pd.Series(0.0, index=port_ret_oos.index)
        prev_w = weights_oos.iloc[0]
        for i in range(1, len(weights_oos)):
            curr_w = weights_oos.iloc[i]
            cost_bps = _estimate_rebalance_cost_bps(prev_w, curr_w)
            cost_series.iloc[i] = cost_bps
            prev_w = curr_w
        port_ret_oos = port_ret_oos - cost_series

    # --- Metrics ---
    metrics_oos = compute_all_metrics(port_ret_oos, n_trials=max(n_trials, 1))
    metrics_is = compute_all_metrics(port_ret_is, n_trials=1) if len(port_ret_is) > 10 else None

    # --- Equity curve ---
    equity = (1 + port_ret_oos).cumprod() * portfolio_value
    equity.name = "equity"

    # --- Trade log ---
    trade_log = _build_trade_log(weights_oos, prices_test, portfolio_value)

    # --- DSR gate (Section 6.4 of mandate) ---
    status = "candidate"
    rejection_reason = None

    if metrics_oos.sharpe_ratio < 0:
        status = "rejected"
        rejection_reason = f"Negative Sharpe: {metrics_oos.sharpe_ratio:.4f}"
    elif metrics_oos.deflated_sharpe < DSR_REJECTION_THRESHOLD:
        status = "rejected"
        rejection_reason = f"DSR below threshold: {metrics_oos.deflated_sharpe:.4f} < {DSR_REJECTION_THRESHOLD}"

    # --- Instability check: IS vs OOS Sharpe divergence ---
    if status == "candidate" and metrics_is is not None:
        sharpe_drop = metrics_is.sharpe_ratio - metrics_oos.sharpe_ratio
        if sharpe_drop > 1.5:
            status = "rejected"
            rejection_reason = (
                f"Unstable: IS Sharpe={metrics_is.sharpe_ratio:.2f} → "
                f"OOS Sharpe={metrics_oos.sharpe_ratio:.2f} (drop={sharpe_drop:.2f})"
            )

    return StrategyResult(
        strategy_name=strategy.name,
        category=strategy.category,
        parameters=strategy.parameters,
        status=status,
        rejection_reason=rejection_reason,
        metrics_oos=metrics_oos,
        metrics_is=metrics_is,
        equity_curve=equity,
        trade_log=trade_log,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        n_trials=n_trials,
        run_timestamp=timestamp,
    )


def run_batch(
    strategies: list[Strategy],
    prices: pd.DataFrame,
    train_frac: float = 0.6,
    portfolio_value: float = 5000.0,
) -> list[StrategyResult]:
    """Run a batch of strategies, tracking cumulative n_trials for DSR.

    The n_trials counter increments with each strategy, penalizing
    later strategies more heavily via the DSR multiple-testing correction.
    """
    results: list[StrategyResult] = []

    for i, strategy in enumerate(strategies):
        n_trials = i + 1  # cumulative count for DSR penalty
        log.info("--- Running %d/%d: %s ---", n_trials, len(strategies), strategy)
        result = run_strategy(
            strategy=strategy,
            prices=prices,
            train_frac=train_frac,
            n_trials=n_trials,
            portfolio_value=portfolio_value,
        )
        results.append(result)

        status_icon = "✓" if result.status == "candidate" else "✗"
        sr = result.metrics_oos.sharpe_ratio if result.metrics_oos else 0
        dsr = result.metrics_oos.deflated_sharpe if result.metrics_oos else 0
        log.info(
            "  %s %s | SR=%.2f | DSR=%.3f | %s",
            status_icon, strategy.name, sr, dsr, result.status,
        )

    n_candidates = sum(1 for r in results if r.status == "candidate")
    log.info(
        "Batch complete: %d/%d candidates (%d rejected)",
        n_candidates, len(results), len(results) - n_candidates,
    )
    return results
