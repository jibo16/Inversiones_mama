"""Walk-forward backtest engine for the 6-factor + Risk-Constrained Kelly strategy.

Integrates all Agent 1 and Agent 2 deliverables into a single loop that:

1. At each rebalance date (monthly by default), fits the 6-factor OLS on the
   trailing ``lookback_days`` window, computes composite mu and sample
   covariance, and solves Risk-Constrained Kelly for target weights.
2. Applies Agent 2's IBKR Tiered commission + slippage model to the
   rebalance trade list, deducting the total cost from portfolio wealth.
3. Between rebalances, weights drift with realized asset returns (dollar
   holdings compound; no free intraday rebalancing).
4. Produces a daily return series that downstream metrics and Monte Carlo
   gates consume.

The engine is strategy-agnostic in principle but wires the v1a defaults
(6-factor + RCK) directly. A future refactor can parameterize the
"select + size" functions for alternative strategies.

Public API
----------
``BacktestConfig`` — tunable parameters (dates, cadence, caps, lookback).
``BacktestResult`` — daily returns, wealth path, weights history, cost log.
``walk_forward_backtest(prices, factors, config) -> BacktestResult``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..backtest.costs import RebalanceCost, portfolio_rebalance_cost
from ..config import (
    KELLY_FRACTION,
    LOOKBACK_DAYS,
    MAX_WEIGHT_PER_NAME,
    RCK_MAX_DRAWDOWN_PROBABILITY,
    RCK_MAX_DRAWDOWN_THRESHOLD,
)
from ..models.factor_regression import (
    compute_composite_mu,
    factor_premia,
    fit_factor_loadings,
)
from ..sizing.kelly import solve_rck

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration and result types                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BacktestConfig:
    """Knobs for a walk-forward backtest run.

    Defaults mirror :mod:`inversiones_mama.config` so a zero-arg
    ``BacktestConfig()`` produces the Jorge-approved v1a strategy.
    """

    initial_capital: float = 5000.0
    rebalance_freq: str = "ME"  # pandas freq alias: "ME" = month-end, "QE" = quarter-end
    lookback_days: int = LOOKBACK_DAYS
    kelly_fraction: float = KELLY_FRACTION
    per_name_cap: float = MAX_WEIGHT_PER_NAME
    rck_alpha: float = RCK_MAX_DRAWDOWN_THRESHOLD
    rck_beta: float = RCK_MAX_DRAWDOWN_PROBABILITY
    # Restrict the backtest date range (None = use full prices range)
    start: datetime | None = None
    end: datetime | None = None
    # Apply transaction costs? (set False for a zero-cost accounting sanity test)
    apply_costs: bool = True


@dataclass(frozen=True)
class RebalanceRecord:
    """Snapshot of one rebalance event."""

    date: pd.Timestamp
    current_weights: pd.Series     # weights BEFORE the rebalance (drifted)
    target_weights: pd.Series      # weights AFTER the rebalance
    mu_estimated: pd.Series        # composite mu used for this rebalance
    kelly_growth_rate: float       # Kelly g(w*) the solver produced
    kelly_status: str              # CVXPY status
    cost: RebalanceCost            # cost model output


@dataclass(frozen=True)
class BacktestResult:
    """Output of :func:`walk_forward_backtest`."""

    config: BacktestConfig
    daily_returns: pd.Series       # day-by-day portfolio returns
    wealth: pd.Series              # $ value of the portfolio by day
    weights_history: pd.DataFrame  # target weights at each rebalance date
    rebalance_records: list[RebalanceRecord] = field(default_factory=list)

    @property
    def final_wealth(self) -> float:
        return float(self.wealth.iloc[-1]) if len(self.wealth) else self.config.initial_capital

    @property
    def cumulative_cost(self) -> float:
        return sum(r.cost.total_cost for r in self.rebalance_records)

    @property
    def annualized_turnover_cost(self) -> float:
        """Total transaction costs / (initial capital) / years elapsed."""
        if not self.rebalance_records or len(self.daily_returns) < 2:
            return 0.0
        span_days = (self.daily_returns.index[-1] - self.daily_returns.index[0]).days
        years = max(span_days / 365.25, 1e-6)
        return self.cumulative_cost / self.config.initial_capital / years

    @property
    def rebalance_dates(self) -> list[pd.Timestamp]:
        return [r.date for r in self.rebalance_records]


# --------------------------------------------------------------------------- #
# Walk-forward loop                                                           #
# --------------------------------------------------------------------------- #


def _rebalance_schedule(
    dates: pd.DatetimeIndex,
    freq: str,
    skip_warmup: int,
) -> set[pd.Timestamp]:
    """Pick the rebalance dates: last available trading day in each period.

    ``dates`` is the full series of trading dates available. ``skip_warmup``
    drops the first N trading days so we have enough history for the first
    fit.
    """
    if len(dates) <= skip_warmup:
        return set()
    eligible = dates[skip_warmup:]
    # Group by period, take the last trading date in each
    grouped = pd.Series(eligible, index=eligible).groupby(
        pd.Grouper(freq=freq)
    ).last()
    return set(pd.Timestamp(d) for d in grouped.dropna())


def walk_forward_backtest(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run the full 6-factor + RCK walk-forward backtest.

    Parameters
    ----------
    prices : DataFrame
        Daily adjusted closes, index=date, columns=ticker.
    factors : DataFrame
        6-factor + RF panel from :func:`data.factors.load_factor_returns`.
    config : BacktestConfig, optional
        Defaults to ``BacktestConfig()`` (v1a defaults from config.py).
    """
    config = config or BacktestConfig()

    if prices.empty:
        raise ValueError("prices DataFrame is empty")
    if factors.empty:
        raise ValueError("factors DataFrame is empty")

    tickers = list(prices.columns)
    n = len(tickers)

    # Compute asset returns and align to factors
    returns = prices.pct_change().dropna(how="all")
    common = returns.index.intersection(factors.index)
    if len(common) < config.lookback_days + 5:
        raise ValueError(
            f"After aligning prices and factors, only {len(common)} days overlap; "
            f"need at least lookback_days + 5 = {config.lookback_days + 5}."
        )
    returns = returns.loc[common]
    factors_al = factors.loc[common]
    prices_al = prices.loc[common]

    # Optional date clipping
    if config.start is not None:
        mask = returns.index >= pd.Timestamp(config.start)
        returns = returns.loc[mask]
        factors_al = factors_al.loc[mask]
        prices_al = prices_al.loc[mask]
    if config.end is not None:
        mask = returns.index <= pd.Timestamp(config.end)
        returns = returns.loc[mask]
        factors_al = factors_al.loc[mask]
        prices_al = prices_al.loc[mask]

    if len(returns) <= config.lookback_days:
        raise ValueError(
            f"After date clipping, only {len(returns)} days remain; need > lookback_days."
        )

    # Rebalance schedule: last trading day of each period, skipping warmup
    rebal_set = _rebalance_schedule(returns.index, config.rebalance_freq, config.lookback_days)

    # State: dollar holdings per asset + cash
    weights_dollar = np.zeros(n)
    cash = float(config.initial_capital)
    wealth = float(config.initial_capital)

    # Logs
    wealth_records: dict[pd.Timestamp, float] = {}
    return_records: dict[pd.Timestamp, float] = {}
    weights_history: dict[pd.Timestamp, np.ndarray] = {}
    rebalance_records: list[RebalanceRecord] = []

    dates_arr = returns.index
    returns_arr = returns.to_numpy(dtype=np.float64)

    for t, date in enumerate(dates_arr):
        asset_r = returns_arr[t]
        # Apply today's asset returns to dollar holdings
        weights_dollar = weights_dollar * (1.0 + asset_r)
        wealth_before = wealth
        wealth = float(weights_dollar.sum() + cash)
        daily_ret = (wealth - wealth_before) / wealth_before if wealth_before > 0 else 0.0

        wealth_records[date] = wealth
        return_records[date] = daily_ret

        # Rebalance at end of day if scheduled AND warmup satisfied
        if date in rebal_set and t >= config.lookback_days:
            lb = config.lookback_days
            train_returns = returns.iloc[t - lb + 1 : t + 1]
            train_factors = factors_al.iloc[t - lb + 1 : t + 1]

            try:
                loadings = fit_factor_loadings(train_returns, train_factors)
                premia = factor_premia(train_factors, lookback_days=lb)
                mu = compute_composite_mu(loadings, premia).reindex(tickers).fillna(0.0)
                Sigma = train_returns.cov().reindex(index=tickers, columns=tickers)

                result = solve_rck(
                    mu,
                    Sigma,
                    fraction=config.kelly_fraction,
                    cap=config.per_name_cap,
                    alpha=config.rck_alpha,
                    beta=config.rck_beta,
                )
            except Exception as exc:  # noqa: BLE001 - we want to continue on any solver failure
                log.warning("RCK fit/solve failed at %s: %s; holding current weights.", date, exc)
                continue

            target_w = result.weights.reindex(tickers).fillna(0.0)

            # Current (drifted) weights
            if wealth > 0:
                current_w = pd.Series(weights_dollar / wealth, index=tickers)
            else:
                current_w = pd.Series(np.zeros(n), index=tickers)

            # Cost
            cost: RebalanceCost
            if config.apply_costs:
                cost = portfolio_rebalance_cost(
                    current_weights=current_w,
                    target_weights=target_w,
                    portfolio_value=wealth,
                    prices=prices_al.loc[date],
                )
                wealth -= cost.total_cost
            else:
                cost = RebalanceCost(
                    total_commission=0.0,
                    total_slippage=0.0,
                    total_cost=0.0,
                    cost_pct_portfolio=0.0,
                    cost_pct_turnover=0.0,
                    turnover_value=0.0,
                    turnover_pct=0.0,
                    n_trades=0,
                    trade_details=[],
                )

            # Snap to target allocations at post-cost wealth
            weights_dollar = target_w.to_numpy(dtype=np.float64) * wealth
            cash = float(wealth - weights_dollar.sum())

            weights_history[date] = target_w.to_numpy(dtype=np.float64).copy()
            rebalance_records.append(
                RebalanceRecord(
                    date=pd.Timestamp(date),
                    current_weights=current_w,
                    target_weights=target_w,
                    mu_estimated=mu,
                    kelly_growth_rate=result.growth_rate,
                    kelly_status=result.status,
                    cost=cost,
                )
            )

    # Assemble DataFrames
    daily_returns = pd.Series(return_records, name="daily_returns").sort_index()
    wealth_series = pd.Series(wealth_records, name="wealth").sort_index()
    weights_df = pd.DataFrame.from_dict(weights_history, orient="index", columns=tickers)
    weights_df.index.name = "date"

    return BacktestResult(
        config=config,
        daily_returns=daily_returns,
        wealth=wealth_series,
        weights_history=weights_df,
        rebalance_records=rebalance_records,
    )
