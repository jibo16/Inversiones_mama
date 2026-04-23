"""Paper-trading orchestrator for deployment validation.

One rebalance cycle of a configurable allocation strategy:

  1. Pull current positions + cash from the ``ExecutionClient``.
  2. Pull latest prices and factor returns.
  3. Compute target weights by running a single training+solve against
     the trailing-lookback window (same logic as ``backtest/engine.py``
     but for one date instead of a historical path).
  4. Build the order list: signed share counts implied by
     (target_weights * portfolio_value) - current_positions.
  5. For each order: record a ``SignalRecord``, submit through the
     client, record the resulting ``FillRecord``, append to the
     ``TradeLog``.
  6. Persist the trade log and optionally an account snapshot.

``ExecutionClient`` is a Protocol — the orchestrator is entirely broker-
agnostic. ``DryRunClient`` is the zero-risk default that simulates
perfect fills at the expected price and is the only client used in
automated tests. Agent 3's concrete IBKR Client Portal client will
satisfy the same Protocol once wired.

Public API
----------
``ExecutionClient`` Protocol.
``DryRunClient`` — simulated-fills implementation.
``PaperTradingOrchestrator`` — the rebalance driver.
``PaperRebalanceSummary`` — output record with fill stats + weight diff.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..backtest.costs import portfolio_rebalance_cost
from ..config import (
    KELLY_FRACTION,
    LOOKBACK_DAYS,
    MAX_WEIGHT_PER_NAME,
    MAX_WEIGHT_PER_SECTOR,
    MU_SHRINKAGE,
    RCK_MAX_DRAWDOWN_PROBABILITY,
    RCK_MAX_DRAWDOWN_THRESHOLD,
)
from ..models.factor_regression import (
    compute_composite_mu,
    factor_premia,
    fit_factor_loadings,
)
from ..sizing.kelly import solve_rck
from .circuit_breaker import CircuitBreaker
from .ibkr import OrderIntent
from .pdt import PDTTracker
from .trade_log import FillRecord, SignalRecord, TradeLog

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Protocol                                                                    #
# --------------------------------------------------------------------------- #


@runtime_checkable
class ExecutionClient(Protocol):
    """Minimum contract a broker client must satisfy for paper trading.

    All methods run synchronously. Implementations that wrap async libs
    (like ``ib_insync``) should block internally so callers do not need
    to know.
    """

    def get_positions(self) -> dict[str, int | float]:
        """Current positions — mapping ticker -> signed share count."""
        ...

    def get_cash(self) -> float:
        """Available cash balance in account currency (USD)."""
        ...

    def get_latest_price(self, ticker: str) -> float | None:
        """Latest known price for a ticker; None if unavailable."""
        ...

    def submit_order(self, intent: OrderIntent) -> FillRecord:
        """Submit an order and return a FillRecord (may be partial/rejected)."""
        ...


# --------------------------------------------------------------------------- #
# DryRunClient                                                                #
# --------------------------------------------------------------------------- #


class DryRunClient:
    """In-memory simulated broker for testing and zero-risk smoke tests.

    * Fills every order instantly at the expected price (no slippage).
    * Tracks positions and cash internally.
    * Rejects orders it cannot afford (buying > cash + proceeds-from-sells).
    """

    def __init__(
        self,
        starting_cash: float = 5_000.0,
        latest_prices: dict[str, float] | None = None,
    ) -> None:
        self._cash: float = float(starting_cash)
        # int or float — fractional shares supported for parity with the
        # fractional-share path in the real Alpaca client.
        self._positions: dict[str, int | float] = {}
        self._prices: dict[str, float] = dict(latest_prices or {})

    # --- Introspection ---------------------------------------------------
    def get_positions(self) -> dict[str, int | float]:
        return {k: v for k, v in self._positions.items() if v != 0}

    def get_cash(self) -> float:
        return self._cash

    def get_latest_price(self, ticker: str) -> float | None:
        return self._prices.get(ticker)

    def set_latest_price(self, ticker: str, price: float) -> None:
        self._prices[ticker] = float(price)

    # --- Execution -------------------------------------------------------
    def submit_order(self, intent: OrderIntent) -> FillRecord:
        now_order = datetime.now(tz=timezone.utc)
        px = self._prices.get(intent.ticker)
        if px is None or px <= 0:
            log.warning("DryRunClient: no valid price for %s; rejecting", intent.ticker)
            return FillRecord(
                order_time=now_order,
                fill_time=None,
                fill_price=None,
                filled_quantity=0,
                status="rejected",
                broker_order_id=None,
                context={"reason": "no_price"},
            )
        trade_value = abs(intent.shares) * px
        # Check affordability for buys (margin would be a separate check; we're
        # modeling a cash-only account)
        if intent.shares > 0 and trade_value > self._cash + 1e-6:
            return FillRecord(
                order_time=now_order,
                fill_time=None,
                fill_price=None,
                filled_quantity=0,
                status="rejected",
                broker_order_id=None,
                context={"reason": "insufficient_cash", "needed": trade_value, "cash": self._cash},
            )
        # Apply the trade. Preserve fractional qty if the intent used it.
        qty_raw = float(intent.shares)
        qty_is_frac = qty_raw != int(qty_raw)
        new_pos = float(self._positions.get(intent.ticker, 0)) + qty_raw
        if abs(new_pos) < 1e-9:
            self._positions.pop(intent.ticker, None)
        else:
            self._positions[intent.ticker] = (
                new_pos if qty_is_frac or isinstance(self._positions.get(intent.ticker), float)
                else int(round(new_pos))
            )
        self._cash -= qty_raw * px  # buy reduces cash, sell increases
        now_fill = datetime.now(tz=timezone.utc)
        filled_quantity = qty_raw if qty_is_frac else int(qty_raw)
        return FillRecord(
            order_time=now_order,
            fill_time=now_fill,
            fill_price=px,
            filled_quantity=filled_quantity,
            status="filled",
            broker_order_id=str(uuid.uuid4()),
        )


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PaperRebalanceSummary:
    """Output of one rebalance cycle."""

    rebalance_time: datetime
    tickers: tuple[str, ...]
    target_weights: pd.Series
    current_weights_before: pd.Series
    order_count: int
    total_fill_value: float
    estimated_cost: float
    trade_log: TradeLog = field(default_factory=TradeLog)
    # Safety-gate status (populated when a gate halts the cycle)
    halted: bool = False
    halt_reason: str | None = None
    breaker_drawdown: float | None = None

    @property
    def fill_rate(self) -> float:
        return self.trade_log.summary().get("fill_rate", 0.0)


class PaperTradingOrchestrator:
    """Runs ONE rebalance cycle end-to-end against an ``ExecutionClient``.

    Does not schedule — cadence is the caller's responsibility (cron, APScheduler,
    Windows Task Scheduler, etc.). This class is intentionally stateless across
    calls; all state lives in the broker client.
    """

    def __init__(
        self,
        client: ExecutionClient,
        prices_history: pd.DataFrame,
        factors_history: pd.DataFrame,
        *,
        weight_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.Series] | None = None,
        lookback_days: int = LOOKBACK_DAYS,
        kelly_fraction: float = KELLY_FRACTION,
        per_name_cap: float = MAX_WEIGHT_PER_NAME,
        rck_alpha: float = RCK_MAX_DRAWDOWN_THRESHOLD,
        rck_beta: float = RCK_MAX_DRAWDOWN_PROBABILITY,
        mu_shrinkage: float = MU_SHRINKAGE,
        per_sector_cap: float = MAX_WEIGHT_PER_SECTOR,
        sector_map: dict[str, str] | None = None,
        max_deploy_capital: float | None = None,
        fractional_shares: bool = False,
        min_order_notional: float = 1.0,
        strategy_id: str | None = None,
    ) -> None:
        if prices_history.empty:
            raise ValueError("prices_history must be non-empty")
        if factors_history.empty:
            raise ValueError("factors_history must be non-empty")
        if max_deploy_capital is not None and max_deploy_capital <= 0:
            raise ValueError(
                f"max_deploy_capital must be positive or None, got {max_deploy_capital}"
            )

        self.client = client
        self._weight_fn = weight_fn
        self.prices_history = prices_history
        self.factors_history = factors_history
        self.lookback_days = int(lookback_days)
        self.kelly_fraction = float(kelly_fraction)
        self.per_name_cap = float(per_name_cap)
        self.rck_alpha = float(rck_alpha)
        self.rck_beta = float(rck_beta)
        self.mu_shrinkage = float(mu_shrinkage)
        self.per_sector_cap = float(per_sector_cap)
        # Build the sector map once at construction so each rebalance cycle
        # reuses it. If the caller passed one explicitly, honor it.
        if sector_map is not None:
            self.sector_map: dict[str, str] | None = dict(sector_map)
        elif self.per_sector_cap < 1.0:
            try:
                from ..data.sectors import build_sector_map  # lazy

                self.sector_map = build_sector_map(list(prices_history.columns))
            except Exception as exc:  # noqa: BLE001
                log.warning("sector map build failed (%s); sector cap disabled.", exc)
                self.sector_map = None
        else:
            self.sector_map = None
        # Optional deployment cap — when set, target weights are applied
        # only to min(portfolio_value, max_deploy_capital). Useful when the
        # broker seeds a paper account with more cash than the strategy's
        # intended capital (Alpaca paper gives $100k; we care about $5k).
        self.max_deploy_capital = (
            float(max_deploy_capital) if max_deploy_capital is not None else None
        )
        # Fractional-share support. When True, target weights that imply
        # sub-1-share deltas are submitted as fractional qty instead of
        # rounded to 0. Critical for small deployment capital on high-priced
        # equity ETFs (e.g., SPY at $711 with a 0.9% target weight on $5k
        # = $45 budget -> 0.063 shares, which would round to 0 without this).
        self.fractional_shares = bool(fractional_shares)
        # Skip any order whose notional value is below this (dust orders).
        self.min_order_notional = float(min_order_notional)
        # Optional tag attached to the signal_context of every order, so
        # multi-strategy ledger consumers can attribute fills back to a
        # specific strategy bag.
        self.strategy_id = str(strategy_id) if strategy_id else None

    def _compute_target_weights(self) -> pd.Series:
        """Compute target weights using the configured strategy.

        If ``weight_fn`` was provided at construction, delegates to it.
        Otherwise falls back to the default 6-factor + RCK solver.
        """
        if self._weight_fn is not None:
            w = self._weight_fn(self.prices_history, self.factors_history)
            tickers = list(self.prices_history.columns)
            return w.reindex(tickers).fillna(0.0)

        return self._compute_rck_weights()

    def _compute_rck_weights(self) -> pd.Series:
        """Fit 6-factor + RCK on the trailing lookback and return target weights."""
        tickers = list(self.prices_history.columns)
        returns = self.prices_history.pct_change().dropna(how="all")
        common = returns.index.intersection(self.factors_history.index)
        if len(common) < self.lookback_days:
            raise ValueError(
                f"Only {len(common)} aligned rows between prices/factors; need >= {self.lookback_days}"
            )
        train_returns = returns.loc[common].tail(self.lookback_days)
        train_factors = self.factors_history.loc[common].tail(self.lookback_days)

        loadings = fit_factor_loadings(train_returns, train_factors)
        premia = factor_premia(train_factors, lookback_days=self.lookback_days)
        mu = compute_composite_mu(loadings, premia).reindex(tickers).fillna(0.0)
        if self.mu_shrinkage > 0.0:
            mu = (1.0 - self.mu_shrinkage) * mu + self.mu_shrinkage * float(mu.mean())
        Sigma = train_returns.cov().reindex(index=tickers, columns=tickers)

        result = solve_rck(
            mu,
            Sigma,
            fraction=self.kelly_fraction,
            cap=self.per_name_cap,
            alpha=self.rck_alpha,
            beta=self.rck_beta,
            sector_map=self.sector_map,
            sector_cap=self.per_sector_cap,
        )
        return result.weights.reindex(tickers).fillna(0.0)

    def rebalance(
        self,
        signal_context: dict | None = None,
        *,
        circuit_breaker: CircuitBreaker | None = None,
        pdt_tracker: PDTTracker | None = None,
        prior_trade_log: TradeLog | None = None,
    ) -> PaperRebalanceSummary:
        """Execute one rebalance cycle.

        Parameters
        ----------
        signal_context : optional dict attached to every SignalRecord.
        circuit_breaker : optional auto-halt gate. If set, the rebalance
            updates the breaker with current wealth BEFORE placing any
            orders; if it trips, the cycle returns halted=True with
            order_count=0.
        pdt_tracker : optional Pattern-Day-Trader gate. If set, and the
            account is below the PDT equity threshold, same-day round-trip
            orders that would breach the 3-in-5 rule are skipped (logged
            but not submitted).
        prior_trade_log : optional previous TradeLog (only used by
            pdt_tracker so the PDT count carries over across cycles).

        Returns a ``PaperRebalanceSummary`` with the weight vectors, the
        generated order list, the ``TradeLog`` of every signal+fill, and
        the estimated transaction cost.
        """
        now = datetime.now(tz=timezone.utc)
        tickers = list(self.prices_history.columns)

        # Gather current state
        current_positions = self.client.get_positions()
        cash = self.client.get_cash()
        latest_prices = {t: self.client.get_latest_price(t) for t in tickers}

        # Portfolio value (cash + positions valued at latest prices)
        position_value = sum(
            qty * latest_prices[t]
            for t, qty in current_positions.items()
            if t in latest_prices and latest_prices[t] is not None
        )
        portfolio_value = cash + position_value
        if portfolio_value <= 0:
            raise ValueError(f"Non-positive portfolio value: {portfolio_value}")

        # Apply the deployment cap (Alpaca paper seeds $100k; we only want
        # to allocate the strategy's intended capital). Unallocated cash
        # stays as cash at the broker and is untouched by the rebalance.
        if self.max_deploy_capital is not None:
            deployable = min(portfolio_value, self.max_deploy_capital)
        else:
            deployable = portfolio_value
        # Fractional-share buffer: a single-ticker strategy at 100% target
        # weight can compute delta_shares so that qty*px exceeds available
        # cash by a few cents due to floating-point rounding. Leave a 0.5%
        # cushion in that case so the broker can't reject on insufficient
        # cash. For whole-share orders the int(round(...)) path already
        # truncates, so no buffer needed.
        if self.fractional_shares:
            deployable = deployable * 0.995

        # --- Circuit-breaker gate ---
        breaker_dd: float | None = None
        if circuit_breaker is not None:
            status = circuit_breaker.update(portfolio_value, as_of=now)
            breaker_dd = status.current_drawdown
            if status.tripped:
                log.warning(
                    "Circuit breaker TRIPPED at %s — dd=%.2f%% >= threshold=%.2f%%. "
                    "Halting rebalance.",
                    now, status.current_drawdown * 100, status.threshold * 100,
                )
                return PaperRebalanceSummary(
                    rebalance_time=now,
                    tickers=tuple(tickers),
                    target_weights=pd.Series(np.zeros(len(tickers)), index=tickers),
                    current_weights_before=pd.Series(
                        {t: current_positions.get(t, 0) * (latest_prices[t] or 0.0) / portfolio_value
                         for t in tickers},
                        index=tickers,
                    ),
                    order_count=0,
                    total_fill_value=0.0,
                    estimated_cost=0.0,
                    trade_log=TradeLog(),
                    halted=True,
                    halt_reason=f"circuit_breaker_tripped (dd={status.current_drawdown:.2%})",
                    breaker_drawdown=status.current_drawdown,
                )

        # Current weights
        current_w = pd.Series(
            {
                t: (current_positions.get(t, 0) * (latest_prices[t] or 0.0)) / portfolio_value
                for t in tickers
            },
            index=tickers,
        )

        # Target weights from the strategy
        target_w = self._compute_target_weights()

        # Cost estimate (diagnostic; real cost is measured via the trade log)
        prices_series = pd.Series({t: (latest_prices[t] or 0.0) for t in tickers})
        try:
            cost_est = portfolio_rebalance_cost(
                current_weights=current_w,
                target_weights=target_w,
                portfolio_value=portfolio_value,
                prices=prices_series,
            )
            estimated_cost = cost_est.total_cost
        except Exception as exc:  # noqa: BLE001
            log.warning("Cost estimator failed: %s", exc)
            estimated_cost = 0.0

        # Build orders: share deltas from weight deltas at current prices.
        # `deployable` caps target notional (target weights applied to
        # min(portfolio_value, max_deploy_capital) so Alpaca's $100k default
        # doesn't force us to deploy more than the strategy's intended slice).
        # Fractional shares (self.fractional_shares=True) preserve the equity
        # floor on small deployment capital with high-priced ETFs.
        trade_log = TradeLog()
        total_fill_value = 0.0
        n_orders = 0
        pdt_skipped = 0
        for t in tickers:
            px = latest_prices[t]
            if px is None or px <= 0:
                continue
            target_dollar = float(target_w[t]) * deployable
            current_dollar = current_positions.get(t, 0) * px
            delta_dollar = target_dollar - current_dollar

            if self.fractional_shares:
                # Preserve full target granularity. Round to 6 decimals to
                # avoid floating-point noise in the order body.
                delta_shares: float = round(delta_dollar / px, 6)
                if abs(delta_shares * px) < self.min_order_notional:
                    # Dust order; skip.
                    continue
            else:
                # Legacy integer-share behavior for backward compat.
                delta_shares = int(round(delta_dollar / px))
                if delta_shares == 0:
                    continue

            # --- PDT gate ---
            # A rebalance order becomes a "day trade" only if it closes a same-day
            # position opened earlier. For our monthly cadence this is rare, but
            # we guard against it defensively. We treat the PDT check as a soft
            # gate: skip the order and keep going.
            if pdt_tracker is not None and not pdt_tracker.exempt:
                merged_log = _merge_logs(prior_trade_log, trade_log)
                if not pdt_tracker.can_execute_new_day_trade(merged_log, as_of=now.date()):
                    log.warning(
                        "PDT gate blocked order: ticker=%s delta_shares=%.4f, "
                        "window day-trade count exhausted.", t, delta_shares,
                    )
                    pdt_skipped += 1
                    continue

            sig_ctx = dict(signal_context or {})
            if self.strategy_id is not None:
                sig_ctx["strategy_id"] = self.strategy_id
            signal = SignalRecord(
                ticker=t,
                signal_time=now,
                expected_price=px,
                expected_size=delta_shares,
                context=sig_ctx,
            )
            intent = OrderIntent(ticker=t, shares=delta_shares, order_type="MKT")
            fill = self.client.submit_order(intent)
            trade_log.record(signal, fill)
            if fill.status in ("filled", "partial") and fill.fill_price is not None:
                total_fill_value += abs(fill.filled_quantity) * fill.fill_price
            n_orders += 1

        return PaperRebalanceSummary(
            rebalance_time=now,
            tickers=tuple(tickers),
            target_weights=target_w,
            current_weights_before=current_w,
            order_count=n_orders,
            total_fill_value=total_fill_value,
            estimated_cost=float(estimated_cost),
            trade_log=trade_log,
            halted=False,
            halt_reason=(f"pdt_skipped_{pdt_skipped}_orders" if pdt_skipped else None),
            breaker_drawdown=breaker_dd,
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _merge_logs(prior: TradeLog | None, current: TradeLog) -> TradeLog:
    """Return a TradeLog containing entries from both ``prior`` and ``current``."""
    merged = TradeLog()
    if prior is not None:
        for entry in prior:
            merged.append(entry)
    for entry in current:
        merged.append(entry)
    return merged
