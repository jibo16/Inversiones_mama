"""Strategy catalog — the 20 concurrent paper-trading bags.

Defines the meta-portfolio: each ``StrategySpec`` is one bag in the
multi-strategy ledger, with its own allocator, universe, starting cash,
and rebalance cadence. The meta-portfolio runner iterates this catalog,
instantiates an orchestrator per spec, submits orders tagged with the
spec's ``strategy_id``, and records every fill against the ledger.

Design choices (confirmed with Jorge):

  * Equal $5,000 per strategy (20 x $5k = $100k Alpaca paper cap).
  * Staggered monthly rebalance: day-of-month = (index % 28) + 1.
    Spreads execution load over the month and reveals intra-month
    performance dispersion across strategies.
  * Strategy 0 (``invvol_eqfloor_etfs``) is ALREADY LIVE from the
    2026-04-23 09:30 ET deployment. The ledger absorbs its existing
    42 Alpaca positions as-is; no new orders are placed for it
    until its next rebalance day.

Each spec's ``weight_fn`` is a callable ``(prices, factors) -> pd.Series``
that the orchestrator invokes at rebalance time. Factories below wrap
the library's existing allocators with the right parameters.

Public API
----------
``StrategySpec``        — dataclass describing one bag.
``STRATEGY_CATALOG``    — the canonical list of 20 specs.
``get_spec(strategy_id)`` — lookup helper.
``due_today(catalog, today)`` — specs whose rebalance day is today.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd


UniverseKey = str  # "v1a" | "etfs" | "sp100" | "sp500" | "spy_agg" | "spy"
AllocatorKey = str
WeightFn = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class StrategySpec:
    """One bag in the meta-portfolio."""
    strategy_id: str
    allocator: AllocatorKey
    universe: UniverseKey
    starting_cash: float
    rebalance_day: int  # 1-28 (28 is the max to avoid missing Feb)
    notes: str = ""


# --- Allocator factories ----------------------------------------------------
# Each returns a ``weight_fn(prices, factors) -> pd.Series`` that emits a
# weight vector summing to <=1 over the prices columns. Factors are only
# needed for rck_6factor; exploration strategies ignore them.


def _make_vol_targeting(target_vol: float = 0.15, vol_lookback: int = 60) -> WeightFn:
    from inversiones_mama.exploration.strategies.vol_targeting import VolatilityTargeting

    def _fn(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
        strat = VolatilityTargeting(vol_lookback=vol_lookback, target_vol=target_vol)
        weights = strat.generate_signals(prices)
        return weights.iloc[-1]

    return _fn


def _make_inverse_vol(vol_lookback: int = 60, equity_floor: float | None = None) -> WeightFn:
    from inversiones_mama.sizing.inverse_vol import generate_current_weights

    def _fn(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
        return generate_current_weights(
            prices,
            vol_lookback=vol_lookback,
            per_name_cap=0.15,
            equity_floor=equity_floor,
        )

    return _fn


def _make_hrp(per_name_cap: float | None = 0.15, equity_floor: float | None = None) -> WeightFn:
    from inversiones_mama.sizing.hrp import hrp_weights

    def _fn(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
        returns = prices.pct_change().iloc[1:].tail(252)
        active = returns.dropna(axis=1, how="any")
        var = active.var()
        active = active[var[var > 0].index]
        w = hrp_weights(active)
        # Align back to the caller's column order with zeros for dropped tickers
        out = pd.Series(0.0, index=prices.columns)
        out.loc[w.index] = w.values
        # Apply per-name cap (iterative redistribution)
        if per_name_cap is not None and per_name_cap > 0:
            for _ in range(10):
                excess_mask = out > per_name_cap
                if not excess_mask.any():
                    break
                excess = (out[excess_mask] - per_name_cap).sum()
                out[excess_mask] = per_name_cap
                below = out[~excess_mask]
                if below.sum() > 0:
                    out[~excess_mask] += excess * below / below.sum()
        # Apply equity floor
        if equity_floor is not None and equity_floor > 0:
            from inversiones_mama.sizing.inverse_vol import _apply_equity_floor
            out = _apply_equity_floor(out, equity_floor)
        total = out.sum()
        if total > 0:
            out = out / total
        return out

    return _fn


def _make_equal_weight() -> WeightFn:
    def _fn(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
        n = prices.shape[1]
        return pd.Series(1.0 / n, index=prices.columns)
    return _fn


def _make_single_ticker(ticker: str) -> WeightFn:
    def _fn(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
        w = pd.Series(0.0, index=prices.columns)
        if ticker in w.index:
            w[ticker] = 1.0
        return w
    return _fn


def _make_fixed_allocation(allocations: dict[str, float]) -> WeightFn:
    def _fn(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
        w = pd.Series(0.0, index=prices.columns)
        for t, weight in allocations.items():
            if t in w.index:
                w[t] = weight
        total = w.sum()
        if total > 0:
            w = w / total
        return w
    return _fn


def _make_momentum_ts(lookback: int = 120) -> WeightFn:
    from inversiones_mama.exploration.strategies.momentum_ts import TimeSeriesMomentum

    def _fn(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
        strat = TimeSeriesMomentum(lookback=lookback)
        weights = strat.generate_signals(prices)
        if weights.empty:
            return pd.Series(0.0, index=prices.columns)
        return weights.iloc[-1].reindex(prices.columns).fillna(0.0)

    return _fn


def _make_momentum_xsec(lookback: int = 120, top_k: int = 3) -> WeightFn:
    from inversiones_mama.exploration.strategies.momentum_xsec import CrossSectionalMomentum

    def _fn(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
        strat = CrossSectionalMomentum(lookback=lookback, top_k=top_k)
        weights = strat.generate_signals(prices)
        if weights.empty:
            return pd.Series(0.0, index=prices.columns)
        return weights.iloc[-1].reindex(prices.columns).fillna(0.0)

    return _fn


def _make_dual_momentum(lookback: int = 120, top_k: int = 3, risk_off: str = "TLT") -> WeightFn:
    from inversiones_mama.exploration.strategies.dual_momentum import DualMomentum

    def _fn(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
        strat = DualMomentum(lookback=lookback, top_k=top_k, risk_off_asset=risk_off)
        weights = strat.generate_signals(prices)
        if weights.empty:
            return pd.Series(0.0, index=prices.columns)
        return weights.iloc[-1].reindex(prices.columns).fillna(0.0)

    return _fn


def _make_mean_reversion(rsi_period: int = 14, oversold: float = 30, overbought: float = 70) -> WeightFn:
    from inversiones_mama.exploration.strategies.mean_reversion import RSIMeanReversion

    def _fn(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
        strat = RSIMeanReversion(rsi_period=rsi_period, oversold=oversold, overbought=overbought)
        weights = strat.generate_signals(prices)
        if weights.empty:
            return pd.Series(0.0, index=prices.columns)
        return weights.iloc[-1].reindex(prices.columns).fillna(0.0)

    return _fn


def _make_rck_6factor() -> WeightFn:
    from inversiones_mama.backtest.engine import BacktestConfig, walk_forward_backtest

    def _fn(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
        cov_method = "sample" if prices.shape[1] <= 15 else "lw_diagonal"
        cfg = BacktestConfig(covariance_method=cov_method)
        result = walk_forward_backtest(prices, factors, cfg)
        if not result.rebalance_records:
            return pd.Series(0.0, index=prices.columns)
        last = result.rebalance_records[-1]
        return last.target_weights.reindex(prices.columns).fillna(0.0)

    return _fn


# --- The catalog ------------------------------------------------------------


ALLOCATOR_FACTORIES: dict[str, WeightFn] = {
    "vol_targeting":       _make_vol_targeting(target_vol=0.15, vol_lookback=60),
    "inverse_vol":         _make_inverse_vol(vol_lookback=60, equity_floor=None),
    "invvol_eqfloor":      _make_inverse_vol(vol_lookback=60, equity_floor=0.40),
    "hrp_capped":          _make_hrp(per_name_cap=0.15, equity_floor=None),
    "hrp_eqfloor":         _make_hrp(per_name_cap=0.15, equity_floor=0.40),
    "equal_weight":        _make_equal_weight(),
    "sixty_forty":         _make_fixed_allocation({"SPY": 0.6, "AGG": 0.4}),
    "spy_hold":            _make_single_ticker("SPY"),
    "momentum_ts_l60":     _make_momentum_ts(lookback=60),
    "momentum_ts_l120":    _make_momentum_ts(lookback=120),
    "momentum_ts_l252":    _make_momentum_ts(lookback=252),
    "momentum_xsec":       _make_momentum_xsec(lookback=120, top_k=3),
    "dual_momentum":       _make_dual_momentum(lookback=120, top_k=3, risk_off="TLT"),
    "mean_reversion":      _make_mean_reversion(rsi_period=14, oversold=30, overbought=70),
    "rck_6factor":         _make_rck_6factor(),
}


STRATEGY_CATALOG: list[StrategySpec] = [
    # 0: already live from 2026-04-23 09:30 ET deployment
    StrategySpec("invvol_eqfloor_etfs",   "invvol_eqfloor",   "etfs",   5_000.0, 23,
                 "LIVE since 2026-04-23 (bag 0; absorbed from prior deploy)"),
    StrategySpec("vol_targeting_etfs",    "vol_targeting",    "etfs",   5_000.0,  2),
    StrategySpec("inverse_vol_etfs",      "inverse_vol",      "etfs",   5_000.0,  3),
    StrategySpec("hrp_capped_etfs",       "hrp_capped",       "etfs",   5_000.0,  4),
    StrategySpec("hrp_eqfloor_etfs",      "hrp_eqfloor",      "etfs",   5_000.0,  5),
    StrategySpec("equal_weight_etfs",     "equal_weight",     "etfs",   5_000.0,  6),
    StrategySpec("sixty_forty",           "sixty_forty",      "etfs",   5_000.0,  7,
                 "classic 60% SPY / 40% AGG benchmark"),
    StrategySpec("spy_hold",              "spy_hold",         "etfs",   5_000.0,  8,
                 "pure SPY buy-and-hold market benchmark"),
    StrategySpec("momentum_ts_l120_etfs", "momentum_ts_l120", "etfs",   5_000.0,  9),
    StrategySpec("momentum_ts_l120_sp500","momentum_ts_l120", "sp500",  5_000.0, 10),
    StrategySpec("momentum_xsec_sp500",   "momentum_xsec",    "sp500",  5_000.0, 11),
    StrategySpec("dual_momentum_etfs",    "dual_momentum",    "etfs",   5_000.0, 12,
                 "fixed: market-wide median filter replaces per-asset filter"),
    StrategySpec("rck_6factor_v1a",       "rck_6factor",      "v1a",    5_000.0, 13),
    StrategySpec("rck_6factor_etfs",      "rck_6factor",      "etfs",   5_000.0, 14),
    StrategySpec("rck_6factor_sp100",     "rck_6factor",      "sp100",  5_000.0, 15),
    StrategySpec("momentum_ts_l60_etfs",  "momentum_ts_l60",  "etfs",   5_000.0, 16),
    StrategySpec("momentum_ts_l252_etfs", "momentum_ts_l252", "etfs",   5_000.0, 17),
    StrategySpec("vol_targeting_sp100",   "vol_targeting",    "sp100",  5_000.0, 18),
    StrategySpec("inverse_vol_sp100",     "inverse_vol",      "sp100",  5_000.0, 19),
    StrategySpec("mean_reversion_sp100",  "mean_reversion",   "sp100",  5_000.0, 20),
]


def get_spec(strategy_id: str) -> StrategySpec | None:
    for s in STRATEGY_CATALOG:
        if s.strategy_id == strategy_id:
            return s
    return None


def due_today(catalog: list[StrategySpec], today: date) -> list[StrategySpec]:
    """Specs whose rebalance_day equals today's day-of-month."""
    return [s for s in catalog if s.rebalance_day == today.day]
