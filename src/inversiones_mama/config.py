"""Shared configuration constants for Inversiones_mama.

All tunable defaults live here so strategy code never hard-codes
magic numbers and so tests can monkey-patch values cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Filesystem layout ------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
RAW_DIR: Path = DATA_DIR / "raw"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# --- Asset universe (Jorge-approved 2026-04-21, Decision A) -----------------
# 10 ETFs after pruning redundant pairs (VWO~AVEM, QUAL~SPY, VLUE~AVUV,
# IEF~TLT, VNQ high equity correlation). Kelly+RCK picks 6-12 active holdings
# at any time; rest get zero weight when they don't improve the objective.
UNIVERSE: dict[str, str] = {
    # Equity factor core (5)
    "AVUV": "Avantis US Small-Cap Value",
    "AVDV": "Avantis International Small-Cap Value",
    "AVEM": "Avantis Emerging Markets Equity",
    "MTUM": "iShares MSCI USA Momentum",
    "IMTM": "iShares MSCI International Momentum",
    # Defensive factor (1)
    "USMV": "iShares MSCI USA Minimum Volatility",
    # Macro diversifiers (3)
    "GLD":  "SPDR Gold Shares",
    "DBC":  "Invesco DB Commodity Index",
    "TLT":  "iShares 20+ Year Treasury",
    # Neutral market anchor (1)
    "SPY":  "SPDR S&P 500 ETF",
}
BENCHMARK: str = "SPY"

# --- Strategy parameters ----------------------------------------------------
KELLY_FRACTION: float = 0.65             # 0.65 multiplier (Jorge, 2026-04-21)
MAX_WEIGHT_PER_NAME: float = 0.15        # per-name cap — lowered from 0.35 on
                                         # 2026-04-22 (Jorge). Forces >=7 active
                                         # positions so expanded universes
                                         # (SP100/SP500/+ETFs) actually exercise
                                         # diversification rather than collapsing
                                         # to the top 3 by in-sample Sharpe.
REBALANCE_FREQ: str = "monthly"          # "monthly" | "quarterly"
LOOKBACK_DAYS: int = 252                 # ~1 year rolling window for mu/Sigma
MIN_HISTORY_DAYS: int = 60               # drop assets with less than 60 obs

# RCK risk-budget dials (feed into lambda = log(beta) / log(alpha))
RCK_MAX_DRAWDOWN_THRESHOLD: float = 0.50  # alpha: tolerate at most 50% DD
RCK_MAX_DRAWDOWN_PROBABILITY: float = 0.10  # beta: with at most 10% probability

# --- IBKR Tiered cost model (Agent 2 owns implementation; constants live here)
# Source: IBKR fee schedule for US stocks/ETFs as of 2026 (verify before live)
IBKR_FIXED_PER_SHARE: float = 0.0035     # $/share
IBKR_MIN_PER_ORDER: float = 0.35         # min per order
IBKR_MAX_PCT_TRADE: float = 0.01         # cap at 1% of trade value
IBKR_SLIPPAGE_BPS: float = 5.0           # 5bps slippage default (tight for liquid ETFs)


# --- Sanity gates (Jorge defaults, 2026-04-21) ------------------------------
@dataclass(frozen=True)
class SanityGates:
    """Pass/fail thresholds. v1a strategy must clear these to ship."""

    # Probability the end-of-year portfolio is below a loss threshold (Monte Carlo)
    max_prob_loss_40pct: float = 0.30    # P(final < $3k) < 30%
    max_prob_loss_60pct: float = 0.10    # P(final < $2k) < 10%
    # Worst-case drawdown tolerance
    max_dd_95th_pct: float = 0.50        # 95th-percentile max DD < 50%
    # Out-of-sample performance
    min_oos_sharpe: float = 0.0          # Sharpe on held-out data > 0
    # Cost leakage
    max_annual_turnover_cost: float = 0.015  # annual commissions+slippage < 1.5% capital


GATES: SanityGates = SanityGates()
