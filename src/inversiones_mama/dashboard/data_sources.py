"""Dashboard data loaders — cached artifacts + live pipeline runs.

Every function returns either a pandas / numpy / dataclass object or
``None`` when the underlying artifact is missing. The Streamlit app
handles rendering the ``None`` cases with "Run the pipeline first"
hints so the dashboard never crashes on an empty project.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, RESULTS_DIR, UNIVERSE
from ..execution.trade_log import TradeLog


# --------------------------------------------------------------------------- #
# Cached artifact loaders                                                     #
# --------------------------------------------------------------------------- #


def load_verdict_text() -> str | None:
    """Return the pretty-printed v1a verdict, or None if the CLI has not run."""
    path = RESULTS_DIR / "v1a_verdict.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def load_paper_trade_log(path: Path | str | None = None) -> TradeLog | None:
    """Load ``results/paper_trades.json`` into a TradeLog, or return None."""
    p = Path(path) if path is not None else RESULTS_DIR / "paper_trades.json"
    if not p.exists():
        return None
    try:
        return TradeLog.load(p)
    except (ValueError, KeyError):
        return None


def load_paper_summary(path: Path | str | None = None) -> dict | None:
    """Load the per-rebalance JSON summary (target weights, fill stats)."""
    import json

    p = Path(path) if path is not None else RESULTS_DIR / "paper_trades_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_universe_stats(path: Path | str | None = None) -> pd.DataFrame | None:
    """Load `results/universe_stats.csv` (output of verify_universe.py)."""
    p = Path(path) if path is not None else RESULTS_DIR / "universe_stats.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0)


def load_universe_corr(path: Path | str | None = None) -> pd.DataFrame | None:
    """Load `results/universe_corr.csv` — pairwise daily-return correlation."""
    p = Path(path) if path is not None else RESULTS_DIR / "universe_corr.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0)


def load_alpha_pipeline_summary(path: Path | str | None = None) -> pd.DataFrame | None:
    """Load `results/alpha_pipeline_summary.csv` (factor loadings + mu)."""
    p = Path(path) if path is not None else RESULTS_DIR / "alpha_pipeline_summary.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0)


def available_artifacts() -> dict[str, bool]:
    """Report which cached artifacts exist, for dashboard status pane."""
    return {
        "v1a_verdict.txt": (RESULTS_DIR / "v1a_verdict.txt").exists(),
        "paper_trades.json": (RESULTS_DIR / "paper_trades.json").exists(),
        "paper_trades_summary.json": (RESULTS_DIR / "paper_trades_summary.json").exists(),
        "universe_stats.csv": (RESULTS_DIR / "universe_stats.csv").exists(),
        "universe_corr.csv": (RESULTS_DIR / "universe_corr.csv").exists(),
        "alpha_pipeline_summary.csv": (RESULTS_DIR / "alpha_pipeline_summary.csv").exists(),
    }


# --------------------------------------------------------------------------- #
# Live pipeline helpers                                                       #
# --------------------------------------------------------------------------- #


def fetch_prices_and_factors(years: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull cached prices + Ken French factors for the v1a universe."""
    from ..data.factors import load_factor_returns
    from ..data.prices import load_prices

    tickers = list(UNIVERSE.keys())
    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=int(years * 365) + 14)
    prices = load_prices(tickers, start, end, use_cache=True)
    factors = load_factor_returns(start=start, end=end, use_cache=True)
    return prices, factors


def run_walk_forward_now(years: float = 5.0):
    """Run the walk-forward backtest on live cached data. Returns BacktestResult."""
    from ..backtest.engine import walk_forward_backtest

    prices, factors = fetch_prices_and_factors(years)
    return walk_forward_backtest(prices, factors)


def run_mc_now(
    weights: pd.Series | None = None,
    *,
    n_paths: int = 5000,
    horizon_days: int = 252,
    initial_capital: float = 5000.0,
    years: float = 5.0,
    seed: int = 20260422,
):
    """Run a Monte Carlo RCK validation using live cached prices."""
    from ..simulation.monte_carlo import run_mc_rck_validation

    prices, _ = fetch_prices_and_factors(years)
    returns = prices.pct_change().dropna(how="all")
    return run_mc_rck_validation(
        returns,
        weights=weights,
        n_paths=n_paths,
        horizon_days=horizon_days,
        initial_capital=initial_capital,
        rng=np.random.default_rng(seed),
    )


# --------------------------------------------------------------------------- #
# Miscellaneous                                                               #
# --------------------------------------------------------------------------- #


def project_root() -> Path:
    return PROJECT_ROOT


def tickers() -> list[str]:
    return list(UNIVERSE.keys())


def ticker_name(t: str) -> str:
    return UNIVERSE.get(t, t)


# --------------------------------------------------------------------------- #
# Alpaca live state (optional; returns None if credentials are unavailable)   #
# --------------------------------------------------------------------------- #


def load_alpaca_account_snapshot() -> dict | None:
    """Fetch a summary of the Alpaca account if credentials are available.

    Returns None when Alpaca is not configured OR the API is unreachable —
    the dashboard treats None as "section unavailable" and shows a hint
    rather than crashing.
    """
    try:
        from ..execution.alpaca import (  # noqa: PLC0415  (lazy import)
            AlpacaAPIError,
            AlpacaAuthError,
            AlpacaClient,
        )
    except ImportError:
        return None
    try:
        client = AlpacaClient.from_env()
    except AlpacaAuthError:
        return None
    try:
        acct = client.check_auth()
    except (AlpacaAuthError, AlpacaAPIError):
        return None
    try:
        positions = client.get_positions()
    except AlpacaAPIError:
        positions = {}
    return {
        "status": acct.get("status"),
        "currency": acct.get("currency", "USD"),
        "cash": float(acct.get("cash", 0) or 0),
        "equity": float(acct.get("equity", 0) or 0),
        "buying_power": float(acct.get("buying_power", 0) or 0),
        "daytrading_buying_power": float(acct.get("daytrading_buying_power", 0) or 0),
        "pattern_day_trader": bool(acct.get("pattern_day_trader")),
        "base_url": client.config.base_url,
        "is_paper": client.config.is_paper,
        "positions": positions,
        "n_positions": len(positions),
    }


def compute_breaker_state(
    current_wealth: float,
    peak_wealth: float,
    threshold_dd: float,
    warn_threshold_dd: float | None = None,
) -> dict:
    """Compute a lightweight breaker snapshot for the dashboard.

    Self-contained so the dashboard doesn't need to persist the full
    CircuitBreaker instance across reruns.
    """
    if warn_threshold_dd is None:
        warn_threshold_dd = round(threshold_dd * 0.8, 4)
    dd = max(0.0, 1.0 - current_wealth / peak_wealth) if peak_wealth > 0 else 0.0
    if dd >= threshold_dd:
        state = "tripped"
    elif dd >= warn_threshold_dd:
        state = "warn"
    else:
        state = "ok"
    return {
        "state": state,
        "current_wealth": float(current_wealth),
        "peak_wealth": float(peak_wealth),
        "current_drawdown": dd,
        "threshold": threshold_dd,
        "warn_threshold": warn_threshold_dd,
    }
