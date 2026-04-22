"""Performance metrics including the Deflated Sharpe Ratio.

Implements standard portfolio performance metrics plus the critical
Deflated Sharpe Ratio (DSR) from Bailey & López de Prado (2014).

The DSR corrects the observed Sharpe Ratio for:
  1. Non-normality of returns (skewness and kurtosis)
  2. Multiple testing / selection bias (number of trials)
  3. Track-record length

This is the primary statistical gate for determining whether a backtest's
Sharpe ratio represents genuine alpha or is a statistical mirage.

Mathematical Formulation
------------------------
The standard error of the Sharpe Ratio under non-normality:

    SE(SR) = sqrt[ (1 - γ₁·SR + (γ₂-1)/4 · SR²) / (T-1) ]

where:
    γ₁ = skewness of returns
    γ₂ = excess kurtosis of returns
    T  = number of observations

The expected maximum Sharpe from N independent trials (haircut):

    E[max SR | N] ≈ (1 - γ) · Φ⁻¹(1 - 1/N) + γ · Φ⁻¹(1 - 1/(N·e))

where γ ≈ 0.5772 (Euler-Mascheroni constant).

The Deflated Sharpe Ratio is then:

    DSR = Φ[ (SR_obs - E[max SR | N]) / SE(SR) ]

A DSR > 0.95 (or the chosen significance level) means the observed SR
exceeds what you'd expect from random chance given N trials.

Public API
----------
sharpe_ratio(returns, rf=0, periods=252) -> float
sortino_ratio(returns, rf=0, periods=252) -> float
max_drawdown(returns_or_equity) -> float
max_drawdown_series(returns_or_equity) -> pd.Series
calmar_ratio(returns, periods=252) -> float
expected_max_sharpe(n_trials, T, skew=0, kurt=3) -> float
deflated_sharpe_ratio(observed_sr, n_trials, T, skew, kurt) -> float
compute_all_metrics(returns, rf=0, periods=252, n_trials=1) -> PerformanceMetrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# --------------------------------------------------------------------------- #
# Result container                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PerformanceMetrics:
    """Comprehensive performance metrics for a return series.

    All annualized values use the specified ``periods`` (default 252 = daily).
    """

    # --- Basic returns ---
    total_return: float             # cumulative total return
    annualized_return: float        # CAGR
    annualized_volatility: float    # annual std dev
    # --- Risk-adjusted ---
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    # --- Drawdown ---
    max_drawdown: float             # worst peak-to-trough (positive = loss)
    avg_drawdown: float
    max_drawdown_duration: int      # max number of periods in a drawdown
    # --- Distribution ---
    skewness: float
    excess_kurtosis: float
    # --- Deflated Sharpe ---
    deflated_sharpe: float          # DSR p-value
    n_trials: int                   # number of backtests / strategies tested
    expected_max_sharpe: float      # E[max SR | N trials]
    sharpe_se: float                # standard error of SR under non-normality
    # --- Other ---
    n_observations: int
    hit_rate: float                 # % of positive-return periods
    profit_factor: float            # sum(gains) / sum(losses)
    tail_ratio: float               # 95th percentile / |5th percentile|


# --------------------------------------------------------------------------- #
# Core metrics                                                                #
# --------------------------------------------------------------------------- #


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    rf: float = 0.0,
    periods: int = 252,
) -> float:
    """Annualized Sharpe Ratio.

    SR = (mean(r - rf) / std(r - rf)) * sqrt(periods)

    Parameters
    ----------
    returns : array-like
        Simple returns per period.
    rf : float
        Risk-free rate per period (same frequency as returns).
    periods : int
        Number of periods per year (252 for daily, 12 for monthly).

    Returns
    -------
    float
        Annualized Sharpe Ratio.
    """
    r = np.asarray(returns, dtype=np.float64)
    excess = r - rf
    mu = np.nanmean(excess)
    sigma = np.nanstd(excess, ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return float(mu / sigma * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series | np.ndarray,
    rf: float = 0.0,
    periods: int = 252,
    target: float = 0.0,
) -> float:
    """Annualized Sortino Ratio (downside deviation denominator).

    Sortino = (mean(r - rf) / downside_std) * sqrt(periods)

    Downside std uses only returns below the target threshold.
    """
    r = np.asarray(returns, dtype=np.float64)
    excess = r - rf
    mu = np.nanmean(excess)

    # Downside deviation: std of returns below target
    downside = np.minimum(r - target, 0.0)
    downside_std = np.sqrt(np.nanmean(downside ** 2))

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(mu / downside_std * np.sqrt(periods))


def max_drawdown(
    returns_or_equity: pd.Series | np.ndarray,
    is_equity: bool = False,
) -> float:
    """Compute the maximum drawdown (positive value = magnitude of loss).

    Parameters
    ----------
    returns_or_equity : array-like
        If ``is_equity=False`` (default), interpret as simple returns and
        compute the equity curve internally.
        If ``is_equity=True``, interpret as the equity curve directly.

    Returns
    -------
    float
        Maximum drawdown as a positive fraction (e.g. 0.25 = 25% DD).
    """
    if is_equity:
        equity = np.asarray(returns_or_equity, dtype=np.float64)
    else:
        r = np.asarray(returns_or_equity, dtype=np.float64)
        equity = np.cumprod(1.0 + r)

    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    return float(np.nanmax(drawdowns))


def max_drawdown_series(
    returns_or_equity: pd.Series | np.ndarray,
    is_equity: bool = False,
) -> pd.Series | np.ndarray:
    """Return the full drawdown series (for plotting / analysis).

    Returns the same type as input (Series preserves index).
    """
    is_series = isinstance(returns_or_equity, pd.Series)

    if is_equity:
        if is_series:
            equity = returns_or_equity.astype(np.float64)
        else:
            equity = np.asarray(returns_or_equity, dtype=np.float64)
    else:
        if is_series:
            equity = (1.0 + returns_or_equity).cumprod()
        else:
            equity = np.cumprod(1.0 + np.asarray(returns_or_equity, dtype=np.float64))

    if is_series:
        running_max = equity.cummax()
    else:
        running_max = np.maximum.accumulate(equity)

    dd = (running_max - equity) / running_max
    return dd


def calmar_ratio(
    returns: pd.Series | np.ndarray,
    periods: int = 252,
) -> float:
    """Annualized Calmar Ratio = CAGR / Max Drawdown."""
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    if n == 0:
        return 0.0

    total = np.prod(1.0 + r) - 1.0
    years = n / periods
    if years <= 0:
        return 0.0
    cagr = (1.0 + total) ** (1.0 / years) - 1.0

    mdd = max_drawdown(r, is_equity=False)
    if mdd == 0:
        return 0.0
    return float(cagr / mdd)


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio (Bailey & López de Prado, 2014)                       #
# --------------------------------------------------------------------------- #


def _sharpe_se(
    sr: float,
    T: int,
    skew: float,
    excess_kurt: float,
) -> float:
    """Standard error of the Sharpe Ratio under non-normality.

    SE(SR) = sqrt[ (1 - γ₁·SR + (γ₂-1)/4 · SR²) / (T-1) ]

    where γ₁ = skewness, γ₂ = excess kurtosis.
    Note: some papers define γ₂ as raw kurtosis (=excess+3); we use
    **excess kurtosis** here (Gaussian excess kurtosis = 0).
    """
    if T <= 1:
        return float("inf")
    # Using excess kurtosis directly (γ₂ in BLP's paper is excess kurtosis)
    numerator = 1.0 - skew * sr + (excess_kurt / 4.0) * sr ** 2
    # Clamp to prevent sqrt of negative from extreme skew/kurt
    numerator = max(numerator, 1e-12)
    return float(np.sqrt(numerator / (T - 1)))


def expected_max_sharpe(
    n_trials: int,
    T: int,
    skew: float = 0.0,
    excess_kurt: float = 0.0,
) -> float:
    """Expected maximum Sharpe Ratio from N independent trials.

    Uses the Bailey & López de Prado (2014) approximation based on
    the expected value of the maximum of N draws from a standard normal:

        E[max SR | N] ≈ (1-γ) · Z(1 - 1/N) + γ · Z(1 - 1/(N·e))

    where γ ≈ 0.5772 (Euler–Mascheroni) and Z = Φ⁻¹ (normal quantile),
    then scaled by the SE(SR) to account for non-normality.

    Parameters
    ----------
    n_trials : int
        Number of independent strategies/backtests tested.
    T : int
        Number of return observations.
    skew : float
        Skewness of returns.
    excess_kurt : float
        Excess kurtosis of returns.

    Returns
    -------
    float
        Expected maximum Sharpe Ratio (annualized if SR was annualized).
    """
    if n_trials <= 1:
        return 0.0

    euler_mascheroni = 0.5772156649015329

    # Quantile arguments — clamp to avoid Φ⁻¹(0) = -inf or Φ⁻¹(1) = +inf
    p1 = 1.0 - 1.0 / n_trials
    p2 = 1.0 - 1.0 / (n_trials * np.e)
    p1 = np.clip(p1, 1e-10, 1.0 - 1e-10)
    p2 = np.clip(p2, 1e-10, 1.0 - 1e-10)

    z1 = sp_stats.norm.ppf(p1)
    z2 = sp_stats.norm.ppf(p2)

    # Expected max of N standard normals
    e_max = (1 - euler_mascheroni) * z1 + euler_mascheroni * z2

    # Scale by the SE to get the expected max SR
    se = _sharpe_se(0.0, T, skew, excess_kurt)  # SE at SR=0 (conservative)
    return float(e_max * se * np.sqrt(T - 1))  # undo the 1/sqrt(T-1) in SE


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    T: int,
    skew: float,
    excess_kurt: float,
) -> float:
    """Compute the Deflated Sharpe Ratio (DSR).

    DSR = Φ[ (SR_obs - E[max SR | N]) / SE(SR) ]

    A DSR > 0.95 means the observed Sharpe is statistically significant
    at the 5% level even after accounting for multiple testing and
    non-normal returns.

    Parameters
    ----------
    observed_sr : float
        The observed annualized Sharpe Ratio.
    n_trials : int
        Number of independent strategies/backtests tested.
    T : int
        Number of return observations.
    skew : float
        Skewness of the return series.
    excess_kurt : float
        Excess kurtosis of the return series.

    Returns
    -------
    float
        DSR value in [0, 1] (p-value interpretation).
    """
    if T <= 1:
        return 0.0

    # De-annualize SR to per-period for the SE calculation
    # Assuming daily returns and 252 annualization
    sr_per_period = observed_sr / np.sqrt(252)

    se = _sharpe_se(sr_per_period, T, skew, excess_kurt)
    e_max = expected_max_sharpe(n_trials, T, skew, excess_kurt)

    # De-annualize E[max SR] too for consistent comparison
    e_max_per_period = e_max / np.sqrt(252)

    if se <= 0 or np.isinf(se):
        return 0.0

    z_score = (sr_per_period - e_max_per_period) / se
    return float(sp_stats.norm.cdf(z_score))


# --------------------------------------------------------------------------- #
# Drawdown duration                                                           #
# --------------------------------------------------------------------------- #


def _max_drawdown_duration(returns: np.ndarray) -> int:
    """Maximum consecutive periods spent in a drawdown."""
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    in_drawdown = equity < running_max * (1 - 1e-10)  # small tolerance

    max_dur = 0
    current_dur = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return max_dur


def _avg_drawdown(returns: np.ndarray) -> float:
    """Average drawdown magnitude (average of all peak-to-trough moves)."""
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    dd = (running_max - equity) / running_max
    # Average of the drawdown series (including zeros when at peaks)
    return float(np.nanmean(dd))


# --------------------------------------------------------------------------- #
# Aggregate metrics                                                           #
# --------------------------------------------------------------------------- #


def compute_all_metrics(
    returns: pd.Series | np.ndarray,
    rf: float = 0.0,
    periods: int = 252,
    n_trials: int = 1,
) -> PerformanceMetrics:
    """Compute all performance metrics for a return series.

    Parameters
    ----------
    returns : array-like
        Simple returns per period (daily by default).
    rf : float
        Risk-free rate per period.
    periods : int
        Number of periods per year.
    n_trials : int
        Number of independent strategies tested (for DSR).

    Returns
    -------
    PerformanceMetrics
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]  # drop NaN
    T = len(r)

    if T < 2:
        return PerformanceMetrics(
            total_return=0.0, annualized_return=0.0,
            annualized_volatility=0.0, sharpe_ratio=0.0,
            sortino_ratio=0.0, calmar_ratio=0.0,
            max_drawdown=0.0, avg_drawdown=0.0,
            max_drawdown_duration=0, skewness=0.0,
            excess_kurtosis=0.0, deflated_sharpe=0.0,
            n_trials=n_trials, expected_max_sharpe=0.0,
            sharpe_se=0.0, n_observations=T,
            hit_rate=0.0, profit_factor=0.0, tail_ratio=0.0,
        )

    # --- Basic ---
    total_ret = float(np.prod(1.0 + r) - 1.0)
    years = T / periods
    cagr = float((1.0 + total_ret) ** (1.0 / max(years, 1e-6)) - 1.0)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(periods))

    # --- Risk-adjusted ---
    sr = sharpe_ratio(r, rf=rf, periods=periods)
    sort = sortino_ratio(r, rf=rf, periods=periods)
    calm = calmar_ratio(r, periods=periods)

    # --- Drawdown ---
    mdd = max_drawdown(r)
    avg_dd = _avg_drawdown(r)
    mdd_dur = _max_drawdown_duration(r)

    # --- Distribution ---
    skw = float(sp_stats.skew(r, bias=False))
    # scipy's kurtosis with fisher=True gives excess kurtosis
    ekurt = float(sp_stats.kurtosis(r, fisher=True, bias=False))

    # --- Deflated Sharpe ---
    e_max_sr = expected_max_sharpe(n_trials, T, skw, ekurt)
    se_sr = _sharpe_se(sr / np.sqrt(periods), T, skw, ekurt)
    dsr = deflated_sharpe_ratio(sr, n_trials, T, skw, ekurt)

    # --- Other ---
    hit = float(np.sum(r > 0) / T)

    gains = r[r > 0]
    losses = r[r < 0]
    pf = float(np.sum(gains) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else 0.0

    p95 = np.percentile(r, 95)
    p5 = np.percentile(r, 5)
    tail = float(p95 / abs(p5)) if abs(p5) > 1e-10 else 0.0

    return PerformanceMetrics(
        total_return=total_ret,
        annualized_return=cagr,
        annualized_volatility=ann_vol,
        sharpe_ratio=sr,
        sortino_ratio=sort,
        calmar_ratio=calm,
        max_drawdown=mdd,
        avg_drawdown=avg_dd,
        max_drawdown_duration=mdd_dur,
        skewness=skw,
        excess_kurtosis=ekurt,
        deflated_sharpe=dsr,
        n_trials=n_trials,
        expected_max_sharpe=e_max_sr,
        sharpe_se=se_sr,
        n_observations=T,
        hit_rate=hit,
        profit_factor=pf,
        tail_ratio=tail,
    )
