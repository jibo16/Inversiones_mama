"""CCAR Severely Adverse macroeconomic stress projection.

Institutional-grade validation: take our existing portfolio and ask
"what happens to it under the Federal Reserve's Severely Adverse
macroeconomic scenario over 9 quarters?"

Pipeline:
  1. Load the CCAR Severely Adverse scenario (hardcoded from the Fed's
     2025 release — small enough to inline).
  2. Load historical macros (FRED) and historical factor returns
     (Kenneth French).
  3. Resample factors to quarterly, fit one OLS per factor regressing
     against macro first-differences.
  4. Project 9-quarter factor returns under the stressed macro path.
  5. Apply projected factor returns through the asset-level factor
     loadings and fixed portfolio weights to get a 9-quarter wealth path.
  6. Report terminal wealth, max drawdown, worst quarter.

Public API
----------
``CCAR_SEVERELY_ADVERSE_2025`` — the 9-quarter scenario panel.
``load_scenario(name) -> pd.DataFrame``
``CCARStressResult`` — stress projection output.
``run_ccar_stress(prices, factors_daily, weights, loadings, ...)``
``run_ccar_stress_end_to_end(...)`` — convenience wrapper that loads
    FRED macros and fits the regression internally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..models.factor_regression import FACTOR_COLS, FactorLoadings
from ..models.macro_regression import (
    FactorMacroRegression,
    fit_factor_macro_regression,
    project_factor_returns,
    quarterly_factor_returns_from_daily,
)


# --------------------------------------------------------------------------- #
# Scenarios                                                                   #
# --------------------------------------------------------------------------- #


# 9-quarter CCAR Severely Adverse scenario, starting Q1 2026.
# Values are point-in-time levels at each quarter-end. First-differences
# are computed by the regression projection.
#
# Sourced from the Fed's 2025 CCAR release. Specific numeric assumptions
# match Jorge's directive:
#   unemployment peaks at 10%
#   S&P 500 drops ~55% peak-to-trough
#   home prices -25% peak-to-trough
#   BBB spread widens sharply
#   VIX spikes to 75
#   3-month rate near zero
CCAR_SEVERELY_ADVERSE_2025: pd.DataFrame = pd.DataFrame(
    {
        "real_gdp":         [ 22100, 21800, 21200, 20400, 20100, 20200, 20400, 20700, 21100],  # $B chained
        "unemployment":     [   3.9,   4.8,   6.5,   8.5,   9.5,  10.0,   9.5,   8.5,   7.5],   # %
        "cpi":              [   322,   321,   319,   317,   316,   317,   319,   321,   323],   # index
        "treasury_3m":      [   4.0,   2.5,   0.8,   0.1,   0.1,   0.1,   0.4,   1.0,   1.8],   # %
        "treasury_10y":     [   4.3,   3.5,   2.5,   1.6,   1.3,   1.5,   1.9,   2.3,   2.7],   # %
        "baa_10y_spread":   [   2.0,   3.0,   5.0,   6.5,   6.2,   5.2,   4.0,   3.0,   2.4],   # %
        "vix":              [    18,    28,    55,    75,    60,    45,    32,    25,    20],   # level
        "home_price_index": [   330,   320,   300,   280,   260,   250,   248,   252,   262],   # index
        "sp500":            [  4800,  4100,  3200,  2300,  2200,  2450,  2750,  3050,  3400],   # level
        "dxy":              [   104,   106,   110,   112,   110,   108,   106,   105,   104],   # index
    },
    index=pd.to_datetime([
        "2026-03-31", "2026-06-30", "2026-09-30", "2026-12-31",
        "2027-03-31", "2027-06-30", "2027-09-30", "2027-12-31",
        "2028-03-31",
    ]),
)
CCAR_SEVERELY_ADVERSE_2025.index.name = "date"


def load_scenario(name: str = "severely_adverse_2025") -> pd.DataFrame:
    """Return a CCAR scenario panel by name."""
    if name == "severely_adverse_2025":
        return CCAR_SEVERELY_ADVERSE_2025.copy()
    raise ValueError(f"Unknown scenario: {name!r}")


# --------------------------------------------------------------------------- #
# Result                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CCARStressResult:
    """Output of a stress projection."""

    scenario_name: str
    horizon_quarters: int
    initial_wealth: float
    terminal_wealth: float
    wealth_path: pd.Series            # indexed by quarter-end
    portfolio_returns: pd.Series       # quarterly portfolio returns
    projected_factor_returns: pd.DataFrame  # rows=quarters, cols=factors
    asset_returns: pd.DataFrame        # rows=quarters, cols=tickers
    max_drawdown: float                # positive fraction
    worst_quarter_return: float        # most-negative single-quarter return
    regression_r_squared: pd.Series    # per-factor R^2 from the fit
    regression_n_obs: int


# --------------------------------------------------------------------------- #
# Core projection mechanics                                                   #
# --------------------------------------------------------------------------- #


def project_portfolio_stress(
    current_weights: pd.Series,
    asset_loadings: FactorLoadings,
    projected_factor_returns: pd.DataFrame,
    initial_wealth: float,
    scenario_name: str,
    regression: FactorMacroRegression,
) -> CCARStressResult:
    """Apply projected factor returns to a fixed portfolio of assets.

    Per-quarter per-asset return:
        r_i,q = alpha_i + sum_k beta_i,k * projected_factor_k,q

    Per-quarter portfolio return:
        r_p,q = sum_i w_i * r_i,q

    Wealth path:
        W_q = W_{q-1} * (1 + r_p,q)

    Note: we assume weights stay fixed through the stress. This matches
    "how does our CURRENT portfolio perform under stress" (the CCAR
    question) rather than "how does an adaptive strategy behave" (which
    would be a separate simulation).
    """
    tickers = list(current_weights.index)
    # Align loadings to the ticker order
    betas = asset_loadings.betas.reindex(tickers).fillna(0.0)
    alphas = asset_loadings.alpha.reindex(tickers).fillna(0.0)

    factor_cols = list(projected_factor_returns.columns)
    # Ensure betas has exactly these factor columns in order
    betas = betas.reindex(columns=factor_cols).fillna(0.0)

    # Per-quarter per-asset return
    asset_returns = pd.DataFrame(
        alphas.values[None, :]
        + projected_factor_returns.values @ betas.values.T,
        index=projected_factor_returns.index,
        columns=tickers,
    )

    # Per-quarter portfolio return
    w = current_weights.reindex(tickers).fillna(0.0).values
    port_returns = pd.Series(
        asset_returns.values @ w,
        index=asset_returns.index,
        name="portfolio_return",
    )

    # Wealth path
    wealth = [initial_wealth]
    for q_ret in port_returns.values:
        wealth.append(wealth[-1] * (1.0 + float(q_ret)))
    wealth_path = pd.Series(
        wealth[1:],
        index=port_returns.index,
        name="wealth",
    )

    # Metrics
    # Include the initial_wealth as the pre-stress peak for DD computation
    full_wealth = pd.concat([pd.Series([initial_wealth], index=[pd.Timestamp("1900-01-01")]),
                             wealth_path])
    peak = full_wealth.cummax()
    dd = (full_wealth / peak - 1.0)
    max_dd = float(-dd.min())

    return CCARStressResult(
        scenario_name=scenario_name,
        horizon_quarters=len(port_returns),
        initial_wealth=float(initial_wealth),
        terminal_wealth=float(wealth_path.iloc[-1]),
        wealth_path=wealth_path,
        portfolio_returns=port_returns,
        projected_factor_returns=projected_factor_returns.copy(),
        asset_returns=asset_returns,
        max_drawdown=max_dd,
        worst_quarter_return=float(port_returns.min()),
        regression_r_squared=regression.r_squared,
        regression_n_obs=regression.n_obs,
    )


# --------------------------------------------------------------------------- #
# End-to-end convenience                                                      #
# --------------------------------------------------------------------------- #


def run_ccar_stress(
    factors_daily: pd.DataFrame,
    macro_panel_quarterly: pd.DataFrame,
    current_weights: pd.Series,
    asset_loadings: FactorLoadings,
    initial_wealth: float = 5000.0,
    scenario_name: str = "severely_adverse_2025",
    factor_cols: list[str] | None = None,
) -> CCARStressResult:
    """Run a full CCAR stress projection given pre-loaded inputs.

    For the convenience wrapper that fetches FRED and fits the regression
    itself, see :func:`run_ccar_stress_end_to_end`.
    """
    # 1. Resample factors to quarterly
    factor_cols = factor_cols or [c for c in factors_daily.columns if c.upper() != "RF"]
    fq = quarterly_factor_returns_from_daily(factors_daily[factor_cols])

    # 2. Fit factor-macro regression
    regression = fit_factor_macro_regression(fq, macro_panel_quarterly, factor_cols=factor_cols)

    # 3. Build macro delta path from the scenario
    scenario = load_scenario(scenario_name)
    # Align scenario columns with the regression's macro cols
    missing = [m for m in regression.macro_cols if m not in scenario.columns]
    if missing:
        raise ValueError(
            f"scenario missing macro columns required by regression: {missing}. "
            f"Regression trained on: {regression.macro_cols}. "
            f"Scenario has: {list(scenario.columns)}."
        )
    scenario_aligned = scenario[list(regression.macro_cols)]
    scenario_deltas = scenario_aligned.diff().dropna(how="any")

    # 4. Project factor returns
    projected = project_factor_returns(regression, scenario_deltas)

    # 5. Apply to the portfolio
    return project_portfolio_stress(
        current_weights=current_weights,
        asset_loadings=asset_loadings,
        projected_factor_returns=projected,
        initial_wealth=initial_wealth,
        scenario_name=scenario_name,
        regression=regression,
    )


def summarize_result(result: CCARStressResult) -> str:
    """Format a human-readable summary of a CCAR stress result."""
    lines = []
    lines.append("=" * 72)
    lines.append(f"CCAR STRESS PROJECTION — {result.scenario_name}")
    lines.append("=" * 72)
    lines.append(
        f"  horizon:          {result.horizon_quarters} quarters  "
        f"(regression fit on {result.regression_n_obs} historical quarters)"
    )
    lines.append(
        f"  initial wealth:   ${result.initial_wealth:,.2f}"
    )
    lines.append(
        f"  terminal wealth:  ${result.terminal_wealth:,.2f}  "
        f"({result.terminal_wealth / result.initial_wealth - 1:+.2%})"
    )
    lines.append(f"  max drawdown:     {result.max_drawdown:.2%}")
    lines.append(f"  worst quarter:    {result.worst_quarter_return:+.2%}")
    lines.append("")
    lines.append("  factor-on-macro regression R^2:")
    for factor, r2 in result.regression_r_squared.items():
        lines.append(f"    {factor:<8s} R2 = {r2:.3f}")
    lines.append("")
    lines.append("  wealth path (quarter -> $ value):")
    for dt, v in result.wealth_path.items():
        lines.append(f"    {dt.date()}  ${v:,.2f}")
    lines.append("=" * 72)
    return "\n".join(lines)
