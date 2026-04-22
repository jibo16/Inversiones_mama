"""6-factor OLS regression for ETF composite-mu estimation.

For each ETF ``i``, fits the Fama-French-Carhart model:

    r_{i,t} - r_{f,t} = alpha_i
                     + beta_{i,Mkt} * (Mkt-RF)_t
                     + beta_{i,SMB} * SMB_t
                     + beta_{i,HML} * HML_t
                     + beta_{i,RMW} * RMW_t
                     + beta_{i,CMA} * CMA_t
                     + beta_{i,MOM} * MOM_t
                     + epsilon_{i,t}

Then computes the composite expected return:

    mu_i = alpha_i + sum_k beta_{i,k} * E[F_k]

where ``E[F_k]`` is the mean of factor ``k`` over the configured window.

The output ``mu`` vector (one per ETF, in daily decimals) is the primary
handoff to the Risk-Constrained Kelly solver (Agent 2, Step 4). Kelly also
needs Sigma = Cov(asset_returns); that is computed separately from the
same ``asset_returns`` DataFrame — we do not fabricate Sigma from factors
in v1a to keep the estimator uncertainty contained.

Public API
----------
fit_factor_loadings(asset_returns, factors, factor_cols=None) -> FactorLoadings
factor_premia(factors, lookback_days=None, factor_cols=None) -> pd.Series
compute_composite_mu(loadings, premia, factor_cols=None) -> pd.Series
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# Canonical column order for the 6-factor model
FACTOR_COLS: list[str] = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]

MIN_OBS: int = 30  # minimum regression observations per asset


@dataclass(frozen=True)
class FactorLoadings:
    """Per-asset OLS output.

    Attributes
    ----------
    alpha : pd.Series
        Intercept per ticker.
    betas : pd.DataFrame
        Factor loadings; index=ticker, columns=factor names.
    r_squared : pd.Series
        Coefficient of determination per ticker.
    residual_std : pd.Series
        Residual standard error sqrt(MSE_resid) per ticker.
    n_obs : int
        Observations used (post-alignment and dropna).
    factor_cols : tuple[str, ...]
        Which factor columns were fit (canonical order).
    """

    alpha: pd.Series
    betas: pd.DataFrame
    r_squared: pd.Series
    residual_std: pd.Series
    n_obs: int
    factor_cols: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def align_excess_returns(
    asset_returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Intersect dates; convert raw asset returns to excess by subtracting RF.

    Returns (excess_returns_df, factors_aligned_df).
    """
    if "RF" not in factors.columns:
        raise ValueError("factors must include 'RF' column (daily risk-free rate)")
    common = asset_returns.index.intersection(factors.index)
    if len(common) < MIN_OBS:
        raise ValueError(
            f"Insufficient overlapping observations: {len(common)} (need >= {MIN_OBS})"
        )
    assets_aligned = asset_returns.loc[common]
    factors_aligned = factors.loc[common]
    excess = assets_aligned.sub(factors_aligned["RF"], axis=0)
    return excess, factors_aligned


# --------------------------------------------------------------------------- #
# Regression                                                                  #
# --------------------------------------------------------------------------- #


def fit_factor_loadings(
    asset_returns: pd.DataFrame,
    factors: pd.DataFrame,
    factor_cols: list[str] | None = None,
) -> FactorLoadings:
    """Fit the 6-factor model per asset via OLS.

    Parameters
    ----------
    asset_returns : DataFrame
        Daily raw returns; columns are tickers, index is date.
    factors : DataFrame
        Output of ``data.factors.load_factor_returns`` — must include the
        factor columns and ``RF``.
    factor_cols : optional list of factor column names (default: FACTOR_COLS).

    Returns
    -------
    FactorLoadings

    Raises
    ------
    ValueError
        If ``factors`` lacks any required column, or if fewer than ``MIN_OBS``
        rows survive alignment and NaN-dropping for any asset.
    """
    cols = list(factor_cols) if factor_cols is not None else list(FACTOR_COLS)

    missing = [c for c in cols if c not in factors.columns]
    if missing:
        raise ValueError(f"factors DataFrame missing required columns: {missing}")

    excess, fac = align_excess_returns(asset_returns, factors)

    # Build combined DataFrame so per-asset NaN drop is consistent
    combined = pd.concat([excess, fac[cols]], axis=1).dropna()
    if len(combined) < MIN_OBS:
        raise ValueError(
            f"After dropna, only {len(combined)} rows remain (need >= {MIN_OBS})."
        )

    y_all = combined[excess.columns]
    X = add_constant(combined[cols].values, has_constant="add")  # intercept column

    alphas: dict[str, float] = {}
    betas_by_ticker: dict[str, np.ndarray] = {}
    r2: dict[str, float] = {}
    resid_se: dict[str, float] = {}

    for ticker in y_all.columns:
        y = y_all[ticker].values
        model = OLS(y, X).fit()
        alphas[ticker] = float(model.params[0])
        betas_by_ticker[ticker] = model.params[1:]
        r2[ticker] = float(model.rsquared)
        # sqrt of residual mean squared error = standard error of regression
        resid_se[ticker] = float(np.sqrt(model.mse_resid))

    betas_df = pd.DataFrame(betas_by_ticker, index=cols).T
    betas_df.index.name = "ticker"
    betas_df.columns.name = "factor"

    return FactorLoadings(
        alpha=pd.Series(alphas, name="alpha"),
        betas=betas_df,
        r_squared=pd.Series(r2, name="r_squared"),
        residual_std=pd.Series(resid_se, name="residual_std"),
        n_obs=len(combined),
        factor_cols=tuple(cols),
    )


# --------------------------------------------------------------------------- #
# Composite mu                                                                #
# --------------------------------------------------------------------------- #


def factor_premia(
    factors: pd.DataFrame,
    lookback_days: int | None = None,
    factor_cols: list[str] | None = None,
) -> pd.Series:
    """Estimate E[F_k] as the arithmetic mean over the specified tail window.

    Parameters
    ----------
    factors : DataFrame with at least the factor columns.
    lookback_days : if given, use only the last N rows; otherwise full history.
    factor_cols : columns to average (default: FACTOR_COLS).

    Returns daily-decimal means, one per factor.
    """
    cols = list(factor_cols) if factor_cols is not None else list(FACTOR_COLS)
    missing = [c for c in cols if c not in factors.columns]
    if missing:
        raise ValueError(f"factors DataFrame missing columns: {missing}")
    df = factors[cols]
    if lookback_days is not None:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        df = df.tail(lookback_days)
    return df.mean()


def compute_composite_mu(
    loadings: FactorLoadings,
    premia: pd.Series,
    factor_cols: list[str] | None = None,
) -> pd.Series:
    """Compute mu_i = alpha_i + sum_k beta_{i,k} * E[F_k] per ticker.

    Returns a Series indexed by ticker with the daily-decimal composite mu.
    """
    cols = list(factor_cols) if factor_cols is not None else list(loadings.factor_cols)

    missing_betas = [c for c in cols if c not in loadings.betas.columns]
    if missing_betas:
        raise ValueError(f"loadings missing columns: {missing_betas}")
    missing_premia = [c for c in cols if c not in premia.index]
    if missing_premia:
        raise ValueError(f"premia missing factors: {missing_premia}")

    betas = loadings.betas[cols]
    premia_aligned = premia.reindex(cols)
    beta_contrib = betas.mul(premia_aligned, axis=1).sum(axis=1)
    mu = loadings.alpha + beta_contrib
    mu.name = "mu"
    return mu
