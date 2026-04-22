"""Factor-on-macro regression for stress-test projection.

Fits one linear regression per Fama-French-Carhart factor against a
panel of macroeconomic variables (quarterly). The fitted intercepts and
betas are then consumed by ``simulation.ccar_stress`` to project factor
returns under a stressed macro scenario.

Math per factor ``k``:

    factor_return_q,k = alpha_k + sum_m beta_k,m * delta_macro_q,m + epsilon

where ``delta_macro_q,m`` is the quarter-over-quarter change in macro
variable ``m``. Using first differences (deltas) rather than levels
matches how CCAR scenarios are interpreted — "unemployment rises 6
percentage points" is a change, not a level.

Public API
----------
``FactorMacroRegression``
``fit_factor_macro_regression(factor_returns, macro_panel) -> FactorMacroRegression``
``project_factor_returns(regression, macro_delta_path) -> pd.DataFrame``
``quarterly_factor_returns_from_daily(factors) -> pd.DataFrame``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


@dataclass(frozen=True)
class FactorMacroRegression:
    """Fitted per-factor regression on macro variables."""

    intercepts: pd.Series          # (n_factors,)
    betas: pd.DataFrame            # (n_factors, n_macros)
    r_squared: pd.Series           # (n_factors,)
    residual_std: pd.Series        # (n_factors,)
    factor_cols: tuple[str, ...]
    macro_cols: tuple[str, ...]
    n_obs: int
    train_window: tuple[pd.Timestamp, pd.Timestamp] | None = None


def quarterly_factor_returns_from_daily(factors: pd.DataFrame) -> pd.DataFrame:
    """Compound daily factor returns to quarter-end returns.

    Factors in the Ken French panel are daily decimals; quarterly
    compounded equivalents are ``(1 + r_d).prod() - 1`` per quarter.
    """
    # Work on the factor columns (exclude RF if present)
    return (1.0 + factors).resample("QE").prod() - 1.0


def fit_factor_macro_regression(
    factor_returns_quarterly: pd.DataFrame,
    macro_panel: pd.DataFrame,
    factor_cols: list[str] | None = None,
    min_obs: int = 20,
) -> FactorMacroRegression:
    """Fit one OLS per factor against first-differences of the macro panel.

    Parameters
    ----------
    factor_returns_quarterly : DataFrame
        Quarter-end indexed, columns = factor names (``Mkt-RF``, ``SMB``,
        ``HML``, ``RMW``, ``CMA``, ``MOM``). Typically the output of
        :func:`quarterly_factor_returns_from_daily`.
    macro_panel : DataFrame
        Quarter-end indexed, columns = macro variable names. See
        :mod:`data.macro` for the canonical set.
    factor_cols : optional subset of factors to fit.
    min_obs : raise ValueError if fewer aligned rows remain.
    """
    # Align on quarter-end dates
    common = factor_returns_quarterly.index.intersection(macro_panel.index)
    if len(common) < min_obs + 1:  # +1 because we difference
        raise ValueError(
            f"Only {len(common)} aligned quarters; need >= {min_obs + 1} for stable fit."
        )
    f = factor_returns_quarterly.loc[common]
    m = macro_panel.loc[common]

    # First-difference the macros: delta_macro_q = macro_q - macro_{q-1}
    dm = m.diff().dropna(how="any")
    # Align factors to dm's index (first row of each is NaN from diff)
    fy = f.loc[dm.index]
    cols_factors = list(factor_cols) if factor_cols is not None else [
        c for c in fy.columns if c.upper() != "RF"
    ]
    macros = list(dm.columns)

    # Pre-build design matrix
    X = add_constant(dm.values, has_constant="add")

    intercepts: dict[str, float] = {}
    betas_rows: dict[str, np.ndarray] = {}
    r_squared: dict[str, float] = {}
    resid_se: dict[str, float] = {}

    for k in cols_factors:
        y = fy[k].values
        model = OLS(y, X, missing="drop").fit()
        intercepts[k] = float(model.params[0])
        betas_rows[k] = model.params[1:]
        r_squared[k] = float(model.rsquared)
        resid_se[k] = float(np.sqrt(model.mse_resid))

    betas_df = pd.DataFrame(betas_rows, index=macros).T  # rows = factors, cols = macros
    betas_df.index.name = "factor"
    betas_df.columns.name = "macro"

    return FactorMacroRegression(
        intercepts=pd.Series(intercepts, name="intercept"),
        betas=betas_df,
        r_squared=pd.Series(r_squared, name="r_squared"),
        residual_std=pd.Series(resid_se, name="residual_std"),
        factor_cols=tuple(cols_factors),
        macro_cols=tuple(macros),
        n_obs=len(fy),
        train_window=(pd.Timestamp(fy.index.min()), pd.Timestamp(fy.index.max())),
    )


def project_factor_returns(
    regression: FactorMacroRegression,
    macro_delta_path: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the fitted regression to a stressed macro delta path.

    Parameters
    ----------
    regression : FactorMacroRegression
        Output of :func:`fit_factor_macro_regression`.
    macro_delta_path : DataFrame
        Quarter-by-quarter first differences of the stressed macros.
        Index = quarter-end dates; columns must include every macro
        in ``regression.macro_cols``.

    Returns
    -------
    pd.DataFrame
        Projected factor returns per quarter (rows = quarters, cols = factors).
    """
    missing = [m for m in regression.macro_cols if m not in macro_delta_path.columns]
    if missing:
        raise ValueError(f"macro_delta_path missing required columns: {missing}")

    dm = macro_delta_path[list(regression.macro_cols)]
    # projected = intercept + dm @ betas.T (broadcast over quarters)
    projected = (
        regression.intercepts.values[None, :]
        + dm.values @ regression.betas.values.T
    )
    return pd.DataFrame(
        projected,
        index=dm.index,
        columns=list(regression.factor_cols),
    )
