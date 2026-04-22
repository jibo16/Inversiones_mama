"""Inversiones_mama — high-risk quantitative portfolio package.

Architecture (see docs/architecture — TBD):
    data/          Price + factor data loaders with parquet caching.
    models/        Factor regression, SDE generators, HMM, Bayesian inference.
    sizing/        Risk-Constrained Kelly (CVXPY) and position sizing.
    backtest/      Walk-forward engine, IBKR Tiered cost model, portfolio P/L.
    simulation/    Monte Carlo, block bootstrap, performance metrics.
    validation/    Sanity gates and report generators.
    execution/     IBKR adapter (stubbed until live account is connected).

Build increments:
    v1a: data + factor_regression + kelly + costs + bootstrap MC + metrics + engine + gates.
    v1b: HMM + Merton jumps + Ornstein-Uhlenbeck bond paths.
    v2:  ABC-MCMC Bayesian inference + Black-Litterman.
"""

__version__ = "0.1.0"
