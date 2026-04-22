"""Consolidated v1a validation — Step 10, final verdict.

Runs the full pipeline end-to-end and evaluates every sanity gate defined
in :mod:`inversiones_mama.config`:

  1. Walk-forward backtest across the full price history (Agent 1 + 2 + Step 8).
  2. Split into in-sample and out-of-sample slices; compute performance
     metrics (including Deflated Sharpe) on each.
  3. Run Monte Carlo RCK validation on the most recent rebalance's weights.
  4. Collect pass/fail booleans into a single ``ValidationReport``.

The report includes both raw numbers and a pretty-printable table. Public
callers typically invoke ``run_full_validation`` and then ``report.render()``.

Public API
----------
``GateVerdict``           — one gate's observed, threshold, pass/fail, description.
``ValidationReport``      — aggregate report + pretty renderer.
``run_full_validation(prices, factors, ...)`` — the primary entry point.
``render_report(report)`` — pretty-print helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestConfig, BacktestResult, RebalanceFailure, walk_forward_backtest
from ..config import GATES
from ..simulation.metrics import PerformanceMetrics, compute_all_metrics
from ..simulation.monte_carlo import MCValidationResult, run_mc_rck_validation


# --------------------------------------------------------------------------- #
# Types                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateVerdict:
    """One gate's status."""

    name: str
    observed: float
    threshold: float
    passed: bool
    description: str
    comparator: str = "<"  # "<", ">", "<=", ">=", "=="


@dataclass(frozen=True)
class ValidationReport:
    """Aggregate v1a validation report."""

    config: BacktestConfig

    # Engine outputs
    n_rebalances: int
    initial_wealth: float
    final_wealth: float
    annualized_turnover_cost: float
    cumulative_cost: float

    # Full-sample performance
    metrics_full: PerformanceMetrics

    # In-sample / out-of-sample split
    oos_split_date: pd.Timestamp | None
    metrics_is: PerformanceMetrics | None
    metrics_oos: PerformanceMetrics | None

    # Monte Carlo on most recent rebalance
    mc_result: MCValidationResult | None

    # Gate verdicts
    gates: list[GateVerdict] = field(default_factory=list)

    # Rebalance exceptions swallowed by the engine (surfaced for diagnosis)
    rebalance_failures: list[RebalanceFailure] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return all(g.passed for g in self.gates)

    def render(self) -> str:
        """Pretty-print the full report as a string."""
        return render_report(self)


# --------------------------------------------------------------------------- #
# Core driver                                                                 #
# --------------------------------------------------------------------------- #


def run_full_validation(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    config: BacktestConfig | None = None,
    oos_split_date: datetime | pd.Timestamp | None = None,
    mc_n_paths: int = 5_000,
    mc_horizon_days: int = 252,
    mc_mean_block_length: int = 20,
    n_trials: int = 20,
    rng: np.random.Generator | None = None,
) -> ValidationReport:
    """Run the full v1a pipeline and collect all gate verdicts.

    Parameters
    ----------
    prices, factors : inputs to the walk-forward engine.
    config : ``BacktestConfig`` (defaults to v1a from config.py).
    oos_split_date : timestamp for the IS/OOS boundary. If None and there
        are >= 504 daily observations, defaults to the midpoint date.
    mc_n_paths, mc_horizon_days, mc_mean_block_length : MC settings.
    n_trials : number of strategy trials assumed when computing the
        Deflated Sharpe Ratio (N in the Bailey/López de Prado formula).
    rng : numpy Generator for reproducibility.
    """
    config = config or BacktestConfig()
    rng = rng if rng is not None else np.random.default_rng(20260422)

    # 1) Run walk-forward
    engine_result = walk_forward_backtest(prices, factors, config)

    # 2) Full-sample metrics
    daily = engine_result.daily_returns.dropna()
    if len(daily) < 60:
        raise ValueError(
            f"Daily-return series too short for metrics: {len(daily)} observations."
        )

    metrics_full = compute_all_metrics(daily, rf=0.0, periods=252, n_trials=n_trials)

    # 3) IS/OOS split
    split_ts: pd.Timestamp | None = None
    metrics_is: PerformanceMetrics | None = None
    metrics_oos: PerformanceMetrics | None = None
    if oos_split_date is not None or len(daily) >= 504:
        if oos_split_date is None:
            mid_idx = len(daily) // 2
            split_ts = pd.Timestamp(daily.index[mid_idx])
        else:
            split_ts = pd.Timestamp(oos_split_date)
        is_slice = daily.loc[daily.index < split_ts]
        oos_slice = daily.loc[daily.index >= split_ts]
        if len(is_slice) >= 60:
            metrics_is = compute_all_metrics(is_slice, rf=0.0, periods=252, n_trials=n_trials)
        if len(oos_slice) >= 60:
            metrics_oos = compute_all_metrics(oos_slice, rf=0.0, periods=252, n_trials=n_trials)

    # 4) Monte Carlo on last rebalance's weights
    mc_result: MCValidationResult | None = None
    if engine_result.rebalance_records:
        last_weights = engine_result.rebalance_records[-1].target_weights
        # v1a hardening (2026-04-22): use the FULL price history as the
        # bootstrap source, not the trailing `mc_horizon_days` window. The
        # trailing window induces recency bias and blinds the MC to
        # regime shifts like the 2022 bond/equity joint selloff. Full
        # history makes the stationary bootstrap resample across *all*
        # observed regimes proportional to their length.
        returns = prices.pct_change().dropna(how="all")
        bootstrap_source = returns.reindex(columns=prices.columns).dropna(how="all")
        if len(bootstrap_source) >= max(mc_horizon_days, 20):
            mc_result = run_mc_rck_validation(
                bootstrap_source,
                weights=last_weights,
                n_paths=mc_n_paths,
                horizon_days=mc_horizon_days,
                initial_capital=config.initial_capital,
                alpha=config.rck_alpha,
                beta=config.rck_beta,
                mean_block_length=mc_mean_block_length,
                rng=rng,
            )

    # 5) Assemble gate verdicts
    gates = _evaluate_gates(
        engine_result,
        metrics_full,
        metrics_is,
        metrics_oos,
        mc_result,
    )

    return ValidationReport(
        config=config,
        n_rebalances=len(engine_result.rebalance_records),
        initial_wealth=config.initial_capital,
        final_wealth=engine_result.final_wealth,
        annualized_turnover_cost=engine_result.annualized_turnover_cost,
        cumulative_cost=engine_result.cumulative_cost,
        metrics_full=metrics_full,
        oos_split_date=split_ts,
        metrics_is=metrics_is,
        metrics_oos=metrics_oos,
        mc_result=mc_result,
        gates=gates,
        rebalance_failures=engine_result.rebalance_failures,
    )


# --------------------------------------------------------------------------- #
# Gate evaluation                                                             #
# --------------------------------------------------------------------------- #


def _evaluate_gates(
    engine_result: BacktestResult,
    metrics_full: PerformanceMetrics,
    metrics_is: PerformanceMetrics | None,
    metrics_oos: PerformanceMetrics | None,
    mc_result: MCValidationResult | None,
) -> list[GateVerdict]:
    """Populate the list of gate verdicts."""
    gates: list[GateVerdict] = []

    # --- Turnover cost gate ---
    gates.append(
        GateVerdict(
            name="annualized_turnover_cost",
            observed=float(engine_result.annualized_turnover_cost),
            threshold=GATES.max_annual_turnover_cost,
            passed=bool(engine_result.annualized_turnover_cost < GATES.max_annual_turnover_cost),
            description="Commissions + slippage per year stay under 1.5% of capital",
            comparator="<",
        )
    )

    # --- OOS Sharpe gate ---
    oos_sharpe = metrics_oos.sharpe_ratio if metrics_oos is not None else metrics_full.sharpe_ratio
    gates.append(
        GateVerdict(
            name="oos_sharpe_positive",
            observed=float(oos_sharpe),
            threshold=GATES.min_oos_sharpe,
            passed=bool(oos_sharpe > GATES.min_oos_sharpe),
            description="Out-of-sample Sharpe ratio is strictly positive",
            comparator=">",
        )
    )

    # --- Monte Carlo gates (only if MC ran) ---
    if mc_result is not None:
        gates.append(
            GateVerdict(
                name="mc_prob_loss_40pct",
                observed=float(mc_result.prob_loss_40pct),
                threshold=GATES.max_prob_loss_40pct,
                passed=bool(mc_result.prob_loss_40pct < GATES.max_prob_loss_40pct),
                description="P(end-of-year wealth < $3,000 i.e. 40% loss) under 30%",
                comparator="<",
            )
        )
        gates.append(
            GateVerdict(
                name="mc_prob_loss_60pct",
                observed=float(mc_result.prob_loss_60pct),
                threshold=GATES.max_prob_loss_60pct,
                passed=bool(mc_result.prob_loss_60pct < GATES.max_prob_loss_60pct),
                description="P(end-of-year wealth < $2,000 i.e. 60% loss) under 10%",
                comparator="<",
            )
        )
        gates.append(
            GateVerdict(
                name="mc_dd_95th_pct",
                observed=float(mc_result.dd_p95),
                threshold=GATES.max_dd_95th_pct,
                passed=bool(mc_result.dd_p95 < GATES.max_dd_95th_pct),
                description="95th-percentile max drawdown under 50%",
                comparator="<",
            )
        )
        gates.append(
            GateVerdict(
                name="mc_rck_bound_honored",
                observed=float(mc_result.prob_dd_exceeds_rck_alpha),
                threshold=float(mc_result.rck_beta + 0.02),
                passed=bool(mc_result.prob_dd_exceeds_rck_alpha <= mc_result.rck_beta + 0.02),
                description=(
                    f"Empirical P(max DD >= {mc_result.rck_alpha}) within 2pp of the "
                    f"theoretical beta = {mc_result.rck_beta}"
                ),
                comparator="<=",
            )
        )

    return gates


# --------------------------------------------------------------------------- #
# Pretty printer                                                              #
# --------------------------------------------------------------------------- #


def render_report(report: ValidationReport) -> str:
    """Return a human-readable multi-line string summary of the report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("INVERSIONES_MAMA v1a VALIDATION REPORT")
    lines.append("=" * 78)
    lines.append(
        f"Capital: ${report.initial_wealth:,.0f}  ->  ${report.final_wealth:,.2f} "
        f"  (n_rebalances = {report.n_rebalances})"
    )
    lines.append(
        f"Kelly fraction: {report.config.kelly_fraction}   |   "
        f"Per-name cap: {report.config.per_name_cap}   |   "
        f"Lookback: {report.config.lookback_days}d   |   "
        f"Rebalance: {report.config.rebalance_freq}"
    )
    lines.append("")
    lines.append("--- Performance (annualized) " + "-" * 48)
    lines.append(
        f"  Full sample:  Sharpe = {report.metrics_full.sharpe_ratio:+.3f}    "
        f"Sortino = {report.metrics_full.sortino_ratio:+.3f}    "
        f"MaxDD = {report.metrics_full.max_drawdown:.3f}"
    )
    lines.append(
        f"                Ret = {report.metrics_full.annualized_return:+.3f}   "
        f"Vol = {report.metrics_full.annualized_volatility:.3f}   "
        f"Calmar = {report.metrics_full.calmar_ratio:+.3f}"
    )
    lines.append(
        f"                Deflated Sharpe = {report.metrics_full.deflated_sharpe:.3f}   "
        f"Skew = {report.metrics_full.skewness:+.2f}   "
        f"ExKurt = {report.metrics_full.excess_kurtosis:+.2f}"
    )
    if report.metrics_is is not None and report.metrics_oos is not None:
        lines.append(
            f"  In-sample:    Sharpe = {report.metrics_is.sharpe_ratio:+.3f}    "
            f"DSR = {report.metrics_is.deflated_sharpe:.3f}    "
            f"MaxDD = {report.metrics_is.max_drawdown:.3f}"
        )
        lines.append(
            f"  OUT-OF-SAMPLE Sharpe = {report.metrics_oos.sharpe_ratio:+.3f}    "
            f"DSR = {report.metrics_oos.deflated_sharpe:.3f}    "
            f"MaxDD = {report.metrics_oos.max_drawdown:.3f}    "
            f"(split = {report.oos_split_date.date() if report.oos_split_date else 'n/a'})"
        )
    lines.append("")
    lines.append("--- Costs " + "-" * 66)
    lines.append(
        f"  Annualized turnover cost: {report.annualized_turnover_cost*100:.3f}% of capital"
    )
    lines.append(
        f"  Cumulative cost over backtest: ${report.cumulative_cost:.2f}"
    )

    if report.mc_result is not None:
        mc = report.mc_result
        lines.append("")
        lines.append(
            f"--- Monte Carlo ({mc.n_paths} paths, horizon {mc.horizon_days}d) "
            + "-" * max(1, 28 - len(str(mc.n_paths)) - len(str(mc.horizon_days)))
        )
        lines.append(
            f"  Terminal wealth p05/p50/p95: "
            f"${mc.terminal_p05:,.0f} / ${mc.terminal_median:,.0f} / ${mc.terminal_p95:,.0f}"
        )
        lines.append(
            f"  Max drawdown p50/p95/p99: "
            f"{mc.dd_median:.3f} / {mc.dd_p95:.3f} / {mc.dd_p99:.3f}"
        )
        lines.append(
            f"  P(loss > 40%) = {mc.prob_loss_40pct*100:.2f}%   "
            f"P(loss > 60%) = {mc.prob_loss_60pct*100:.2f}%   "
            f"P(DD >= {mc.rck_alpha}) = {mc.prob_dd_exceeds_rck_alpha*100:.2f}%"
        )

    lines.append("")
    lines.append("--- Gate verdicts " + "-" * 60)
    for g in report.gates:
        mark = "PASS" if g.passed else "FAIL"
        lines.append(
            f"  [{mark}] {g.name:30s} observed={g.observed:+.4f} "
            f"{g.comparator} threshold={g.threshold:+.4f}"
        )
        lines.append(f"          {g.description}")

    lines.append("")
    overall = "PASS" if report.all_pass else "FAIL"
    lines.append(f"OVERALL v1a VERDICT: {overall}")
    lines.append("=" * 78)
    return "\n".join(lines)
