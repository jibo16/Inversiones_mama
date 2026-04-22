"""Report generator for exploration strategy results.

Produces the three mandated outputs:
  1. JSON summary report
  2. Equity curve CSV
  3. Trade log CSV

All outputs are saved to results/exploration/{strategy_name}_{timestamp}/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from inversiones_mama.config import RESULTS_DIR
from inversiones_mama.exploration.runner import StrategyResult

log = logging.getLogger(__name__)

EXPLORATION_RESULTS_DIR = RESULTS_DIR / "exploration"


def save_strategy_result(
    result: StrategyResult,
    output_dir: Path | None = None,
) -> Path:
    """Save a single strategy result to disk.

    Parameters
    ----------
    result : StrategyResult
        The result to save.
    output_dir : Path, optional
        Override output directory. Defaults to
        results/exploration/{strategy_name}_{timestamp}/

    Returns
    -------
    Path
        The directory where results were saved.
    """
    if output_dir is None:
        ts = result.run_timestamp.replace(" ", "_").replace(":", "-")
        safe_name = result.strategy_name.replace(" ", "_").lower()
        output_dir = EXPLORATION_RESULTS_DIR / f"{safe_name}_{ts}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON summary
    summary = result.to_summary_dict()
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Saved summary: %s", summary_path)

    # 2. Equity curve
    if result.equity_curve is not None and len(result.equity_curve) > 0:
        equity_path = output_dir / "equity_curve.csv"
        eq_df = result.equity_curve.reset_index()
        eq_df.columns = ["date", "equity"]
        eq_df.to_csv(equity_path, index=False)
        log.info("Saved equity curve: %s", equity_path)

    # 3. Trade log
    if result.trade_log:
        trade_path = output_dir / "trade_log.csv"
        trade_df = pd.DataFrame([
            {
                "date": t.date,
                "ticker": t.ticker,
                "direction": t.direction,
                "weight_before": t.weight_before,
                "weight_after": t.weight_after,
                "dollar_amount": t.dollar_amount,
                "shares": t.shares,
                "commission": t.commission,
                "slippage": t.slippage,
            }
            for t in result.trade_log
        ])
        trade_df.to_csv(trade_path, index=False)
        log.info("Saved trade log (%d trades): %s", len(result.trade_log), trade_path)

    return output_dir


def save_batch_results(
    results: list[StrategyResult],
    batch_name: str = "batch",
) -> Path:
    """Save an entire batch of strategy results.

    Creates a batch summary table plus individual strategy directories.

    Returns
    -------
    Path
        The batch output directory.
    """
    ts = results[0].run_timestamp.replace(" ", "_").replace(":", "-") if results else "unknown"
    batch_dir = EXPLORATION_RESULTS_DIR / f"{batch_name}_{ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save each strategy
    for result in results:
        safe_name = result.strategy_name.replace(" ", "_").lower()
        strat_dir = batch_dir / safe_name
        save_strategy_result(result, output_dir=strat_dir)

    # Batch summary table
    summaries = [r.to_summary_dict() for r in results]
    summary_df = pd.DataFrame(summaries)

    # Sort: candidates first, then by Sharpe descending
    summary_df["_sort"] = summary_df["status"].map(
        {"candidate": 0, "rejected": 1}
    )
    summary_df = summary_df.sort_values(
        ["_sort", "sharpe"], ascending=[True, False]
    ).drop(columns=["_sort"])

    summary_path = batch_dir / "batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    summary_json_path = batch_dir / "batch_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summaries, f, indent=2, default=str)

    log.info("Saved batch summary (%d strategies): %s", len(results), batch_dir)

    return batch_dir


def format_results_table(results: list[StrategyResult]) -> str:
    """Format results as a human-readable ASCII table for console output."""
    lines = []
    header = f"{'Strategy':<35} {'Cat':<12} {'SR':>7} {'DSR':>7} {'MaxDD':>7} {'CAGR':>8} {'Status':<12}"
    lines.append(header)
    lines.append("─" * len(header))

    for r in results:
        m = r.metrics_oos
        sr = f"{m.sharpe_ratio:7.2f}" if m else "   N/A"
        dsr = f"{m.deflated_sharpe:7.3f}" if m else "   N/A"
        mdd = f"{m.max_drawdown:7.1%}" if m else "   N/A"
        cagr = f"{m.annualized_return:8.1%}" if m else "    N/A"
        icon = "✓" if r.status == "candidate" else "✗"
        status = f"{icon} {r.status}"
        lines.append(f"{r.strategy_name:<35} {r.category:<12} {sr} {dsr} {mdd} {cagr} {status:<12}")

    n_cand = sum(1 for r in results if r.status == "candidate")
    lines.append("─" * len(header))
    lines.append(f"Total: {len(results)} strategies | {n_cand} candidates | {len(results) - n_cand} rejected")

    return "\n".join(lines)
