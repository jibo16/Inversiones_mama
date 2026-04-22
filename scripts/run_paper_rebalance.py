"""Run one paper-trading rebalance cycle of the v1a strategy.

Zero-budget deployment driver. By default uses :class:`DryRunClient` —
simulated perfect fills, no real broker connection. When Agent 3 finishes
the IBKR Client Portal live-data adapter, swap ``--client ibkr`` to route
to the live paper account.

Usage
-----
    .venv\\Scripts\\python.exe scripts\\run_paper_rebalance.py
    .venv\\Scripts\\python.exe scripts\\run_paper_rebalance.py --cash 10000
    .venv\\Scripts\\python.exe scripts\\run_paper_rebalance.py --log results/paper_trades.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from inversiones_mama.config import UNIVERSE
from inversiones_mama.data.factors import load_factor_returns
from inversiones_mama.data.prices import load_prices
from inversiones_mama.execution.paper_trader import (
    DryRunClient,
    PaperTradingOrchestrator,
)


def _build_dryrun_client(cash: float, prices_df) -> DryRunClient:
    client = DryRunClient(starting_cash=cash)
    latest = prices_df.iloc[-1]
    for ticker, price in latest.items():
        if price > 0:
            client.set_latest_price(str(ticker), float(price))
    return client


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paper-trading rebalance cycle for v1a.")
    parser.add_argument("--client", choices=["dry", "ibkr"], default="dry",
                        help="Broker client (default: dry, simulated fills).")
    parser.add_argument("--cash", type=float, default=5_000.0,
                        help="Starting cash for the dry-run client (ignored for ibkr).")
    parser.add_argument("--log", type=str, default="results/paper_trades.json",
                        help="Destination JSON file for the trade log.")
    parser.add_argument("--years", type=float, default=5.0,
                        help="Years of price history to pull for training.")
    args = parser.parse_args(argv)

    tickers = list(UNIVERSE.keys())
    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=int(args.years * 365) + 14)

    print(f"[1/3] Loading prices for {len(tickers)} tickers: {start.date()} -> {end.date()}...")
    prices = load_prices(tickers, start, end, use_cache=True)
    print(f"       shape: {prices.shape}")

    print("[2/3] Loading factors...")
    factors = load_factor_returns(start=start, end=end, use_cache=True)
    print(f"       shape: {factors.shape}")

    if args.client == "dry":
        client = _build_dryrun_client(args.cash, prices)
    else:
        print("[!] --client ibkr is not yet wired. Waiting for Agent 3's Client Portal adapter to commit.")
        print("    Falling back to --client dry.")
        client = _build_dryrun_client(args.cash, prices)

    print("[3/3] Running one rebalance cycle...")
    orch = PaperTradingOrchestrator(client, prices, factors)
    summary = orch.rebalance(signal_context={"mode": args.client})

    print()
    print(f"  Rebalance time:       {summary.rebalance_time.isoformat()}")
    print(f"  Orders placed:        {summary.order_count}")
    print(f"  Trade log entries:    {len(summary.trade_log)}")
    print(f"  Estimated cost:       ${summary.estimated_cost:.2f}")
    print(f"  Fill rate:            {summary.fill_rate*100:.1f}%")
    print(f"  Total fill value:     ${summary.total_fill_value:,.2f}")
    print()
    print("  Target weights:")
    nonzero = summary.target_weights[summary.target_weights > 1e-4].sort_values(ascending=False)
    for t, w in nonzero.items():
        print(f"    {t:<6s} {w*100:5.2f}%")
    cash_w = max(0.0, 1.0 - nonzero.sum())
    print(f"    CASH   {cash_w*100:5.2f}%")

    log_summary = summary.trade_log.summary()
    print()
    print("  Execution stats:")
    for k, v in log_summary.items():
        print(f"    {k:<28s} {v}")

    # Persist the trade log
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary.trade_log.save(log_path)
    print(f"\nSaved -> {log_path}")

    # Also save a summary JSON for easy downstream parsing
    summary_path = log_path.with_name(log_path.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "rebalance_time": summary.rebalance_time.isoformat(),
                "order_count": summary.order_count,
                "total_fill_value": summary.total_fill_value,
                "estimated_cost": summary.estimated_cost,
                "fill_rate": summary.fill_rate,
                "target_weights": {t: float(w) for t, w in summary.target_weights.items()},
                "execution_stats": log_summary,
            },
            fh,
            indent=2,
            default=str,
        )
    print(f"Saved -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
