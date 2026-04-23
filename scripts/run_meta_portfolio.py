"""Run the multi-strategy meta-portfolio against one Alpaca paper account.

Reads the 20-strategy catalog from
``inversiones_mama.execution.strategy_catalog.STRATEGY_CATALOG``, and
for each spec scheduled to rebalance today (or all if ``--all``),
computes target weights from the spec's allocator and submits orders
to the chosen broker. Every fill is tagged with ``strategy_id`` and
recorded in the multi-strategy ledger (``results/ledger.db``).

Typical use cases
-----------------
    # Initial deployment of all 20 strategies against Alpaca paper
    python scripts/run_meta_portfolio.py --all --broker alpaca \\
        --seed-from-alpaca invvol_eqfloor_etfs

    # Dry-run against the DryRunClient (no broker calls)
    python scripts/run_meta_portfolio.py --all --broker dry

    # Scheduled monthly run (picks only specs due today)
    python scripts/run_meta_portfolio.py --broker alpaca

Flags
-----
  --all                  run every spec, not just today's
  --strategies ID,...    subset to a comma-separated list
  --broker {dry,alpaca}  execution venue (default: dry)
  --seed-from-alpaca ID  on first run: record current Alpaca positions
                         as fills for this strategy_id (only once)
  --reconcile            after execution, reconcile ledger vs. Alpaca
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from inversiones_mama.config import UNIVERSE as V1A_UNIVERSE
from inversiones_mama.data.factors import load_factor_returns
from inversiones_mama.data.liquid_universe import (
    LIQUID_ETFS,
    SP100_CORE,
    fetch_sp500_tickers,
)
from inversiones_mama.data.prices import load_prices
from inversiones_mama.execution.multi_strategy_ledger import (
    LedgerBackedClient,
    MultiStrategyLedger,
)
from inversiones_mama.execution.paper_trader import (
    DryRunClient,
    PaperTradingOrchestrator,
)
from inversiones_mama.execution.strategy_catalog import (
    ALLOCATOR_FACTORIES,
    STRATEGY_CATALOG,
    StrategySpec,
    due_today,
    get_spec,
)

log = logging.getLogger(__name__)

# Universe resolution
UNIVERSE_RESOLVERS = {
    "v1a":   lambda: list(V1A_UNIVERSE.keys()),
    "etfs":  lambda: list(LIQUID_ETFS),
    "sp100": lambda: list(SP100_CORE),
    "sp500": lambda: list(fetch_sp500_tickers()),
}


def _load_universe(key: str) -> list[str]:
    if key not in UNIVERSE_RESOLVERS:
        raise ValueError(f"unknown universe key: {key!r}")
    return UNIVERSE_RESOLVERS[key]()


def _build_alpaca_client():
    from inversiones_mama.execution.alpaca import AlpacaClient
    return AlpacaClient.from_env()


def _seed_from_alpaca(ledger: MultiStrategyLedger, strategy_id: str) -> int:
    """Record current Alpaca positions as initial fills for the strategy.

    Uses Alpaca's avg_entry_price for each position, fills it as a
    single synthetic buy at that price. The ledger's cash balance then
    matches: starting_cash - sum(qty * avg_entry_price).

    Only runs if the strategy has zero recorded fills.
    """
    existing_positions = ledger.positions(strategy_id)
    if existing_positions:
        print(f"  [seed] skipping {strategy_id}: already has {len(existing_positions)} positions")
        return 0

    key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("alpaca_key")
    secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("alpaca_secret")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    import requests
    r = requests.get("https://paper-api.alpaca.markets/v2/positions",
                     headers=headers, timeout=15)
    r.raise_for_status()
    positions = r.json()
    if not positions:
        print(f"  [seed] no Alpaca positions to absorb into {strategy_id}")
        return 0

    fills = []
    for p in positions:
        try:
            qty = float(p["qty"])
            price = float(p["avg_entry_price"])
        except (TypeError, ValueError, KeyError):
            continue
        if qty <= 0 or price <= 0:
            continue
        fills.append({
            "ticker":          p["symbol"],
            "side":            "buy",
            "qty":             qty,
            "fill_price":      price,
            "broker_order_id": f"seed_{p['symbol']}",
        })
    ids = ledger.bulk_record_fills(strategy_id, fills)
    print(f"  [seed] recorded {len(ids)} seed fills into {strategy_id}")
    return len(ids)


def _run_one_strategy(
    spec: StrategySpec,
    broker: str,
    ledger: MultiStrategyLedger,
    years: float = 5.0,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "strategy_id": spec.strategy_id,
        "allocator":   spec.allocator,
        "universe":    spec.universe,
        "rebal_day":   spec.rebalance_day,
        "status":      "pending",
    }
    print(f"\n=== {spec.strategy_id}  (allocator={spec.allocator}, universe={spec.universe}) ===",
          flush=True)

    # 1. Resolve universe + load prices
    try:
        tickers = _load_universe(spec.universe)
    except Exception as exc:
        result.update(status="universe_error", detail=str(exc))
        print(f"  [skip] {exc}")
        return result

    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=int(years * 365) + 14)
    try:
        prices = load_prices(tickers, start, end, use_cache=True)
    except Exception as exc:
        result.update(status="data_error", detail=str(exc)[:200])
        print(f"  [skip] data load failed: {exc}")
        return result

    # Coverage filter
    coverage = prices.notna().sum()
    min_cov = int(0.95 * len(prices))
    sparse = coverage[coverage < min_cov].index.tolist()
    if sparse:
        prices = prices.drop(columns=sparse)
    prices = prices.ffill().dropna(how="any")
    if prices.empty or prices.shape[1] < 2:
        result.update(status="data_error",
                      detail=f"insufficient survivors after coverage filter "
                             f"(shape={prices.shape})")
        print(f"  [skip] insufficient data: {prices.shape}")
        return result
    print(f"  prices: {prices.shape}  ({prices.index[0].date()} -> {prices.index[-1].date()})")

    # 2. Factors (rck_6factor needs them)
    try:
        factors = load_factor_returns(start=start, end=end, use_cache=True)
    except Exception as exc:
        factors = pd.DataFrame()
        print(f"  [warn] factors unavailable: {exc}")

    # 3. Build weight_fn from allocator key
    if spec.allocator not in ALLOCATOR_FACTORIES:
        result.update(status="allocator_error",
                      detail=f"unknown allocator: {spec.allocator}")
        return result
    weight_fn = ALLOCATOR_FACTORIES[spec.allocator]

    # 4. Build broker client, then wrap in LedgerBackedClient so the
    # orchestrator sees ONLY this strategy's bag for positions/cash,
    # not the commingled Alpaca account total.
    if broker == "dry":
        real_client = DryRunClient(starting_cash=ledger.cash(spec.strategy_id))
        latest = prices.iloc[-1]
        for t, px in latest.items():
            if pd.notna(px) and px > 0:
                real_client.set_latest_price(str(t), float(px))
    elif broker == "alpaca":
        real_client = _build_alpaca_client()
    else:
        raise ValueError(f"unknown broker: {broker}")
    client = LedgerBackedClient(real_client, ledger, spec.strategy_id)

    # 5. Run the rebalance
    orch = PaperTradingOrchestrator(
        client, prices, factors,
        weight_fn=weight_fn,
        max_deploy_capital=spec.starting_cash,
        fractional_shares=True,
        min_order_notional=1.0,
        strategy_id=spec.strategy_id,
        per_sector_cap=1.0,  # sector cap handled by allocator-specific logic
    )
    try:
        summary = orch.rebalance(
            signal_context={"strategy_id": spec.strategy_id, "broker": broker},
        )
    except Exception as exc:
        result.update(status="rebalance_error", detail=str(exc)[:300])
        print(f"  [error] rebalance failed: {exc}")
        return result

    # 6. Record fills into the ledger
    n_recorded = 0
    for entry in summary.trade_log:
        fill = entry.fill
        if fill.status not in ("filled", "partial"):
            continue
        if fill.filled_quantity == 0 or fill.fill_price is None:
            continue
        side = "buy" if (entry.signal.expected_size or 0) >= 0 else "sell"
        qty = abs(float(fill.filled_quantity))
        try:
            ledger.record_fill(
                strategy_id=spec.strategy_id,
                ticker=entry.signal.ticker,
                side=side,
                qty=qty,
                fill_price=float(fill.fill_price),
                fill_time=fill.fill_time,
                broker_order_id=fill.broker_order_id,
            )
            n_recorded += 1
        except Exception as exc:
            print(f"  [warn] ledger record failed for {entry.signal.ticker}: {exc}")

    result.update(
        status="ok",
        order_count=summary.order_count,
        fill_rate=summary.fill_rate,
        total_fill_value=summary.total_fill_value,
        n_ledger_rows=n_recorded,
    )
    print(f"  orders: {summary.order_count}  filled: {n_recorded}  "
          f"fill_rate: {summary.fill_rate*100:.1f}%  "
          f"fill_value: ${summary.total_fill_value:,.2f}")
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="Run every spec in the catalog, ignoring rebalance_day.")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated subset of strategy_ids to run.")
    parser.add_argument("--broker", choices=["dry", "alpaca"], default="dry",
                        help="Execution venue (default: dry, no real orders).")
    parser.add_argument("--seed-from-alpaca", type=str, default=None,
                        metavar="STRATEGY_ID",
                        help="Record current Alpaca positions as initial fills "
                             "for STRATEGY_ID (only if its ledger is empty).")
    parser.add_argument("--reconcile", action="store_true",
                        help="After execution, reconcile ledger vs. Alpaca positions.")
    parser.add_argument("--ledger-path", type=str, default="results/ledger.db")
    parser.add_argument("--env", type=str, default=".env")
    args = parser.parse_args(argv)

    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(env_path)

    # 1. Initialise ledger + register all specs (if_exists=skip makes re-runs safe)
    ledger = MultiStrategyLedger(db_path=args.ledger_path)
    for spec in STRATEGY_CATALOG:
        ledger.create_strategy(
            strategy_id=spec.strategy_id,
            allocator=spec.allocator,
            universe=spec.universe,
            starting_cash=spec.starting_cash,
            notes=spec.notes,
            if_exists="skip",
        )

    # 2. Optional: seed a strategy from current Alpaca positions
    if args.seed_from_alpaca:
        sid = args.seed_from_alpaca
        if get_spec(sid) is None:
            print(f"[error] unknown strategy_id for seeding: {sid}")
            return 2
        _seed_from_alpaca(ledger, sid)

    # 3. Select today's specs
    if args.strategies:
        wanted = {s.strip() for s in args.strategies.split(",") if s.strip()}
        specs = [s for s in STRATEGY_CATALOG if s.strategy_id in wanted]
        if not specs:
            print(f"[error] none of --strategies matched: {wanted}")
            return 2
    elif args.all:
        specs = list(STRATEGY_CATALOG)
    else:
        specs = due_today(STRATEGY_CATALOG, date.today())
        if not specs:
            print(f"[info] no strategies due on day-of-month {date.today().day}")
            return 0

    print(f"\n{'=' * 72}")
    print(f"META-PORTFOLIO RUN  |  {datetime.now().isoformat(timespec='seconds')}")
    print(f"  broker={args.broker}  strategies_to_run={len(specs)}  ledger={args.ledger_path}")
    print(f"{'=' * 72}")

    # 4. Execute each spec
    results = []
    for spec in specs:
        results.append(_run_one_strategy(spec, args.broker, ledger))

    # 5. Summarise
    print(f"\n{'=' * 72}")
    print(f"META-PORTFOLIO SUMMARY")
    print(f"{'=' * 72}")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_fail = len(results) - n_ok
    total_orders = sum(r.get("order_count", 0) for r in results)
    total_recorded = sum(r.get("n_ledger_rows", 0) for r in results)
    total_value = sum(r.get("total_fill_value", 0) for r in results)
    print(f"  strategies run:    {len(results)}")
    print(f"  ok:                {n_ok}")
    print(f"  failed:            {n_fail}")
    print(f"  orders submitted:  {total_orders}")
    print(f"  fills recorded:    {total_recorded}")
    print(f"  total fill value:  ${total_value:,.2f}")
    print()
    print("  Per-strategy ledger state:")
    for spec in STRATEGY_CATALOG:
        summary = ledger.strategy_summary(spec.strategy_id)
        if summary["n_positions"] > 0 or summary["cash"] != summary["starting_cash"]:
            print(f"    {spec.strategy_id:<30} cash=${summary['cash']:>9,.2f}  "
                  f"positions={summary['n_positions']:>3}")

    # 6. Persist the run record
    out_path = Path("results/meta_portfolio_runs") / \
               f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "timestamp": datetime.now().isoformat(),
            "broker": args.broker,
            "results": results,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  run record: {out_path}")

    # 7. Optional reconciliation
    if args.reconcile and args.broker == "alpaca":
        print(f"\n{'=' * 72}")
        print(f"LEDGER RECONCILIATION")
        print(f"{'=' * 72}")
        import requests
        key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("alpaca_key")
        secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("alpaca_secret")
        r = requests.get("https://paper-api.alpaca.markets/v2/positions",
                         headers={"APCA-API-KEY-ID": key,
                                  "APCA-API-SECRET-KEY": secret}, timeout=15)
        alpaca_positions = {p["symbol"].upper(): float(p["qty"]) for p in r.json()}
        report = ledger.reconcile_against_broker(alpaca_positions, tolerance=0.01)
        print(report.to_text())
        if not report.in_sync:
            print("[warn] ledger out of sync with Alpaca -- investigate drift")

    ledger.close()
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
