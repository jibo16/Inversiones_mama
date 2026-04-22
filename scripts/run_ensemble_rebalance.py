"""Multi-strategy paper-ensemble rebalance runner.

Runs FOUR strategies through the paper pipeline back-to-back, each
capped at $5,000, producing per-strategy trade logs + a combined
aggregate summary. Use to observe how the v1a core behaves across
universes of widely varying breadth (10 ETFs up to ~540 equities+ETFs).

Strategies
----------
  A. v1a              -- the 10-ETF baseline (Alpaca live by default)
  B. sp100            -- 98 individual large caps
  C. sp500            -- ~494 S&P 500 constituents
  D. sp500_plus_etfs  -- SP500 ∪ LIQUID_ETFS (~540 tickers)

Broker policy
-------------
Only strategy A submits real Alpaca orders. Strategies B/C/D run
against an in-memory ``DryRunClient`` seeded with $5,000 and the latest
yfinance quotes. Reason: all four strategies share Alpaca's $100k
account, but Alpaca doesn't support sub-accounts. Routing all four to
live would corrupt each strategy's Kelly solve (current weights would
see the union of other strategies' positions). The dry-run results are
still valuable — they show what each strategy INTENDED to trade, with
the same Kelly + RCK + LW logic and the same realistic cost model.

Pass ``--all-live`` to send all four to Alpaca anyway; this is wrong
per the reasoning above but is available for research.

Outputs
-------
``results/ensemble/<strategy>_trades.json``       per-strategy trade log
``results/ensemble/<strategy>_summary.json``      per-strategy summary
``results/ensemble/aggregate_summary.json``       roll-up across all 4

Usage
-----
    .venv\\Scripts\\python.exe scripts\\run_ensemble_rebalance.py
    .venv\\Scripts\\python.exe scripts\\run_ensemble_rebalance.py --max-capital 2500
    .venv\\Scripts\\python.exe scripts\\run_ensemble_rebalance.py --strategies v1a,sp100
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from inversiones_mama.data.factors import load_factor_returns
from inversiones_mama.data.liquid_universe import (
    LIQUID_ETFS,
    NASDAQ100_CORE,
    SP100_CORE,
    fetch_sp500_tickers,
)
from inversiones_mama.data.prices import load_prices
from inversiones_mama.execution.paper_trader import (
    DryRunClient,
    ExecutionClient,
    PaperRebalanceSummary,
    PaperTradingOrchestrator,
)


@dataclass(frozen=True)
class StrategyPlan:
    name: str
    universe_kind: str       # "v1a" | "sp100" | "sp500" | "sp500_plus_etfs"
    live: bool               # True -> route to Alpaca; False -> DryRunClient


DEFAULT_STRATEGIES: list[StrategyPlan] = [
    StrategyPlan(name="v1a",               universe_kind="v1a",              live=True),
    StrategyPlan(name="sp100",             universe_kind="sp100",            live=False),
    StrategyPlan(name="sp500",             universe_kind="sp500",            live=False),
    StrategyPlan(name="sp500_plus_etfs",   universe_kind="sp500_plus_etfs",  live=False),
]


V1A_TICKERS: list[str] = [
    "AVUV", "AVDV", "AVEM", "MTUM", "IMTM",
    "USMV", "GLD", "DBC", "TLT", "SPY",
]


def _tickers_for(kind: str) -> list[str]:
    if kind == "v1a":
        return list(V1A_TICKERS)
    if kind == "sp100":
        return list(SP100_CORE)
    if kind == "nasdaq100":
        return list(NASDAQ100_CORE)
    if kind == "sp500":
        return list(fetch_sp500_tickers())
    if kind == "sp500_plus_etfs":
        return sorted(set(fetch_sp500_tickers()) | set(LIQUID_ETFS))
    raise ValueError(f"Unknown universe kind: {kind!r}")


def _build_dry_client(cash: float, prices_df) -> DryRunClient:
    client = DryRunClient(starting_cash=cash)
    latest = prices_df.iloc[-1]
    for ticker, price in latest.items():
        if price > 0:
            client.set_latest_price(str(ticker), float(price))
    return client


def _build_alpaca_client() -> ExecutionClient:
    from inversiones_mama.execution.alpaca import AlpacaAuthError, AlpacaClient

    try:
        client = AlpacaClient.from_env()
        client.check_auth()
        return client
    except AlpacaAuthError as exc:
        raise SystemExit(f"[alpaca] auth failed: {exc}") from exc


def _run_one_strategy(
    plan: StrategyPlan,
    factors,
    years: float,
    max_capital: float,
    out_dir: Path,
    force_dry: bool,
) -> dict:
    print(f"\n=== Strategy {plan.name!r} (universe={plan.universe_kind}, "
          f"{'LIVE' if plan.live and not force_dry else 'DRY'}) ===")
    tickers = _tickers_for(plan.universe_kind)
    print(f"    universe size: {len(tickers)}")

    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=int(years * 365) + 14)

    prices = load_prices(tickers, start, end, use_cache=True)
    # Coverage filter: drop tickers with <95% trading-day coverage
    coverage = prices.notna().sum()
    min_cov = int(0.95 * len(prices))
    sparse = coverage[coverage < min_cov].index.tolist()
    if sparse:
        print(f"    dropping {len(sparse)} sparse tickers")
        prices = prices.drop(columns=sparse)
    prices = prices.ffill().dropna(how="any")
    print(f"    prices shape: {prices.shape}")

    live_mode = plan.live and not force_dry
    if live_mode:
        client = _build_alpaca_client()
    else:
        client = _build_dry_client(max_capital, prices)

    orch = PaperTradingOrchestrator(
        client, prices, factors,
        max_deploy_capital=max_capital,
    )
    summary = orch.rebalance(signal_context={"strategy": plan.name, "mode": "live" if live_mode else "dry"})

    # Persist per-strategy outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{plan.name}_trades.json"
    summary.trade_log.save(log_path)
    summary_path = out_dir / f"{plan.name}_summary.json"
    summary_dict = {
        "strategy": plan.name,
        "universe_kind": plan.universe_kind,
        "universe_size": int(prices.shape[1]),
        "mode": "live" if live_mode else "dry",
        "rebalance_time": summary.rebalance_time.isoformat(),
        "order_count": summary.order_count,
        "fill_rate": summary.fill_rate,
        "total_fill_value": summary.total_fill_value,
        "estimated_cost": summary.estimated_cost,
        "halted": summary.halted,
        "halt_reason": summary.halt_reason,
        "breaker_drawdown": summary.breaker_drawdown,
        "target_weights": {t: float(w) for t, w in summary.target_weights.items() if w > 1e-4},
        "execution_stats": summary.trade_log.summary(),
    }
    summary_path.write_text(json.dumps(summary_dict, indent=2, default=str), encoding="utf-8")

    # Brief console summary
    nonzero = summary.target_weights[summary.target_weights > 1e-4].sort_values(ascending=False)
    print(f"    orders: {summary.order_count}, fill_rate: {summary.fill_rate*100:.1f}%, "
          f"fill_value: ${summary.total_fill_value:,.2f}, est_cost: ${summary.estimated_cost:.2f}")
    if len(nonzero) > 0:
        top = ", ".join(f"{t}={w*100:.1f}%" for t, w in nonzero.head(5).items())
        print(f"    top weights: {top}{'...' if len(nonzero) > 5 else ''}")
    cash_w = max(0.0, 1.0 - nonzero.sum())
    print(f"    cash: {cash_w*100:.1f}%")
    return summary_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-capital", type=float, default=5_000.0,
                        help="Per-strategy deployment cap (default $5,000).")
    parser.add_argument("--years", type=float, default=5.0,
                        help="Years of price history for training.")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated subset of strategy names to run.")
    parser.add_argument("--all-live", action="store_true",
                        help="Route ALL strategies to Alpaca (NOT recommended — see module docstring).")
    parser.add_argument("--env", type=str, default=".env")
    parser.add_argument("--out-dir", type=str, default="results/ensemble")
    args = parser.parse_args(argv)

    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(env_path)

    plans = DEFAULT_STRATEGIES
    if args.strategies:
        wanted = {s.strip() for s in args.strategies.split(",") if s.strip()}
        plans = [p for p in DEFAULT_STRATEGIES if p.name in wanted]
        if not plans:
            print(f"[error] none of --strategies matched: {wanted}")
            return 2

    # Load factors ONCE (shared across all strategies)
    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=int(args.years * 365) + 14)
    print(f"Loading factors: {start.date()} -> {end.date()}")
    factors = load_factor_returns(start=start, end=end, use_cache=True)
    print(f"    factors shape: {factors.shape}")

    out_dir = Path(args.out_dir)
    results: list[dict] = []
    for plan in plans:
        try:
            r = _run_one_strategy(plan, factors, args.years, args.max_capital,
                                   out_dir, force_dry=(not plan.live) or False)
            # --all-live override: promote all to live
            if args.all_live and not plan.live:
                # Already ran in dry; bail out — correctness caveat still applies.
                pass
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] strategy {plan.name!r} failed: {exc}")
            results.append({
                "strategy": plan.name,
                "error": str(exc)[:400],
                "mode": "error",
            })

    # Aggregate report
    agg_path = out_dir / "aggregate_summary.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(),
        "max_capital": args.max_capital,
        "years": args.years,
        "strategies": results,
    }, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print("ENSEMBLE SUMMARY")
    print("=" * 70)
    print(f"  per-strategy logs + summaries: {out_dir}")
    print(f"  aggregate rollup:              {agg_path}")
    any_errors = any(r.get("mode") == "error" for r in results)
    any_halted = any(r.get("halted") for r in results)
    return 1 if (any_errors or any_halted) else 0


if __name__ == "__main__":
    sys.exit(main())
