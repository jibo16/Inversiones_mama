"""Run one paper-trading rebalance cycle.

Supports multiple allocation strategies via ``--allocator``:

* ``rck`` (default)            — 6-factor Risk-Constrained Kelly.
* ``vol_targeting``            — Volatility-targeting inverse-vol (60d, 15% target vol).
* ``invvol_eqfloor``           — Inverse-vol with 15% per-name cap + 40% equity floor.

Supports multiple universes via ``--universe``:

* ``v1a`` (default)            — 10 ETFs from config.UNIVERSE.
* ``etfs``                     — 91 liquid ETFs from LIQUID_ETFS.

Supports three brokers via the ``--broker`` flag:

* ``dry`` (default) — simulated perfect fills, no external service.
* ``alpaca``        — Alpaca paper, REST adapter (no 2FA, always available).
* ``ibkr``          — IBKR Client Portal Gateway (requires a 2FA'd session).

Optional ``--max-capital`` caps how many dollars the strategy allocates.
Alpaca seeds paper accounts with $100,000; pass ``--max-capital 5000``
to match Jorge's intended $5k account size so realized fills, slippage,
and turnover costs reflect what mom's eventual IBKR account will see.

Auto-halt circuit breaker: if ``--breaker-threshold`` is set, the cycle
installs a CircuitBreaker whose trip level is that fraction of peak
wealth and refuses to place orders once tripped. Integrates with the
PDT tracker so sub-$25k accounts stay below FINRA's 3-in-5 limit.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from inversiones_mama.data.factors import load_factor_returns
from inversiones_mama.data.prices import load_prices
from inversiones_mama.execution.circuit_breaker import CircuitBreaker
from inversiones_mama.execution.paper_trader import (
    DryRunClient,
    ExecutionClient,
    PaperTradingOrchestrator,
)
from inversiones_mama.execution.pdt import PDTTracker


# --------------------------------------------------------------------------- #
# Allocator weight functions                                                  #
# --------------------------------------------------------------------------- #


def _make_vol_targeting_fn():
    """Return a weight_fn for VolatilityTargeting (60d lookback, 15% target vol)."""
    from inversiones_mama.exploration.strategies.vol_targeting import VolatilityTargeting

    strategy = VolatilityTargeting(vol_lookback=60, target_vol=0.15)

    def weight_fn(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
        weights_df = strategy.generate_signals(prices)
        if weights_df.empty:
            return pd.Series(0.0, index=prices.columns)
        return weights_df.iloc[-1]

    return weight_fn


def _make_invvol_eqfloor_fn():
    """Return a weight_fn for inverse-vol with 15% cap + 40% equity floor."""
    from inversiones_mama.sizing.inverse_vol import generate_current_weights

    def weight_fn(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
        return generate_current_weights(
            prices,
            vol_lookback=60,
            per_name_cap=0.15,
            equity_floor=0.40,
        )

    return weight_fn


def _build_dryrun_client(cash: float, prices_df) -> DryRunClient:
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
    except AlpacaAuthError as exc:
        raise SystemExit(
            f"[alpaca] auth failed: {exc}\n"
            "  Configure alpaca_key / alpaca_secret in .env "
            "(or ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY)."
        ) from exc
    return client


def _build_ibkr_client() -> ExecutionClient:
    # Lazy import — ib_insync / Client Portal deps shouldn't block --broker dry
    from inversiones_mama.execution.ibkr import IBKRAdapter, IBKRConnectionError

    try:
        return IBKRAdapter.connect()
    except IBKRConnectionError as exc:
        raise SystemExit(
            f"[ibkr] Gateway not reachable: {exc}\n"
            "  Start the Gateway and complete 2FA at "
            "https://localhost:5000/sso/Login (see docs/IBKR_SETUP.md)."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocator",
                        choices=["rck", "vol_targeting", "invvol_eqfloor"],
                        default="rck",
                        help="Allocation strategy: rck (6-factor Kelly), "
                             "vol_targeting (60d/15%% target), "
                             "invvol_eqfloor (inv-vol + 15%% cap + 40%% eq floor). "
                             "Default: rck.")
    parser.add_argument("--universe",
                        choices=["v1a", "etfs"],
                        default="v1a",
                        help="Asset universe: v1a (10 ETFs), etfs (91 liquid ETFs). "
                             "Default: v1a. vol_targeting and invvol_eqfloor work "
                             "best on etfs.")
    parser.add_argument("--broker", choices=["dry", "alpaca", "ibkr"], default="dry",
                        help="Broker client (default: dry; simulated fills).")
    parser.add_argument("--cash", type=float, default=5_000.0,
                        help="Starting cash for --broker dry (ignored for alpaca/ibkr).")
    parser.add_argument("--max-capital", type=float, default=None,
                        help="Cap strategy deployment at this $ amount "
                             "(e.g. 5000 when running on a $100k Alpaca paper account).")
    parser.add_argument("--breaker-threshold", type=float, default=None,
                        help="If set, trip the circuit breaker at this drawdown "
                             "(positive fraction, e.g. 0.30 for 30%%).")
    parser.add_argument("--pdt-equity", type=float, default=None,
                        help="Account equity for PDT tracker. If unset, uses broker's cash balance.")
    parser.add_argument("--log", type=str, default="results/paper_trades.json",
                        help="Destination JSON file for the trade log.")
    parser.add_argument("--years", type=float, default=5.0,
                        help="Years of price history to pull for training.")
    parser.add_argument("--env", type=str, default=".env",
                        help="Path to .env file to load.")
    args = parser.parse_args(argv)

    # Load .env so os.getenv() sees the Alpaca / IBKR / etc. secrets
    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(env_path)

    # Resolve universe
    if args.universe == "etfs":
        from inversiones_mama.data.liquid_universe import LIQUID_ETFS
        tickers = sorted(set(LIQUID_ETFS))
        universe_label = f"LIQUID_ETFS ({len(tickers)} tickers)"
    else:
        from inversiones_mama.config import UNIVERSE
        tickers = list(UNIVERSE.keys())
        universe_label = f"v1a ({len(tickers)} tickers)"

    # Resolve allocator
    weight_fn = None
    if args.allocator == "vol_targeting":
        weight_fn = _make_vol_targeting_fn()
    elif args.allocator == "invvol_eqfloor":
        weight_fn = _make_invvol_eqfloor_fn()
    # else: rck (default, weight_fn=None triggers RCK path)

    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=int(args.years * 365) + 14)

    print(f"[1/4] Loading prices for {universe_label}: {start.date()} -> {end.date()}...")
    prices = load_prices(tickers, start, end, use_cache=True)
    print(f"       shape: {prices.shape}")

    print("[2/4] Loading factors...")
    factors = load_factor_returns(start=start, end=end, use_cache=True)
    print(f"       shape: {factors.shape}")

    print(f"[3/4] Connecting broker: {args.broker}...")
    if args.broker == "dry":
        client = _build_dryrun_client(args.cash, prices)
    elif args.broker == "alpaca":
        client = _build_alpaca_client()
    else:  # ibkr
        client = _build_ibkr_client()
    print(f"       connected. cash=${client.get_cash():,.2f}, positions={len(client.get_positions())}")

    # Optional breaker + PDT
    breaker = None
    if args.breaker_threshold is not None:
        breaker = CircuitBreaker(
            threshold=float(args.breaker_threshold),
            initial_wealth=args.max_capital or client.get_cash(),
        )
        print(f"       circuit_breaker threshold = {breaker.threshold:.2%}")
    pdt_equity = args.pdt_equity if args.pdt_equity is not None else client.get_cash()
    pdt = PDTTracker(account_equity=pdt_equity)
    print(f"       PDT tracker: equity=${pdt_equity:,.0f} exempt={pdt.exempt}")

    print(f"[4/4] Running one rebalance cycle (allocator={args.allocator}, max_capital={args.max_capital})...")
    orch = PaperTradingOrchestrator(
        client, prices, factors,
        weight_fn=weight_fn,
        max_deploy_capital=args.max_capital,
    )
    summary = orch.rebalance(
        signal_context={"broker": args.broker, "max_capital": args.max_capital},
        circuit_breaker=breaker,
        pdt_tracker=pdt,
    )

    print()
    if summary.halted:
        print(f"  !! HALTED: {summary.halt_reason}")
        print(f"     breaker_drawdown = {summary.breaker_drawdown}")
    print(f"  Rebalance time:       {summary.rebalance_time.isoformat()}")
    print(f"  Orders placed:        {summary.order_count}")
    print(f"  Trade log entries:    {len(summary.trade_log)}")
    print(f"  Estimated cost:       ${summary.estimated_cost:.2f}")
    print(f"  Fill rate:            {summary.fill_rate*100:.1f}%")
    print(f"  Total fill value:     ${summary.total_fill_value:,.2f}")
    deploy_desc = (f"${args.max_capital:,.0f} capped" if args.max_capital else "full")
    print(f"  Deployment basis:     {deploy_desc}")
    print()
    print("  Target weights (non-zero):")
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

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary.trade_log.save(log_path)
    print(f"\nSaved -> {log_path}")

    summary_path = log_path.with_name(log_path.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "broker": args.broker,
                "max_capital": args.max_capital,
                "rebalance_time": summary.rebalance_time.isoformat(),
                "order_count": summary.order_count,
                "total_fill_value": summary.total_fill_value,
                "estimated_cost": summary.estimated_cost,
                "fill_rate": summary.fill_rate,
                "halted": summary.halted,
                "halt_reason": summary.halt_reason,
                "breaker_drawdown": summary.breaker_drawdown,
                "target_weights": {t: float(w) for t, w in summary.target_weights.items()},
                "execution_stats": log_summary,
            },
            fh,
            indent=2,
            default=str,
        )
    print(f"Saved -> {summary_path}")
    return 0 if not summary.halted else 1


if __name__ == "__main__":
    sys.exit(main())
