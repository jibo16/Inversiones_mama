"""Pre-deploy gate for the Alpaca paper pipeline.

Runs every check that must be GREEN before invoking
``run_paper_rebalance.py --broker alpaca``. On any failure, prints the
specific reason and exits non-zero so a shell pipeline can abort.

Checks (in order, fail-fast):

  1. Alpaca auth + network reachable
  2. Account status == ACTIVE
  3. PDT flag == False  (we don't want the $25k margin-locked path)
  4. Market is open  (--skip-clock disables this for offline smoke tests)
  5. Open orders == 0  (no pending fills that would conflict)
  6. Positions == 0    (fully-flat account; use --allow-positions to
                        waive this check when rebalancing vs. a non-zero book)
  7. Cash >= --min-cash (default $5,000 — the deployable capital slice)
  8. Buying power >= --min-cash

Exit codes
----------
0   all gates green; deploy is safe
1   a gate failed; do NOT deploy
2   network / auth error reaching Alpaca

Usage
-----
    python scripts/verify_deploy_ready.py
    python scripts/verify_deploy_ready.py --min-cash 5000 --json
    python scripts/verify_deploy_ready.py --allow-positions  # skip gate 6
    python scripts/verify_deploy_ready.py --skip-clock       # offline test

Intended invocation in the deploy loop:

    python scripts/verify_deploy_ready.py --min-cash 5000 && \\
    python scripts/run_paper_rebalance.py \\
        --allocator invvol_eqfloor --universe etfs \\
        --broker alpaca --max-capital 5000 \\
        --breaker-threshold 0.30 \\
        --log results/paper_invvol_eqfloor_alpaca.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import requests
from dotenv import load_dotenv


ALPACA_BASE = "https://paper-api.alpaca.markets"


def _headers() -> dict[str, str]:
    key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("alpaca_key")
    secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("alpaca_secret")
    if not key or not secret:
        raise SystemExit(
            "[fatal] ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY missing in .env"
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _get(path: str, headers: dict[str, str], timeout: float = 10.0) -> Any:
    url = f"{ALPACA_BASE}{path}"
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def run_checks(
    min_cash: float,
    allow_positions: bool,
    skip_clock: bool,
    json_out: bool,
) -> int:
    load_dotenv(".env")
    headers = _headers()

    results: list[dict] = []
    pass_all = True

    def record(name: str, ok: bool, detail: str, *, critical: bool = True) -> None:
        nonlocal pass_all
        results.append({"name": name, "ok": ok, "detail": detail, "critical": critical})
        if critical and not ok:
            pass_all = False

    # 1. Auth + account
    try:
        account = _get("/v2/account", headers)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        print(f"[fatal] Alpaca unreachable or auth failed: {exc} (HTTP {status})")
        return 2
    except Exception as exc:
        print(f"[fatal] Alpaca unreachable: {exc}")
        return 2

    record("alpaca_reachable", True, "GET /v2/account responded OK")

    # 2. Account status
    acct_status = str(account.get("status", "")).upper()
    record(
        "account_active", acct_status == "ACTIVE",
        f"account.status={acct_status}",
    )

    # 3. PDT flag
    pdt = bool(account.get("pattern_day_trader", False))
    record(
        "pdt_clear", not pdt,
        f"pattern_day_trader={pdt} (must be False for the unrestricted path)",
    )

    # 4. Market open
    if not skip_clock:
        clock = _get("/v2/clock", headers)
        is_open = bool(clock.get("is_open", False))
        detail = (
            f"is_open={is_open}  next_open={clock.get('next_open')}  "
            f"next_close={clock.get('next_close')}"
        )
        record("market_open", is_open, detail)
    else:
        record("market_open", True, "skipped (--skip-clock)", critical=False)

    # 5. Open orders
    open_orders = _get("/v2/orders?status=open&limit=200", headers)
    n_orders = len(open_orders)
    detail_orders = (
        f"open_orders={n_orders}"
        + ("" if n_orders == 0 else f" -- {[o['symbol'] for o in open_orders[:10]]}")
    )
    record("no_open_orders", n_orders == 0, detail_orders)

    # 6. Positions
    positions = _get("/v2/positions", headers)
    n_positions = len(positions)
    detail_positions = (
        f"positions={n_positions}"
        + ("" if n_positions == 0
           else f" -- {[p['symbol'] for p in positions[:10]]}")
    )
    record(
        "no_positions", n_positions == 0 or allow_positions,
        detail_positions
        + (" (waived by --allow-positions)" if allow_positions and n_positions > 0
           else ""),
    )

    # 7/8. Cash + buying power
    try:
        cash = float(account.get("cash", 0.0))
        bp = float(account.get("buying_power", 0.0))
    except (TypeError, ValueError):
        cash = 0.0
        bp = 0.0
    record(
        "cash_sufficient", cash >= min_cash,
        f"cash=${cash:,.2f} >= min_cash=${min_cash:,.2f}",
    )
    record(
        "buying_power_sufficient", bp >= min_cash,
        f"buying_power=${bp:,.2f} >= min_cash=${min_cash:,.2f}",
    )

    # Emit report
    if json_out:
        out = {
            "checked_at_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "all_pass": pass_all,
            "checks": results,
            "account": {
                "status": acct_status,
                "cash": cash,
                "buying_power": bp,
                "pattern_day_trader": pdt,
            },
        }
        print(json.dumps(out, indent=2))
    else:
        print("=" * 72)
        print("DEPLOY-READINESS CHECK")
        print("=" * 72)
        for r in results:
            mark = "PASS" if r["ok"] else ("FAIL" if r["critical"] else "skip")
            print(f"  [{mark}] {r['name']:<26} {r['detail']}")
        print("=" * 72)
        verdict = "GREEN -- safe to deploy" if pass_all else "RED -- do NOT deploy"
        print(f"  {verdict}")
        print("=" * 72)

    return 0 if pass_all else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-cash", type=float, default=5_000.0,
        help="Minimum cash + buying power required (default $5,000).",
    )
    parser.add_argument(
        "--allow-positions", action="store_true",
        help="Do NOT fail if the account holds positions "
             "(useful when rebalancing vs. an already-live book).",
    )
    parser.add_argument(
        "--skip-clock", action="store_true",
        help="Do NOT check that the market is open "
             "(for offline pre-flight tests).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit a machine-readable JSON report instead of the text summary.",
    )
    args = parser.parse_args(argv)
    return run_checks(
        min_cash=float(args.min_cash),
        allow_positions=bool(args.allow_positions),
        skip_clock=bool(args.skip_clock),
        json_out=bool(args.json),
    )


if __name__ == "__main__":
    sys.exit(main())
