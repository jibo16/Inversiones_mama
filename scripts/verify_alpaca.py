"""Smoke-test the Alpaca paper-trading credentials in .env.

Loads .env via python-dotenv, calls ``/v2/account``, prints a redacted
summary that confirms the session is valid without echoing the
account ID or credentials.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from inversiones_mama.execution.alpaca import (
    AlpacaAPIError,
    AlpacaAuthError,
    AlpacaClient,
)


def main() -> int:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)

    try:
        client = AlpacaClient.from_env()
    except AlpacaAuthError as exc:
        print(f"[FAIL] Missing credentials: {exc}")
        return 2

    print(f"Connecting to: {client.config.base_url}  (paper={client.config.is_paper})")
    try:
        acct = client.check_auth()
    except AlpacaAuthError as exc:
        print(f"[FAIL] Auth rejected by Alpaca: {exc}")
        return 3
    except AlpacaAPIError as exc:
        print(f"[FAIL] API error contacting Alpaca: {exc}")
        return 4

    # Redact account id; print only safe diagnostic fields
    safe_fields = [
        "status", "currency", "pattern_day_trader",
        "cash", "equity", "portfolio_value",
        "buying_power", "daytrading_buying_power", "regt_buying_power",
        "last_equity", "last_maintenance_margin",
        "created_at",
    ]
    print("\n[OK] Alpaca paper session is authenticated. Account summary:")
    for key in safe_fields:
        if key in acct:
            print(f"  {key:<30s} {acct[key]}")

    # Also test the positions + latest-price endpoints so we confirm
    # the data API works too
    try:
        positions = client.get_positions()
        print(f"\n  open positions                 : {len(positions)} "
              f"({', '.join(sorted(positions.keys())[:5])}{'...' if len(positions) > 5 else ''})")
    except AlpacaAPIError as exc:
        print(f"  [warn] positions endpoint      : {exc}")

    try:
        spy_px = client.get_latest_price("SPY")
        print(f"  latest SPY quote mid           : {spy_px}")
    except AlpacaAPIError as exc:
        print(f"  [warn] latest-price endpoint   : {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
