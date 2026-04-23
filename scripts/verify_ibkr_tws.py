"""Live smoke-test for the IBKR TWS adapter.

Requires a running TWS or IB Gateway instance with API access enabled
and the matching port:

    Paper:  7497
    Live:   7496

Reads ``IBKR_TWS_HOST / IBKR_TWS_PORT / IBKR_TWS_CLIENT_ID`` from .env.
Does NOT place any order — only:

1. Connects.
2. Reads positions + cash.
3. Fetches the latest SPY quote.
4. Disconnects.

Exit codes: 0 = reachable, 1 = connect failure, 2 = other error.

Usage
-----
    python scripts/verify_ibkr_tws.py
    python scripts/verify_ibkr_tws.py --symbol AVDV
    python scripts/verify_ibkr_tws.py --port 7496    # go live (do not do this lightly)
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from inversiones_mama.execution.ibkr_tws import (
    IBKRTWSClient,
    IBKRTWSConfig,
    IBKRTWSConnectError,
    IBKRTWSError,
)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("IBKR_TWS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("IBKR_TWS_PORT", "7497")))
    parser.add_argument("--client-id", type=int, default=int(os.getenv("IBKR_TWS_CLIENT_ID", "1")))
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    cfg = IBKRTWSConfig(
        host=args.host, port=args.port, client_id=args.client_id,
        timeout_seconds=args.timeout,
    )
    print(f"Connecting to TWS at {cfg.host}:{cfg.port} (clientId={cfg.client_id})...")
    print(f"  paper={cfg.is_paper}")

    try:
        with IBKRTWSClient(config=cfg) as client:
            cash = client.get_cash()
            print(f"[OK] cash = ${cash:,.2f}")

            positions = client.get_positions()
            print(f"[OK] positions ({len(positions)}):")
            for sym, qty in sorted(positions.items()):
                print(f"       {sym:6s} {qty:+d}")

            price = client.get_latest_price(args.symbol)
            if price is not None:
                print(f"[OK] latest {args.symbol} mid = ${price:.2f}")
            else:
                print(f"[warn] could not fetch quote for {args.symbol} "
                      f"(market closed or no subscription)")
        print("\nSUCCESS: TWS reachable + adapter functional.")
        return 0
    except IBKRTWSConnectError as exc:
        print(f"\nCONNECT FAILED: {exc}", file=sys.stderr)
        print("\nChecklist:", file=sys.stderr)
        print("  1. Is TWS or IB Gateway running?", file=sys.stderr)
        print("  2. In TWS -> File -> Global Config -> API -> Settings:", file=sys.stderr)
        print("     - 'Enable ActiveX and Socket Clients' = ON", file=sys.stderr)
        print(f"     - Socket port = {cfg.port}", file=sys.stderr)
        print("     - 'Read-Only API' unchecked (if you want to trade)", file=sys.stderr)
        print("  3. Is the clientId free? (TWS enforces uniqueness per session)", file=sys.stderr)
        return 1
    except IBKRTWSError as exc:
        print(f"\nADAPTER ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
