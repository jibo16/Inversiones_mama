"""Reconcile Alpaca live fills against the ADV-aware sqrt-impact cost model.

Runs post-market-open and answers: did the backtest's slippage model
match reality? If not, by how much and in what direction?

Pipeline
--------
1. GET /v2/orders?status=filled&after=<since> from Alpaca.
2. For each filled order:
   * Reference price = yfinance daily Open for the fill date (what an
     MOO order nominally pays; fair reference when Alpaca queues orders
     overnight for the opening auction).
   * realized_slippage_bps = 10000 * (fill - open) / open, signed by side
     (buy: positive means we paid more than the open; sell: positive
     means we got less than the open — "friction against us").
   * predicted_slippage_bps = estimate_slippage(qty, price, adv, ...)
     from backtest/costs.py, same function used in the walk-forward.
3. Emit per-order CSV + aggregate summary text.
4. Gate: p95 of |realized - predicted| must be < 20 bps (V2 acceptance
   gate: live slippage distribution matches the IBKR Tiered cost model
   baseline within 20 bps at the 95th percentile).

Usage
-----
    python scripts/reconcile_alpaca_fills.py
    python scripts/reconcile_alpaca_fills.py --since 2026-04-22T22:00:00Z
    python scripts/reconcile_alpaca_fills.py --gate-bps 30

Exit codes
----------
0 = PASS (p95 |realized - predicted| within the gate)
1 = FAIL (p95 exceeds the gate)
2 = NO_DATA (no filled orders in the window)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from inversiones_mama.backtest.costs import estimate_slippage
from inversiones_mama.data.prices import load_prices
from inversiones_mama.data.volume import load_adv_shares

ALPACA_BASE = "https://paper-api.alpaca.markets"


def _alpaca_headers() -> dict[str, str]:
    key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("alpaca_key")
    secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("alpaca_secret")
    if not key or not secret:
        raise SystemExit(
            "Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in .env"
        )
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }


def fetch_filled_orders(since_iso: str) -> list[dict]:
    headers = _alpaca_headers()
    url = f"{ALPACA_BASE}/v2/orders"
    params = {
        "status": "filled",
        "after": since_iso,
        "limit": 200,
        "direction": "asc",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_open_price(ticker: str, date: pd.Timestamp) -> float | None:
    """yfinance daily Open for the fill date. Cached via load_prices."""
    start = date - timedelta(days=7)
    end = date + timedelta(days=1)
    try:
        import yfinance as yf  # noqa: PLC0415

        df = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] yfinance fetch failed for {ticker}: {exc}")
        return None
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    target = pd.Timestamp(date).normalize().tz_localize(None)
    if target not in df.index:
        nearest = df.index[df.index <= target]
        if len(nearest) == 0:
            return None
        target = nearest[-1]
    return float(df.loc[target, "Open"])


def reconcile(since: datetime, gate_bps: float) -> int:
    load_dotenv(".env")
    since_iso = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Fetching Alpaca fills since {since_iso} ...")
    orders = fetch_filled_orders(since_iso)
    if not orders:
        print("No filled orders in the window.")
        return 2

    tickers = sorted({o["symbol"] for o in orders})
    print(f"Found {len(orders)} fills across {len(tickers)} tickers: {tickers}")

    end = datetime.today() + timedelta(days=1)
    try:
        adv_map = load_adv_shares(tickers, end=end, window_days=30, use_cache=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] ADV load failed ({exc}); predictions will use base bps only.")
        adv_map = {}

    rows: list[dict] = []
    for o in orders:
        sym = o["symbol"]
        side = o["side"]
        qty = int(float(o.get("filled_qty") or o.get("qty") or 0))
        fill_px = float(o.get("filled_avg_price") or 0.0)
        filled_at_raw = o.get("filled_at") or o.get("updated_at") or o.get("submitted_at")
        try:
            filled_at = pd.Timestamp(filled_at_raw).tz_convert("America/New_York")
        except Exception:
            filled_at = pd.Timestamp(filled_at_raw)

        open_px = get_open_price(sym, filled_at)
        if open_px is None or open_px <= 0 or fill_px <= 0 or qty <= 0:
            continue

        direction = 1.0 if side == "buy" else -1.0
        realized_bps = 10_000.0 * direction * (fill_px - open_px) / open_px

        adv = float(adv_map.get(sym, 0.0)) if adv_map else 0.0
        # estimate_slippage returns USD; convert to bps against notional.
        predicted_usd = estimate_slippage(
            shares=qty, price=fill_px,
            adv=adv if adv > 0 else None,
        )
        notional = fill_px * qty
        predicted_bps = (predicted_usd * 10_000.0 / notional) if notional > 0 else 0.0

        rows.append({
            "submitted_at": o.get("submitted_at"),
            "filled_at": filled_at_raw,
            "symbol": sym,
            "side": side,
            "qty": qty,
            "open_px": round(open_px, 4),
            "fill_px": round(fill_px, 4),
            "notional": round(fill_px * qty, 2),
            "realized_bps": round(realized_bps, 2),
            "predicted_bps": round(predicted_bps, 2),
            "diff_bps": round(realized_bps - predicted_bps, 2),
            "abs_diff_bps": round(abs(realized_bps - predicted_bps), 2),
            "adv_shares": int(adv),
            "participation_pct_adv": round(100.0 * qty / adv, 4) if adv else None,
        })

    if not rows:
        print("No reconcilable fills (all failed price/qty validation).")
        return 2

    df = pd.DataFrame(rows).sort_values("filled_at")

    out_dir = Path("results/reconciliation")
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%Y-%m-%d")
    csv_path = out_dir / f"fills_{date_tag}.csv"
    df.to_csv(csv_path, index=False)

    # Aggregates
    realized = df["realized_bps"].to_numpy()
    predicted = df["predicted_bps"].to_numpy()
    abs_diff = df["abs_diff_bps"].to_numpy()

    stats = {
        "n_fills": int(len(df)),
        "realized_bps": {
            "median": float(np.median(realized)),
            "mean": float(realized.mean()),
            "p05": float(np.percentile(realized, 5)),
            "p95": float(np.percentile(realized, 95)),
            "max_abs": float(np.abs(realized).max()),
        },
        "predicted_bps": {
            "median": float(np.median(predicted)),
            "mean": float(predicted.mean()),
            "p05": float(np.percentile(predicted, 5)),
            "p95": float(np.percentile(predicted, 95)),
        },
        "abs_diff_bps": {
            "median": float(np.median(abs_diff)),
            "mean": float(abs_diff.mean()),
            "p95": float(np.percentile(abs_diff, 95)),
            "max": float(abs_diff.max()),
        },
        "pearson_corr_realized_vs_predicted": float(
            np.corrcoef(realized, predicted)[0, 1]
        ) if len(df) > 2 else None,
        "gate_bps": gate_bps,
        "gate_verdict": "PASS" if float(np.percentile(abs_diff, 95)) < gate_bps else "FAIL",
    }

    # Text summary
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"ALPACA FILL RECONCILIATION — {date_tag}")
    lines.append("=" * 72)
    lines.append(f"  fills reconciled:       {stats['n_fills']}")
    lines.append(f"  tickers:                {', '.join(sorted(df['symbol'].unique()))}")
    lines.append("")
    lines.append("  --- Realized slippage (bps, signed against us) ---")
    rb = stats["realized_bps"]
    lines.append(f"    median={rb['median']:+.2f}  mean={rb['mean']:+.2f}  p05={rb['p05']:+.2f}  p95={rb['p95']:+.2f}")
    lines.append("")
    lines.append("  --- Predicted slippage (bps, from sqrt-impact model) ---")
    pb = stats["predicted_bps"]
    lines.append(f"    median={pb['median']:+.2f}  mean={pb['mean']:+.2f}  p05={pb['p05']:+.2f}  p95={pb['p95']:+.2f}")
    lines.append("")
    lines.append("  --- |realized - predicted| ---")
    ab = stats["abs_diff_bps"]
    lines.append(f"    median={ab['median']:.2f}  mean={ab['mean']:.2f}  p95={ab['p95']:.2f}  max={ab['max']:.2f}")
    if stats["pearson_corr_realized_vs_predicted"] is not None:
        lines.append(f"    Pearson corr(realized, predicted) = {stats['pearson_corr_realized_vs_predicted']:+.3f}")
    lines.append("")
    verdict = stats["gate_verdict"]
    marker = "[PASS]" if verdict == "PASS" else "[FAIL]"
    lines.append(f"  {marker} p95 |realized - predicted| = {ab['p95']:.2f} bps  vs gate = {gate_bps:.2f} bps")
    lines.append("")
    lines.append("  per-order detail:")
    cols = ["filled_at", "symbol", "side", "qty", "open_px", "fill_px",
            "realized_bps", "predicted_bps", "diff_bps", "participation_pct_adv"]
    tbl = df[cols].to_string(index=False)
    for ln in tbl.splitlines():
        lines.append(f"    {ln}")
    lines.append("=" * 72)
    text = "\n".join(lines)
    print(text)

    txt_path = out_dir / f"summary_{date_tag}.txt"
    txt_path.write_text(text + "\n", encoding="utf-8")
    json_path = out_dir / f"stats_{date_tag}.json"
    json_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    print(f"\nWROTE: {csv_path}")
    print(f"WROTE: {txt_path}")
    print(f"WROTE: {json_path}")
    return 0 if verdict == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_since = datetime.now(tz=timezone.utc) - timedelta(hours=18)
    parser.add_argument(
        "--since",
        type=str,
        default=default_since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        help=(
            "ISO-8601 UTC timestamp: only fills submitted at or after this "
            "time are reconciled. Default: 18 hours ago."
        ),
    )
    parser.add_argument(
        "--gate-bps",
        type=float,
        default=20.0,
        help="Pass/fail gate: p95 |realized - predicted| must be under this.",
    )
    args = parser.parse_args(argv)
    since = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
    return reconcile(since=since, gate_bps=float(args.gate_bps))


if __name__ == "__main__":
    sys.exit(main())
