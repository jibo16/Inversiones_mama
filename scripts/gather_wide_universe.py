"""Multi-source wide-universe data gatherer.

Aggregates ticker symbols from every free source we have access to, pulls
maximum-history price data via yfinance in batches with retries, caches
to parquet, and reports the final coverage.

Sources (all free, no paid tier required):

* **US equities wide**: iShares IWV (Russell 3000) + IWB (Russell 1000)
  + IWM (Russell 2000) holdings CSVs, deduplicated.
* **Liquid ETFs**: the full ``LIQUID_ETFS`` list (broad equity, sector,
  country, bond, commodity, factor, real-estate, volatility ETFs).
* **International ADRs**: curated list of large non-US companies listed
  on US exchanges (TSM, ASML, BABA, NVO, etc.).
* **Commodity futures proxies** (yfinance ``=F`` suffix): gold, silver,
  copper, crude, natgas, corn, soy, wheat, sugar, coffee.
* **FX major pairs** (yfinance ``=X`` suffix): EURUSD, GBPUSD, USDJPY,
  AUDUSD, USDCAD, USDCHF, NZDUSD, plus cross pairs.
* **Volatility products**: ^VIX index, VXX, UVXY, SVXY, VIXM, VIXY.

Crypto is intentionally excluded per Jorge's portfolio mandate (no
crypto exposure). Crypto prices could be included here for research
purposes but the user explicitly forbade crypto in the deployable
portfolio, so we keep the eligible test universe consistent with what
we can actually trade.

Outputs
-------
* ``results/wide_gather/universe.txt`` — one ticker per line, all assets
  we attempted to fetch.
* ``results/wide_gather/coverage.csv`` — per-ticker coverage (start
  date, end date, # trading days, % non-null).
* ``results/wide_gather/summary.json`` — aggregate: # tickers requested,
  # successful, total data points, per-class breakdown.
* ``results/wide_gather/failed.txt`` — tickers yfinance could not return.

Usage
-----
    python scripts/gather_wide_universe.py
    python scripts/gather_wide_universe.py --start 1990-01-01 --batch-size 250
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from inversiones_mama.data.liquid_universe import (
    LIQUID_ETFS,
    fetch_russell3000_tickers,
    fetch_sp500_tickers,
)
from inversiones_mama.data.prices import fetch_prices_yfinance


# --- Asset-class definitions ------------------------------------------------


# Generic iShares holdings CSV URL template. Different funds' stable IDs.
_ISHARES_URLS: dict[str, str] = {
    "IWV": ("https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/"
            "1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"),
    "IWB": ("https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
            "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"),
    "IWM": ("https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
            "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"),
}


INTERNATIONAL_ADRS: tuple[str, ...] = (
    # Tech / semis
    "TSM", "ASML", "BABA", "JD", "PDD", "BIDU", "NTES", "SE",
    # Consumer / staples
    "NVO", "TM", "SONY", "HMC", "UL", "DEO", "BTI", "PM", "NSRGY",
    # Energy / materials
    "BP", "SHEL", "TTE", "RIO", "BHP", "VALE", "SCCO",
    # Financials
    "HSBC", "BCS", "BBVA", "SAN", "ING", "MUFG", "SMFG", "RY", "BMO", "BNS", "TD",
    # Healthcare / pharma
    "NVS", "AZN", "GSK", "SNY",
    # Other
    "SAP", "PHG", "NOK", "ERIC", "STM", "E", "ENI",
    "INFY", "WIT", "HDB", "IBN", "ICICI",
)


COMMODITY_FUTURES: tuple[str, ...] = (
    # Metals
    "GC=F",   # Gold
    "SI=F",   # Silver
    "PL=F",   # Platinum
    "PA=F",   # Palladium
    "HG=F",   # Copper
    # Energy
    "CL=F",   # Crude WTI
    "BZ=F",   # Brent
    "NG=F",   # Nat gas
    "HO=F",   # Heating oil
    "RB=F",   # Gasoline
    # Agriculture
    "ZC=F",   # Corn
    "ZS=F",   # Soybeans
    "ZW=F",   # Wheat
    "KC=F",   # Coffee
    "SB=F",   # Sugar
    "CT=F",   # Cotton
    "CC=F",   # Cocoa
    # Meat
    "LE=F",   # Live cattle
    "HE=F",   # Lean hogs
)


FX_PAIRS: tuple[str, ...] = (
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
    "USDCAD=X", "USDCHF=X", "NZDUSD=X",
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X",
    "USDCNY=X", "USDMXN=X", "USDBRL=X", "USDINR=X",
    "USDSEK=X", "USDNOK=X", "USDZAR=X", "USDTRY=X",
)


VOLATILITY_PRODUCTS: tuple[str, ...] = (
    "^VIX", "VXX", "UVXY", "SVXY", "VIXM", "VIXY", "VXZ",
)


ADDITIONAL_ETFS: tuple[str, ...] = (
    # Country ETFs not in LIQUID_ETFS
    "EWA", "EWC", "EWH", "EWI", "EWP", "EWQ", "EWS", "EWT", "EWY",
    "EWW", "EZA", "TUR", "THD", "VNM",
    # Sector deep cuts
    "XBI", "KRE", "SMH", "SOXX", "IBB", "XOP", "OIH", "XME", "KBE",
    "ITB", "XHB", "XAR", "IYT", "JETS", "MJ", "MOO", "TAN", "ICLN",
    # Smart beta / factor niches
    "SPHD", "RSP", "FNDB", "MGV", "MGK", "IJT", "IJJ", "IJK",
    # Dividend
    "VIG", "DGRO", "SCHD", "NOBL", "SDY",
    # Private credit / BDC
    "BIZD", "SRLN",
    # Preferred / convertibles
    "PFF", "CWB",
    # Real assets / infrastructure
    "IFRA", "PAVE", "GII",
)


def _unique(tickers: list[str]) -> list[str]:
    return sorted({t.upper().strip() for t in tickers if t and t.strip() and t != "-"})


# --- iShares fetchers --------------------------------------------------------


def _parse_ishares_csv(text: str) -> list[str]:
    """Return the Equity tickers from an iShares holdings CSV payload."""
    lines = text.splitlines()
    header_idx: int | None = None
    for i, ln in enumerate(lines[:30]):
        if ln.startswith("Ticker"):
            header_idx = i
            break
    if header_idx is None:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    out: list[str] = []
    for row in reader:
        t = (row.get("Ticker") or "").strip().upper().replace(".", "-")
        asset_class = (row.get("Asset Class") or "").strip()
        if not t or t == "-" or len(t) > 8:  # drop blanks, cash rows, iShares footer
            continue
        if asset_class and asset_class != "Equity":
            continue
        out.append(t)
    return out


def fetch_ishares(fund_id: str, timeout: float = 30.0) -> list[str]:
    url = _ISHARES_URLS[fund_id]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; inversiones-mama/0.1; research)"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] iShares {fund_id} fetch failed: {exc}", file=sys.stderr)
        return []
    return _parse_ishares_csv(r.text)


# --- Main driver -------------------------------------------------------------


def gather(
    start: datetime,
    end: datetime,
    batch_size: int,
    out_dir: Path,
    max_retries: int,
    pause_between_batches: float,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("WIDE UNIVERSE GATHER")
    print("=" * 78)

    # 1. Aggregate tickers from all sources
    print("\n[1/3] collecting ticker symbols from all free sources...")

    per_class: dict[str, list[str]] = {}

    print("  iShares IWV (Russell 3000)...", flush=True)
    per_class["russell3000"] = fetch_ishares("IWV")
    print(f"    -> {len(per_class['russell3000'])} tickers")

    print("  iShares IWB (Russell 1000)...", flush=True)
    per_class["russell1000"] = fetch_ishares("IWB")
    print(f"    -> {len(per_class['russell1000'])} tickers")

    print("  iShares IWM (Russell 2000)...", flush=True)
    per_class["russell2000"] = fetch_ishares("IWM")
    print(f"    -> {len(per_class['russell2000'])} tickers")

    print("  Wikipedia SP500 (redundancy)...", flush=True)
    try:
        per_class["sp500"] = list(fetch_sp500_tickers())
    except Exception as exc:
        print(f"    [warn] SP500 fetch failed: {exc}")
        per_class["sp500"] = []
    print(f"    -> {len(per_class['sp500'])} tickers")

    per_class["liquid_etfs"] = list(LIQUID_ETFS)
    print(f"  LIQUID_ETFS: {len(per_class['liquid_etfs'])} tickers")

    per_class["additional_etfs"] = list(ADDITIONAL_ETFS)
    print(f"  ADDITIONAL_ETFS: {len(per_class['additional_etfs'])} tickers")

    per_class["international_adrs"] = list(INTERNATIONAL_ADRS)
    print(f"  INTERNATIONAL_ADRS: {len(per_class['international_adrs'])} tickers")

    per_class["commodity_futures"] = list(COMMODITY_FUTURES)
    print(f"  COMMODITY_FUTURES: {len(per_class['commodity_futures'])} tickers")

    per_class["fx_pairs"] = list(FX_PAIRS)
    print(f"  FX_PAIRS: {len(per_class['fx_pairs'])} tickers")

    per_class["volatility"] = list(VOLATILITY_PRODUCTS)
    print(f"  VOLATILITY: {len(per_class['volatility'])} tickers")

    all_tickers: list[str] = _unique([
        t for ts in per_class.values() for t in ts
    ])
    # Also keep only equity-ticker-like symbols: drop pathological ones
    all_tickers = [t for t in all_tickers if len(t) <= 12]

    print(f"\n  TOTAL UNIQUE TICKERS: {len(all_tickers)}")
    (out_dir / "universe.txt").write_text(
        "\n".join(all_tickers) + "\n", encoding="utf-8",
    )
    (out_dir / "per_class.json").write_text(
        json.dumps({k: len(v) for k, v in per_class.items()}, indent=2),
        encoding="utf-8",
    )

    # 2. Bulk-download prices in batches with retries
    print(f"\n[2/3] yfinance bulk download {start.date()} -> {end.date()}")
    print(f"      {len(all_tickers)} tickers in batches of {batch_size}")

    all_frames: list[pd.DataFrame] = []
    failed_tickers: list[str] = []
    t0 = time.monotonic()
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(all_tickers) + batch_size - 1) // batch_size
        print(f"  batch {batch_num}/{total_batches}  "
              f"({len(batch)} tickers, cumulative elapsed {time.monotonic()-t0:.0f}s)",
              flush=True)
        attempt = 0
        df = pd.DataFrame()
        while attempt <= max_retries:
            try:
                df = fetch_prices_yfinance(batch, start=start, end=end, auto_adjust=True)
                break
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                print(f"    [retry {attempt}/{max_retries}] {exc}", file=sys.stderr)
                time.sleep(pause_between_batches * attempt)
        if df.empty:
            failed_tickers.extend(batch)
            continue
        missing_in_batch = [t for t in batch if t not in df.columns]
        if missing_in_batch:
            failed_tickers.extend(missing_in_batch)
        all_frames.append(df)
        time.sleep(pause_between_batches)

    if not all_frames:
        print("\n[error] all batches failed", file=sys.stderr)
        return 2

    # 3. Merge and report
    print(f"\n[3/3] merging {len(all_frames)} batch frames...")
    combined = pd.concat(all_frames, axis=1)
    # Deduplicate columns (in case of overlapping batches)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]
    combined = combined.sort_index()

    # Per-ticker coverage stats
    coverage_rows: list[dict] = []
    total_non_null = 0
    for col in combined.columns:
        s = combined[col].dropna()
        n = len(s)
        total_non_null += n
        coverage_rows.append({
            "ticker":       col,
            "first_date":   str(s.index[0].date()) if n else "",
            "last_date":    str(s.index[-1].date()) if n else "",
            "n_days":       n,
            "pct_coverage": round(100.0 * n / len(combined), 2) if len(combined) else 0.0,
        })
    coverage_df = pd.DataFrame(coverage_rows).sort_values("n_days", ascending=False)
    coverage_df.to_csv(out_dir / "coverage.csv", index=False)

    # Failed-tickers file
    (out_dir / "failed.txt").write_text(
        "\n".join(sorted(set(failed_tickers))) + "\n", encoding="utf-8",
    )

    # Summary
    summary = {
        "generated_at":        datetime.now().isoformat(timespec="seconds"),
        "requested_start":     str(start.date()),
        "requested_end":       str(end.date()),
        "n_tickers_requested": int(len(all_tickers)),
        "n_tickers_returned":  int(combined.shape[1]),
        "n_tickers_failed":    int(len(set(failed_tickers))),
        "n_trading_days":      int(len(combined)),
        "total_data_points":   int(total_non_null),
        "per_class_requested": {k: len(v) for k, v in per_class.items()},
        "earliest_date":       str(combined.index.min().date()) if len(combined) else None,
        "latest_date":         str(combined.index.max().date()) if len(combined) else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save the combined price dataframe as a parquet for re-use
    combined.to_parquet(out_dir / "prices.parquet")

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Requested: {summary['n_tickers_requested']} tickers")
    print(f"  Returned:  {summary['n_tickers_returned']} tickers")
    print(f"  Failed:    {summary['n_tickers_failed']} tickers")
    print(f"  Date range: {summary['earliest_date']} -> {summary['latest_date']}  "
          f"({summary['n_trading_days']} rows)")
    print(f"  TOTAL DATA POINTS (non-null): {summary['total_data_points']:,}")
    print()
    print(f"  Artifacts in {out_dir}:")
    for p in sorted(out_dir.iterdir()):
        print(f"    {p.name}  ({p.stat().st_size // 1024} KB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, default="1990-01-01")
    parser.add_argument("--end", type=str,
                        default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--out-dir", type=str, default="results/wide_gather")
    args = parser.parse_args(argv)

    return gather(
        start=datetime.fromisoformat(args.start),
        end=datetime.fromisoformat(args.end),
        batch_size=int(args.batch_size),
        out_dir=Path(args.out_dir),
        max_retries=int(args.max_retries),
        pause_between_batches=float(args.pause_seconds),
    )


if __name__ == "__main__":
    sys.exit(main())
