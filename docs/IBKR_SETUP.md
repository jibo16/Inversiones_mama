# IBKR Client Portal Gateway — Setup & 2FA Flow

> **Audience:** Jorge, running the paper-trading deployment locally.
> **Last updated:** 2026-04-22 for the `source="ibkr"` data pipeline.

This doc describes the end-to-end flow to get the local Client Portal
Gateway running, authenticated (2FA), and reachable from the project's
code. Once set up, the `scripts/backfill_ibkr_history.py` CLI and the
`data.prices.load_prices(..., source="ibkr")` entry point both work.

---

## 1. Prerequisites

- **IBKR paper account** with real-time US-equity data permissions
  (default for paper; no monthly fee).
- **Java 17+** installed and on PATH (the Gateway is a Java app).
- **IBKR Mobile app** or SMS-2FA configured on your IBKR profile.
- The project's `.env` file populated with the Client Portal settings
  (already present — see `.env.example`):
  ```
  IBKR_CP_BASE_URL=https://localhost:5000/v1/api
  IBKR_CP_WS_URL=wss://localhost:5000/v1/api/ws
  IBKR_CP_VERIFY_SSL=false
  IBKR_ACCOUNT=DU<your_paper_account_id>
  IBKR_MARKET_DATA_FIELDS=31,84,85,86,88,7059,6509
  ```

---

## 2. One-time Gateway install

Download the Client Portal API package from IBKR:

- <https://www.interactivebrokers.com/en/trading/ib-api.php#clientportal>
- Unzip to e.g. `C:\Users\jorge\clientportal.gw`.

Inside you'll find `bin\run.bat` (Windows) and `bin\run.sh` (Linux/macOS),
plus a `root\conf.yaml` configuration.

The default config binds to port 5000 with a self-signed cert. Our `.env`
already points at `https://localhost:5000` — no config changes needed.

---

## 3. Daily launch flow

**Every trading day (or whenever the session has expired):**

1. **Start the Gateway** in a dedicated terminal:
   ```
   cd C:\Users\jorge\clientportal.gw
   bin\run.bat root\conf.yaml
   ```
   Keep this terminal open — the Gateway must stay running for our code to work.

2. **Authenticate via browser.** Open:
   ```
   https://localhost:5000/sso/Login
   ```
   (Accept the self-signed cert warning — safe on localhost.)

   - Enter your paper username + password.
   - The Gateway triggers 2FA: **push notification to IBKR Mobile**, or
     SMS, or key-fob code depending on your setup.
   - Approve on your phone.
   - Browser shows "Client login succeeds."

3. **Session is live for ~6–24 hours.** IBKR's session-expiry policy is
   activity-dependent; expect to re-authenticate at least once per
   trading day.

---

## 4. Verify from the project

With the Gateway running and authenticated:

```
.venv\Scripts\python.exe -c "from inversiones_mama.data.ibkr_historical import IBKRHistoricalLoader; l = IBKRHistoricalLoader.from_env(); l.ensure_authenticated(); print('✓ IBKR session authenticated')"
```

If it prints the checkmark, you're good to go. Otherwise the error
message will tell you what's wrong (usually: Gateway not running, or
session needs re-auth).

---

## 5. Bulk backfill

With the Gateway authenticated:

```
.venv\Scripts\python.exe scripts\backfill_ibkr_history.py --kind all --period 10y
```

This downloads 10 years of daily bars for the full curated universe
(~230 tickers: SP100 + NASDAQ100 + LIQUID_ETFS). Expected runtime: ~5–10
minutes with the 0.5-second inter-request pause.

Use `--resume` to skip tickers already in the cache, and `--limit N` to
cap for a quick smoke test:

```
.venv\Scripts\python.exe scripts\backfill_ibkr_history.py --kind etfs --limit 5 --delay 0.25
```

Results land in the parquet cache under `data/cache/` and a summary CSV
at `results/ibkr_backfill_summary.csv`.

---

## 6. Switching backtests to IBKR data

After at least one successful backfill run, routing the strategy's
backtests through IBKR data is a one-keyword change:

```python
from inversiones_mama.data.prices import load_prices

prices = load_prices(tickers, start, end, source="ibkr")
```

The `source="ibkr"` path calls `IBKRHistoricalLoader.from_env()`,
re-verifies the session, pulls fresh bars at the requested period, trims
to the exact `[start, end]` window, and caches the result. If the
Gateway has died mid-run you'll see a clear `IBKRConnectionError` with
the "re-auth and retry" hint.

To default the whole project to IBKR data, swap `source="yfinance"` for
`source="ibkr"` in:

- `scripts/run_v1a_verdict.py`
- `scripts/run_paper_rebalance.py`
- `scripts/demo_alpha_pipeline.py`

(Currently these all use yfinance for zero-friction development; the
switch is left manual so yfinance stays available as a free fallback
when the Gateway isn't running.)

---

## 7. Troubleshooting

| Symptom                                                 | Fix                                                                            |
|---------------------------------------------------------|--------------------------------------------------------------------------------|
| `IBKRConnectionError: Gateway unreachable`              | Gateway not running. Relaunch `bin\run.bat`.                                   |
| `... session is not authenticated`                      | Visit `https://localhost:5000/sso/Login` again and complete 2FA.               |
| Browser login hangs                                     | Restart the Gateway (sometimes it needs a kick after idle >12h).               |
| `IBKRDataError: could not resolve contract id`          | Ticker is mis-spelled or not available in your market-data subscription.      |
| `HTTP 429` / rate-limit warnings                        | Increase `--delay` on backfill (try 1.0 or 2.0 seconds).                       |
| Prices look wrong (~100× off)                           | Check `priceFactor` handling — the loader divides by it automatically.         |
| SSL cert warnings drowning logs                         | Expected on localhost self-signed; `urllib3` warnings are suppressed in code. |

---

## 8. Security notes

- The Gateway binds to `localhost:5000` only — no LAN exposure.
- Your IBKR credentials never pass through this project; they live in
  the Gateway's browser session.
- `.env` in the repo is `.gitignore`'d; API keys never leak.
- The Gateway's self-signed cert is only trusted for localhost.
  If you see a cert warning on a URL that's NOT `localhost`,
  **stop and verify the URL** — something's wrong.
