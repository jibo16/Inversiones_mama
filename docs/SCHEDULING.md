# Scheduling the paper rebalance

The paper orchestrator is **stateless across runs** — cadence is the caller's
responsibility. This doc shows how to register the rebalance on Windows
(Task Scheduler) and Unix (cron) so the ensemble runs monthly without
human intervention.

> The safety triad (PDT tracker + circuit breaker) runs inside the
> orchestrator on every call. A scheduled run is equivalent to a manual
> run in terms of guardrails.

---

## Windows — Task Scheduler (recommended for Jorge's laptop)

### Option A: monthly, last business day, 20:00 local

Save this as `scripts/register_scheduled_rebalance.ps1` and run it from an
**admin PowerShell**:

```powershell
# register_scheduled_rebalance.ps1
$Repo = "C:\Users\jorge\OneDrive\Desktop\Inversiones_mama"
$Python = "$Repo\.venv\Scripts\python.exe"
$Script = "$Repo\scripts\run_ensemble_rebalance.py"

# Trigger: last business day of each month at 20:00 local (after market close)
$Trigger = New-ScheduledTaskTrigger `
    -Weekly -DaysOfWeek Friday -At 20:00 `
    -WeeksInterval 4    # approx monthly; adjust to taste

$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "$Script --max-capital 5000" `
    -WorkingDirectory $Repo

$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask `
    -TaskName "InversionesMama_Rebalance" `
    -Trigger $Trigger -Action $Action `
    -Principal $Principal -Settings $Settings `
    -Description "Monthly ensemble rebalance across v1a / sp100 / sp500 / sp500+etfs"
```

Verify with:

```powershell
Get-ScheduledTask -TaskName InversionesMama_Rebalance | Format-List *
Get-ScheduledTaskInfo -TaskName InversionesMama_Rebalance
```

Remove later with:

```powershell
Unregister-ScheduledTask -TaskName InversionesMama_Rebalance -Confirm:$false
```

### Option B: daily observation (dry-run only, no live orders)

If you want the dashboard populated with daily "what would the strategy
do" snapshots without actually submitting orders, schedule the single-
strategy CLI with `--broker dry`:

```powershell
$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "$Script --broker dry --max-capital 5000 --log results/daily_dry/run_$(Get-Date -Format yyyy-MM-dd).json" `
    -WorkingDirectory $Repo

$Trigger = New-ScheduledTaskTrigger -Daily -At 17:00
# ... rest as Option A
```

---

## Unix (Linux / macOS) — cron

Edit your crontab (`crontab -e`) and add **one** of:

```cron
# Monthly ensemble on the 1st at 20:00
0 20 1 * * cd ~/Inversiones_mama && ./.venv/bin/python scripts/run_ensemble_rebalance.py --max-capital 5000 >> logs/rebalance.log 2>&1

# Daily dry-run snapshot at 17:00 (weekdays)
0 17 * * 1-5 cd ~/Inversiones_mama && ./.venv/bin/python scripts/run_paper_rebalance.py --broker dry --max-capital 5000 --log results/daily_dry/run_$(date +\%Y-\%m-\%d).json >> logs/rebalance.log 2>&1
```

---

## IBKR gotcha (when / if you promote from Alpaca paper to IBKR live)

Scheduled runs will fail the moment your Gateway session expires (every
6–24 h). Two mitigations:

1. **Renewal ping**: add a pre-rebalance step that POSTs `/iserver/auth/status`
   and exits with an error if not authenticated. Your scheduler logs will
   make this loud immediately.
2. **Auto-logout protection**: IBKR's Gateway can be configured to
   auto-logout after idle. Set `conf.yaml`'s `listenPort` and disable
   `autoLogout` if you want the session to live until you kill it.

---

## Dashboard integration

The Streamlit dashboard's "Data status" sidebar already shows whether
`paper_trades.json` and the ensemble's `aggregate_summary.json` exist.
When the scheduled task writes new files, the dashboard picks them up on
the next page refresh (or click "Clear cache & refresh all" in the
sidebar). For a live watchwall, run the dashboard with
`streamlit run --server.runOnSave=true` so it hot-reloads.

---

## What NOT to automate (yet)

* **Promoting from Alpaca paper to IBKR live.** Live trading involves real
  money. The circuit breaker + PDT tracker catch many problems, but live
  orders should ride a manual "approve and run" for the first 2–4 weeks
  at minimum (see `docs/ARCHITECTURE_V2.md` §9).
* **Key rotation.** Manual one-off per portal dashboard.
* **Universe changes.** Adding/removing tickers changes strategy behavior —
  these are versioned config changes, not scheduled tasks.
