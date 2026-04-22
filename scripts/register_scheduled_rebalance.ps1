# Register the monthly Inversiones_mama paper rebalance as a Windows Scheduled Task.
#
# Usage (admin PowerShell):
#     .\scripts\register_scheduled_rebalance.ps1
#     .\scripts\register_scheduled_rebalance.ps1 -Daily      # daily dry-run variant
#     .\scripts\register_scheduled_rebalance.ps1 -Remove     # unregister
#
# Why market-close local time (15:45 ET = 20:45 Madrid):
#   Alpaca paper only fills orders during regular market hours. Running
#   at 20:00 UTC (after US close) leaves orders "accepted" overnight and
#   fills them at next open — introducing an overnight-gap slippage we
#   don't want. 15:45 ET gives us 15 minutes of live-market execution
#   and clean fills. Schedule the task in local time that corresponds.

[CmdletBinding()]
param(
    [switch]$Daily,
    [switch]$Remove
)

$TaskName = "InversionesMama_Rebalance"
if ($Daily) { $TaskName = "InversionesMama_Rebalance_Daily" }

if ($Remove) {
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
        Write-Host "Unregistered '$TaskName'." -ForegroundColor Green
    } catch {
        Write-Warning "Could not unregister '$TaskName': $($_.Exception.Message)"
    }
    return
}

# --- paths -----------------------------------------------------------------
$Repo   = (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$EnsembleScript = Join-Path $Repo "scripts\run_ensemble_rebalance.py"
$DryScript      = Join-Path $Repo "scripts\run_paper_rebalance.py"
$LogDir = Join-Path $Repo "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# --- sanity ---------------------------------------------------------------
if (-not (Test-Path $Python))         { throw "venv not found: $Python" }
if (-not (Test-Path $EnsembleScript)) { throw "script not found: $EnsembleScript" }

# --- action ---------------------------------------------------------------
if ($Daily) {
    # Daily DRY-run at 16:30 ET (=21:30 Madrid/CEST, =22:30 Madrid/CET)
    # Adjust -At to whatever wall-clock suits your locale.
    $Argument = "$DryScript --broker dry --max-capital 5000 " +
                "--log `"$Repo\results\daily_dry\run_`$(Get-Date -Format yyyy-MM-dd).json`""
    $Trigger = New-ScheduledTaskTrigger -Daily -At (Get-Date "21:30")
    $Description = "Daily dry-run snapshot of the v1a strategy for dashboard time-series."
} else {
    # Monthly ensemble, last business day (approx: weekly friday, interval 4)
    $Argument = "$EnsembleScript --max-capital 5000 " +
                "2>&1 | Tee-Object -FilePath `"$LogDir\ensemble_rebalance.log`" -Append"
    $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At (Get-Date "20:45") -WeeksInterval 4
    $Description = "Monthly paper-ensemble rebalance across v1a / sp100 / sp500 / sp500+etfs."
}

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"& '$Python' $Argument`"" `
    -WorkingDirectory $Repo

$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew

# --- register -------------------------------------------------------------
try {
    # If a task with this name exists, update it
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Write-Host "Task '$TaskName' already exists; overwriting." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Trigger $Trigger -Action $Action `
        -Principal $Principal -Settings $Settings `
        -Description $Description
    Write-Host "Registered '$TaskName'." -ForegroundColor Green
    Get-ScheduledTaskInfo -TaskName $TaskName | Format-List LastRunTime, NextRunTime, LastTaskResult
} catch {
    Write-Error "Failed to register scheduled task: $($_.Exception.Message)"
    throw
}
