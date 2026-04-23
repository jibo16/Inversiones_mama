# CASE — Continuous Autonomous Simulation Engine

> **Status**: mandate handed to the sibling agent on **2026-04-22** by Jorge.
> **Runtime**: local desktop 24/7. **Budget**: zero. **Execution**: RESEARCH ONLY.
>
> Companion docs:
> - [`ARCHITECTURE_V2.md`](ARCHITECTURE_V2.md) — zero-budget data-sourcing rules (still authoritative)
> - [`ARCHITECTURE_V3_LOCAL_SIMULATION.md`](ARCHITECTURE_V3_LOCAL_SIMULATION.md) — simulation-engine building blocks CASE composes

---

## Directive (verbatim)

> **CEASE REACTIVE EXECUTION.** The sibling agent is no longer to await manual
> user prompts to execute backtests, generate data, or optimize portfolios.
> It must encapsulate the full quantitative pipeline
> (Data Loader → Alpha Generator → Risk/Sizing Optimizer → CPCV Backtester)
> inside a **continuous asynchronous loop** (`while True` or an automated
> message broker queue like Celery/Redis).
>
> Objective: permanently search the combinatorial space of assets, factors,
> and regime states to surface statistically significant portfolios.
> Interrupt the user **only** when an anomaly passes the strict statistical
> gatekeepers below.

---

## Direct instructions to the sibling agent

**Implementation order:**

1. **Supervisor daemon first.** A thin `systemd`-style wrapper (or Windows
   equivalent — `nssm` / Task Scheduler keep-alive) that (re)launches the
   main loop on crash. Verified with a forced `kill -9` before anything
   else is built.
2. **Message broker.** Redis + Celery via `docker-compose up` on the local
   machine. No cloud. Queue names: `case.hypothesis`, `case.synth`,
   `case.backtest`, `case.verdict`.
3. **Persistence.** Append-only: every generated hypothesis, every
   CPCV path, every rejected strategy lands in local SQLite (trial
   metadata) + Parquet (bulk metrics). Schema enforced — see "Trial
   ledger" below. This ledger is what the DSR multiple-testing penalty
   reads from.
4. **Main `while True` loop.** Pulls tasks from the broker in order:
   hypothesis → synth paths → CPCV → gatekeeper → (log | alert).
5. **Alpha sub-routines** (local LLM BLM + Bayesian ARD) second.
6. **Synthetic market sub-routines** (TimeGAN + jump-diffusion + OU) third.
7. **CPCV + DSR gatekeeper** last, because it depends on 1–6.

**Do not:**
- Connect CASE to the live Alpaca order endpoint under any circumstance.
- Call paid LLM APIs (Anthropic, OpenAI, etc.).
- Pull paid market data (Refinitiv, FMP, EODHD, Polygon paid tiers).
- Expand the universe beyond US equities / Russell 3000 + free factor data.
- Share in-process state with `scripts/run_paper_rebalance.py` or the
  current v1a production pipeline. CASE is parallel infrastructure.

---

## Module 1 — Autonomous alpha hypothesis generation

CASE cannot wait for the user to define investment strategies. The
hypothesis generator continuously produces candidate View vectors for
Black-Litterman, with two complementary sources.

### 1a. LLM-Enhanced Black-Litterman (local models only)

Zero-budget rule overrides the mandate's default to OpenAI/Anthropic
APIs. Use **open-weight models running on the local machine**:
- **Llama-3.1-8B-Instruct** (primary — best instruction-following at 8B)
- **Qwen-2-7B-Instruct** (diverse architecture, independent second view)
- **Gemma-7B-Instruct** (smaller ensemble member)

Run all three via `llama.cpp` (CPU-friendly) or `ollama` (GPU if
available). Quantize to Q4_K_M at minimum to fit in consumer VRAM.

**Pipeline per hypothesis cycle:**

1. Assemble a prompt with trailing 252d data summary (returns, vol,
   correlations) for a rotating subset of the universe.
2. Query each of the three models ≥ 5 times with slight paraphrasing.
3. Parse structured output → populate:
   - **View vector $q$**: expected return deltas vs. equilibrium
   - **Picking matrix $P$**: which assets each view applies to
   - **Confidence matrix $\Omega$**: variance of the N model×query responses
     per view. High variance → low confidence → Black-Litterman shrinks
     that view back toward the market equilibrium prior $\pi$.
4. Solve for posterior expected returns $\mu_{BL}$ via the standard
   BL master equation.

**Critical**: LLMs are **not calibrated forecasters**. Treat their output
as *one alpha source among many*, not a primary driver. Every LLM-derived
portfolio must pass the exact same CPCV + DSR gauntlet as the 6-factor
OLS portfolios (Module 4). If the LLM persistently biases
pessimistic/optimistic, the $\Omega$ matrix will penalize those views
and $\mu_{BL}$ reverts toward $\pi$ automatically.

### 1b. Bayesian Logistic Regression with Automatic Relevance Determination (BLR-ARD)

Run BLR-ARD continuously in the background over the full factor +
technical + fundamental feature panel. ARD priors $\alpha_k$ on each
feature automatically drive irrelevant features' posterior precision to
infinity → effective zero weight. Survivors are the features with
current predictive power.

Output: a sparse posterior distribution over factor loadings. Feed the
posterior mean $\hat{\beta}$ as a secondary view into the BL step above
(stacked alongside the LLM views), with $\Omega$ entries derived from
the posterior variance of each coefficient.

Re-fit nightly. Features dropped for ≥ 30 consecutive nights are
archived out of the active feature set; revisited weekly via a small
sentinel fit to catch regime changes.

---

## Module 2 — Continuous synthetic market generation

Historical data is finite. To run thousands of backtests without
overfitting to the single realised path, CASE continuously synthesises
new market data.

### 2a. TimeGAN / VAE (local training)

Permanently train Generative Adversarial Networks (TimeGAN, Yoon 2019)
on the rolling asset universe.

**Local-hardware accommodations** (per zero-budget rule):
- Batch size: 64 (not 512 as in the paper)
- Epochs per training session: 500 (not 10,000); compensate with more
  frequent retraining cycles
- Sequence length: 30 days (not 90) to fit in VRAM
- Train one model per sector (IT, Industrials, Treasuries, ...) rather
  than a single global model — smaller and more useful
- Accept longer wall-clock training times as the cost of zero budget

Use the trained generator to emit synthetic return sequences that
preserve:
- Heavy-tailed distributions
- Volatility clustering
- Non-stationary dynamics of extreme market crises (the discriminator
  learns these implicitly from the 2008 + 2020 + 2022 windows)

### 2b. Jump-Diffusion + Ornstein-Uhlenbeck (closed-form)

Complement TimeGAN with parametric processes that are cheap to simulate:

**Equities — Merton Jump-Diffusion:**
$$
dS_t = \mu S_t\, dt + \sigma S_t\, dW_t + S_t\, J\, dN_t
$$

Calibrate $\lambda$ (jump intensity) and $J$ (jump-size distribution) on
crisis windows so simulated paths include violent localized shocks.

**Bonds — Ornstein-Uhlenbeck:**
$$
dX_t = \theta(\mu - X_t)\, dt + \sigma\, dW_t
$$

Preserves the mean-reverting nature of interest rates.

Both generators run on CPU, take milliseconds per path. Use for bulk
Monte Carlo; use TimeGAN sparingly for a smaller number of
higher-fidelity paths.

---

## Module 3 — Automated Combinatorial Purged Cross-Validation

Standard $k$-fold CV is fatal in finance — it assumes IID and leaks
future information. CASE must automate CPCV so thousands of strategies
can be validated safely, without human oversight per-trial.

- **Splits**: partition the combined (real + synthetic) dataset into
  $N$ sequential groups; select $k$ groups as test sets for every
  possible combination. Paths per run:
  $$\phi(N, k) = \frac{k}{N}\binom{N}{k}$$
- **Purging**: drop any training observation whose label horizon
  overlaps temporally with the test set.
- **Embargo**: discard an automated temporal dead-zone (default 10–20
  trading days) immediately following each test set to prevent serial
  correlation leakage into subsequent training blocks.

See `ARCHITECTURE_V3_LOCAL_SIMULATION.md` Module 2 for the mathematical
detail. CASE calls into the `PurgedKFold` class from V3 rather than
reimplementing.

---

## Module 4 — The Gatekeeper (DSR > 0.95)

Running thousands of simulations continuously **guarantees** finding
strategies that look phenomenal by pure chance (data snooping).
The gatekeeper blocks those.

### Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)

For every candidate that reaches this stage:
$$
\mathrm{DSR} = \Phi\!\left(\frac{(\hat{SR} - \mathbb{E}[\widehat{SR}_0])\sqrt{T - 1}}
{\sqrt{1 - \gamma_3 \hat{SR} + \frac{\gamma_4 - 1}{4}\hat{SR}^2}}\right)
$$

Where:
- $\hat{SR}$ = observed Sharpe on CPCV out-of-sample paths
- $\mathbb{E}[\widehat{SR}_0]$ = expected null Sharpe adjusted for the
  **total number of trials** the CASE ledger records (see §Trial ledger)
- $\gamma_3$, $\gamma_4$ = skew and kurtosis of the return distribution
- $T$ = number of return observations

**DSR is a probability** in [0, 1]. The gate is **DSR > 0.95** — i.e.
≥ 95% probability that the observed Sharpe reflects real edge after
accounting for selection bias and non-normality.

### Notification protocol (strict AND)

A candidate only triggers an alert when **all three** hold:

1. **Out-of-Sample DSR > 0.95** across the CPCV path distribution
   (5th-percentile of DSR across paths, not the mean).
2. **Risk-Constrained Kelly drawdown bound honored** — $P(\text{DD} \ge 0.5) \le 0.10$
   — strictly across **every** TimeGAN + jump-diffusion + OU synthetic
   stress path, not just the historical path.
3. **HRP clustering** indicates structural diversification across
   **≥ 4 distinct, uncorrelated asset branches** at the root of the
   dendrogram. (Branches defined by the HRP quasi-diagonalisation step;
   "uncorrelated" = inter-branch correlation $\le 0.3$ on the OOS window.)

All three. No two-out-of-three promotions.

### Trial ledger (multiple-testing bookkeeping)

Append-only record of every strategy CASE has evaluated. Schema:

```
trial_id (ULID), timestamp, hypothesis_source (llm_blm | ard | ff6f | ...),
universe_hash, factor_set_hash, tickers_used, n_synth_paths, n_cpcv_paths,
observed_sharpe, sharpe_ci_5_95, skew, kurt, dsr, rck_dd_p, hrp_branches,
gate_verdict (pass | fail_dsr | fail_rck | fail_hrp), rejection_reason
```

Stored in `results/case/ledger.sqlite` (metadata) + `results/case/paths/`
Parquet shards (bulk OOS return series per trial). Never mutated, never
pruned — the DSR penalty in future cycles reads the total count from
this ledger.

---

## Operational constraints

### Zero-budget rule

Strictly in force. No paid APIs, no cloud GPUs, no paid data. Local LLMs
only (§1a). Local TimeGAN training accepting longer wall-clock (§2a).
Redis + Celery on local Docker, not a managed broker.

### Local desktop 24/7

CASE runs as a persistent daemon on Jorge's local machine. The
supervisor script must:
- Auto-restart the main loop on any crash (kill -9 survival test
  mandatory before launch)
- Auto-restart Celery workers independently
- Cap memory (OOM-kill watchdog) so a leaked TimeGAN model can't brick
  the machine
- Cap CPU to leave headroom for Jorge's normal desktop usage
  (e.g. `cpulimit -l 70` or equivalent)
- Log structured events to `results/case/logs/YYYY-MM-DD.jsonl`

### Data scope — US universe only

- **Equities**: Russell 3000 constituents (or wider subsets of it),
  scraped from free public sources. Wikipedia Russell 3000 list +
  SEC EDGAR for delisting timestamps. Accept the survivorship bias
  (Wikipedia shows *current* constituents) as a known limitation until
  Phase 3 of V2 unlocks paid point-in-time data.
- **Factors**: Kenneth French library (already cached locally)
- **Macros**: FRED (already in `data/macro.py`, free)

### Research–production boundary (NON-NEGOTIABLE)

CASE does **not** trade live capital. Ever. When a candidate passes the
three gates, CASE:

1. Writes the candidate as a self-contained JSON spec to
   `results/case/candidates/YYYY-MM-DD_<trial_id>.json`
2. Fires a **desktop notification** via `plyer` / `win10toast` (Windows)
   / `notify-send` (Linux) summarising the candidate
3. Optionally updates a minimal Streamlit dashboard at
   `scripts/case_dashboard.py` that polls the candidates directory

Jorge (or this agent) then:
1. Manually reviews the candidate spec
2. Decides to approve / reject
3. If approved, **manually** copies parameters into
   `scripts/run_paper_rebalance.py` or `BacktestConfig` and runs the
   existing pipeline

**The Alpaca order endpoint remains driven exclusively by
`scripts/run_paper_rebalance.py`** — human-in-the-loop, monthly cadence.
CASE has no code path that reaches Alpaca.

---

## Acceptance gates before CASE goes online

In order:

1. **Supervisor survival** — process killed with `kill -9` or hard OS
   reboot recovers automatically within 60 seconds.
2. **No-trade audit** — static analysis (ripgrep / AST walker) proves no
   import of `execution.alpaca` anywhere in the CASE package, and no
   HTTP call to `*.alpaca.markets` from any CASE process.
3. **Trial ledger integrity** — kill the process mid-CPCV, verify the
   ledger is consistent on restart (no dangling rows, no lost trials).
4. **DSR reproducibility** — given a fixed seed, the same hypothesis
   produces identical DSR to within 1e-9 across runs.
5. **CPCV reproduces V3 gate #2** — v1a CPCV run produces ≥ 50 paths and
   a DSR distribution whose 5/50/95 percentiles are reported.
6. **CPU/memory bounds** — 72-hour soak test on the local machine
   without OOM, without crashing Jorge's normal desktop use.
7. **Alert flow** — at least one synthetic candidate passes the three
   gates (can be a rigged test) and a desktop notification fires
   end-to-end.

Until all seven pass, CASE stays in `DRY_RUN=1` mode where all gate
outputs route to logs only; no notifications, no candidate writes.

---

## What CASE does NOT do (explicit scope boundary)

- Does not trade any capital, paper or live.
- Does not call paid APIs.
- Does not expand the universe beyond US / Russell 3000 + free factors.
- Does not replace or modify the v1a monthly rebalance pipeline.
- Does not share state with `paper_trader.py`.
- Does not promote candidates to production autonomously — every
  approval is a human decision.
- Does not re-backtest candidates it has already evaluated within the
  last 7 days (dedup via hypothesis hash).

---

## Deliverable checklist for the sibling agent

- [ ] `docker-compose.yml` — Redis + Celery workers, local only
- [ ] `src/inversiones_mama/case/daemon.py` — main loop + supervisor hook
- [ ] `src/inversiones_mama/case/ledger.py` — SQLite + Parquet ledger
- [ ] `src/inversiones_mama/case/alpha/llm_blm.py` — local LLM BLM
- [ ] `src/inversiones_mama/case/alpha/bayesian_ard.py` — BLR-ARD
- [ ] `src/inversiones_mama/case/synth/timegan.py` — training + sampling
- [ ] `src/inversiones_mama/case/synth/jump_diffusion.py`
- [ ] `src/inversiones_mama/case/synth/ou.py`
- [ ] `src/inversiones_mama/case/cpcv.py` — wraps V3's PurgedKFold
- [ ] `src/inversiones_mama/case/gatekeeper.py` — DSR + RCK + HRP gates
- [ ] `src/inversiones_mama/case/notify.py` — desktop notification + candidate JSON writer
- [ ] `scripts/case_daemon.py` — entrypoint
- [ ] `scripts/case_dashboard.py` — optional Streamlit viewer
- [ ] `tests/case/` — unit + property tests; at minimum the seven
  acceptance gates above.

All inside a new `case/` package; no modifications to
`execution/paper_trader.py`, `scripts/run_paper_rebalance.py`, or any
production-path code.
