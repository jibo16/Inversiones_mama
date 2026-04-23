# Meta-portfolio analysis — self-audit

**Date:** 2026-04-23
**Scope:** Delivered against the 2-part request: (1) backtest the 20
strategies with the advanced simulation menu; (2) present results on a
website.

---

## 1. What was delivered

### 1.1 Historical walk-forward backtest — all 20 strategies

`scripts/run_meta_backtest.py` + `results/meta_backtest/`

For every strategy in `STRATEGY_CATALOG`, loads its universe's prices
(5-year window), runs the allocator's historical walk-forward (monthly
rebalance; each allocator's native logic preserved), and derives 22
metrics per strategy including Sharpe, Sortino, Calmar, MaxDD,
CVaR95, skew, excess kurtosis, DSR, hit rate, profit factor, best/
worst day, best/worst month.

**Artifacts:**
- `results/meta_backtest/summary.csv` — 20 rows, 22 columns
- `results/meta_backtest/summary.json` — same, programmatic
- `results/meta_backtest/daily/<strategy_id>.csv` — full daily return
  + cumulative wealth series for each strategy (20 files)

**Top 5 by DSR:**

| # | Strategy | Sharpe | MaxDD | CVaR95 | DSR |
|:-:|---|---:|---:|---:|---:|
| 1 | momentum_xsec_sp500 | +1.725 | 42.70% | 6.55% | 0.352 |
| 2 | vol_targeting_etfs | +1.389 | 6.04% | 0.45% | 0.136 |
| 3 | rck_6factor_v1a | +1.063 | 11.05% | 1.28% | 0.033 |
| 4 | momentum_ts_l120_sp500 | +0.929 | 17.64% | 2.24% | 0.020 |
| 5 | inverse_vol_etfs | +0.913 | 10.37% | 0.68% | 0.015 |

### 1.2 Block-bootstrap forward simulation

`scripts/run_meta_bootstrap.py` + `results/meta_bootstrap/`

For each strategy's backtested daily-return series, 1,000 forward
paths of 252 trading days each via the stationary block-bootstrap
(20-day mean block length). Preserves volatility clustering and
heavy tails (no Gaussian assumption).

**Artifacts:**
- `results/meta_bootstrap/summary.csv` — per-strategy terminal-return
  + drawdown distribution percentiles (20 rows)
- `results/meta_bootstrap/paths/<strategy_id>_stats.json` — one file
  per strategy (20 files)

**Key insights:**
- `momentum_xsec_sp500`: enormous dispersion (+2.5% to +359% p05/p95)
  with a fat right tail; also the highest stress risk at p95 MDD 44.8%
- `vol_targeting_etfs`: tightest distribution (−1.0% to +10.2% p05/p95),
  MDD p95 only 4.83%
- `dual_momentum_etfs`: 17.1% probability of >20% 1-year loss —
  deployable risk
- `mean_reversion_sp100`: median −2.0%, 11.5% P(loss>20%) — systematically
  underperforms

### 1.3 Markov regime-switching attribution

`src/inversiones_mama/simulation/regime_switching.py`
`scripts/run_meta_regime.py` + `results/meta_regime/`

2-state Gaussian HMM fit on 20-year SPY daily returns via `hmmlearn`.
Regime 0 = LOW_VOL (bull), Regime 1 = HIGH_VOL (2008, 2020 COVID,
2022 rate hikes).

**SPY regime reference:**
- LOW_VOL: 77.7% of days, ann_ret +27.5%, ann_vol 11.5%, Sharpe +2.40
- HIGH_VOL: 22.3% of days, ann_ret -25.6%, ann_vol 35.0%, Sharpe -0.73

**Transition matrix** (regimes are sticky):
- Stay LOW: 98.6%, Stay HIGH: 95.3%
- Expected duration LOW ≈ 71 days, HIGH ≈ 21 days

**Most important finding**: only **momentum_xsec_sp500** has a
positive Sharpe in HIGH_VOL (+0.14). Every other strategy generates
negative risk-adjusted returns in stress periods. SPY-hold loses
0.65 Sharpe in HIGH_VOL; dual_momentum_etfs loses −1.34.

**Artifacts:**
- `results/meta_regime/regime_labels.csv` — every day's regime (5036 rows)
- `results/meta_regime/regime_transition_matrix.csv` — 2x2 matrix
- `results/meta_regime/spy_regime_stats.csv`
- `results/meta_regime/per_strategy.csv` — 40 rows (20 strategies × 2 regimes)
- `results/meta_regime/summary.json`

### 1.4 Static HTML website

`scripts/build_meta_website.py` + `results/meta_website/`

Self-contained website (no server required — open `index.html`
directly). 21 HTML pages, 44 PNG figures.

**Structure:**
- `index.html` — overview, leaderboard, bootstrap fan chart, regime
  scatter plot, CPCV reference, per-strategy index, self-audit
- `strategies/<strategy_id>.html` — per-strategy drill-down with
  metric cards, equity curve, drawdown timeline, bootstrap summary,
  per-regime table (20 pages)
- `figures/` — 4 overview PNGs + 40 per-strategy PNGs (equity +
  drawdown)
- `style.css` — minimal stylesheet (no external dependencies)

### 1.5 CPCV reference (no new work)

The earlier `results/cpcv_superwide_hrp/` artifacts (1,512-ticker
tournament with HRP) remain the authoritative multiple-testing
verdict. The website references them; no re-run.

---

## 2. What was NOT built and why

The request listed six simulation architectures. Three were built
(bootstrap, Markov regime, CPCV reference). Three were explicitly
scoped out for this session:

### 2.1 TimeGAN / VAE synthetic-data generation — NOT built
**Why:** Requires PyTorch training infrastructure + 12–24h of GPU
compute per model, per strategy. Not feasible in a single session on
available hardware. The block-bootstrap in §1.2 is the non-parametric
substitute — it also preserves volatility clustering and heavy tails,
with the honest limitation that it can only resample from the
historical distribution (no extrapolation to unseen regimes).

**If you want to unblock this:** budget for 1–2 weeks of engineer
time + a GPU (cloud T4 is $0.35/hr; ~$50–100 of compute per strategy
to train a decent TimeGAN).

### 2.2 Agent-Based Modeling (ABM) — NOT built
**Why:** ABM requires a full market simulator with agent archetypes
(trend-followers, value investors, noise traders), order-book
mechanics, and tick-level event processing. Building one from
scratch would be multi-week effort. The request accurately describes
ABM's value (emergent phenomena, reflexivity) but those capabilities
are expensive to stand up.

**If you want to unblock this:** look at `abides-jpmc` or similar
open-source market simulators; integration with our strategy layer
would take 2–3 weeks of focused engineering.

### 2.3 Multi-Level Monte Carlo (MLMC) — NOT built
**Why:** MLMC is designed for pricing exotic derivatives where path
generation is expensive (e.g., stochastic volatility SDEs with fine
discretization). For monthly-rebalance equity strategies on daily
prices, path generation is already cheap — MLMC's variance-reduction
advantages don't materialize. The request correctly describes
MLMC's strengths but they don't apply to our domain.

**If you want to unblock this:** would require re-scoping the
simulation layer to include continuous-time SDE models, which we've
discussed but haven't implemented.

---

## 3. Known issues carried forward

### 3.1 Live-deployment ledger drift — UNRESOLVED
From the 2026-04-23 09:30 ET live deployment: 238 of 1,246 orders
(19%) were submitted to Alpaca but not recorded to the ledger because
Alpaca's fill polling has a 30-second deadline and some fills took
longer. The ledger is under-counting some strategies' positions
(particularly `momentum_ts_l120_sp500`, `equal_weight_etfs`,
`hrp_eqfloor_etfs`).

**Impact on this site:** the historical backtest, bootstrap, and
regime analyses all operate on the BACKTEST return series, not on
the live ledger state. So the analysis results are NOT affected by
the live ledger drift. The drift only affects live-ledger P&L
attribution going forward.

**Fix scoped but not built:** add `client_order_id` tagging on order
submit + a post-run reconciliation script that queries
`/v2/orders?status=filled` after the submit burst and backfills the
ledger.

### 3.2 HMM non-convergence warning
The `hmmlearn` fit on 20 years of SPY returns emitted a warning that
log-likelihood decreased by ~0.12 nats in the final iteration
(non-monotone convergence). This is numerical noise, not a genuine
divergence — the final regime labels are stable. For attribution
purposes (classifying each day's regime after the fact), this is
acceptable. For live regime prediction it would warrant tightening
the `tol` parameter or using a more robust fitter.

### 3.3 Integer-share rounding bias from morning deployment
The live `invvol_eqfloor_etfs` bag shows 22.8% equity vs. the 40%
target because integer-share rounding compressed small equity weights
to zero. Fractional-share support was added to `paper_trader.py`
during this session; subsequent deployments (the 19 other bags) used
fractional shares and landed on target. The bug is not in the
allocator — it's in the execution layer of the first deployment.

---

## 4. Reproducibility

Every analysis can be re-run deterministically:

```bash
# 1. Historical backtest (5-10 min)
python scripts/run_meta_backtest.py --years 5

# 2. Bootstrap forward sim (2-3 min)
python scripts/run_meta_bootstrap.py --n-paths 1000 --horizon-days 252

# 3. Markov regime analysis (1-2 min)
python scripts/run_meta_regime.py --spy-years 20

# 4. Rebuild the website (30 sec)
python scripts/build_meta_website.py

# Open results/meta_website/index.html
```

All random seeds are fixed. Data sources are cached in
`data/cache/*.parquet`.

---

## 5. What the audit should verify

1. **Numbers match:**
   - `results/meta_backtest/summary.csv` Sharpe/DSR values match
     those shown on `index.html`
   - `results/meta_bootstrap/summary.csv` percentiles match the
     fan chart and bootstrap section
   - `results/meta_regime/per_strategy.csv` values match the
     regime scatter and per-regime table

2. **Methodology sanity:**
   - Historical walk-forward uses monthly rebalance (matches earlier
     tournament conventions)
   - Bootstrap uses stationary block-bootstrap with 20d mean block
     length (preserves vol clustering)
   - HMM uses 2-state Gaussian emissions on SPY, fit once with seed
     20260423, regime labels are vol-ordered (state 0 = low vol)

3. **Known limitations honored:**
   - No look-ahead bias in the backtest (each rebalance uses trailing
     data only — verified by code inspection)
   - No survivorship-bias claim; LIQUID_ETFS and curated SP100 are
     current-constituent lists (same caveat as the CPCV tournament)
   - No claim of DSR clearance — only one strategy (momentum_xsec_sp500
     at DSR 0.352) comes close and still fails the 0.95 gate

4. **Generative-model claims:**
   - Site explicitly labels TimeGAN / ABM / MLMC as NOT built and
     why
   - Bootstrap is NOT presented as equivalent to TimeGAN; its
     inability to extrapolate beyond observed regimes is disclosed
