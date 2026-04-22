# Testing Roadmap — toward live-market-applicable validation

> Mandate received 2026-04-22 from Jorge: "To guarantee that a 60% historical
> CAGR does not disintegrate in live markets, our testing must move beyond
> historical walk-forward methods."
>
> This doc scopes the five advanced validation frameworks Jorge called
> out, classifies each by effort + budget + project fit, and sets the
> order I plan to implement them once the Alpaca paper ensemble
> stabilizes.

**Current validation arsenal** (already shipped):

- Walk-forward with IS/OOS chronological split (`backtest/engine.py`).
- Block-bootstrap Monte Carlo preserving autocorrelation (`simulation/monte_carlo.py`, `simulation/bootstrap.py`).
- Permutation-based trade-sequence MC (`simulation/trade_sequence_mc.py`).
- Deflated Sharpe Ratio gate (Bailey & López de Prado, 2014) in `simulation/metrics.py`.
- IBKR Tiered cost model + Almgren-Chriss base slippage in `backtest/costs.py`.
- Auto-halt circuit breaker off MC p95 drawdown (`execution/circuit_breaker.py`).
- Ledoit-Wolf shrinkage + PSD clip for covariance at 500+-asset scale (`models/covariance.py`).

The five Jorge-proposed frameworks below **extend** these, they don't replace them.

---

## 1. Dynamic Market-Impact Modeling (LOB-aware slippage)

**Status of existing infra:** Agent 2's `estimate_slippage()` in
`backtest/costs.py` already has the right *shape*:

```python
cost_bps = base_bps + impact_coeff * sqrt(shares / ADV) * 10000
```

— Almgren-Chriss square-root law. It just isn't receiving real ADV
data today; the orchestrator passes `adv=None` so only `base_bps`
(5 bps) applies.

**What this adds:**
- Pull volume data from yfinance (free) and compute 30-day ADV per ticker.
- Pipe `adv={ticker: adv}` into `portfolio_rebalance_cost()`.
- Add a **non-linear "walk the book" override** when a trade exceeds
  a configurable fraction of ADV (e.g., 1%):
  ```
  if shares > 0.01 * ADV:
      extra_bps = (shares / (0.01 * ADV) - 1) ** 2 * 10
  ```
  This is a crude LOB proxy that captures "the more we gulp, the worse
  the fill gets" without needing real Level-II data.

**Effort:** ~half-day. Volume loader + wiring + tests.
**Budget:** $0 (yfinance has daily volume).
**Blocker:** none.
**Expected impact on results:** reduces SP500 strategy's ~60% CAGR
toward something more plausible; larger rebalance orders (e.g. small-
caps) eat more of their own returns.

**Priority: HIGH** — cheapest and most directly applicable to our
concentration concern.

---

## 2. GAN-generated Out-of-Distribution scenarios (TimeGAN / WGAN-GP)

**Goal:** generate thousands of synthetic "causally plausible" 5-year
market paths the strategy has never seen, including regimes that
don't appear in our 2021-2026 training window.

**Existing alternative we have today:** block bootstrap in
`simulation/bootstrap.py`. Preserves autocorrelation, but can only
resample from observed history — no 1970s stagflation, no 2008
credit freeze.

**What this adds:**
- Train a WGAN-GP (or TimeGAN) on FF5F factor returns to learn the
  joint distribution of market dynamics.
- Condition the generator on macro regimes (low-vol bull, high-vol
  bear, panic).
- Generate N=10,000 synthetic 5-year paths.
- Feed into the existing `run_mc_rck_validation()` with
  `bootstrap_method="synthetic"` (new option).

**Effort:** 3-5 days. GAN training is finicky — needs PyTorch, careful
regularization, hyperparameter sweeps.
**Budget:** $0 for PyTorch. Would ideally use GPU but CPU training
works at reduced sample complexity.
**Blocker:** meaningful training requires more historical factor data
than the 2021-2026 window. Ken French's library goes back to 1963 —
62 years × 252 days = 15,624 daily samples is plausible for a small
TimeGAN (~50 params), not for a 4-layer Transformer.
**Expected impact on results:** MC tail risk estimates become
meaningfully wider (the bootstrap is optimistic because its support
is bounded by observed history).

**Priority: MEDIUM** — high value but significant implementation risk.
Deferred until #1 and #3 are done.

---

## 3. CCAR / DFAST macroeconomic stress scenarios

**Goal:** test portfolio survival under specific severe-adverse macro
paths the Fed publishes annually for bank-holding-company stress tests.

**What this is concretely:** the 2024 CCAR Severely Adverse scenario
specifies 28 macro variables over 9 quarters — unemployment to 10%,
GDP -7.5%, S&P 500 -55%, VIX to 75, yield curve shifts, commercial
real estate -40%, BBB spread to 6.5%, etc. It's a spreadsheet the Fed
publishes; no vendor cost.

**What this adds:**
- `simulation/macro_stress.py` module that loads the CCAR scenario,
  maps its 28 variables onto our 6-factor model (via pre-computed
  regression: market-beta responds to S&P 500, SMB to unemployment,
  HML to yield-curve, RMW to BBB spread, etc.).
- Generates a deterministic 9-quarter-long scenario path for each
  factor, then for each asset in the universe.
- Runs the walk-forward engine with real prices replaced by
  CCAR-driven synthetic prices; reports terminal wealth and max DD.

**Effort:** 2-3 days. Mapping the 28 variables to our 6 factors is the
time sink; the simulation mechanics are standard.
**Budget:** $0. The CCAR scenario is public (Fed Board publishes PDF
+ xlsx annually).
**Blocker:** none.
**Expected impact:** a strategy that passes the MC gates but fails
CCAR Severely Adverse is not safe for live $5k deployment. A strategy
that survives both is.

**Priority: MEDIUM-HIGH** — second-most-valuable validation. Good
companion to #1.

---

## 4. SIP vs. direct-feed latency adjustment (Lee-Ready / RBBO)

**Applicability to our system:** **NONE for v1a/v2 as currently
scoped.**

Lee-Ready, TAQ, RBBO, and the 86%→92% accuracy jump that Jorge cited
only matter when:
- The strategy is intraday / HFT.
- Tick-level trade classification matters (e.g., for signed order flow
  models, effective-spread estimation, or microstructure alpha).

Our strategy:
- Rebalances monthly.
- Uses daily bars.
- Does not care which side initiated a trade.

**Recommendation: skip this entirely unless we pivot to intraday.**
Polygon/tick data is expensive and we wouldn't extract any signal
from it at daily cadence.

**Priority: N/A (out of scope).**

---

## 5. Multi-Agent Reinforcement Learning environments (ABIDES-Gym)

**Goal:** place our strategy in a simulated exchange populated by
heterogeneous agents (fundamental, momentum, noise, market-maker) to
see whether our rebalancing signal decays when other adaptive agents
are actively trying to front-run it.

**What this adds vs. what we have:**
- Block bootstrap assumes our trades don't move the market.
- Real walk-forward assumes our orders fill at historical closes.
- An ABIDES simulation would simulate *endogenous* price formation —
  our order hitting the LOB moves the book, which other agents react
  to, which feeds back into our next fill.

**Effort:** 2+ weeks. ABIDES-Gym is research-grade software, requires
bespoke configuration of the agent mix, and produces simulation logs
that need a non-trivial parser to integrate with our engine.
**Budget:** $0 for ABIDES itself. Compute: each simulated trading day
costs ~minutes of wall time, so a 5y backtest = days of sim time.
**Blocker:** compute + complexity. Worth doing once for a "does the
strategy survive adversarial agents?" sanity check, not for every
rebalance.
**Expected impact:** would reveal whether our alpha decays in adverse
selection. If it does, that's a blocker for live capital at scale.

**Priority: LOW for v2, HIGH once we're considering >$25k live
capital deployment.** At $5k account size, the impact of our rebalance
on market microstructure is literally negligible, so ABIDES buys us
little.

---

## Suggested implementation order

1. **Volume-based dynamic market impact (#1).** Cheapest, highest fit.
2. **CCAR Severely Adverse scenario (#3).** Regulatory-grade stress
   test that exposes macro-factor concentration.
3. **Phase 4 skew-t Kelly** (see `docs/PHASE4_SKEW_T.md`). Address the
   +5.3 kurtosis in SP500 returns that Kelly's Gaussian-wealth
   assumption mishandles.
4. **Synthetic scenarios via TimeGAN (#2).** Do this after #1–3 land.
5. **ABIDES-Gym (#5).** Deferred until live capital discussion.
6. **Lee-Ready / RBBO (#4).** Skipped indefinitely — not applicable.

## What NOT to build

- CCAR-DFAST compliance reporting (we're not a bank).
- Regulatory submission artifacts.
- Real-time tick reconstruction from SIP feeds.
- L3 order-book replay.

These are institutional-grade concerns that don't accrue to a $5k
mom-portfolio.
