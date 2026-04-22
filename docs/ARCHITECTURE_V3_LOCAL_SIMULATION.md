# Inversiones_mama v3 — Large-Scale Local Simulation Engine

> **Status**: mandate handed off to the sibling agent on **2026-04-22** by Jorge.
> This supersedes no earlier doc. It specifies the backtesting/simulation
> engine that will run locally across thousands of multi-asset instruments.
> For the production-deployment phasing and zero-budget data constraints,
> see [`ARCHITECTURE_V2.md`](ARCHITECTURE_V2.md) (still authoritative on data
> sourcing, phases, ownership).

---

## Direct instruction to the sibling agent

**Start here**, in this order:

1. Scaffold the **Event-Driven Engine** types: `MarketEvent`, `SignalEvent`,
   `OrderEvent`, `FillEvent`. The event loop dequeues chronologically
   (`collections.deque` or `heapq`) — no `DataFrame.shift()` anywhere in
   the execution layer.
2. Scaffold the **Data Integrity Loader** with strict point-in-time
   constituency. Delisted / bankrupt tickers must be present on the dates
   they were actually in the index.
3. Implement the **Square-Root Market Impact** fill simulator:
   $I(Q) = \sigma \cdot Y \cdot \sqrt{Q/V}$ — this is what punishes
   over-allocation to illiquid assets.
4. Once the event-driven + point-in-time infra passes unit tests,
   implement the `PurgedKFold` class and then CPCV on top of it.
5. Last: HRP, jump-diffusion / OU synthetic generators, MLMC, DSR
   deflation.

**Do not use vectorized Pandas for the execution layer.** Vectorized
backtests create "at-close bias" by assuming you can trade at the same
price the signal is emitted; the event loop prevents this structurally.

---

## Why standard techniques fail at scale

When scaling to thousands of multi-asset instruments, standard statistical
techniques fail catastrophically. Financial data is non-stationary, highly
autocorrelated, and suffers from the curse of dimensionality. The engine
must implement a pipeline consisting of **five robust layers**:

1. Data Integrity & Asset-Specific Modeling
2. Combinatorial Purged Cross-Validation (CPCV)
3. Event-Driven Execution & Market Microstructure Friction
4. Large-Scale Portfolio Optimization (HRP)
5. Advanced Monte Carlo & Statistical Verification

Together they defend against the "Seven Deadly Sins of Quantitative
Investing": survivorship bias, look-ahead bias, overfitting, data mining,
ignoring transaction costs, ignoring capacity, and ignoring regime shifts.

---

## Module 1 — Data Integrity and Asset-Specific Modeling

When downloading and storing thousands of assets locally, the agent must
treat different asset classes with mathematically appropriate stochastic
processes.

### Point-in-time data

The agent must construct the database using **point-in-time constituents**
to eliminate survivorship bias, which can artificially inflate simulated
returns by up to **2.1% annually**. Delisted and bankrupt companies must
be present in the historical tests on the dates they were actually in
the investable universe.

### Asset-specific Stochastic Differential Equations (SDEs)

For generating synthetic data and simulations, the agent cannot use
standard Geometric Brownian Motion for everything.

**Equities & ETFs — Jump-Diffusion Process:**

$$
dS_t = \mu S_t\, dt + \sigma S_t\, dW_t + S_t\, J\, dN_t
$$

Captures the heavy-tailed kurtosis and violent price gaps characteristic
of extreme market events (earnings, M&A, macro shocks). $J$ is the jump
magnitude distribution, $N_t$ is a Poisson process with intensity
$\lambda$ for the jump arrival rate.

**Bonds — Ornstein-Uhlenbeck (OU) Process:**

$$
dX_t = \theta(\mu - X_t)\, dt + \sigma\, dW_t
$$

Correctly simulates the mean-reverting nature of interest rates. $\theta$
is the speed of reversion toward the long-run mean $\mu$.

---

## Module 2 — Combinatorial Purged Cross-Validation (CPCV)

Standard $k$-fold cross-validation is **fatal in finance** because it
assumes data is Independent and Identically Distributed (IID) and leaks
future information into the training set. The agent must implement
Combinatorial Purged Cross-Validation.

### Purging

The algorithm must explicitly drop training observations whose **labels
overlap temporally** with the test set. If a training label uses 20-day
forward returns and a test fold starts at day $t$, any training
observation with a label window that crosses $t$ must be purged.

### Embargoing

To combat serial correlation, enforce an **embargo**: discard a fixed
temporal buffer (e.g. 10–20 trading days) of training data immediately
following any test set.

### Path generation

CPCV partitions the dataset into $N$ sequential groups and selects $k$
groups as test sets for every possible combination. This generates a
distribution of unique backtest paths mathematically defined as:

$$
\phi(N, k) = \frac{k}{N} \binom{N}{k}
$$

This allows the portfolio to be tested across **hundreds of out-of-sample
scenarios**, generating a distribution of Sharpe ratios rather than a
single point estimate. The mean and dispersion of that distribution are
what feed the Deflated Sharpe Ratio in Module 5.

---

## Module 3 — Event-Driven Execution & Market Microstructure Friction

Vectorized backtesting (e.g. Pandas `shift` tricks) is fast but creates
"at-close bias" by assuming execution happens at the exact price the
signal is generated, ignoring intraday liquidity and latency.

### Event-loop architecture

Build an event-driven engine utilizing a chronological queue
(`collections.deque` or `heapq`). The loop sequentially processes:

- `MarketEvent` — a price / quote / bar arrives
- `SignalEvent` — the strategy emits a trade intent
- `OrderEvent` — the router converts intent to an order
- `FillEvent` — the simulated broker reports partial/full fill

This structurally prevents look-ahead bias and mimics a live production
environment. **No vectorized transforms in the execution layer.**

### Square-Root Law of Market Impact

For thousands of assets, liquidity constraints are absolute. Implement a
dynamic execution-cost model based on the square-root law:

$$
I(Q) = \sigma \cdot Y \cdot \sqrt{Q/V}
$$

where $Q$ is the order size, $V$ is the daily volume, $\sigma$ is the
asset's realized volatility, and $Y$ is a calibrated proportionality
constant. Doubling the order size increases slippage impact by
$\sqrt{2} \approx 1.41$ — this realistically punishes the algorithm for
over-allocating to illiquid assets.

(Already implemented in the current codebase at
[`backtest/costs.py::estimate_slippage`](../src/inversiones_mama/backtest/costs.py)
with the additional LOB-walk quadratic penalty above 1% ADV. Port the
same cost model into the new event-driven fill simulator.)

---

## Module 4 — Large-Scale Portfolio Optimization (HRP)

Applying traditional Markowitz Mean-Variance or unconstrained Kelly
optimization to thousands of assets will trigger **error maximization**.
The $N \times N$ covariance matrix becomes ill-conditioned, and its
inversion $\Sigma^{-1}$ amplifies minute estimation errors into massive,
unstable portfolio allocations.

The agent must bypass covariance inversion entirely by implementing
**Hierarchical Risk Parity** (Lopez de Prado 2016):

### Step 1 — Tree clustering

Map the correlation matrix $\rho$ into a distance matrix using

$$
d_{i,j} = \sqrt{0.5\,(1 - \rho_{i,j})}
$$

and group similar assets using **agglomerative hierarchical clustering**.

### Step 2 — Quasi-diagonalization

Reorganize the covariance matrix so highly correlated assets sit near
the diagonal. This reveals the natural block structure of the market
(equities cluster, bonds cluster, commodities cluster).

### Step 3 — Recursive bisection

Allocate capital top-down based on the **inverse variance** of the
sub-clusters. At each split:

$$
\alpha_1 = \frac{V_2}{V_1 + V_2}, \qquad
\alpha_2 = \frac{V_1}{V_1 + V_2}
$$

This guarantees structural diversification across asset classes
(separating Equities from Bonds inherently) and remains mathematically
robust **even if the covariance matrix is singular**.

HRP is the large-N replacement for the current RCK solver. The RCK +
sector-cap path remains appropriate for the v1a 10-ETF universe and the
SP100/SP500 backtests; HRP is mandatory once the universe exceeds a few
hundred assets or mixes asset classes.

---

## Module 5 — Advanced Simulations & Statistical Verification

A single backtest proves nothing. To verify that the portfolio's edge is
real and not a statistical fluke from data mining, the agent must
implement rigorous simulation and deflation layers.

### Multi-Level Monte Carlo (MLMC)

Standard Monte Carlo is computationally prohibitive for thousands of
assets — $O(\epsilon^{-3})$ complexity to reach error tolerance
$\epsilon$. Implement MLMC, which balances coarse low-cost approximations
with fine high-accuracy simulations and reduces cost to
$O(\epsilon^{-2})$.

### Block bootstrapping

When resampling historical data for Monte Carlo tests, use a **blocked**
bootstrap. Drawing single random days destroys volatility clustering;
drawing **contiguous blocks** of time preserves the heteroscedasticity
and long-memory effects inherent in financial markets. (Stationary
Politis-Romano and circular variants already exist in
[`simulation/bootstrap.py`](../src/inversiones_mama/simulation/bootstrap.py);
reuse.)

### Deflated Sharpe Ratio (DSR)

After CPCV and MC, compute the Deflated Sharpe Ratio (Bailey &
Lopez de Prado 2014). Testing thousands of asset / model / hyperparameter
combinations inevitably produces multiple-testing bias; DSR penalizes
the realized Sharpe by the **total number of trials** and the
**non-normality** (skewness, excess kurtosis) of returns:

$$
\mathrm{DSR} = \Phi\!\left(\frac{(\hat{SR} - \mathbb{E}[\widehat{SR}_0])\sqrt{T - 1}}{\sqrt{1 - \gamma_3 \hat{SR} + \frac{\gamma_4 - 1}{4} \hat{SR}^2}}\right)
$$

where $\gamma_3$ = skew, $\gamma_4$ = kurtosis, $T$ = # observations,
$\mathbb{E}[\widehat{SR}_0]$ = null Sharpe adjusted for # of trials. A DSR
$\ge 0.95$ is the conventional bar for "real edge after deflation." Our
current reporting already surfaces DSR in the verdict files — the new
engine must surface it per-CPCV-path so we get a full distribution.

---

## Interaction with existing code

The current codebase has substantial pieces the sibling agent should
reuse rather than rewrite:

| Existing module | Reuse for |
|---|---|
| [`backtest/costs.py`](../src/inversiones_mama/backtest/costs.py) | Square-root + LOB-walk cost model → port into the event-driven fill simulator |
| [`models/factor_regression.py`](../src/inversiones_mama/models/factor_regression.py) | 6-factor alpha source (still valid) |
| [`models/covariance.py`](../src/inversiones_mama/models/covariance.py) | Ledoit-Wolf shrinkage helpers (still valid for RCK path) |
| [`simulation/bootstrap.py`](../src/inversiones_mama/simulation/bootstrap.py) | Block bootstrap implementations (reuse) |
| [`simulation/monte_carlo.py`](../src/inversiones_mama/simulation/monte_carlo.py) | MC RCK validator (extend to MLMC) |
| [`backtest/metrics.py`](../src/inversiones_mama/backtest/metrics.py) | DSR implementation (reuse directly) |
| [`execution/paper_trader.py`](../src/inversiones_mama/execution/paper_trader.py) | `ExecutionClient` Protocol — the event-driven engine's fill simulator should conform to the same protocol so live paper trading remains a drop-in swap |
| [`data/sectors.py`](../src/inversiones_mama/data/sectors.py) | Sector mapping for HRP cluster labeling / diagnostics |

What **does not** exist and must be built:
- Event-driven engine (Market/Signal/Order/Fill events + chronological queue)
- Point-in-time constituent loader with delisting support
- Jump-Diffusion + OU synthetic generators
- `PurgedKFold` + CPCV splitter
- HRP optimizer
- MLMC simulator

---

## Acceptance gates for v3

The v3 engine is not considered ready until:

1. **Event-driven engine** reproduces the existing RCK walk-forward
   backtest for v1a within 1 bp per rebalance (same mu, same Sigma,
   same fills to the integer share).
2. **CPCV on v1a** produces ≥ 50 backtest paths and the DSR distribution
   is reported with its 5th/50th/95th percentiles.
3. **Point-in-time loader** verified against at least 3 known delistings
   (e.g. LEH 2008-09-15, WCG, XOM index membership changes).
4. **HRP optimizer** reproduces the Lopez de Prado 2016 paper results on
   the 10-asset synthetic dataset.
5. **Jump-diffusion generator** calibrates to SP500 daily returns with
   jump intensity $\lambda$ and jump-size distribution estimated from
   ≥ 2008 crisis + ≥ 2020 COVID crash windows.
6. All CPCV paths through the v1a universe pass DSR $\ge 0.95$ after
   multiple-testing deflation.

Until all six pass, v3 remains offline and v1a paper trading continues
unchanged under the current RCK + sector-cap engine.
