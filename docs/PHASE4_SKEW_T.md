# Phase 4 scope — skew-t parametric Kelly

> Scoping doc per Jorge's 2026-04-22 directive: "You may begin scoping
> Phase 4 (skew-t Kelly) in the background to handle the +5.3 kurtosis,
> but validation of our testing environment is the higher priority."

## Why this matters

The SP500 walk-forward produces:
- Excess kurtosis +5.3 (vs. Gaussian = 0)
- Skewness +0.4 (mild right tail)
- Max DD 33.4%

Kelly's analytical derivation assumes **log-wealth has Gaussian
increments**. When returns have:
- Fat tails (high kurtosis), Kelly UNDERWEIGHTS the drawdown risk in
  both tails.
- Asymmetric skew, Kelly treats positive and negative tail risks
  symmetrically even when they aren't.

For a $5k account, the consequence is real: Kelly's "optimal" fraction
overstates the safe leverage level, meaning we sit closer to the
drawdown bound than the theory claims.

**The fix is a skew-t parametric Kelly** — replace the Gaussian
assumption with a multivariate Skew-t distribution (Azzalini & Capitanio
2003), and maximize expected log-wealth under *that*. No `O(N^4)`
co-kurtosis tensors; the Skew-t's moment structure is parameterized by
a modest number of scalars.

## Concrete deliverables

### `src/inversiones_mama/models/skew_t.py`

```python
@dataclass(frozen=True)
class SkewTParams:
    mu: np.ndarray           # (n,) location
    scale: np.ndarray        # (n, n) dispersion matrix (≈ covariance when df→∞)
    alpha: np.ndarray        # (n,) skewness vector
    df: float                # degrees of freedom (<=30 = fat tails)
    log_likelihood: float    # for diagnostics

def fit_skew_t(
    returns: pd.DataFrame,
    df_grid: tuple[int, ...] = (3, 5, 8, 12, 20),
) -> SkewTParams:
    """Fit a multivariate Skew-t distribution to `returns` via the MM
    estimator (Azzalini-Capitanio), choosing df from a small grid by
    max likelihood."""

def skew_t_logpdf(x: np.ndarray, params: SkewTParams) -> float: ...

def skew_t_sample(params: SkewTParams, n_paths: int, horizon: int) -> np.ndarray: ...
```

### `src/inversiones_mama/sizing/sca_kelly.py`

```python
def solve_sca_kelly(
    skew_t: SkewTParams,
    fraction: float = 0.65,
    cap: float = 0.15,
    alpha_rck: float = 0.50,
    beta_rck: float = 0.10,
    n_iterations: int = 20,
    samples_per_iter: int = 5000,
) -> KellyResult:
    """Successive Convex Approximation (SCA) for maximum-E[log(W)] Kelly
    under a multivariate Skew-t distribution.

    Each iteration:
      1. Draw `samples_per_iter` synthetic returns from `skew_t`.
      2. Linearize E[log(1 + w'r)] around current w.
      3. Solve the resulting QP under Kelly/RCK constraints.
      4. Update w = (1-alpha) * w + alpha * w_new (damped).
    """
```

### Test matrix

- Unit: recover Gaussian Kelly when df=∞ (limit check).
- Unit: recover zero skew when alpha=0.
- Integration: on synthetic data with planted skewness/kurtosis,
  skew-t Kelly produces materially different weights than Gaussian Kelly.
- Integration: on real SP500 data, skew-t Kelly's max DD is *lower*
  than Gaussian Kelly's at the same target fraction (because it
  correctly accounts for fatter tails).

### Wire into the engine

`BacktestConfig` grows `kelly_method: str = "gaussian" | "skew_t"`.
Engine dispatches to the right solver.

## What to watch out for

1. **Skew-t fitting is non-trivial.** The MM estimator has multiple
   local optima; start from `sample_covariance` + zero skew.
2. **SCA convergence.** May oscillate without damping. Use `alpha=0.3`
   to start.
3. **Compute.** 20 iterations × 5000 samples × Sigma-solve = ~seconds
   per rebalance at N=494. Slightly slower than the Gaussian path but
   well within budget.
4. **Validation.** Don't assume the new solver is "better". Run both
   side-by-side on v1a/SP100/SP500 and compare Sharpe / OOS DSR /
   DD-95th. It's only a win if OOS DSR improves.

## Rough effort

- `skew_t.py` + tests: 1 day
- `sca_kelly.py` + tests: 1 day
- Engine wiring + config + side-by-side runs: half a day
- Total: ~2.5 days of focused work

## Start signal

Begin ONLY after the Alpaca paper ensemble has been running cleanly
for 2 weeks with no breaker trips and stable fill behavior. Per Jorge:
"validation of our testing environment is the higher priority."
