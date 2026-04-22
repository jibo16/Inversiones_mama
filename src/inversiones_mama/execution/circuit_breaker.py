"""Auto-halt circuit breaker for live/paper trading.

Core principle of the zero-budget deployment plan: the biggest risk is
"false confidence caused by backtesting assumptions." The Monte Carlo
validator produces an empirical distribution of wealth-drawdown outcomes
that a well-calibrated strategy should stay comfortably within. If the
LIVE realized drawdown punches through the MC 95th-percentile bound,
either:

* the strategy is curve-fitted (the in-sample statistics lied about the
  distribution of future drawdowns), or
* the market has entered a regime the bootstrap didn't sample.

Either way, continuing to trade without a human in the loop is reckless.
This module gives every paper / live run an automatic halt gate.

Usage pattern
-------------
::

    from inversiones_mama.execution.circuit_breaker import CircuitBreaker
    from inversiones_mama.simulation.monte_carlo import run_mc_rck_validation

    mc = run_mc_rck_validation(returns, weights=last_weights, n_paths=5000)
    breaker = CircuitBreaker.from_mc_result(mc, percentile=95)

    # After each rebalance or tick:
    status = breaker.update(current_wealth)
    if status.tripped:
        # Halt all subsequent order submissions; log + alert Jorge.
        ...

Public API
----------
``CircuitBreakerStatus`` — snapshot returned by each ``update`` call.
``CircuitBreaker``       — peak-tracker + threshold gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..simulation.monte_carlo import MCValidationResult


# --------------------------------------------------------------------------- #
# Result types                                                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CircuitBreakerStatus:
    """Snapshot of one `update()` evaluation."""

    state: str                     # "ok", "warn", "tripped"
    current_wealth: float
    peak_wealth: float
    current_drawdown: float        # positive fraction, e.g. 0.23 = -23%
    threshold: float               # positive fraction, e.g. 0.50 = -50%
    warn_threshold: float          # positive fraction, e.g. 0.40 = -40%
    as_of: datetime | None
    tripped_at: datetime | None    # when it first breached (sticky)

    @property
    def tripped(self) -> bool:
        return self.state == "tripped"

    @property
    def warning(self) -> bool:
        return self.state == "warn"


# --------------------------------------------------------------------------- #
# The breaker                                                                 #
# --------------------------------------------------------------------------- #


class CircuitBreaker:
    """Tracks peak wealth, computes drawdown, halts when the MC bound is breached.

    Parameters
    ----------
    threshold : float
        Drawdown (positive fraction) at which the breaker trips. 0.5 means
        "halt if current wealth is 50% below the running peak".
    warn_threshold : float or None
        Drawdown at which to emit "warn" state (no halt yet). Defaults to
        80% of ``threshold`` (rounded).
    initial_wealth : float
        Starting peak. Subsequent ``update`` calls can raise the peak but
        never lower it (peaks are sticky).
    sticky : bool
        If True (default), once tripped the breaker stays tripped until
        ``reset()``. If False, the breaker can auto-recover when wealth
        climbs back above the threshold — **not recommended** for live
        trading because it can oscillate.
    """

    def __init__(
        self,
        threshold: float,
        warn_threshold: float | None = None,
        initial_wealth: float = 5_000.0,
        *,
        sticky: bool = True,
    ) -> None:
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0,1); got {threshold}")
        if initial_wealth <= 0:
            raise ValueError(f"initial_wealth must be positive; got {initial_wealth}")
        if warn_threshold is None:
            warn_threshold = round(threshold * 0.8, 4)
        if not (0.0 < warn_threshold <= threshold):
            raise ValueError(
                f"warn_threshold must be in (0, threshold]; got {warn_threshold}"
            )
        self.threshold: float = float(threshold)
        self.warn_threshold: float = float(warn_threshold)
        self.sticky: bool = bool(sticky)
        self._peak: float = float(initial_wealth)
        self._current: float = float(initial_wealth)
        self._tripped_at: datetime | None = None
        self._last_as_of: datetime | None = None

    # ------------------------------------------------------------ factories

    @classmethod
    def from_mc_result(
        cls,
        mc_result: "MCValidationResult",
        percentile: float = 95.0,
        warn_percentile: float = 75.0,
    ) -> "CircuitBreaker":
        """Build a breaker whose threshold is the MC ``percentile`` drawdown.

        Default: halt at MC's 95th-percentile maximum-drawdown, warn at the
        75th percentile.
        """
        import numpy as np

        threshold = float(np.percentile(mc_result.max_drawdowns, percentile))
        warn = float(np.percentile(mc_result.max_drawdowns, warn_percentile))
        if threshold <= 0:
            raise ValueError(
                f"MC result yielded non-positive p{percentile} drawdown ({threshold}); "
                "cannot build a sensible breaker."
            )
        warn = max(min(warn, threshold * 0.95), 1e-4)
        return cls(
            threshold=threshold,
            warn_threshold=warn,
            initial_wealth=mc_result.initial_capital,
        )

    # --------------------------------------------------------- observation

    def update(self, wealth: float, as_of: datetime | None = None) -> CircuitBreakerStatus:
        """Observe a new wealth value; return the resulting state snapshot."""
        if wealth < 0:
            raise ValueError(f"wealth cannot be negative; got {wealth}")
        as_of = as_of or datetime.now(tz=timezone.utc)
        self._current = float(wealth)
        self._last_as_of = as_of
        if wealth > self._peak:
            self._peak = float(wealth)

        drawdown = self._compute_drawdown()

        # Determine state
        if self.sticky and self._tripped_at is not None:
            state = "tripped"
        elif drawdown >= self.threshold:
            state = "tripped"
            if self._tripped_at is None:
                self._tripped_at = as_of
        elif drawdown >= self.warn_threshold:
            state = "warn"
        else:
            state = "ok"

        return CircuitBreakerStatus(
            state=state,
            current_wealth=self._current,
            peak_wealth=self._peak,
            current_drawdown=drawdown,
            threshold=self.threshold,
            warn_threshold=self.warn_threshold,
            as_of=as_of,
            tripped_at=self._tripped_at,
        )

    def is_tripped(self) -> bool:
        return self._tripped_at is not None

    def current_status(self) -> CircuitBreakerStatus:
        """Return the last snapshot without changing state."""
        drawdown = self._compute_drawdown()
        state = "tripped" if self._tripped_at is not None else (
            "warn" if drawdown >= self.warn_threshold else "ok"
        )
        return CircuitBreakerStatus(
            state=state,
            current_wealth=self._current,
            peak_wealth=self._peak,
            current_drawdown=drawdown,
            threshold=self.threshold,
            warn_threshold=self.warn_threshold,
            as_of=self._last_as_of,
            tripped_at=self._tripped_at,
        )

    # ------------------------------------------------------------ mutation

    def reset(self, initial_wealth: float) -> None:
        """Clear the tripped state and reset the peak."""
        if initial_wealth <= 0:
            raise ValueError(f"initial_wealth must be positive; got {initial_wealth}")
        self._peak = float(initial_wealth)
        self._current = float(initial_wealth)
        self._tripped_at = None
        self._last_as_of = None

    # --------------------------------------------------------- internals

    def _compute_drawdown(self) -> float:
        if self._peak <= 0:
            return 0.0
        return max(0.0, 1.0 - self._current / self._peak)
