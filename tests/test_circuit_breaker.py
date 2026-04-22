"""Tests for inversiones_mama.execution.circuit_breaker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from inversiones_mama.execution.circuit_breaker import CircuitBreaker, CircuitBreakerStatus


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# --------------------------------------------------------------------------- #
# Construction                                                                #
# --------------------------------------------------------------------------- #


def test_init_rejects_bad_threshold():
    with pytest.raises(ValueError, match="threshold"):
        CircuitBreaker(threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        CircuitBreaker(threshold=1.0)
    with pytest.raises(ValueError, match="threshold"):
        CircuitBreaker(threshold=-0.1)


def test_init_rejects_bad_initial_wealth():
    with pytest.raises(ValueError, match="initial_wealth"):
        CircuitBreaker(threshold=0.5, initial_wealth=0.0)


def test_init_rejects_warn_threshold_above_halt():
    with pytest.raises(ValueError, match="warn_threshold"):
        CircuitBreaker(threshold=0.3, warn_threshold=0.5)


def test_default_warn_is_80pct_of_threshold():
    cb = CircuitBreaker(threshold=0.5)
    assert cb.warn_threshold == 0.4


# --------------------------------------------------------------------------- #
# Peak tracking                                                               #
# --------------------------------------------------------------------------- #


def test_peak_starts_at_initial_wealth():
    cb = CircuitBreaker(threshold=0.5, initial_wealth=5000.0)
    status = cb.current_status()
    assert status.peak_wealth == 5000.0
    assert status.current_wealth == 5000.0
    assert status.current_drawdown == 0.0


def test_peak_rises_with_wealth_increase():
    cb = CircuitBreaker(threshold=0.5, initial_wealth=5000.0)
    s = cb.update(6000.0)
    assert s.peak_wealth == 6000.0
    assert s.current_wealth == 6000.0
    assert s.current_drawdown == 0.0


def test_peak_is_sticky_on_wealth_decrease():
    cb = CircuitBreaker(threshold=0.5, initial_wealth=5000.0)
    cb.update(6000.0)
    s = cb.update(4500.0)
    assert s.peak_wealth == 6000.0
    assert s.current_wealth == 4500.0
    assert s.current_drawdown == pytest.approx(0.25)


def test_negative_wealth_rejected():
    cb = CircuitBreaker(threshold=0.5)
    with pytest.raises(ValueError, match="negative"):
        cb.update(-100.0)


# --------------------------------------------------------------------------- #
# State transitions                                                           #
# --------------------------------------------------------------------------- #


def test_ok_state_below_warn_threshold():
    cb = CircuitBreaker(threshold=0.5, warn_threshold=0.3, initial_wealth=1000.0)
    s = cb.update(800.0)  # 20% drawdown — below warn
    assert s.state == "ok"
    assert s.tripped is False
    assert s.warning is False


def test_warn_state_between_warn_and_halt():
    cb = CircuitBreaker(threshold=0.5, warn_threshold=0.3, initial_wealth=1000.0)
    s = cb.update(650.0)  # 35% drawdown — past warn, before halt
    assert s.state == "warn"
    assert s.warning is True
    assert s.tripped is False


def test_tripped_state_at_or_above_threshold():
    cb = CircuitBreaker(threshold=0.5, warn_threshold=0.3, initial_wealth=1000.0)
    s = cb.update(500.0)  # exactly 50% drawdown — trips
    assert s.state == "tripped"
    assert s.tripped is True


def test_tripped_state_well_below_threshold():
    cb = CircuitBreaker(threshold=0.5, warn_threshold=0.3, initial_wealth=1000.0)
    s = cb.update(200.0)  # 80% drawdown
    assert s.state == "tripped"


# --------------------------------------------------------------------------- #
# Stickiness                                                                  #
# --------------------------------------------------------------------------- #


def test_sticky_breaker_stays_tripped_after_recovery():
    cb = CircuitBreaker(threshold=0.5, initial_wealth=1000.0, sticky=True)
    cb.update(400.0)  # 60% DD — tripped
    s = cb.update(950.0)  # mostly recovered; still tripped
    assert s.state == "tripped"
    assert cb.is_tripped() is True


def test_non_sticky_breaker_can_recover():
    cb = CircuitBreaker(threshold=0.5, warn_threshold=0.1,
                         initial_wealth=1000.0, sticky=False)
    cb.update(400.0)  # 60% DD — tripped
    # Recovers above 50% — state should go back to warn/ok since non-sticky and never latched
    # But our implementation sets _tripped_at on the first breach even in non-sticky mode.
    # Non-sticky means subsequent calls re-evaluate based on current drawdown.
    # Let's verify: after recovery, state should not be "tripped"
    s = cb.update(950.0)  # 5% DD
    assert s.state in {"ok", "warn"}


def test_tripped_at_timestamp_recorded():
    cb = CircuitBreaker(threshold=0.5, initial_wealth=1000.0)
    t = _now()
    cb.update(400.0, as_of=t)
    assert cb.is_tripped() is True
    status = cb.current_status()
    assert status.tripped_at == t


def test_tripped_at_not_updated_on_subsequent_updates():
    cb = CircuitBreaker(threshold=0.5, initial_wealth=1000.0)
    t1 = _now()
    cb.update(400.0, as_of=t1)
    t2 = t1 + timedelta(hours=1)
    cb.update(300.0, as_of=t2)  # deeper drawdown
    # Original trip time sticks
    assert cb.current_status().tripped_at == t1


def test_reset_clears_state():
    cb = CircuitBreaker(threshold=0.5, initial_wealth=1000.0)
    cb.update(300.0)
    assert cb.is_tripped() is True
    cb.reset(initial_wealth=5000.0)
    assert cb.is_tripped() is False
    assert cb.current_status().peak_wealth == 5000.0
    assert cb.current_status().current_drawdown == 0.0


def test_reset_rejects_bad_capital():
    cb = CircuitBreaker(threshold=0.5)
    with pytest.raises(ValueError, match="initial_wealth"):
        cb.reset(0.0)


# --------------------------------------------------------------------------- #
# from_mc_result factory                                                      #
# --------------------------------------------------------------------------- #


def _fake_mc_result(dds: np.ndarray, initial_capital: float = 5000.0) -> MagicMock:
    mc = MagicMock()
    mc.max_drawdowns = dds
    mc.initial_capital = initial_capital
    return mc


def test_from_mc_result_uses_p95_by_default():
    # Uniform DDs from 0 to 0.5; p95 ≈ 0.475
    mc = _fake_mc_result(np.linspace(0.0, 0.5, 1000))
    cb = CircuitBreaker.from_mc_result(mc)
    assert 0.45 < cb.threshold < 0.5
    # warn_threshold defaults to p75 ≈ 0.375 (clipped to ≤ 0.95 * threshold)
    assert 0.35 < cb.warn_threshold < cb.threshold


def test_from_mc_result_custom_percentile():
    mc = _fake_mc_result(np.linspace(0.0, 1.0, 1000))
    cb = CircuitBreaker.from_mc_result(mc, percentile=99.0)
    # p99 ≈ 0.99
    assert cb.threshold > 0.95


def test_from_mc_result_rejects_non_positive_threshold():
    mc = _fake_mc_result(np.zeros(100))
    with pytest.raises(ValueError, match="non-positive"):
        CircuitBreaker.from_mc_result(mc)


def test_from_mc_result_preserves_initial_capital():
    mc = _fake_mc_result(np.linspace(0, 0.5, 100), initial_capital=10_000.0)
    cb = CircuitBreaker.from_mc_result(mc)
    assert cb.current_status().peak_wealth == 10_000.0
