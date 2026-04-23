"""Unit tests for Inverse-Volatility portfolio weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.sizing.inverse_vol import (
    _apply_cap,
    generate_current_weights,
    inverse_vol_allocator,
    inverse_vol_weights,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20260423)


def _make_returns(rng, n_assets: int = 10, n_obs: int = 252,
                  vols: list[float] | None = None) -> pd.DataFrame:
    """Generate synthetic daily returns with specified per-asset volatilities."""
    if vols is None:
        vols = list(rng.uniform(0.01, 0.03, n_assets))
    cols = [f"ASSET_{i}" for i in range(len(vols))]
    data = rng.standard_normal((n_obs, len(vols))) * np.array(vols)
    return pd.DataFrame(data, columns=cols,
                        index=pd.bdate_range("2025-01-02", periods=n_obs))


def _make_prices(rng, n_assets: int = 10, n_obs: int = 252) -> pd.DataFrame:
    """Generate synthetic prices from cumulative returns."""
    rets = _make_returns(rng, n_assets, n_obs)
    prices = (1 + rets).cumprod() * 100
    return prices


# --- inverse_vol_weights ----------------------------------------------------


class TestInverseVolWeights:

    def test_sum_to_one(self, rng):
        rets = _make_returns(rng)
        w = inverse_vol_weights(rets)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_nonnegative(self, rng):
        rets = _make_returns(rng)
        w = inverse_vol_weights(rets)
        assert (w >= 0).all()

    def test_bounded(self, rng):
        rets = _make_returns(rng)
        w = inverse_vol_weights(rets)
        assert (w <= 1).all()

    def test_index_matches_columns(self, rng):
        rets = _make_returns(rng)
        w = inverse_vol_weights(rets)
        assert list(w.index) == list(rets.columns)

    def test_reproducible(self, rng):
        rets = _make_returns(rng)
        w1 = inverse_vol_weights(rets)
        w2 = inverse_vol_weights(rets)
        np.testing.assert_allclose(w1.values, w2.values, atol=1e-12)

    def test_high_vol_gets_less_weight(self, rng):
        """Asset with 2x volatility should get roughly half the weight."""
        rets = _make_returns(rng, n_assets=2, vols=[0.01, 0.02])
        w = inverse_vol_weights(rets)
        # w[0] should be roughly 2x w[1]
        ratio = w.iloc[0] / w.iloc[1]
        assert ratio > 1.5  # Allow sampling noise

    def test_equal_vol_gives_equal_weight(self, rng):
        rets = _make_returns(rng, n_assets=4, vols=[0.015, 0.015, 0.015, 0.015])
        w = inverse_vol_weights(rets)
        # Should be close to 0.25 each
        for wt in w.values:
            assert abs(wt - 0.25) < 0.05

    def test_rejects_single_asset(self, rng):
        rets = _make_returns(rng, n_assets=1)
        with pytest.raises(ValueError, match=">=\\s*2"):
            inverse_vol_weights(rets)

    def test_rejects_insufficient_observations(self, rng):
        rets = _make_returns(rng, n_obs=30)
        with pytest.raises(ValueError, match="Need >="):
            inverse_vol_weights(rets, vol_lookback=60)

    def test_zero_variance_asset_gets_zero_weight(self, rng):
        rets = _make_returns(rng, n_assets=4)
        rets["ZERO"] = 0.0
        w = inverse_vol_weights(rets)
        assert w["ZERO"] == pytest.approx(0.0)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_custom_lookback(self, rng):
        rets = _make_returns(rng, n_obs=100)
        w30 = inverse_vol_weights(rets, vol_lookback=30)
        w90 = inverse_vol_weights(rets, vol_lookback=90)
        # Different windows = different weights (generally)
        assert not np.allclose(w30.values, w90.values, atol=0.01)


# --- _apply_cap -------------------------------------------------------------


class TestApplyCap:

    def test_cap_respected(self):
        w = pd.Series([0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01],
                      index=[f"X{i}" for i in range(10)])
        capped = _apply_cap(w, cap=0.15)
        assert (capped <= 0.15 + 1e-4).all()

    def test_sum_to_one_after_cap(self):
        w = pd.Series([0.6, 0.3, 0.1], index=["A", "B", "C"])
        capped = _apply_cap(w, cap=0.40)
        assert capped.sum() == pytest.approx(1.0, abs=1e-9)

    def test_no_change_if_below_cap(self):
        w = pd.Series([0.1] * 10, index=[f"X{i}" for i in range(10)])
        capped = _apply_cap(w, cap=0.15)
        np.testing.assert_allclose(capped.values, w.values, atol=1e-9)

    def test_extreme_concentration(self):
        """99% in one asset, with enough assets that 15% cap is achievable."""
        weights = [0.99] + [0.01 / 9] * 9
        w = pd.Series(weights, index=[f"X{i}" for i in range(10)])
        capped = _apply_cap(w, cap=0.15)
        assert capped["X0"] == pytest.approx(0.15, abs=0.01)
        assert capped.sum() == pytest.approx(1.0, abs=1e-9)


# --- inverse_vol_allocator --------------------------------------------------


class TestInverseVolAllocator:

    def test_returns_correct_shapes(self, rng):
        prices = _make_prices(rng, n_assets=5, n_obs=200)
        weights, port_ret = inverse_vol_allocator(prices, vol_lookback=60)
        assert weights.shape[1] == 5
        assert len(port_ret) > 0
        assert len(port_ret) == len(weights)

    def test_weights_sum_to_one(self, rng):
        prices = _make_prices(rng, n_assets=5, n_obs=200)
        weights, _ = inverse_vol_allocator(prices, vol_lookback=60)
        # After the lookback period, weights should sum to ~1.0
        late_weights = weights.iloc[70:]
        sums = late_weights.sum(axis=1)
        assert (sums > 0.99).all()
        assert (sums < 1.01).all()

    def test_with_per_name_cap(self, rng):
        prices = _make_prices(rng, n_assets=5, n_obs=200)
        weights, _ = inverse_vol_allocator(
            prices, vol_lookback=60, per_name_cap=0.30
        )
        # After lookback, no weight should exceed 30%
        late_weights = weights.iloc[70:]
        assert (late_weights.max(axis=1) <= 0.30 + 0.01).all()


# --- generate_current_weights -----------------------------------------------


class TestGenerateCurrentWeights:

    def test_basic(self, rng):
        prices = _make_prices(rng, n_assets=8, n_obs=100)
        w = generate_current_weights(prices, vol_lookback=60)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert len(w) == 8

    def test_with_cap(self, rng):
        prices = _make_prices(rng, n_assets=8, n_obs=100)
        w = generate_current_weights(prices, vol_lookback=60, per_name_cap=0.15)
        assert (w <= 0.15 + 1e-9).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_without_cap(self, rng):
        prices = _make_prices(rng, n_assets=8, n_obs=100)
        w = generate_current_weights(prices, vol_lookback=60, per_name_cap=None)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_matches_manual_computation(self, rng):
        """Verify generate_current_weights matches manual inverse-vol."""
        prices = _make_prices(rng, n_assets=4, n_obs=100)
        returns = prices.pct_change().iloc[1:]
        sigma = returns.iloc[-60:].std(ddof=0)
        inv = 1.0 / sigma
        expected = inv / inv.sum()
        actual = generate_current_weights(prices, vol_lookback=60, per_name_cap=None)
        np.testing.assert_allclose(actual.values, expected.values, atol=1e-9)


# --- Integration with LIQUID_ETFS universe ----------------------------------


class TestLiquidETFIntegration:

    def test_module_importable(self):
        """Verify the module can be imported from the sizing package."""
        from inversiones_mama.sizing.inverse_vol import (  # noqa: F401
            generate_current_weights,
            inverse_vol_allocator,
            inverse_vol_weights,
        )
