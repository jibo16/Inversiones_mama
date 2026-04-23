"""Unit tests for Inverse-Volatility portfolio weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.sizing.inverse_vol import (
    _apply_cap,
    _apply_equity_floor,
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


# --- Equity-floor constraint -----------------------------------------------


class TestEqualityFloor:
    """Verify the 40% equity floor prevents bond/money-market degeneration."""

    def _equity_non_equity_split(self, w: pd.Series) -> tuple[float, float]:
        from inversiones_mama.data.asset_classes import NON_EQUITY_TICKERS
        is_eq = ~w.index.astype(str).str.upper().isin(NON_EQUITY_TICKERS)
        return float(w[is_eq].sum()), float(w[~is_eq].sum())

    def test_rejects_invalid_floor(self):
        w = pd.Series({"SPY": 0.5, "TLT": 0.5})
        with pytest.raises(ValueError, match="equity_floor must be in"):
            _apply_equity_floor(w, equity_floor=1.5)
        with pytest.raises(ValueError, match="equity_floor must be in"):
            _apply_equity_floor(w, equity_floor=-0.1)

    def test_zero_floor_is_noop(self):
        w = pd.Series({"SPY": 0.2, "TLT": 0.8})
        out = _apply_equity_floor(w, equity_floor=0.0)
        pd.testing.assert_series_equal(w, out)

    def test_floor_already_satisfied(self):
        """If equity_w >= floor, weights unchanged."""
        w = pd.Series({"SPY": 0.6, "QQQ": 0.2, "TLT": 0.2})
        out = _apply_equity_floor(w, equity_floor=0.40)
        pd.testing.assert_series_equal(w, out)

    def test_floor_lifts_equity_exposure(self):
        """When equity_w < floor, equities get scaled up proportionally."""
        # 91-ETF case: bonds dominate because of low vol. 10% equity, 90% bonds.
        w = pd.Series({"SPY": 0.05, "QQQ": 0.05, "TLT": 0.60, "SHY": 0.30})
        out = _apply_equity_floor(w, equity_floor=0.40)
        eq, neq = self._equity_non_equity_split(out)
        assert eq == pytest.approx(0.40, abs=1e-9)
        assert neq == pytest.approx(0.60, abs=1e-9)
        # SPY and QQQ should be scaled 4x each (0.10 -> 0.40)
        assert out["SPY"] == pytest.approx(0.20, abs=1e-9)  # 0.05 * 8 = 0.40 / 2 = 0.20
        assert out["QQQ"] == pytest.approx(0.20, abs=1e-9)
        # TLT and SHY should be scaled 2/3x each (0.90 -> 0.60)
        assert out["TLT"] == pytest.approx(0.60 * (0.60 / 0.90), abs=1e-9)
        assert out["SHY"] == pytest.approx(0.30 * (0.60 / 0.90), abs=1e-9)
        assert out.sum() == pytest.approx(1.0, abs=1e-9)

    def test_floor_preserves_relative_weights_within_partition(self):
        """Within equities, ratios are preserved. Same for non-equities."""
        w = pd.Series({"SPY": 0.10, "QQQ": 0.05, "TLT": 0.50, "SHY": 0.35})
        out = _apply_equity_floor(w, equity_floor=0.40)
        # SPY:QQQ ratio preserved (2:1)
        assert out["SPY"] / out["QQQ"] == pytest.approx(0.10 / 0.05, rel=1e-9)
        # TLT:SHY ratio preserved
        assert out["TLT"] / out["SHY"] == pytest.approx(0.50 / 0.35, rel=1e-9)

    def test_floor_noop_when_no_equities(self):
        """With only non-equity tickers, floor can't be applied."""
        w = pd.Series({"TLT": 0.5, "SHY": 0.5})
        out = _apply_equity_floor(w, equity_floor=0.40)
        pd.testing.assert_series_equal(w, out)

    def test_floor_noop_when_no_non_equities(self):
        """With only equities, floor is already satisfied."""
        w = pd.Series({"SPY": 0.5, "QQQ": 0.5})
        out = _apply_equity_floor(w, equity_floor=0.40)
        pd.testing.assert_series_equal(w, out)

    def test_inverse_vol_weights_honors_floor(self):
        """End-to-end: inverse_vol_weights with equity_floor lifts equity allocation."""
        # Simulate a universe where bonds have very low vol (dominates inverse-vol)
        rng = np.random.default_rng(20260423)
        n = 252
        # SPY ~ 1% daily vol, TLT ~ 0.5%, SHY ~ 0.05% (treasury-like)
        spy = rng.standard_normal(n) * 0.010
        qqq = rng.standard_normal(n) * 0.012
        tlt = rng.standard_normal(n) * 0.005
        shy = rng.standard_normal(n) * 0.0005  # near-zero vol -> inverse-vol blows up
        returns = pd.DataFrame(
            {"SPY": spy, "QQQ": qqq, "TLT": tlt, "SHY": shy},
            index=pd.bdate_range("2025-01-02", periods=n),
        )

        # Without floor: SHY dominates because inverse-vol weight explodes
        w_no_floor = inverse_vol_weights(returns, vol_lookback=60)
        eq_no_floor = w_no_floor["SPY"] + w_no_floor["QQQ"]
        assert eq_no_floor < 0.15, f"expected SHY-dominated, got equity={eq_no_floor:.3f}"

        # With 40% floor: equities lifted to >= 40%
        w_with_floor = inverse_vol_weights(returns, vol_lookback=60, equity_floor=0.40)
        eq_with_floor = w_with_floor["SPY"] + w_with_floor["QQQ"]
        assert eq_with_floor == pytest.approx(0.40, abs=1e-9)
        assert w_with_floor.sum() == pytest.approx(1.0, abs=1e-9)

    def test_generate_current_weights_defaults_to_40pct_floor(self):
        """Deployment entry point defaults equity_floor=0.40."""
        rng = np.random.default_rng(20260423)
        n = 300
        # Prices: 2 equities, 2 bonds. Bonds near-flat (low vol).
        eq_prices = np.cumprod(1 + rng.standard_normal((n, 2)) * 0.01, axis=0) * 100
        bond_prices = np.cumprod(1 + rng.standard_normal((n, 2)) * 0.001, axis=0) * 100
        prices = pd.DataFrame(
            np.hstack([eq_prices, bond_prices]),
            columns=["SPY", "QQQ", "TLT", "SHY"],
            index=pd.bdate_range("2024-01-02", periods=n),
        )
        # With default floor=0.40, equity share should be at least 40%
        w = generate_current_weights(prices, vol_lookback=60)
        eq_w = w["SPY"] + w["QQQ"]
        assert eq_w >= 0.39, f"equity_w={eq_w:.3f} below default 40% floor"
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_generate_current_weights_floor_none_disables(self):
        """equity_floor=None disables the floor (back to raw inverse-vol + cap)."""
        rng = np.random.default_rng(20260423)
        n = 300
        eq_prices = np.cumprod(1 + rng.standard_normal((n, 2)) * 0.01, axis=0) * 100
        bond_prices = np.cumprod(1 + rng.standard_normal((n, 2)) * 0.001, axis=0) * 100
        prices = pd.DataFrame(
            np.hstack([eq_prices, bond_prices]),
            columns=["SPY", "QQQ", "TLT", "SHY"],
            index=pd.bdate_range("2024-01-02", periods=n),
        )
        # With floor=None, bonds should dominate (near-zero vol)
        w = generate_current_weights(prices, vol_lookback=60, equity_floor=None,
                                      per_name_cap=None)
        eq_w = w["SPY"] + w["QQQ"]
        assert eq_w < 0.20, f"expected bond-dominated, got equity={eq_w:.3f}"
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_allocator_floor_stable_across_rebalances(self):
        """Every rebalance enforces the floor."""
        rng = np.random.default_rng(20260423)
        n = 400
        eq = np.cumprod(1 + rng.standard_normal((n, 2)) * 0.01, axis=0) * 100
        bonds = np.cumprod(1 + rng.standard_normal((n, 2)) * 0.001, axis=0) * 100
        prices = pd.DataFrame(
            np.hstack([eq, bonds]),
            columns=["SPY", "QQQ", "TLT", "SHY"],
            index=pd.bdate_range("2024-01-02", periods=n),
        )
        from inversiones_mama.data.asset_classes import NON_EQUITY_TICKERS

        weights_df, _ = inverse_vol_allocator(
            prices, vol_lookback=60, rebal_freq="ME", equity_floor=0.40,
        )
        # Check every rebalance date's equity share
        monthly_idx = prices.resample("ME").last().index
        sampled = weights_df.loc[weights_df.index.isin(monthly_idx)]
        # Drop any leading rows where current_w is still equal-weight (pre-first-rebalance)
        sampled = sampled.iloc[2:]  # skip first couple months (warmup)
        is_eq = ~sampled.columns.astype(str).str.upper().isin(NON_EQUITY_TICKERS)
        equity_sum = sampled.loc[:, is_eq].sum(axis=1)
        # After warmup, equity share should be >= 0.39 (allow tiny drift)
        assert (equity_sum >= 0.39).mean() > 0.9, (
            f"expected >=90% of rebalances to respect floor, got "
            f"{(equity_sum >= 0.39).mean():.2%}"
        )
