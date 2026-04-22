"""Tests for backtest.costs — IBKR Tiered commission + slippage model.

Coverage:
  - ibkr_commission: exact fee schedule (per-share, min, max cap)
  - estimate_slippage: base bps, ADV-scaled market impact
  - total_trade_cost: combined cost breakdown
  - portfolio_rebalance_cost: full rebalance aggregation
  - Edge cases: zero shares, zero price, tiny trades
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.backtest.costs import (
    RebalanceCost,
    TradeCost,
    estimate_slippage,
    ibkr_commission,
    portfolio_rebalance_cost,
    total_trade_cost,
)


# --------------------------------------------------------------------------- #
# LOB-walk non-linear impact (Roadmap #1, 2026-04-22)                         #
# --------------------------------------------------------------------------- #


class TestLOBWalkSlippage:
    """Test the non-linear "walk the book" penalty above 1% of ADV."""

    def test_below_threshold_no_lob_penalty(self):
        """Order at 0.5% of ADV: only sqrt-impact applies."""
        # shares=500, price=$100, ADV=100_000 => participation=0.5%
        s_with_lob = estimate_slippage(500, 100.0, adv=100_000, lob_walk=True)
        s_without = estimate_slippage(500, 100.0, adv=100_000, lob_walk=False)
        assert s_with_lob == pytest.approx(s_without, rel=1e-9)

    def test_above_threshold_adds_penalty(self):
        """Order at 5% of ADV: LOB penalty kicks in."""
        # shares=5000, price=$100, ADV=100_000 => participation=5%
        s_lob = estimate_slippage(5000, 100.0, adv=100_000, lob_walk=True)
        s_sqrt_only = estimate_slippage(5000, 100.0, adv=100_000, lob_walk=False)
        assert s_lob > s_sqrt_only

    def test_penalty_grows_quadratically(self):
        """Doubling participation above threshold should more-than-double extra penalty."""
        base_no_lob_10 = estimate_slippage(2000, 100.0, adv=100_000, lob_walk=False)
        with_lob_10 = estimate_slippage(2000, 100.0, adv=100_000, lob_walk=True)
        extra_10 = with_lob_10 - base_no_lob_10

        base_no_lob_20 = estimate_slippage(4000, 100.0, adv=100_000, lob_walk=False)
        with_lob_20 = estimate_slippage(4000, 100.0, adv=100_000, lob_walk=True)
        extra_20 = with_lob_20 - base_no_lob_20

        # participation 2% -> overshoot = 1 unit.  penalty ~ 10 bps
        # participation 4% -> overshoot = 3 units. penalty ~ 90 bps
        # ratio of penalties should be well above 2 (quadratic growth)
        assert extra_20 / max(extra_10, 1e-9) > 3.0

    def test_no_adv_disables_lob_penalty(self):
        """When ADV is None, LOB penalty cannot be computed — base bps only."""
        s_no_adv = estimate_slippage(1_000_000, 100.0, adv=None, lob_walk=True)
        # base 5 bps of $100M = $50k
        assert s_no_adv == pytest.approx(1_000_000 * 100.0 * 5 / 10_000, rel=1e-6)


# --------------------------------------------------------------------------- #
# ibkr_commission                                                             #
# --------------------------------------------------------------------------- #


class TestIBKRCommission:
    """Test the exact IBKR Tiered fee schedule."""

    def test_standard_order(self):
        """100 shares at $50 → $0.35 (100 × $0.0035 = $0.35 = min)."""
        comm = ibkr_commission(100, 50.0)
        assert comm == pytest.approx(0.35, abs=0.0001)

    def test_min_per_order_kicks_in(self):
        """10 shares at $50 → raw $0.035 < min $0.35 → clamp to $0.35."""
        comm = ibkr_commission(10, 50.0)
        assert comm == pytest.approx(0.35, abs=0.0001)

    def test_above_min(self):
        """200 shares at $50 → 200 × $0.0035 = $0.70 > $0.35."""
        comm = ibkr_commission(200, 50.0)
        assert comm == pytest.approx(0.70, abs=0.0001)

    def test_max_cap_kicks_in(self):
        """10000 shares at $1 → raw $35 vs max 1% of $10k = $100 → $35."""
        comm = ibkr_commission(10000, 1.0)
        assert comm == pytest.approx(35.0, abs=0.01)

    def test_max_cap_low_price(self):
        """1000 shares at $0.50 → raw $3.50 vs max 1% of $500 = $5 → $3.50."""
        comm = ibkr_commission(1000, 0.50)
        assert comm == pytest.approx(3.50, abs=0.01)

    def test_max_cap_binding(self):
        """Scenario where max cap is binding: many shares, low price."""
        # 10000 shares at $0.10 → raw $35 vs max 1% of $1000 = $10 → $10
        comm = ibkr_commission(10000, 0.10)
        assert comm == pytest.approx(10.0, abs=0.01)

    def test_zero_shares(self):
        comm = ibkr_commission(0, 50.0)
        assert comm == 0.0

    def test_negative_shares_uses_abs(self):
        """Selling: negative shares should use absolute value."""
        comm = ibkr_commission(-200, 50.0)
        assert comm == pytest.approx(0.70, abs=0.0001)

    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="[Pp]rice"):
            ibkr_commission(100, -10.0)

    def test_custom_params(self):
        """Override default fee parameters."""
        comm = ibkr_commission(
            100, 50.0,
            per_share=0.005,
            min_per_order=0.50,
            max_pct_trade=0.02,
        )
        # 100 × $0.005 = $0.50 = min → $0.50
        assert comm == pytest.approx(0.50, abs=0.0001)

    def test_large_order_1000_shares(self):
        """1000 shares at $100 → 1000 × $0.0035 = $3.50."""
        comm = ibkr_commission(1000, 100.0)
        assert comm == pytest.approx(3.50, abs=0.01)

    def test_ibkr_realistic_spy(self):
        """Realistic: buy 10 shares SPY at ~$520."""
        comm = ibkr_commission(10, 520.0)
        # 10 × $0.0035 = $0.035 < $0.35 min → $0.35
        assert comm == pytest.approx(0.35, abs=0.01)


# --------------------------------------------------------------------------- #
# estimate_slippage                                                           #
# --------------------------------------------------------------------------- #


class TestEstimateSlippage:
    """Test the slippage model."""

    def test_base_slippage_only(self):
        """Without ADV, only base bps apply."""
        slip = estimate_slippage(100, 50.0, adv=None, base_bps=5.0)
        expected = 100 * 50.0 * 5.0 / 10_000
        assert slip == pytest.approx(expected, abs=0.01)

    def test_zero_shares(self):
        slip = estimate_slippage(0, 50.0)
        assert slip == 0.0

    def test_adv_increases_slippage(self):
        """With ADV, market impact adds to base slippage."""
        base_only = estimate_slippage(1000, 50.0, adv=None, base_bps=5.0)
        with_adv = estimate_slippage(1000, 50.0, adv=50000, base_bps=5.0)
        assert with_adv > base_only

    def test_large_participation_rate(self):
        """Trading a large fraction of ADV → significant impact."""
        # 10000 shares, ADV = 20000 → participation = 0.5
        slip = estimate_slippage(10000, 50.0, adv=20000, base_bps=5.0)
        base = 10000 * 50.0 * 5.0 / 10_000
        assert slip > base  # Impact component should add to base

    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="[Pp]rice"):
            estimate_slippage(100, -10.0)

    def test_slippage_scales_with_trade_value(self):
        """Doubling shares should roughly double base slippage."""
        s1 = estimate_slippage(100, 50.0, adv=None, base_bps=5.0)
        s2 = estimate_slippage(200, 50.0, adv=None, base_bps=5.0)
        assert s2 == pytest.approx(2 * s1, rel=0.01)


# --------------------------------------------------------------------------- #
# total_trade_cost                                                            #
# --------------------------------------------------------------------------- #


class TestTotalTradeCost:
    """Test the combined trade cost breakdown."""

    def test_returns_trade_cost(self):
        tc = total_trade_cost("SPY", 100, 520.0)
        assert isinstance(tc, TradeCost)
        assert tc.ticker == "SPY"
        assert tc.shares == 100

    def test_total_is_commission_plus_slippage(self):
        tc = total_trade_cost("SPY", 100, 520.0)
        assert tc.total_cost == pytest.approx(
            tc.commission + tc.slippage, abs=0.01
        )

    def test_trade_value_correct(self):
        tc = total_trade_cost("SPY", 50, 100.0)
        assert tc.trade_value == pytest.approx(5000.0, abs=0.01)

    def test_zero_shares(self):
        tc = total_trade_cost("SPY", 0, 520.0)
        assert tc.total_cost == 0.0
        assert tc.commission == 0.0
        assert tc.slippage == 0.0

    def test_cost_bps_calculated(self):
        tc = total_trade_cost("SPY", 100, 100.0)
        expected_bps = tc.total_cost / tc.trade_value * 10_000
        assert tc.cost_bps == pytest.approx(expected_bps, abs=0.1)

    def test_sell_order(self):
        """Negative shares = sell order, costs should still compute."""
        tc = total_trade_cost("SPY", -100, 520.0)
        assert tc.commission > 0
        assert tc.slippage > 0
        assert tc.shares == 100  # stored as abs


# --------------------------------------------------------------------------- #
# portfolio_rebalance_cost                                                    #
# --------------------------------------------------------------------------- #


class TestPortfolioRebalanceCost:
    """Test portfolio-level rebalance cost."""

    @pytest.fixture
    def rebal_inputs(self):
        """Standard rebalance scenario: 3 assets, $5000 portfolio."""
        current = pd.Series({"A": 0.40, "B": 0.35, "C": 0.25})
        target = pd.Series({"A": 0.30, "B": 0.40, "C": 0.30})
        prices = pd.Series({"A": 100.0, "B": 50.0, "C": 200.0})
        return current, target, 5000.0, prices

    def test_returns_rebalance_cost(self, rebal_inputs):
        curr, tgt, pv, px = rebal_inputs
        rc = portfolio_rebalance_cost(curr, tgt, pv, px)
        assert isinstance(rc, RebalanceCost)

    def test_total_cost_positive(self, rebal_inputs):
        curr, tgt, pv, px = rebal_inputs
        rc = portfolio_rebalance_cost(curr, tgt, pv, px)
        assert rc.total_cost > 0

    def test_total_is_comm_plus_slip(self, rebal_inputs):
        curr, tgt, pv, px = rebal_inputs
        rc = portfolio_rebalance_cost(curr, tgt, pv, px)
        assert rc.total_cost == pytest.approx(
            rc.total_commission + rc.total_slippage, abs=0.01
        )

    def test_no_change_no_cost(self):
        """Same weights → no trades → zero cost."""
        curr = pd.Series({"A": 0.50, "B": 0.50})
        prices = pd.Series({"A": 100.0, "B": 200.0})
        rc = portfolio_rebalance_cost(curr, curr, 5000.0, prices)
        assert rc.total_cost == 0.0
        assert rc.n_trades == 0

    def test_cost_pct_portfolio(self, rebal_inputs):
        curr, tgt, pv, px = rebal_inputs
        rc = portfolio_rebalance_cost(curr, tgt, pv, px)
        expected = rc.total_cost / pv * 100
        assert rc.cost_pct_portfolio == pytest.approx(expected, abs=0.01)

    def test_trade_details_populated(self, rebal_inputs):
        curr, tgt, pv, px = rebal_inputs
        rc = portfolio_rebalance_cost(curr, tgt, pv, px)
        assert len(rc.trade_details) > 0
        assert all(isinstance(td, TradeCost) for td in rc.trade_details)

    def test_min_trade_value_filter(self):
        """Tiny rebalance should be skipped."""
        curr = pd.Series({"A": 0.500, "B": 0.500})
        tgt = pd.Series({"A": 0.501, "B": 0.499})  # $5 change on $5000
        prices = pd.Series({"A": 100.0, "B": 200.0})
        rc = portfolio_rebalance_cost(
            curr, tgt, 5000.0, prices, min_trade_value=10.0
        )
        # Delta is $5 per name, which is below $10 min → skipped
        assert rc.n_trades == 0

    def test_with_adv(self, rebal_inputs):
        """ADV should affect slippage but not commissions."""
        curr, tgt, pv, px = rebal_inputs
        adv = pd.Series({"A": 1_000_000, "B": 500_000, "C": 100_000})
        rc_with = portfolio_rebalance_cost(curr, tgt, pv, px, adv=adv)
        rc_without = portfolio_rebalance_cost(curr, tgt, pv, px, adv=None)
        # Commissions should be the same
        assert rc_with.total_commission == pytest.approx(
            rc_without.total_commission, abs=0.01
        )
        # Slippage should be higher with ADV (adds impact component)
        assert rc_with.total_slippage >= rc_without.total_slippage - 0.01

    def test_realistic_5k_portfolio(self):
        """Realistic $5k ETF portfolio rebalance."""
        curr = pd.Series({
            "AVUV": 0.20, "MTUM": 0.20, "GLD": 0.15,
            "SPY": 0.15, "DBC": 0.10, "TLT": 0.10, "USMV": 0.10,
        })
        tgt = pd.Series({
            "AVUV": 0.30, "MTUM": 0.25, "GLD": 0.15,
            "SPY": 0.10, "DBC": 0.10, "TLT": 0.00, "USMV": 0.10,
        })
        prices = pd.Series({
            "AVUV": 85.0, "MTUM": 210.0, "GLD": 190.0,
            "SPY": 520.0, "DBC": 25.0, "TLT": 90.0, "USMV": 80.0,
        })
        rc = portfolio_rebalance_cost(curr, tgt, 5000.0, prices)
        # Cost should be very small for liquid ETFs
        assert rc.cost_pct_portfolio < 0.10  # Less than 0.1% of portfolio
        assert rc.n_trades > 0
