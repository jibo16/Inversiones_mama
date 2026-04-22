"""Tests for the strategy exploration framework.

Covers:
  - Base class contract enforcement
  - Runner: train/test split, cost deduction, metric computation, DSR gate
  - Each strategy: signal shape, weight constraints, determinism
  - Report generation
  - Integration: full pipeline from prices → signals → backtest → report
  - Hard constraint: no imports from sizing/execution

Test data is synthetic to keep tests offline and deterministic.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.exploration.base import Strategy, StrategyMeta
from inversiones_mama.exploration.runner import (
    DSR_REJECTION_THRESHOLD,
    StrategyResult,
    _compute_portfolio_returns,
    _estimate_rebalance_cost_bps,
    run_strategy,
    run_batch,
)
from inversiones_mama.exploration.report import (
    save_strategy_result,
    save_batch_results,
    format_results_table,
)
from inversiones_mama.exploration.strategies.momentum_xsec import CrossSectionalMomentum
from inversiones_mama.exploration.strategies.momentum_ts import TimeSeriesMomentum
from inversiones_mama.exploration.strategies.mean_reversion import RSIMeanReversion
from inversiones_mama.exploration.strategies.vol_targeting import VolatilityTargeting
from inversiones_mama.exploration.strategies.dual_momentum import DualMomentum


# --------------------------------------------------------------------------- #
# Synthetic test data                                                          #
# --------------------------------------------------------------------------- #


def _make_prices(n_days: int = 504, n_assets: int = 5, seed: int = 42) -> pd.DataFrame:
    """Create synthetic price data with realistic drift and vol."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days, freq="B")
    tickers = [f"ETF{i}" for i in range(n_assets)]

    prices_data = {}
    for i, ticker in enumerate(tickers):
        # Different drift per asset so momentum strategies have something to rank
        drift = 0.0003 * (i + 1)  # 0.03% to 0.15% daily
        vol = 0.015 + 0.005 * i   # 1.5% to 3.5% daily vol
        log_returns = rng.normal(drift, vol, n_days)
        log_returns[0] = 0.0
        prices_data[ticker] = 100.0 * np.exp(np.cumsum(log_returns))

    return pd.DataFrame(prices_data, index=dates)


def _make_prices_with_tlt(n_days: int = 504, seed: int = 42) -> pd.DataFrame:
    """Create synthetic prices including a TLT-like asset for dual momentum."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days, freq="B")

    tickers = ["AVUV", "MTUM", "GLD", "SPY", "TLT"]
    drifts = [0.0005, 0.0003, 0.0002, 0.0004, 0.0001]
    vols = [0.02, 0.015, 0.01, 0.012, 0.008]

    prices_data = {}
    for ticker, drift, vol in zip(tickers, drifts, vols):
        log_returns = rng.normal(drift, vol, n_days)
        log_returns[0] = 0.0
        prices_data[ticker] = 100.0 * np.exp(np.cumsum(log_returns))

    return pd.DataFrame(prices_data, index=dates)


# --------------------------------------------------------------------------- #
# Concrete Strategy subclass for testing                                       #
# --------------------------------------------------------------------------- #


class ConstantWeightStrategy(Strategy):
    """Test strategy: constant equal-weight across all assets."""

    def __init__(self) -> None:
        super().__init__(StrategyMeta(
            name="ConstantWeight",
            category="test",
            parameters={"weight": "equal"},
            description="Equal-weight all assets constantly.",
        ))

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        n = len(prices.columns)
        w = 1.0 / n if n > 0 else 0.0
        return pd.DataFrame(w, index=prices.index, columns=prices.columns)


class FailingStrategy(Strategy):
    """Test strategy that raises an exception."""

    def __init__(self) -> None:
        super().__init__(StrategyMeta(name="Failing", category="test"))

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise RuntimeError("Intentional failure")


class EmptyStrategy(Strategy):
    """Test strategy that returns empty signals."""

    def __init__(self) -> None:
        super().__init__(StrategyMeta(name="Empty", category="test"))

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Base class tests                                                             #
# --------------------------------------------------------------------------- #


class TestStrategyBase:
    """Test the abstract Strategy base class contract."""

    def test_meta_properties(self):
        strat = ConstantWeightStrategy()
        assert strat.name == "ConstantWeight"
        assert strat.category == "test"
        assert strat.parameters == {"weight": "equal"}
        assert "Equal-weight" in strat.description

    def test_repr(self):
        strat = ConstantWeightStrategy()
        r = repr(strat)
        assert "ConstantWeight" in r

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Strategy(StrategyMeta(name="X", category="Y"))

    def test_generate_signals_returns_dataframe(self):
        prices = _make_prices(100, 3)
        strat = ConstantWeightStrategy()
        signals = strat.generate_signals(prices)
        assert isinstance(signals, pd.DataFrame)
        assert signals.shape[0] == len(prices)
        assert signals.shape[1] == 3


# --------------------------------------------------------------------------- #
# Runner tests                                                                 #
# --------------------------------------------------------------------------- #


class TestPortfolioReturns:
    """Test _compute_portfolio_returns."""

    def test_equal_weight_aggregation(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        rets = pd.DataFrame(
            {"A": [0.01, 0.02, -0.01, 0.03, 0.01],
             "B": [0.02, -0.01, 0.01, 0.01, 0.02]},
            index=dates,
        )
        weights = pd.DataFrame(
            {"A": [0.5, 0.5, 0.5, 0.5, 0.5],
             "B": [0.5, 0.5, 0.5, 0.5, 0.5]},
            index=dates,
        )
        port = _compute_portfolio_returns(weights, rets)
        expected = (rets["A"] * 0.5 + rets["B"] * 0.5)
        pd.testing.assert_series_equal(port, expected, check_names=False)

    def test_zero_weight_is_cash(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        rets = pd.DataFrame({"A": [0.05, 0.05, 0.05]}, index=dates)
        weights = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)
        port = _compute_portfolio_returns(weights, rets)
        assert (port == 0.0).all()


class TestRebalanceCost:
    """Test _estimate_rebalance_cost_bps."""

    def test_no_change_no_cost(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        cost = _estimate_rebalance_cost_bps(w, w)
        assert cost == 0.0

    def test_full_rebalance_costs_something(self):
        old = pd.Series({"A": 1.0, "B": 0.0})
        new = pd.Series({"A": 0.0, "B": 1.0})
        cost = _estimate_rebalance_cost_bps(old, new)
        assert cost > 0


class TestRunStrategy:
    """Test the run_strategy function."""

    def test_returns_strategy_result(self):
        prices = _make_prices(504, 3)
        strat = ConstantWeightStrategy()
        result = run_strategy(strat, prices, n_trials=1)
        assert isinstance(result, StrategyResult)
        assert result.strategy_name == "ConstantWeight"

    def test_train_test_split_dates(self):
        prices = _make_prices(500, 3)
        strat = ConstantWeightStrategy()
        result = run_strategy(strat, prices, train_frac=0.6)
        assert result.train_start != ""
        assert result.test_start != ""
        # Test start should be after train end
        assert result.test_start >= result.train_end

    def test_metrics_populated(self):
        prices = _make_prices(504, 3)
        strat = ConstantWeightStrategy()
        result = run_strategy(strat, prices)
        assert result.metrics_oos is not None
        assert result.metrics_oos.n_observations > 0
        assert result.metrics_oos.sharpe_ratio != 0  # synthetic data has drift

    def test_equity_curve_exists(self):
        prices = _make_prices(504, 3)
        strat = ConstantWeightStrategy()
        result = run_strategy(strat, prices)
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0

    def test_trade_log_populated(self):
        prices = _make_prices(504, 3)
        strat = ConstantWeightStrategy()
        result = run_strategy(strat, prices)
        # Constant weight has minimal trades (only initial allocation)
        assert isinstance(result.trade_log, list)

    def test_failing_strategy_rejected(self):
        prices = _make_prices(504, 3)
        strat = FailingStrategy()
        result = run_strategy(strat, prices)
        assert result.status == "rejected"
        assert "failed" in result.rejection_reason.lower()

    def test_empty_strategy_rejected(self):
        prices = _make_prices(504, 3)
        strat = EmptyStrategy()
        result = run_strategy(strat, prices)
        assert result.status == "rejected"

    def test_insufficient_data_rejected(self):
        prices = _make_prices(50, 3)  # too short
        strat = ConstantWeightStrategy()
        result = run_strategy(strat, prices)
        assert result.status == "rejected"
        assert "Insufficient" in result.rejection_reason

    def test_summary_dict(self):
        prices = _make_prices(504, 3)
        strat = ConstantWeightStrategy()
        result = run_strategy(strat, prices)
        summary = result.to_summary_dict()
        assert "strategy_name" in summary
        assert "sharpe" in summary
        assert "dsr" in summary
        assert "status" in summary

    def test_n_trials_affects_dsr(self):
        prices = _make_prices(504, 3)
        strat = ConstantWeightStrategy()
        r1 = run_strategy(strat, prices, n_trials=1)
        r100 = run_strategy(strat, prices, n_trials=100)
        # More trials → lower DSR (more stringent)
        if r1.metrics_oos and r100.metrics_oos:
            assert r100.metrics_oos.deflated_sharpe <= r1.metrics_oos.deflated_sharpe


class TestRunBatch:
    """Test batch runner."""

    def test_runs_multiple_strategies(self):
        prices = _make_prices(504, 3)
        strategies = [ConstantWeightStrategy(), ConstantWeightStrategy()]
        results = run_batch(strategies, prices)
        assert len(results) == 2

    def test_n_trials_increments(self):
        prices = _make_prices(504, 3)
        strategies = [ConstantWeightStrategy(), ConstantWeightStrategy()]
        results = run_batch(strategies, prices)
        assert results[0].n_trials == 1
        assert results[1].n_trials == 2


# --------------------------------------------------------------------------- #
# Strategy-specific tests                                                      #
# --------------------------------------------------------------------------- #


class TestCrossSectionalMomentum:
    """Test CrossSectionalMomentum signal generation."""

    def test_signal_shape(self):
        prices = _make_prices(300, 5)
        strat = CrossSectionalMomentum(lookback=60, top_k=2)
        signals = strat.generate_signals(prices)
        assert isinstance(signals, pd.DataFrame)
        # Should drop lookback warmup
        assert len(signals) == len(prices) - 60

    def test_weight_sum_leq_one(self):
        prices = _make_prices(300, 5)
        strat = CrossSectionalMomentum(lookback=60, top_k=2)
        signals = strat.generate_signals(prices)
        row_sums = signals.sum(axis=1)
        assert (row_sums <= 1.0 + 1e-10).all()

    def test_nonneg_weights(self):
        prices = _make_prices(300, 5)
        strat = CrossSectionalMomentum(lookback=60, top_k=2)
        signals = strat.generate_signals(prices)
        assert (signals >= -1e-10).all().all()

    def test_top_k_respected(self):
        prices = _make_prices(300, 5)
        strat = CrossSectionalMomentum(lookback=60, top_k=2)
        signals = strat.generate_signals(prices)
        n_nonzero = (signals > 1e-10).sum(axis=1)
        assert (n_nonzero <= 2).all()

    def test_deterministic(self):
        prices = _make_prices(300, 5)
        strat = CrossSectionalMomentum(lookback=60, top_k=2)
        s1 = strat.generate_signals(prices)
        s2 = strat.generate_signals(prices)
        pd.testing.assert_frame_equal(s1, s2)

    def test_meta_correct(self):
        strat = CrossSectionalMomentum(lookback=120, top_k=3)
        assert strat.category == "momentum"
        assert strat.parameters["lookback"] == 120
        assert strat.parameters["top_k"] == 3


class TestTimeSeriesMomentum:
    """Test TimeSeriesMomentum signal generation."""

    def test_signal_shape(self):
        prices = _make_prices(300, 5)
        strat = TimeSeriesMomentum(lookback=60)
        signals = strat.generate_signals(prices)
        assert len(signals) == len(prices) - 60

    def test_weight_sum_leq_one(self):
        prices = _make_prices(300, 5)
        strat = TimeSeriesMomentum(lookback=60)
        signals = strat.generate_signals(prices)
        assert (signals.sum(axis=1) <= 1.0 + 1e-10).all()

    def test_nonneg_weights(self):
        prices = _make_prices(300, 5)
        strat = TimeSeriesMomentum(lookback=60)
        signals = strat.generate_signals(prices)
        assert (signals >= -1e-10).all().all()

    def test_meta_correct(self):
        strat = TimeSeriesMomentum(lookback=120)
        assert strat.category == "momentum"
        assert strat.parameters["lookback"] == 120


class TestRSIMeanReversion:
    """Test RSIMeanReversion signal generation."""

    def test_signal_shape(self):
        prices = _make_prices(300, 5)
        strat = RSIMeanReversion(rsi_period=14)
        signals = strat.generate_signals(prices)
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) > 0

    def test_weight_sum_leq_one(self):
        prices = _make_prices(300, 5)
        strat = RSIMeanReversion(rsi_period=14)
        signals = strat.generate_signals(prices)
        assert (signals.sum(axis=1) <= 1.0 + 1e-10).all()

    def test_nonneg_weights(self):
        prices = _make_prices(300, 5)
        strat = RSIMeanReversion(rsi_period=14)
        signals = strat.generate_signals(prices)
        assert (signals >= -1e-10).all().all()

    def test_meta_correct(self):
        strat = RSIMeanReversion(rsi_period=14, oversold=30, overbought=70)
        assert strat.category == "mean_reversion"
        assert strat.parameters["rsi_period"] == 14


class TestVolatilityTargeting:
    """Test VolatilityTargeting signal generation."""

    def test_signal_shape(self):
        prices = _make_prices(300, 5)
        strat = VolatilityTargeting(vol_lookback=60)
        signals = strat.generate_signals(prices)
        assert len(signals) == len(prices) - 60

    def test_weight_sum_leq_one(self):
        prices = _make_prices(300, 5)
        strat = VolatilityTargeting(vol_lookback=60)
        signals = strat.generate_signals(prices)
        assert (signals.sum(axis=1) <= 1.0 + 1e-10).all()

    def test_nonneg_weights(self):
        prices = _make_prices(300, 5)
        strat = VolatilityTargeting(vol_lookback=60)
        signals = strat.generate_signals(prices)
        assert (signals >= -1e-10).all().all()

    def test_meta_correct(self):
        strat = VolatilityTargeting(vol_lookback=60, target_vol=0.15)
        assert strat.category == "volatility"
        assert strat.parameters["target_vol"] == 0.15


class TestDualMomentum:
    """Test DualMomentum signal generation."""

    def test_signal_shape(self):
        prices = _make_prices_with_tlt(300)
        strat = DualMomentum(lookback=60, top_k=2)
        signals = strat.generate_signals(prices)
        assert len(signals) == len(prices) - 60

    def test_weight_sum_leq_one(self):
        prices = _make_prices_with_tlt(300)
        strat = DualMomentum(lookback=60, top_k=2)
        signals = strat.generate_signals(prices)
        assert (signals.sum(axis=1) <= 1.0 + 1e-10).all()

    def test_nonneg_weights(self):
        prices = _make_prices_with_tlt(300)
        strat = DualMomentum(lookback=60, top_k=2)
        signals = strat.generate_signals(prices)
        assert (signals >= -1e-10).all().all()

    def test_risk_off_uses_tlt(self):
        """If all assets have negative trailing returns, should go to TLT."""
        # Create prices where everything trends down except TLT
        rng = np.random.default_rng(999)
        dates = pd.bdate_range("2020-01-01", periods=300)
        prices = pd.DataFrame({
            "AVUV": 100.0 * np.exp(np.cumsum(rng.normal(-0.003, 0.02, 300))),
            "MTUM": 100.0 * np.exp(np.cumsum(rng.normal(-0.003, 0.02, 300))),
            "TLT":  100.0 * np.exp(np.cumsum(rng.normal(-0.001, 0.005, 300))),
        }, index=dates)

        strat = DualMomentum(lookback=60, top_k=2, risk_off_asset="TLT")
        signals = strat.generate_signals(prices)

        # On rebalance dates where everything is negative, TLT should be 1.0
        has_tlt_only = signals[(signals["TLT"] > 0.99)]
        assert len(has_tlt_only) > 0

    def test_meta_correct(self):
        strat = DualMomentum(lookback=120, top_k=3)
        assert strat.category == "hybrid"
        assert strat.parameters["risk_off_asset"] == "TLT"


# --------------------------------------------------------------------------- #
# Report tests                                                                 #
# --------------------------------------------------------------------------- #


class TestReport:
    """Test report generation."""

    def test_save_strategy_result(self, tmp_path):
        prices = _make_prices(504, 3)
        strat = ConstantWeightStrategy()
        result = run_strategy(strat, prices)
        out_dir = save_strategy_result(result, output_dir=tmp_path / "test_strat")
        assert (out_dir / "summary.json").exists()
        # Check JSON is valid
        with open(out_dir / "summary.json") as f:
            data = json.load(f)
        assert data["strategy_name"] == "ConstantWeight"

    def test_save_equity_curve(self, tmp_path):
        prices = _make_prices(504, 3)
        strat = ConstantWeightStrategy()
        result = run_strategy(strat, prices)
        out_dir = save_strategy_result(result, output_dir=tmp_path / "test_eq")
        assert (out_dir / "equity_curve.csv").exists()
        eq = pd.read_csv(out_dir / "equity_curve.csv")
        assert "date" in eq.columns
        assert "equity" in eq.columns

    def test_save_batch_results(self, tmp_path):
        prices = _make_prices(504, 3)
        strategies = [ConstantWeightStrategy()]
        results = run_batch(strategies, prices)

        import inversiones_mama.exploration.report as report_mod
        original = report_mod.EXPLORATION_RESULTS_DIR
        report_mod.EXPLORATION_RESULTS_DIR = tmp_path
        try:
            batch_dir = save_batch_results(results, batch_name="test_batch")
            assert (batch_dir / "batch_summary.csv").exists()
            assert (batch_dir / "batch_summary.json").exists()
        finally:
            report_mod.EXPLORATION_RESULTS_DIR = original

    def test_format_results_table(self):
        prices = _make_prices(504, 3)
        strategies = [ConstantWeightStrategy()]
        results = run_batch(strategies, prices)
        table = format_results_table(results)
        assert "ConstantWeight" in table
        assert "Strategy" in table  # header


# --------------------------------------------------------------------------- #
# Integration tests                                                            #
# --------------------------------------------------------------------------- #


class TestIntegration:
    """Full pipeline integration tests."""

    def test_momentum_xsec_full_pipeline(self):
        prices = _make_prices(504, 5)
        strat = CrossSectionalMomentum(lookback=60, top_k=2)
        result = run_strategy(strat, prices, n_trials=1)
        assert result.metrics_oos is not None
        assert result.equity_curve is not None
        assert result.strategy_name.startswith("XSMomentum")

    def test_all_strategies_run_cleanly(self):
        prices = _make_prices_with_tlt(504)
        strategies = [
            CrossSectionalMomentum(lookback=60, top_k=2),
            TimeSeriesMomentum(lookback=60),
            RSIMeanReversion(rsi_period=14),
            VolatilityTargeting(vol_lookback=60),
            DualMomentum(lookback=60, top_k=2),
        ]
        results = run_batch(strategies, prices)
        assert len(results) == 5
        for r in results:
            assert r.metrics_oos is not None or r.status == "rejected"

    def test_batch_summary_format(self):
        prices = _make_prices_with_tlt(504)
        strategies = [
            CrossSectionalMomentum(lookback=60, top_k=2),
            TimeSeriesMomentum(lookback=60),
        ]
        results = run_batch(strategies, prices)
        table = format_results_table(results)
        assert "momentum" in table.lower()


# --------------------------------------------------------------------------- #
# Hard constraint: isolation verification                                      #
# --------------------------------------------------------------------------- #


class TestIsolation:
    """Verify the exploration package is isolated from production."""

    def test_no_sizing_imports(self):
        """Exploration MUST NOT import from sizing."""
        import inversiones_mama.exploration.runner as runner
        import inversiones_mama.exploration.report as report
        import inversiones_mama.exploration.base as base

        source_code = ""
        for mod in [runner, report, base]:
            import inspect
            source_code += inspect.getsource(mod)

        assert "from inversiones_mama.sizing" not in source_code
        assert "from ..sizing" not in source_code
        assert "import inversiones_mama.sizing" not in source_code

    def test_no_execution_imports(self):
        """Exploration MUST NOT import from execution."""
        import inversiones_mama.exploration.runner as runner
        import inversiones_mama.exploration.report as report
        import inversiones_mama.exploration.base as base

        source_code = ""
        for mod in [runner, report, base]:
            import inspect
            source_code += inspect.getsource(mod)

        assert "from inversiones_mama.execution" not in source_code
        assert "from ..execution" not in source_code
        assert "import inversiones_mama.execution" not in source_code

    def test_no_validation_imports(self):
        """Exploration MUST NOT import from validation."""
        import inversiones_mama.exploration.runner as runner
        import inversiones_mama.exploration.report as report
        import inversiones_mama.exploration.base as base

        source_code = ""
        for mod in [runner, report, base]:
            import inspect
            source_code += inspect.getsource(mod)

        assert "from inversiones_mama.validation" not in source_code
        assert "from ..validation" not in source_code
        assert "import inversiones_mama.validation" not in source_code
