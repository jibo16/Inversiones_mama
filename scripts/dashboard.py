"""Inversiones_mama local dashboard — run with ``streamlit``.

Launch:
    .venv\\Scripts\\python.exe -m streamlit run scripts\\dashboard.py \\
        --server.address=127.0.0.1 --server.port=8501

Then open http://127.0.0.1:8501 in your browser.

Design principles:
* Localhost-only (no public binding).
* Read from cached artifacts where possible; trigger live pipelines only
  when the user clicks a "Run now" button.
* Every chart is a pure ``charts.*`` function — the Streamlit wrapper is
  intentionally thin so the visualization logic stays testable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the src package importable when Streamlit launches this script directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from inversiones_mama.config import GATES, KELLY_FRACTION, MAX_WEIGHT_PER_NAME, UNIVERSE
from inversiones_mama.dashboard import charts, data_sources


st.set_page_config(
    page_title="Inversiones_mama",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached heavy work                                                           #
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner="Running walk-forward backtest (first load)...")
def _cached_walk_forward(years: float):
    """Cache the backtest so we only run it once per session per years value."""
    return data_sources.run_walk_forward_now(years=years)


@st.cache_data(show_spinner="Loading prices + factors...")
def _cached_prices_factors(years: float):
    return data_sources.fetch_prices_and_factors(years=years)


@st.cache_data(show_spinner="Running Monte Carlo...")
def _cached_mc(n_paths: int, horizon_days: int, initial_capital: float, years: float, seed: int):
    """Cache MC so re-rendering the tab doesn't re-run it."""
    return data_sources.run_mc_now(
        n_paths=n_paths,
        horizon_days=horizon_days,
        initial_capital=initial_capital,
        years=years,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Sidebar                                                                     #
# --------------------------------------------------------------------------- #


with st.sidebar:
    st.title("Inversiones_mama")
    st.caption("v1a paper-deployment dashboard")
    st.markdown(
        f"""
        **Universe:** {len(UNIVERSE)} ETFs
        **Kelly fraction:** {KELLY_FRACTION}
        **Per-name cap:** {MAX_WEIGHT_PER_NAME:.0%}
        """
    )
    st.divider()
    st.subheader("Data status")
    artifacts = data_sources.available_artifacts()
    for name, exists in artifacts.items():
        st.write(f"{'✓' if exists else '—'} `{name}`")

    st.divider()
    st.subheader("Settings")
    backtest_years = st.slider("Backtest window (years)", 1.0, 5.0, 5.0, step=0.5)
    if st.button("🔄 Clear cache & refresh all"):
        st.cache_data.clear()
        st.rerun()


# --------------------------------------------------------------------------- #
# Main tabs                                                                   #
# --------------------------------------------------------------------------- #


tab_verdict, tab_mc, tab_paper, tab_factors, tab_weights = st.tabs(
    ["🛡️ v1a Verdict", "🎲 Monte Carlo", "📝 Paper trades", "📊 Factor analysis", "🥧 Target weights"]
)


# ---- Tab 1: v1a Verdict ----------------------------------------------------


with tab_verdict:
    st.header("v1a strategy verdict")
    st.caption(
        "Live walk-forward backtest on the 10-ETF universe. Every rebalance "
        "refits the 6-factor model and solves Risk-Constrained Kelly (0.65)."
    )

    try:
        result = _cached_walk_forward(backtest_years)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Backtest failed: {exc}")
        st.stop()

    final_wealth = result.final_wealth
    n_rebal = len(result.rebalance_records)
    ann_cost = result.annualized_turnover_cost

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Capital deployed", f"${result.config.initial_capital:,.0f}")
    col2.metric("Final wealth", f"${final_wealth:,.2f}",
                delta=f"{(final_wealth / result.config.initial_capital - 1) * 100:+.1f}%")
    col3.metric("Rebalances", f"{n_rebal}")
    col4.metric("Annual turnover cost", f"{ann_cost * 100:.2f}%",
                delta="✓ within gate" if ann_cost < GATES.max_annual_turnover_cost else "⚠ over gate",
                delta_color="normal" if ann_cost < GATES.max_annual_turnover_cost else "inverse")

    st.subheader("Equity curve")
    rebal_dates = [r.date for r in result.rebalance_records]
    st.plotly_chart(
        charts.equity_curve_chart(result.wealth, rebalance_dates=rebal_dates),
        use_container_width=True,
    )

    st.subheader("Drawdown")
    st.plotly_chart(charts.drawdown_chart(result.wealth), use_container_width=True)

    with st.expander("Raw text verdict (from `scripts/run_v1a_verdict.py`)"):
        text = data_sources.load_verdict_text()
        if text:
            st.code(text)
        else:
            st.info(
                "No cached verdict text found. Run "
                "`.venv\\Scripts\\python.exe scripts\\run_v1a_verdict.py` to generate."
            )


# ---- Tab 2: Monte Carlo ----------------------------------------------------


with tab_mc:
    st.header("Monte Carlo stress test")
    st.caption(
        "Block-bootstrap the full price history, apply the most recent "
        "RCK-solved weights, and validate the drawdown bound empirically."
    )

    colA, colB, colC, colD = st.columns(4)
    n_paths = colA.slider("Paths", 500, 20_000, 5_000, step=500)
    horizon = colB.slider("Horizon (days)", 30, 504, 252, step=30)
    init_cap = colC.number_input("Initial capital", value=5000.0, min_value=100.0)
    seed = colD.number_input("Random seed", value=20260422, step=1)

    if st.button("▶️ Run Monte Carlo", type="primary"):
        mc = _cached_mc(int(n_paths), int(horizon), float(init_cap), backtest_years, int(seed))
        st.session_state["_mc_last"] = mc

    mc = st.session_state.get("_mc_last")
    if mc is None:
        st.info("Click **Run Monte Carlo** to generate a simulation.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Median terminal", f"${mc.terminal_median:,.0f}")
        col2.metric("5th percentile", f"${mc.terminal_p05:,.0f}")
        col3.metric("95th percentile", f"${mc.terminal_p95:,.0f}")
        col4.metric("95th-pct max DD", f"{mc.dd_p95 * 100:.1f}%",
                    delta="✓" if mc.gate_dd_95th_pass else "⚠",
                    delta_color="normal" if mc.gate_dd_95th_pass else "inverse")

        st.subheader("Terminal wealth distribution")
        st.plotly_chart(
            charts.mc_terminal_histogram(mc.terminal_wealth, mc.initial_capital),
            use_container_width=True,
        )

        st.subheader("Maximum drawdown distribution")
        st.plotly_chart(charts.mc_drawdown_histogram(mc.max_drawdowns), use_container_width=True)

        st.subheader("Gate verdicts")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Gate": "P(loss > 40%)",
                        "Observed": f"{mc.prob_loss_40pct * 100:.2f}%",
                        "Threshold": f"{GATES.max_prob_loss_40pct * 100:.1f}%",
                        "Pass": "✓" if mc.gate_prob_loss_40pct_pass else "✗",
                    },
                    {
                        "Gate": "P(loss > 60%)",
                        "Observed": f"{mc.prob_loss_60pct * 100:.2f}%",
                        "Threshold": f"{GATES.max_prob_loss_60pct * 100:.1f}%",
                        "Pass": "✓" if mc.gate_prob_loss_60pct_pass else "✗",
                    },
                    {
                        "Gate": "95th-pct max DD",
                        "Observed": f"{mc.dd_p95 * 100:.2f}%",
                        "Threshold": f"{GATES.max_dd_95th_pct * 100:.1f}%",
                        "Pass": "✓" if mc.gate_dd_95th_pass else "✗",
                    },
                    {
                        "Gate": f"RCK bound (alpha={mc.rck_alpha})",
                        "Observed": f"{mc.prob_dd_exceeds_rck_alpha * 100:.2f}%",
                        "Threshold": f"{(mc.rck_beta + 0.02) * 100:.1f}%",
                        "Pass": "✓" if mc.gate_rck_bound_pass else "✗",
                    },
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


# ---- Tab 3: Paper trades ---------------------------------------------------


with tab_paper:
    st.header("Paper-trading execution log")

    paper_summary = data_sources.load_paper_summary()
    paper_log = data_sources.load_paper_trade_log()

    if paper_summary is None and paper_log is None:
        st.info(
            "No paper-trading log yet. Run "
            "`.venv\\Scripts\\python.exe scripts\\run_paper_rebalance.py` to populate."
        )
    else:
        if paper_summary:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Orders placed", paper_summary.get("order_count", "—"))
            col2.metric("Total fill value", f"${paper_summary.get('total_fill_value', 0):,.2f}")
            col3.metric("Est. cost", f"${paper_summary.get('estimated_cost', 0):.2f}")
            col4.metric("Fill rate", f"{paper_summary.get('fill_rate', 0) * 100:.1f}%")

        if paper_log and len(paper_log) > 0:
            col_g, col_s = st.columns([1, 2])
            with col_g:
                st.plotly_chart(
                    charts.fill_rate_gauge(paper_log.summary()["fill_rate"]),
                    use_container_width=True,
                )
            with col_s:
                slip = [
                    e.slippage_bps for e in paper_log
                    if e.slippage_bps is not None
                ]
                st.plotly_chart(charts.slippage_histogram(slip), use_container_width=True)

            st.subheader("Trade log")
            st.dataframe(paper_log.to_frame(), use_container_width=True, hide_index=True)

            st.subheader("Execution stats")
            st.json(paper_log.summary())


# ---- Tab 4: Factor analysis -------------------------------------------------


with tab_factors:
    st.header("Factor analysis")

    # Prefer the cached alpha-pipeline CSV; fall back to live
    alpha_df = data_sources.load_alpha_pipeline_summary()
    corr_df = data_sources.load_universe_corr()
    stats_df = data_sources.load_universe_stats()

    if alpha_df is None:
        st.info(
            "No factor-analysis cache found. Run "
            "`.venv\\Scripts\\python.exe scripts\\demo_alpha_pipeline.py` to populate."
        )
    else:
        beta_cols = [c for c in alpha_df.columns if c.startswith("b_")]
        if beta_cols:
            betas = alpha_df[beta_cols].copy()
            betas.columns = [c.removeprefix("b_") for c in betas.columns]
            st.subheader("Factor loadings")
            st.plotly_chart(
                charts.factor_loadings_heatmap(betas),
                use_container_width=True,
            )

        if "mu_ann_252d" in alpha_df.columns:
            st.subheader("Composite μ (annualized, 252-day lookback)")
            st.plotly_chart(
                charts.composite_mu_bars(alpha_df["mu_ann_252d"]),
                use_container_width=True,
            )

    if corr_df is not None:
        st.subheader("Pairwise correlation matrix")
        st.plotly_chart(charts.correlation_heatmap(corr_df), use_container_width=True)

    if stats_df is not None:
        st.subheader("Universe statistics (5-year trailing)")
        st.dataframe(stats_df, use_container_width=True)


# ---- Tab 5: Target weights -------------------------------------------------


with tab_weights:
    st.header("Current target weights")
    st.caption("Most recent walk-forward rebalance + RCK-solved allocation.")

    try:
        result = _cached_walk_forward(backtest_years)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not compute target weights: {exc}")
    else:
        if not result.rebalance_records:
            st.info("No rebalance events in the backtest. Extend the window.")
        else:
            last = result.rebalance_records[-1]
            cash_w = max(0.0, 1.0 - float(last.target_weights.sum()))
            col_pie, col_meta = st.columns([2, 1])
            with col_pie:
                st.plotly_chart(
                    charts.weights_pie(last.target_weights, cash_weight=cash_w),
                    use_container_width=True,
                )
            with col_meta:
                st.metric("Rebalance date", str(last.date.date()))
                st.metric("Cash allocation", f"{cash_w * 100:.1f}%")
                st.metric("Kelly growth rate (ann)", f"{last.kelly_growth_rate * 252 * 100:+.2f}%")
                st.metric("Solver status", last.kelly_status)
            st.subheader("Weights table")
            weights_tbl = last.target_weights.to_frame(name="Weight").assign(
                Name=[UNIVERSE.get(t, t) for t in last.target_weights.index],
                Dollars=last.target_weights.values * result.final_wealth,
            )[["Name", "Weight", "Dollars"]]
            weights_tbl["Weight"] = (weights_tbl["Weight"] * 100).round(2).astype(str) + "%"
            weights_tbl["Dollars"] = weights_tbl["Dollars"].round(2).map(lambda v: f"${v:,.2f}")
            st.dataframe(weights_tbl, use_container_width=True)


# --------------------------------------------------------------------------- #
# Footer                                                                      #
# --------------------------------------------------------------------------- #


st.divider()
st.caption(
    "Inversiones_mama v1a · zero-budget deployment dashboard · "
    "built with Streamlit + Plotly · localhost-only, no public binding"
)
