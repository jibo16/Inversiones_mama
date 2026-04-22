"""Plotly chart helpers for the Streamlit dashboard.

Each function returns a ``plotly.graph_objects.Figure`` and is pure —
no Streamlit state, no I/O — so the same chart code can be reused in
static exports later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# --------------------------------------------------------------------------- #
# Backtest / wealth curves                                                    #
# --------------------------------------------------------------------------- #


def equity_curve_chart(
    wealth: pd.Series,
    title: str = "Portfolio wealth",
    rebalance_dates: list[pd.Timestamp] | None = None,
) -> go.Figure:
    """Line chart of wealth over time with optional rebalance markers."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wealth.index,
            y=wealth.values,
            mode="lines",
            name="Wealth",
            line={"color": "#1f77b4", "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    if rebalance_dates:
        rebal_values = [float(wealth.asof(d)) for d in rebalance_dates if d in wealth.index or True]
        rebal_x = list(rebalance_dates)
        fig.add_trace(
            go.Scatter(
                x=rebal_x,
                y=rebal_values,
                mode="markers",
                name="Rebalance",
                marker={"size": 6, "color": "#ff7f0e", "symbol": "diamond"},
                hovertemplate="Rebalance<br>%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Wealth ($)",
        hovermode="x unified",
        template="plotly_white",
        height=420,
    )
    return fig


def drawdown_chart(wealth: pd.Series, title: str = "Drawdown") -> go.Figure:
    """Underwater chart — shaded region of peak-to-current drawdown (%)."""
    cum_max = wealth.cummax()
    dd = (wealth / cum_max - 1.0) * 100.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line={"color": "#d62728", "width": 1},
            fillcolor="rgba(214, 39, 40, 0.25)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis={"ticksuffix": "%"},
        template="plotly_white",
        height=260,
    )
    return fig


# --------------------------------------------------------------------------- #
# Monte Carlo                                                                 #
# --------------------------------------------------------------------------- #


def mc_terminal_histogram(
    terminal_wealth: np.ndarray,
    initial_capital: float,
    title: str = "Terminal wealth (1-year, MC)",
) -> go.Figure:
    """Histogram of terminal wealth with loss-threshold reference lines."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=terminal_wealth,
            nbinsx=60,
            name="Terminal wealth",
            marker={"color": "#1f77b4"},
            hovertemplate="$%{x:,.0f}<br>count %{y}<extra></extra>",
        )
    )
    # Reference lines at common loss thresholds
    for pct, label, color in [
        (1.0, f"Initial ${initial_capital:,.0f}", "#2ca02c"),
        (0.6, f"-40% (${initial_capital*0.6:,.0f})", "#ff7f0e"),
        (0.4, f"-60% (${initial_capital*0.4:,.0f})", "#d62728"),
    ]:
        fig.add_vline(
            x=initial_capital * pct,
            line_width=2,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top",
        )
    fig.update_layout(
        title=title,
        xaxis_title="Terminal wealth ($)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=420,
        bargap=0.03,
    )
    return fig


def mc_drawdown_histogram(max_drawdowns: np.ndarray, title: str = "Max drawdown distribution") -> go.Figure:
    """Histogram of max drawdowns (fractions)."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=max_drawdowns * 100.0,
            nbinsx=50,
            marker={"color": "#d62728"},
            name="Max DD",
            hovertemplate="%{x:.1f}%<br>count %{y}<extra></extra>",
        )
    )
    # 95th and 99th percentile markers
    p95 = float(np.percentile(max_drawdowns, 95)) * 100.0
    p99 = float(np.percentile(max_drawdowns, 99)) * 100.0
    for pct, label, color in [
        (p95, f"p95 = {p95:.1f}%", "#ff7f0e"),
        (p99, f"p99 = {p99:.1f}%", "#8c564b"),
    ]:
        fig.add_vline(
            x=pct,
            line_width=2,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top",
        )
    fig.update_layout(
        title=title,
        xaxis_title="Max drawdown (%)",
        yaxis_title="Frequency",
        xaxis={"ticksuffix": "%"},
        template="plotly_white",
        height=420,
    )
    return fig


def mc_fan_chart(
    wealth_paths: np.ndarray,
    initial_capital: float,
    percentiles: tuple[int, ...] = (5, 25, 50, 75, 95),
    title: str = "Simulated wealth paths (percentile fan)",
) -> go.Figure:
    """Fan chart: percentile bands of wealth over simulation horizon.

    ``wealth_paths`` shape is ``(n_paths, horizon_days + 1)`` where column 0 is the
    starting capital.
    """
    n_paths, horizon_plus_1 = wealth_paths.shape
    days = np.arange(horizon_plus_1)
    pct_arr = np.percentile(wealth_paths, percentiles, axis=0)

    fig = go.Figure()
    # Outer band (5-95)
    if 5 in percentiles and 95 in percentiles:
        lo = pct_arr[percentiles.index(5)]
        hi = pct_arr[percentiles.index(95)]
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([days, days[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.15)",
                line={"color": "rgba(0,0,0,0)"},
                name="5-95%",
                hoverinfo="skip",
            )
        )
    # Inner band (25-75)
    if 25 in percentiles and 75 in percentiles:
        lo = pct_arr[percentiles.index(25)]
        hi = pct_arr[percentiles.index(75)]
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([days, days[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.35)",
                line={"color": "rgba(0,0,0,0)"},
                name="25-75%",
                hoverinfo="skip",
            )
        )
    # Median line
    if 50 in percentiles:
        fig.add_trace(
            go.Scatter(
                x=days,
                y=pct_arr[percentiles.index(50)],
                mode="lines",
                line={"color": "#1f77b4", "width": 2.5},
                name="Median",
            )
        )
    fig.add_hline(
        y=initial_capital,
        line_dash="dot",
        line_color="#2ca02c",
        annotation_text=f"Initial ${initial_capital:,.0f}",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Days from now",
        yaxis_title="Wealth ($)",
        template="plotly_white",
        height=420,
        hovermode="x unified",
    )
    return fig


# --------------------------------------------------------------------------- #
# Target weights / portfolio composition                                      #
# --------------------------------------------------------------------------- #


def weights_pie(weights: pd.Series, cash_weight: float | None = None, title: str = "Target weights") -> go.Figure:
    """Donut chart of non-zero target weights (plus cash slice if provided)."""
    nonzero = weights[weights > 1e-4].sort_values(ascending=False)
    labels = list(nonzero.index)
    values = list(nonzero.values)
    if cash_weight is None:
        cash_weight = max(0.0, 1.0 - float(nonzero.sum()))
    if cash_weight > 1e-6:
        labels.append("CASH")
        values.append(cash_weight)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                textinfo="label+percent",
                hovertemplate="%{label}<br>%{value:.2%}<extra></extra>",
            )
        ]
    )
    fig.update_layout(title=title, template="plotly_white", height=420)
    return fig


# --------------------------------------------------------------------------- #
# Correlation / factor loadings                                               #
# --------------------------------------------------------------------------- #


def correlation_heatmap(corr: pd.DataFrame, title: str = "Pairwise correlation") -> go.Figure:
    """Heatmap of a correlation matrix."""
    fig = px.imshow(
        corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(title=title, template="plotly_white", height=500)
    return fig


def factor_loadings_heatmap(betas: pd.DataFrame, title: str = "Factor loadings (beta)") -> go.Figure:
    """Heatmap of factor betas (rows=tickers, cols=factors)."""
    fig = px.imshow(
        betas.values,
        x=list(betas.columns),
        y=list(betas.index),
        color_continuous_scale="RdBu",
        zmin=-1.5,
        zmax=1.5,
        aspect="auto",
    )
    fig.update_layout(title=title, template="plotly_white", height=400)
    return fig


def composite_mu_bars(mu: pd.Series, title: str = "Composite annualized mu") -> go.Figure:
    """Bar chart of composite expected returns per ticker (annualized)."""
    s = mu.sort_values(ascending=True)
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in s.values]
    fig = go.Figure(
        go.Bar(
            x=s.values,
            y=list(s.index),
            orientation="h",
            marker={"color": colors},
            hovertemplate="%{y}: %{x:+.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="mu (annualized)",
        yaxis_title="Ticker",
        xaxis={"tickformat": ".1%"},
        template="plotly_white",
        height=400,
    )
    return fig


# --------------------------------------------------------------------------- #
# Trade log / execution                                                       #
# --------------------------------------------------------------------------- #


def slippage_histogram(slippage_bps: list[float], title: str = "Slippage distribution (bps)") -> go.Figure:
    """Histogram of slippage in basis points."""
    if not slippage_bps:
        fig = go.Figure()
        fig.update_layout(
            title=title + " — no filled orders yet",
            template="plotly_white",
            height=320,
            annotations=[
                {"text": "No fills to plot", "xref": "paper", "yref": "paper",
                 "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 14}}
            ],
        )
        return fig
    fig = go.Figure(
        go.Histogram(
            x=slippage_bps,
            nbinsx=30,
            marker={"color": "#9467bd"},
            hovertemplate="%{x:.1f} bps<br>count %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title=title,
        xaxis_title="Slippage (bps)",
        yaxis_title="Count",
        template="plotly_white",
        height=320,
    )
    return fig


def fill_rate_gauge(fill_rate: float, title: str = "Fill rate") -> go.Figure:
    """Gauge chart for fill rate in [0, 1]."""
    pct = float(max(0.0, min(1.0, fill_rate))) * 100.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ca02c" if pct >= 95 else ("#ff7f0e" if pct >= 80 else "#d62728")},
                "steps": [
                    {"range": [0, 80], "color": "#fadbd8"},
                    {"range": [80, 95], "color": "#fef5e7"},
                    {"range": [95, 100], "color": "#d5f5e3"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": 95},
            },
            title={"text": title},
        )
    )
    fig.update_layout(template="plotly_white", height=260)
    return fig
