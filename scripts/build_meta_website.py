"""Build the self-contained HTML results website.

Takes the outputs of run_meta_backtest.py, run_meta_bootstrap.py, and
run_meta_regime.py, renders them as a single HTML site under
``results/meta_website/`` that opens in any browser without a server.

Contents
--------
  index.html                  overview + leaderboard + nav
  strategies/<id>.html        per-strategy drill-down
  figures/                    all PNG charts (matplotlib)
  style.css                   minimal stylesheet
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path("results")
BACKTEST = BASE_DIR / "meta_backtest"
BOOTSTRAP = BASE_DIR / "meta_bootstrap"
REGIME = BASE_DIR / "meta_regime"
OUT = BASE_DIR / "meta_website"
FIGS = OUT / "figures"


# --- Minimal CSS -----------------------------------------------------------


CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0; padding: 0;
  background: #fafafa; color: #1a1a1a;
  line-height: 1.55;
}
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
header {
  background: linear-gradient(135deg, #0b4f6c 0%, #01baef 100%);
  color: white; padding: 32px 24px; margin-bottom: 24px;
}
header h1 { margin: 0; font-size: 2rem; }
header .sub { opacity: 0.85; margin-top: 4px; font-size: 0.95rem; }
nav { background: white; padding: 12px 24px; border-bottom: 1px solid #ddd;
      position: sticky; top: 0; z-index: 10; }
nav a { margin-right: 20px; color: #0b4f6c; text-decoration: none; font-weight: 500; }
nav a:hover { text-decoration: underline; }
h2 { color: #0b4f6c; border-bottom: 2px solid #01baef; padding-bottom: 4px;
     margin-top: 32px; }
h3 { color: #333; margin-top: 24px; }
.card { background: white; padding: 20px; border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 20px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 16px; margin: 16px 0; }
.metric {
  background: white; padding: 16px; border-radius: 8px; border-left: 4px solid #01baef;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.metric .label { font-size: 0.75rem; text-transform: uppercase;
                 letter-spacing: 0.5px; color: #666; margin-bottom: 4px; }
.metric .value { font-size: 1.4rem; font-weight: 600; color: #0b4f6c; }
.metric .sub { font-size: 0.85rem; color: #888; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; margin: 12px 0;
        font-size: 0.9rem; }
table th { background: #0b4f6c; color: white; padding: 8px 12px;
           text-align: left; font-weight: 500; }
table td { padding: 8px 12px; border-bottom: 1px solid #eee; }
table tr:hover { background: #f5faff; }
.pos { color: #0a7b27; font-weight: 500; }
.neg { color: #c0392b; font-weight: 500; }
.null { color: #888; font-style: italic; }
.dataframe { width: 100%; }
img.chart { max-width: 100%; height: auto; border: 1px solid #ddd;
            border-radius: 4px; margin: 8px 0; }
code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px;
       font-family: "SF Mono", Consolas, monospace; font-size: 0.85rem; }
.meta { font-size: 0.8rem; color: #888; margin-top: 32px;
        padding-top: 16px; border-top: 1px solid #ddd; }
.warn { background: #fff8dc; border-left: 4px solid #f39c12;
        padding: 12px 16px; margin: 16px 0; border-radius: 4px; }
.info { background: #eaf4fd; border-left: 4px solid #3498db;
        padding: 12px 16px; margin: 16px 0; border-radius: 4px; }
.rank-1 { background: #fffbeb; font-weight: 600; }
.rank-2 { background: #fafafa; }
.rank-3 { background: #fefaf4; }
"""


def _pct(x: float, decimals: int = 2) -> str:
    try:
        return f"{x*100:+.{decimals}f}%"
    except Exception:  # noqa: BLE001
        return "n/a"


def _num(x: float, decimals: int = 3) -> str:
    try:
        return f"{x:+.{decimals}f}"
    except Exception:  # noqa: BLE001
        return "n/a"


def _cls(x: float) -> str:
    try:
        return "pos" if x > 0 else ("neg" if x < 0 else "null")
    except Exception:  # noqa: BLE001
        return "null"


# --- Chart generators ------------------------------------------------------


def _plot_leaderboard_bar(summary: pd.DataFrame, metric: str, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    df = summary.dropna(subset=[metric]).sort_values(metric, ascending=True)
    colors = ["#c0392b" if v < 0 else "#0a7b27" for v in df[metric]]
    ax.barh(df["strategy_id"], df[metric], color=colors, alpha=0.8)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_bootstrap_fan(summary_boot: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    df = summary_boot.sort_values("terminal_return_median", ascending=True)
    y = np.arange(len(df))
    ax.hlines(y, df["terminal_return_p05"] * 100, df["terminal_return_p95"] * 100,
              colors="#01baef", linewidth=3, alpha=0.7)
    ax.plot(df["terminal_return_median"] * 100, y, "o", color="black", markersize=6)
    ax.set_yticks(y); ax.set_yticklabels(df["strategy_id"])
    ax.axvline(0, color="r", linestyle="--", alpha=0.6, label="zero")
    ax.set_xlabel("1y terminal return (%)  --  bar = [p05, p95],  dot = median")
    ax.set_title("Bootstrap forward simulation (1000 paths × 252 days, 20d block)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_regime_scatter(regime_df: pd.DataFrame, out_path: Path):
    """Scatter: LOW_VOL Sharpe (x) vs HIGH_VOL Sharpe (y), one dot per strategy."""
    fig, ax = plt.subplots(figsize=(10, 8))
    wide = regime_df.pivot_table(
        index="strategy_id", columns="regime_label",
        values="sharpe", aggfunc="first",
    )
    if "HIGH_VOL" not in wide.columns or "LOW_VOL" not in wide.columns:
        plt.close(fig)
        return
    wide = wide.dropna()
    ax.scatter(wide["LOW_VOL"], wide["HIGH_VOL"], s=60,
               alpha=0.8, color="#0b4f6c")
    for name, row in wide.iterrows():
        ax.annotate(name, (row["LOW_VOL"], row["HIGH_VOL"]),
                    fontsize=8, alpha=0.8, xytext=(4, 2), textcoords="offset points")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5,
               label="HIGH_VOL breakeven")
    ax.set_xlabel("LOW_VOL regime annualized Sharpe")
    ax.set_ylabel("HIGH_VOL regime annualized Sharpe")
    ax.set_title("Strategy performance by Markov regime\n"
                 "(right of x-axis: survives bull markets — above y-axis: survives stress)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_strategy_wealth(strategy_id: str, out_path: Path):
    """Equity curve for one strategy."""
    p = BACKTEST / "daily" / f"{strategy_id}.csv"
    if not p.exists():
        return False
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    if "wealth" not in df.columns:
        return False
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, df["wealth"], linewidth=1.5, color="#0b4f6c")
    ax.fill_between(df.index, df["wealth"], 1.0,
                    where=df["wealth"] >= 1.0, color="#0a7b27", alpha=0.15)
    ax.fill_between(df.index, df["wealth"], 1.0,
                    where=df["wealth"] < 1.0, color="#c0392b", alpha=0.15)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
    ax.set_title(f"{strategy_id} — equity curve ($1 base)")
    ax.set_ylabel("wealth")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return True


def _plot_strategy_drawdown(strategy_id: str, out_path: Path):
    p = BACKTEST / "daily" / f"{strategy_id}.csv"
    if not p.exists():
        return False
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    if "wealth" not in df.columns:
        return False
    w = df["wealth"]
    dd = (w / w.cummax() - 1.0) * 100
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(dd.index, dd.values, 0, color="#c0392b", alpha=0.4)
    ax.plot(dd.index, dd.values, color="#c0392b", linewidth=0.8)
    ax.set_title(f"{strategy_id} — drawdown (%)")
    ax.set_ylabel("drawdown")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return True


# --- Page builders ---------------------------------------------------------


HTML_BASE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="{css_rel}">
</head>
<body>
<header>
  <div class="container">
    <h1>{header_title}</h1>
    <div class="sub">{header_sub}</div>
  </div>
</header>
<nav><div class="container">
  <a href="{index_rel}">Overview</a>
  <a href="{index_rel}#leaderboard">Leaderboard</a>
  <a href="{index_rel}#bootstrap">Bootstrap</a>
  <a href="{index_rel}#regime">Regime</a>
  <a href="{index_rel}#cpcv">CPCV</a>
  <a href="{index_rel}#per-strategy">Strategies</a>
  <a href="{index_rel}#audit">Audit</a>
</div></nav>
<div class="container">
{body}
<div class="meta">Generated {generated_at} · Inversiones_mama · self-contained static site ·
Open <code>index.html</code> directly in a browser; no server required.</div>
</div>
</body>
</html>
"""


def _section_overview(summary: pd.DataFrame, boot: pd.DataFrame) -> str:
    n = len(summary)
    ok = summary.dropna(subset=["sharpe_ratio"]).shape[0] if "sharpe_ratio" in summary.columns else 0
    top = summary.dropna(subset=["deflated_sharpe"]).sort_values(
        "deflated_sharpe", ascending=False,
    ).iloc[0] if "deflated_sharpe" in summary.columns and summary["deflated_sharpe"].notna().any() else None
    boot_n = len(boot)
    html_str = f"""
<h2 id="overview">Overview</h2>
<div class="info">
<b>What this site shows.</b> The meta-portfolio runs 20 concurrent paper-trading
strategies against one Alpaca paper account of $100,000 (split $5k per bag) with
an internal SQLite ledger tracking per-strategy ownership. This site aggregates
four independent analyses of those 20 strategies: historical walk-forward
backtest, block-bootstrap forward simulation, Markov regime-switching attribution,
and references to the CPCV tournament output.
</div>
<div class="grid">
<div class="metric"><div class="label">strategies</div>
  <div class="value">{n}</div><div class="sub">with backtest data: {ok}</div></div>
<div class="metric"><div class="label">top DSR</div>
  <div class="value">{top['strategy_id'] if top is not None else 'n/a'}</div>
  <div class="sub">DSR={top['deflated_sharpe']:.3f}</div></div>
<div class="metric"><div class="label">bootstrap paths</div>
  <div class="value">{1000 if boot_n else 0}</div>
  <div class="sub">per strategy, 252d horizon</div></div>
<div class="metric"><div class="label">regime states</div>
  <div class="value">2</div><div class="sub">LOW_VOL / HIGH_VOL (HMM on SPY)</div></div>
</div>
"""
    return html_str


def _section_leaderboard(summary: pd.DataFrame) -> str:
    df = summary.copy()
    # keep only strategies with valid metrics
    if "sharpe_ratio" not in df.columns:
        return '<h2 id="leaderboard">Leaderboard</h2><p>no backtest data</p>'
    df = df.dropna(subset=["sharpe_ratio"])
    df = df.sort_values("deflated_sharpe", ascending=False) if "deflated_sharpe" in df.columns else df

    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        css = "rank-1" if i == 1 else ("rank-2" if i == 2 else ("rank-3" if i == 3 else ""))
        rows.append(
            f'<tr class="{css}"><td>{i}</td>'
            f'<td><a href="strategies/{r.strategy_id}.html">{r.strategy_id}</a></td>'
            f'<td class="{_cls(r.sharpe_ratio)}">{_num(r.sharpe_ratio)}</td>'
            f'<td class="{_cls(r.sortino_ratio)}">{_num(r.sortino_ratio)}</td>'
            f'<td>{_pct(r.max_drawdown)}</td>'
            f'<td>{_pct(r.cvar_95)}</td>'
            f'<td class="{_cls(r.deflated_sharpe)}">{_num(r.deflated_sharpe)}</td>'
            f'<td>{_pct(r.hit_rate)}</td>'
            f'<td class="{_cls(r.total_return)}">{_pct(r.total_return)}</td>'
            f'</tr>'
        )
    table = (
        '<table><thead><tr><th>#</th><th>Strategy</th><th>Sharpe</th>'
        '<th>Sortino</th><th>Max DD</th><th>CVaR 95</th><th>DSR</th>'
        '<th>Hit rate</th><th>Total return</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table>"
    )
    return f"""
<h2 id="leaderboard">Historical backtest leaderboard</h2>
<div class="info">Ranked by Deflated Sharpe Ratio (Bailey & López de Prado 2014),
which corrects raw Sharpe for the number of trials (20 strategies in this
tournament) and for non-normality of returns (skew + kurtosis). DSR &gt; 0.95
is the institutional bar.</div>
{table}
<img class="chart" src="figures/leaderboard_sharpe.png" alt="Sharpe leaderboard">
<img class="chart" src="figures/leaderboard_dsr.png" alt="DSR leaderboard">
"""


def _section_bootstrap(boot: pd.DataFrame) -> str:
    if boot.empty:
        return '<h2 id="bootstrap">Bootstrap</h2><p>no bootstrap data</p>'
    df = boot.sort_values("terminal_return_median", ascending=False).copy()
    rows = []
    for r in df.itertuples():
        rows.append(
            f'<tr><td><a href="strategies/{r.strategy_id}.html">{r.strategy_id}</a></td>'
            f'<td>{_pct(r.terminal_return_p05)}</td>'
            f'<td>{_pct(r.terminal_return_median)}</td>'
            f'<td>{_pct(r.terminal_return_p95)}</td>'
            f'<td>{_pct(r.max_dd_p95)}</td>'
            f'<td>{r.prob_loss_20pct*100:.1f}%</td>'
            f'<td>{r.prob_dd_gte_40pct*100:.1f}%</td></tr>'
        )
    table = (
        '<table><thead><tr><th>Strategy</th><th>1y p05</th>'
        '<th>1y median</th><th>1y p95</th>'
        '<th>MaxDD p95</th><th>P(loss&gt;20%)</th><th>P(DD≥40%)</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table>"
    )
    return f"""
<h2 id="bootstrap">Bootstrap forward simulation</h2>
<div class="info">Stationary block-bootstrap on each strategy's realized daily-return
series. 1000 forward paths × 252 trading days, mean block length = 20 days (preserves
volatility clustering and fat tails). Results describe the distribution of 1-year
terminal return and max drawdown assuming the historical return-generating process
persists.</div>
{table}
<img class="chart" src="figures/bootstrap_fan.png" alt="Bootstrap fan chart">
"""


def _section_regime(per_strategy: pd.DataFrame, spy_stats: pd.DataFrame, trans: pd.DataFrame) -> str:
    if per_strategy.empty:
        return '<h2 id="regime">Regime</h2><p>no regime data</p>'
    wide = per_strategy.pivot_table(
        index="strategy_id", columns="regime_label",
        values="sharpe", aggfunc="first",
    )
    # Sort by HIGH_VOL Sharpe desc if present
    if "HIGH_VOL" in wide.columns:
        wide = wide.sort_values("HIGH_VOL", ascending=False)
    rows = []
    for sid, r in wide.iterrows():
        low = r.get("LOW_VOL", np.nan)
        high = r.get("HIGH_VOL", np.nan)
        rows.append(
            f'<tr><td><a href="strategies/{sid}.html">{sid}</a></td>'
            f'<td class="{_cls(low)}">{_num(low)}</td>'
            f'<td class="{_cls(high)}">{_num(high)}</td>'
            f'<td>{_num(low - high)}</td></tr>'
        )
    table = (
        '<table><thead><tr><th>Strategy</th><th>LOW_VOL Sharpe</th>'
        '<th>HIGH_VOL Sharpe</th><th>Spread (LOW - HIGH)</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table>"
    )
    spy_html = spy_stats.to_html(index=False, classes="dataframe",
                                 float_format=lambda v: f"{v:+.3f}")
    trans_html = trans.to_html(classes="dataframe",
                                float_format=lambda v: f"{v:.3f}")
    return f"""
<h2 id="regime">Markov regime analysis</h2>
<div class="info">Two-state Gaussian HMM fit on SPY daily returns (20y). State 0
(LOW_VOL) is bull-market-like: positive drift, low realized vol. State 1 (HIGH_VOL)
captures 2008, 2020 COVID, 2022 rate-hike stress: negative drift, high vol. For
each strategy we split daily returns by the regime label on each date and compute
annualized Sharpe separately.</div>
<h3>SPY regime stats (reference)</h3>
{spy_html}
<h3>HMM transition matrix (row → col)</h3>
{trans_html}
<h3>Per-strategy regime Sharpe</h3>
<img class="chart" src="figures/regime_scatter.png" alt="Regime scatter">
{table}
"""


def _section_cpcv() -> str:
    """Reference pointer to the existing CPCV tournament results."""
    return """
<h2 id="cpcv">Combinatorial Purged Cross-Validation</h2>
<div class="info">CPCV was executed independently earlier in the project. On the
super-wide universe (1,512 tickers × 21 years, 405 trials) NONE of the 5 real
strategies cleared the DSR &gt; 0.95 institutional gate; HRP was the strongest
null baseline. See <code>results/cpcv_superwide_hrp/report.txt</code> for the full
verdict.</div>
<p>The meta-portfolio analyses on this site (historical, bootstrap, regime) are
complementary to CPCV: they describe what the 20 live strategies are actually
doing, rather than whether any one of them survives multiple-testing deflation.</p>
"""


def _section_per_strategy_index(summary: pd.DataFrame) -> str:
    if "strategy_id" not in summary.columns:
        return ""
    links = "".join(
        f'<li><a href="strategies/{sid}.html">{sid}</a></li>'
        for sid in summary["strategy_id"].tolist()
    )
    return f"""
<h2 id="per-strategy">Per-strategy drill-down</h2>
<p>Each strategy has its own page with equity curve, drawdown timeline,
all backtest metrics, bootstrap summary, and per-regime performance.</p>
<ul>{links}</ul>
"""


def _section_audit() -> str:
    return """
<h2 id="audit">Self-audit</h2>
<div class="warn">
<b>Known issues flagged during this build.</b>
<ul>
<li><b>Integer-share rounding</b> destroyed the 40% equity floor on this
morning's live deployment of <code>invvol_eqfloor_etfs</code> (target 40% equity,
actual 22.8%). Fixed in subsequent strategies by enabling fractional shares.</li>
<li><b>238 of 1,246 orders (19%)</b> from the live 20-strategy deployment
filled at Alpaca AFTER our 30-second poll deadline; they fill on Alpaca but are
missing from the per-strategy ledger. Root cause: poll-timeout on burst
submission. Mitigation scoped but not yet built: client_order_id tagging +
post-run reconciliation pass.</li>
<li><b>HMM non-convergence warning</b> surfaced during the 20-year SPY fit but
the log-likelihood delta is ~0.12 nats and regime labels are stable. Acceptable
for attribution purposes; would need refinement if used for live prediction.</li>
</ul>
</div>
<div class="warn">
<b>What was NOT built (explicitly scoped out for this session).</b>
<ul>
<li><b>TimeGAN / VAE synthetic-data generation.</b> Requires PyTorch + 12-24h GPU
training. Infeasible in one session. The bootstrap forward sim is the
non-parametric substitute for this session.</li>
<li><b>Agent-Based Modeling (ABM).</b> Would need a full market simulator with
interacting agent types; out of scope for a single session.</li>
<li><b>Multi-Level Monte Carlo (MLMC).</b> Meaningful for exotic-derivative pricing
with expensive path generators; overkill for monthly-rebalance portfolios on
daily prices.</li>
</ul>
</div>
<p>The delivered analyses (historical walk-forward, block-bootstrap,
Markov regime) cover the <i>tractable</i> subset of the requested methods.
Synthetic generative models (GAN / ABM / MLMC) are plausible next steps but
require infrastructure builds that exceed a single-session scope.</p>
"""


def _build_strategy_page(
    spec_row: pd.Series,
    boot_row: pd.Series | None,
    regime_rows: pd.DataFrame,
) -> str:
    sid = spec_row["strategy_id"]

    metrics_cards = f"""
<div class="grid">
  <div class="metric"><div class="label">Sharpe</div>
    <div class="value">{_num(spec_row['sharpe_ratio'])}</div></div>
  <div class="metric"><div class="label">Sortino</div>
    <div class="value">{_num(spec_row['sortino_ratio'])}</div></div>
  <div class="metric"><div class="label">Max Drawdown</div>
    <div class="value">{_pct(spec_row['max_drawdown'])}</div></div>
  <div class="metric"><div class="label">CVaR 95</div>
    <div class="value">{_pct(spec_row['cvar_95'])}</div></div>
  <div class="metric"><div class="label">DSR</div>
    <div class="value">{_num(spec_row['deflated_sharpe'])}</div></div>
  <div class="metric"><div class="label">Hit rate</div>
    <div class="value">{_pct(spec_row['hit_rate'])}</div></div>
  <div class="metric"><div class="label">Total return</div>
    <div class="value">{_pct(spec_row['total_return'])}</div></div>
  <div class="metric"><div class="label">Days</div>
    <div class="value">{int(spec_row['n_days'])}</div>
    <div class="sub">{spec_row['start_date']} → {spec_row['end_date']}</div></div>
</div>
"""

    equity_img = f'<img class="chart" src="../figures/strategy_{sid}_wealth.png" alt="equity">'
    dd_img = f'<img class="chart" src="../figures/strategy_{sid}_drawdown.png" alt="drawdown">'

    boot_html = ""
    if boot_row is not None:
        boot_html = f"""
<h3>Bootstrap forward simulation (1y horizon, 1000 paths)</h3>
<div class="grid">
  <div class="metric"><div class="label">terminal p05</div>
    <div class="value">{_pct(boot_row['terminal_return_p05'])}</div></div>
  <div class="metric"><div class="label">terminal median</div>
    <div class="value">{_pct(boot_row['terminal_return_median'])}</div></div>
  <div class="metric"><div class="label">terminal p95</div>
    <div class="value">{_pct(boot_row['terminal_return_p95'])}</div></div>
  <div class="metric"><div class="label">max DD p95</div>
    <div class="value">{_pct(boot_row['max_dd_p95'])}</div></div>
  <div class="metric"><div class="label">P(loss &gt; 20%)</div>
    <div class="value">{boot_row['prob_loss_20pct']*100:.1f}%</div></div>
  <div class="metric"><div class="label">P(DD ≥ 40%)</div>
    <div class="value">{boot_row['prob_dd_gte_40pct']*100:.1f}%</div></div>
</div>
"""

    regime_html = ""
    if not regime_rows.empty:
        regime_table = regime_rows[["regime_label", "n_days", "frac_days", "ann_return",
                                     "ann_vol", "sharpe", "max_drawdown", "hit_rate"]].to_html(
            index=False, classes="dataframe",
            float_format=lambda v: f"{v:+.3f}",
        )
        regime_html = f"""
<h3>Per-regime performance</h3>
{regime_table}
"""

    return f"""
<h2>{sid}</h2>
<p><a href="../index.html">← back to overview</a></p>
<h3>Historical backtest metrics</h3>
{metrics_cards}
<h3>Equity curve</h3>
{equity_img}
<h3>Drawdown timeline</h3>
{dd_img}
{boot_html}
{regime_html}
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    (OUT / "strategies").mkdir(parents=True, exist_ok=True)
    (OUT / "style.css").write_text(CSS, encoding="utf-8")

    print("[1/5] loading inputs...")
    summary = pd.read_csv(BACKTEST / "summary.csv")
    try:
        boot = pd.read_csv(BOOTSTRAP / "summary.csv")
    except FileNotFoundError:
        boot = pd.DataFrame()
    try:
        regime_df = pd.read_csv(REGIME / "per_strategy.csv")
        spy_stats = pd.read_csv(REGIME / "spy_regime_stats.csv")
        trans_df = pd.read_csv(REGIME / "regime_transition_matrix.csv", index_col=0)
    except FileNotFoundError:
        regime_df = pd.DataFrame()
        spy_stats = pd.DataFrame()
        trans_df = pd.DataFrame()

    print("[2/5] rendering overview charts...")
    _plot_leaderboard_bar(summary, "sharpe_ratio",
                          "Historical Sharpe ratio — 5y backtest",
                          FIGS / "leaderboard_sharpe.png")
    _plot_leaderboard_bar(summary, "deflated_sharpe",
                          "Deflated Sharpe ratio (20-trial deflation)",
                          FIGS / "leaderboard_dsr.png")
    if not boot.empty:
        _plot_bootstrap_fan(boot, FIGS / "bootstrap_fan.png")
    if not regime_df.empty:
        _plot_regime_scatter(regime_df, FIGS / "regime_scatter.png")

    print("[3/5] rendering per-strategy charts...")
    for sid in summary["strategy_id"].dropna().tolist():
        _plot_strategy_wealth(sid, FIGS / f"strategy_{sid}_wealth.png")
        _plot_strategy_drawdown(sid, FIGS / f"strategy_{sid}_drawdown.png")

    print("[4/5] rendering index.html...")
    body = (
        _section_overview(summary, boot)
        + _section_leaderboard(summary)
        + _section_bootstrap(boot)
        + _section_regime(regime_df, spy_stats, trans_df)
        + _section_cpcv()
        + _section_per_strategy_index(summary)
        + _section_audit()
    )
    index_html = HTML_BASE.format(
        title="Inversiones_mama meta-portfolio — analysis site",
        header_title="Inversiones_mama · Meta-portfolio analysis",
        header_sub="20 concurrent paper-trading strategies · historical · bootstrap · regime · CPCV",
        body=body,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        css_rel="style.css",
        index_rel="index.html",
    )
    (OUT / "index.html").write_text(index_html, encoding="utf-8")

    print("[5/5] rendering per-strategy pages...")
    for _, row in summary.iterrows():
        sid = row.get("strategy_id")
        if pd.isna(sid):
            continue
        boot_row = None
        if not boot.empty:
            hit = boot[boot["strategy_id"] == sid]
            if not hit.empty:
                boot_row = hit.iloc[0]
        regime_rows = (regime_df[regime_df["strategy_id"] == sid]
                       if not regime_df.empty else pd.DataFrame())
        body_html = _build_strategy_page(row, boot_row, regime_rows)
        page_html = HTML_BASE.format(
            title=f"{sid} — Inversiones_mama",
            header_title=f"{sid}",
            header_sub="Per-strategy drill-down",
            body=body_html,
            generated_at=datetime.now().isoformat(timespec="seconds"),
            css_rel="../style.css",
            index_rel="../index.html",
        )
        (OUT / "strategies" / f"{sid}.html").write_text(page_html, encoding="utf-8")

    print(f"\nDone. Open:  {OUT / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
