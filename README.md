# High-Risk Factor Investing Portfolio ($5,000)

This repository tracks the strategy, execution, and performance of a highly concentrated, aggressive $5,000 portfolio dedicated purely to high-risk growth.

## Core Strategy

The strategy Abandons traditional Modern Portfolio Theory (which seeks to minimize volatility) and broad ETF diversification in favor of maximizing the geometric growth rate through a **Maximum Return / Kelly Sizing Strategy** combined with **Factor Screening**.

### 1. Sizing: Fractional Kelly Criterion
Instead of arbitrary position sizing, we use a Risk-Constrained "Half-Kelly" approach to balance aggressive growth with protection from total ruin.
- **Kelly Formula:** `f* = (μ - rf) / σ^2`
- We divide the optimal `f*` by two (Half-Kelly) to significantly reduce drawdown volatility while capturing most of the growth potential.

### 2. Asset Selection: Factor Screening
We screen the market for individual equities exhibiting specific risk factors known to generate higher market premiums. No ETFs are used.
- **Small-Cap and Value (SMB & HML):** Targeting companies with small capitalizations and high book-to-market ratios.
- **Robust Profitability (RMW):** Filtering the small-cap/value space for firms with robust operating profitability.
- **Momentum (MOM):** Applying an intermediate-term momentum filter (3 to 12 months) to identify current winners. Short-term (under 1 month) momentum is ignored due to mean-reversion tendencies.

### 3. Portfolio Construction
- **High Concentration:** The portfolio is highly concentrated in roughly 5 to 10 stocks that pass the screens.
- **Sector Limits:** Exposure to any single sector is capped at 25% to 30% (roughly $1,250 to $1,500 max per sector) to ensure a diversified factor bet rather than a localized sector bet.

### 4. Active Management & Risk Isolation
- **Strict Rebalancing:** The portfolio is re-evaluated and rebalanced monthly or quarterly to maintain optimal Kelly position sizes and capture time-sensitive Momentum.
- **Risk Isolation:** The $5,000 capital block is strictly isolated. Heavy drawdowns (30%+), which are expected in this strategy, will NOT be offset by mingling funds from safer investments.

## NotebookLM Reference
Reference material for quantitative portfolio management, Fama-French models, Hierarchical Risk Parity (HRP), and Kelly criterion:
[NotebookLM Reference - 6b519c7a-675b-4072-aab3-dd374c55ddab](https://notebooklm.google.com/notebook/6b519c7a-675b-4072-aab3-dd374c55ddab)
