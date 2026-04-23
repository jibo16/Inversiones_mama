"""Per-regime performance attribution for every strategy.

Fits a 2-state Markov regime-switching model on SPY daily returns over
the backtest window, classifies every historical trading day as
LOW_VOL or HIGH_VOL, and for each strategy splits its daily returns by
regime to derive regime-specific Sharpe / MDD / hit-rate / CVaR.

This answers: "when the market is in a high-volatility regime (2008,
2020 COVID, 2022 rate hikes), which of the 20 strategies actually
work, and which collapse?"

Outputs
-------
results/meta_regime/
  regime_labels.csv                    date -> regime_id
  regime_transition_matrix.csv         2x2 HMM transition probabilities
  per_strategy.csv                     per-(strategy, regime) metrics
  summary.json                         overall regime stats
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from inversiones_mama.data.prices import load_prices
from inversiones_mama.execution.strategy_catalog import STRATEGY_CATALOG
from inversiones_mama.simulation.regime_switching import (
    REGIME_LABELS,
    RegimeClassifier,
)


def _load_daily(strategy_id: str, backtest_dir: Path) -> pd.Series | None:
    path = backtest_dir / "daily" / f"{strategy_id}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.get("daily_return").dropna() if "daily_return" in df.columns else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spy-years", type=float, default=20.0,
                        help="Years of SPY history to fit the HMM (default 20).")
    parser.add_argument("--backtest-dir", type=str, default="results/meta_backtest")
    parser.add_argument("--out-dir", type=str, default="results/meta_regime")
    args = parser.parse_args(argv)

    backtest_dir = Path(args.backtest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fit regime classifier on SPY
    print("=" * 72)
    print("META-PORTFOLIO REGIME ANALYSIS")
    print("=" * 72)
    print(f"[1/3] Fitting 2-state Gaussian HMM on SPY ({args.spy_years}y)...")
    end = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=int(args.spy_years * 365) + 14)
    spy_prices = load_prices(["SPY"], start, end, use_cache=True)
    if "SPY" not in spy_prices.columns or spy_prices["SPY"].dropna().empty:
        print("[error] SPY price data unavailable", file=sys.stderr)
        return 2
    spy_returns = spy_prices["SPY"].pct_change().dropna()
    classifier = RegimeClassifier()
    classifier.fit(spy_returns)

    # 2. Persist regime labels + transition matrix
    print("[2/3] Classifying history + saving labels...")
    spy_labels = classifier.predict_regimes(spy_returns)
    labels_df = pd.DataFrame({
        "date":         [d.date() for d in spy_labels.index],
        "regime_id":    spy_labels.values,
        "regime_label": [REGIME_LABELS[r] for r in spy_labels.values],
        "spy_return":   spy_returns.loc[spy_labels.index].values,
    })
    labels_df.to_csv(out_dir / "regime_labels.csv", index=False)

    trans = classifier.transition_matrix()
    trans_df = pd.DataFrame(
        trans,
        index=[REGIME_LABELS[i] for i in range(classifier.n_states)],
        columns=[REGIME_LABELS[i] for i in range(classifier.n_states)],
    )
    trans_df.to_csv(out_dir / "regime_transition_matrix.csv")

    # Regime stats on SPY itself (reference)
    regime_stats_spy = classifier.regime_stats(spy_returns)
    regime_stats_spy.to_csv(out_dir / "spy_regime_stats.csv", index=False)

    print("\n  SPY regime stats (reference):")
    for r in regime_stats_spy.itertuples():
        print(f"    {r.regime_label:<9}  n={r.n_days:>5}  "
              f"frac={r.frac_time*100:5.1f}%  "
              f"ann_ret={r.ann_return*100:+7.2f}%  "
              f"ann_vol={r.ann_vol*100:5.2f}%  SR={r.sharpe:+.2f}")
    print("\n  HMM transition matrix (row=from, col=to):")
    for name in trans_df.index:
        row = trans_df.loc[name]
        print(f"    {name:<9}  -> " + "  ".join(
            f"{col}: {row[col]*100:5.1f}%" for col in trans_df.columns
        ))

    # 3. Per-strategy per-regime stats
    print("\n[3/3] Per-strategy regime attribution...")
    per_strategy_rows: list[dict] = []
    for spec in STRATEGY_CATALOG:
        daily = _load_daily(spec.strategy_id, backtest_dir)
        if daily is None or len(daily) < 30:
            continue
        # Align strategy daily returns with regime labels by date
        common = daily.index.intersection(spy_labels.index)
        if len(common) < 30:
            continue
        r = daily.loc[common]
        g = spy_labels.loc[common]
        for rid in sorted(g.unique()):
            mask = g == rid
            slice_ret = r.loc[g.index[mask]]
            if len(slice_ret) < 10:
                continue
            mu = float(slice_ret.mean())
            sd = float(slice_ret.std(ddof=1))
            ann_ret = float((1 + mu) ** 252 - 1)
            ann_vol = sd * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            wealth = (1 + slice_ret).cumprod()
            peak = wealth.cummax()
            dd = (1 - wealth / peak).max()
            q05 = float(np.percentile(slice_ret, 5))
            tail = slice_ret[slice_ret <= q05]
            cvar = float(-tail.mean()) if tail.size else np.nan
            per_strategy_rows.append({
                "strategy_id":  spec.strategy_id,
                "regime_id":    int(rid),
                "regime_label": REGIME_LABELS[int(rid)],
                "n_days":       int(mask.sum()),
                "frac_days":    float(mask.sum() / len(r)),
                "mean_return":  mu,
                "ann_return":   ann_ret,
                "ann_vol":      ann_vol,
                "sharpe":       float(sharpe),
                "max_drawdown": float(dd),
                "hit_rate":     float((slice_ret > 0).mean()),
                "cvar_95":      cvar,
            })

    df = pd.DataFrame(per_strategy_rows)
    df.to_csv(out_dir / "per_strategy.csv", index=False)

    # Headline view: Sharpe per regime per strategy, sorted by HIGH_VOL Sharpe
    wide = df.pivot_table(
        index="strategy_id", columns="regime_label",
        values="sharpe", aggfunc="first",
    )
    if "HIGH_VOL" in wide.columns:
        wide = wide.sort_values("HIGH_VOL", ascending=False)

    print("\n" + "=" * 72)
    print("SHARPE BY REGIME  (ranked by HIGH_VOL Sharpe)")
    print("=" * 72)
    print(wide.to_string(float_format=lambda v: f"{v:+.3f}"))

    # Persist metadata + summary JSON
    summary = {
        "generated_at":    datetime.now().isoformat(timespec="seconds"),
        "spy_years":       args.spy_years,
        "spy_start":       str(spy_returns.index[0].date()),
        "spy_end":         str(spy_returns.index[-1].date()),
        "n_states":        classifier.n_states,
        "transition_matrix": trans_df.to_dict(orient="index"),
        "spy_regime_stats":  regime_stats_spy.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                          encoding="utf-8")
    print(f"\nSaved: {out_dir}/per_strategy.csv + regime_labels.csv + summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
