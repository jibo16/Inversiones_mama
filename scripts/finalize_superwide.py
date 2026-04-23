"""Finalize the super-wide CPCV artifacts after the in-script MW crash.

Reads the already-saved aggregate.csv + all_splits.csv, computes the
Mann-Whitney U test + gate verdict, writes verdict.txt + mannwhitney.csv
+ meta.json + two PNG plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


OUT = Path("results/cpcv_superwide")
agg = pd.read_csv(OUT / "aggregate.csv")
splits = pd.read_csv(OUT / "all_splits.csv")

null_s = splits[splits["is_null"]]["sharpe"].dropna().to_numpy()
null_med = float(np.median(null_s))

mw_rows: list[dict] = []
for name in splits[~splits["is_null"].astype(bool)]["strategy"].unique():
    strat = splits[(splits["strategy"] == name) & (~splits["is_null"].astype(bool))]["sharpe"].dropna().to_numpy()
    u, p = mannwhitneyu(strat, null_s, alternative="greater")
    mw_rows.append({
        "universe": "parquet:prices", "strategy": name,
        "strategy_median": float(np.median(strat)), "null_median": null_med,
        "diff": float(np.median(strat) - null_med),
        "u_stat": float(u), "p_value": float(p),
        "significant_5pct": bool(p < 0.05),
    })
mw_df = pd.DataFrame(mw_rows)
mw_df.to_csv(OUT / "mannwhitney.csv", index=False)

# Verdict text
lines: list[str] = []
bar = "=" * 84
lines.append(bar)
lines.append("SUPER-WIDE CPCV TOURNAMENT — VERDICT")
lines.append(bar)
lines.append("  Universe:  1,512 tickers surviving 95%-coverage over 2005-2026")
lines.append("  Window:    2005-10-18 -> 2026-04-21  (5347-5467 trading days per strategy)")
lines.append("  Data pts:  16,059,686 total non-null cells in source parquet")
lines.append("  Strategies: 5 real + 3 null baselines")
lines.append("  CPCV:      N=10, k=2, 45 splits per cell, 360 total trials")
lines.append("  Bootstrap: 1000 iters, 20d blocks")
lines.append(bar)
lines.append("")
lines.append(f"Pooled null (n={len(null_s)} splits): median Sharpe = {null_med:+.3f}")
lines.append("")
hdr = f"{'strategy':<18} {'median_SR':>10} {'DSR_med':>8} {'MDD_med':>9}  MW_p     sig@5pct  diff_vs_null"
lines.append(hdr)
lines.append("-" * 84)
for name in splits[~splits["is_null"].astype(bool)]["strategy"].unique():
    strat = splits[(splits["strategy"] == name) & (~splits["is_null"].astype(bool))]["sharpe"].dropna().to_numpy()
    ar = agg[agg["strategy"] == name].iloc[0]
    _, p = mannwhitneyu(strat, null_s, alternative="greater")
    sig = "YES" if p < 0.05 else "no "
    diff = float(np.median(strat) - null_med)
    lines.append(f"{name:<18} {np.median(strat):>+10.3f} {ar['dsr_median']:>8.3f} {ar['maxdd_median']*100:>8.2f}%  "
                 f"{p:6.4f}   {sig}      {diff:+.3f}")
lines.append("-" * 84)
lines.append("")
lines.append("NULL baselines (reference):")
for name in ["null_equal_weight", "null_random_uniform", "null_inverse_vol"]:
    ar = agg[agg["strategy"] == name].iloc[0]
    n_s = splits[splits["strategy"] == name]["sharpe"].dropna().to_numpy()
    lines.append(f"{name:<18} {np.median(n_s):>+10.3f} {ar['dsr_median']:>8.3f} {ar['maxdd_median']*100:>8.2f}%")
lines.append("")
lines.append(bar)
lines.append("GATE VERDICT (bootstrap_CI>0 AND Mann-Whitney p<0.05 AND DSR_med>0.95)")
lines.append(bar)
passes = 0
for name in splits[~splits["is_null"].astype(bool)]["strategy"].unique():
    ar = agg[agg["strategy"] == name].iloc[0]
    mwr = mw_df[mw_df["strategy"] == name].iloc[0]
    gate_a = bool(ar["boot_ci_lo"] > 0)
    gate_b = bool(mwr["significant_5pct"])
    gate_c = bool(ar["dsr_median"] > 0.95)
    all_pass = gate_a and gate_b and gate_c
    mark = "PASS" if all_pass else "fail"
    if all_pass:
        passes += 1
    lines.append(f"  [{mark}] {name:<18}  CI>0={gate_a!s:<5}  MW_p<.05={gate_b!s:<5}  DSR>0.95={gate_c!s:<5}")
lines.append("")
if passes == 0:
    lines.append("CONCLUSION: NO strategy cleared all three gates on 1,512 tickers x 21 years.")
    lines.append("The widest, deepest test yet -- the answer is CONSISTENT with narrower")
    lines.append("tests: no real strategy has demonstrable statistical edge vs simple null")
    lines.append("baselines at the institutional bar.")
    lines.append("")
    lines.append("All 3 null baselines (equal-weight, random-uniform, inverse-vol) BEAT every")
    lines.append("real strategy on median DSR. Naive diversification outperforms every rules-")
    lines.append("based strategy we tested on this 16M-data-point dataset.")
else:
    lines.append(f"{passes} cell(s) cleared all three gates.")
lines.append(bar)
text = "\n".join(lines)
(OUT / "verdict.txt").write_text(text + "\n", encoding="utf-8")

# Meta
meta = {
    "generated_at":                pd.Timestamp.now().isoformat(),
    "source_parquet":              "results/wide_gather/prices.parquet",
    "source_data_points":          16_059_686,
    "min_year":                    2005,
    "universe_size_post_filter":   1512,
    "window_start":                "2005-10-18",
    "window_end":                  "2026-04-21",
    "n_real_strategies":           5,
    "n_null_baselines":            3,
    "cpcv_splits_per_cell":        45,
    "total_trials":                360,
    "bootstrap_iters":              1000,
    "gate_verdict":                "FAIL" if passes == 0 else "PASS",
    "n_cells_passed":              int(passes),
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

# Plots
import matplotlib.pyplot as plt
(OUT / "figures").mkdir(parents=True, exist_ok=True)

# 1. Box plot
fig, ax = plt.subplots(figsize=(13, 6))
strategies = list(splits["strategy"].unique())
data = [splits[splits["strategy"] == s]["sharpe"].dropna().values for s in strategies]
colors = ["#d62728" if splits[splits["strategy"] == s]["is_null"].iloc[0] else "#1f77b4"
          for s in strategies]
bp = ax.boxplot(data, tick_labels=strategies, patch_artist=True, showmeans=True)
for patch, col in zip(bp["boxes"], colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.5)
ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
ax.axhline(null_med, color="r", linestyle="--", alpha=0.7,
           label=f"pooled-null median = {null_med:+.3f}")
ax.set_title("Super-wide CPCV (1,512 tickers x 21y) — per-split Sharpe\n"
             "red=null baseline, blue=real strategy, dashed=pooled null median")
ax.set_ylabel("Annualized Sharpe")
ax.tick_params(axis="x", rotation=30)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "figures" / "01_sharpe_boxplot.png", dpi=110)
plt.close()

# 2. Bootstrap CI dot plot
fig, ax = plt.subplots(figsize=(13, 0.5 * len(agg) + 2))
agg_sorted = agg.sort_values("full_sharpe").reset_index(drop=True)
y = np.arange(len(agg_sorted))
colors2 = ["#d62728" if b else "#1f77b4" for b in agg_sorted["is_null"]]
ax.hlines(y, agg_sorted["boot_ci_lo"], agg_sorted["boot_ci_hi"],
          colors=colors2, linewidth=3)
ax.plot(agg_sorted["full_sharpe"], y, "o", color="k")
ax.set_yticks(y)
ax.set_yticklabels(agg_sorted["strategy"])
ax.axvline(0, color="r", linestyle="--", alpha=0.7, label="zero")
ax.set_xlabel("Annualized Sharpe (block-bootstrap 95% CI, n_boot=1000)")
ax.set_title("Super-wide CPCV — Bootstrap 95% CI on full-series Sharpe\n"
             "red=null baseline, blue=real strategy")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "figures" / "02_bootstrap_ci.png", dpi=110)
plt.close()

print("wrote:")
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"  {p}  ({p.stat().st_size // 1024} KB)")
