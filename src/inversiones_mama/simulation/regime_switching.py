"""Markov Regime Switching classifier for financial return series.

Fits a 2-state Gaussian HMM on a benchmark return series (typically
SPY daily returns). Each state captures a latent market regime:

  Regime 0 (LOW_VOL):  lower realized vol, typically positive drift
  Regime 1 (HIGH_VOL): higher realized vol, typically negative drift
                       (bear / correction / crisis windows)

The classifier exposes ``predict_regimes(returns)`` which emits a
per-date regime label, enabling downstream per-regime performance
attribution: split any strategy's daily returns by regime date, compute
Sharpe / MDD within each regime, and report how a strategy behaves in
different market states.

Design notes
------------
* Two states, not more. Three-state models overfit on 20-25 years of
  daily data and the incremental interpretability is marginal.
* Gaussian emissions on scaled daily returns. A more faithful model
  would use Student-t or a mixture, but Gaussian-HMM is stable and
  good-enough for regime labeling (we care about WHEN the regime
  changes, not about pricing).
* Seed fixed for reproducibility: same seed + same SPY history =>
  same regime labels.
* Classification is OFFLINE: we fit on the full SPY history, then
  label every day. For a live-trading classifier we'd need an online
  filter; out of scope here.

Public API
----------
``RegimeClassifier.fit(returns)``
``RegimeClassifier.predict_regimes(returns) -> pd.Series[int]``
``RegimeClassifier.regime_stats(returns) -> pd.DataFrame`` summary
``fit_spy_classifier(start, end) -> RegimeClassifier`` convenience
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


REGIME_LABELS = {0: "LOW_VOL", 1: "HIGH_VOL"}


@dataclass
class RegimeClassifier:
    n_states: int = 2
    random_seed: int = 20260423
    _fitted: bool = field(default=False, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _state_vol_order: list[int] = field(default_factory=list, init=False, repr=False)

    def fit(self, returns: pd.Series) -> "RegimeClassifier":
        """Fit a Gaussian HMM on the supplied daily return series."""
        from hmmlearn.hmm import GaussianHMM  # lazy import

        r = returns.dropna()
        if len(r) < 60:
            raise ValueError(f"need >= 60 observations to fit HMM, got {len(r)}")

        x = r.values.reshape(-1, 1).astype(np.float64)
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=500,
            tol=1e-4,
            random_state=self.random_seed,
        )
        model.fit(x)

        # Order states by realized vol ascending so state 0 = LOW_VOL
        vols = [float(np.sqrt(model.covars_[i].ravel()[0])) for i in range(self.n_states)]
        self._state_vol_order = list(np.argsort(vols))

        self._model = model
        self._fitted = True
        return self

    def predict_regimes(self, returns: pd.Series) -> pd.Series:
        """Return a Series of regime labels (0=LOW_VOL, 1=HIGH_VOL)."""
        if not self._fitted:
            raise RuntimeError("call .fit() first")
        r = returns.dropna()
        if len(r) == 0:
            return pd.Series(dtype=int)
        raw = self._model.predict(r.values.reshape(-1, 1).astype(np.float64))
        # Re-map state indices to vol-ordered labels
        remap = {s: i for i, s in enumerate(self._state_vol_order)}
        labels = np.array([remap[s] for s in raw], dtype=int)
        return pd.Series(labels, index=r.index, name="regime")

    def regime_stats(self, returns: pd.Series) -> pd.DataFrame:
        """Summary of regime durations + return stats."""
        regimes = self.predict_regimes(returns)
        stats = []
        for rid in sorted(regimes.unique()):
            mask = regimes == rid
            slice_ret = returns.loc[regimes.index[mask]]
            if len(slice_ret) == 0:
                continue
            ann_ret = float((1 + slice_ret.mean()) ** 252 - 1)
            ann_vol = float(slice_ret.std(ddof=1) * np.sqrt(252))
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            stats.append({
                "regime_id":    int(rid),
                "regime_label": REGIME_LABELS.get(int(rid), f"R{rid}"),
                "n_days":       int(mask.sum()),
                "frac_time":    float(mask.mean()),
                "mean_return":  float(slice_ret.mean()),
                "std_return":   float(slice_ret.std(ddof=1)),
                "ann_return":   ann_ret,
                "ann_vol":      ann_vol,
                "sharpe":       float(sharpe),
            })
        return pd.DataFrame(stats)

    def transition_matrix(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call .fit() first")
        raw = self._model.transmat_
        # Re-map to vol-ordered labels
        perm = self._state_vol_order
        return raw[np.ix_(perm, perm)]


def fit_spy_classifier(
    start: datetime | None = None,
    end: datetime | None = None,
) -> RegimeClassifier:
    """Convenience: load SPY, fit a 2-state HMM, return the classifier."""
    from inversiones_mama.data.prices import load_prices
    end = end or (datetime.today() - timedelta(days=1))
    start = start or (end - timedelta(days=20 * 365))
    prices = load_prices(["SPY"], start, end, use_cache=True)
    returns = prices["SPY"].pct_change().dropna()
    classifier = RegimeClassifier()
    classifier.fit(returns)
    return classifier
