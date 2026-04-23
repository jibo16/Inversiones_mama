"""Combinatorial Purged Cross-Validation (López de Prado 2018, Chapter 12).

Standard k-fold cross-validation is fatal in finance: it assumes IID
data and leaks future information into training via both random
shuffling and label overlap. CPCV fixes both:

1. **Combinatorial splits.** Partition a time-ordered series into ``N``
   contiguous groups. For every combination of ``k`` groups-as-test,
   emit a (train, test) pair. That yields ``C(N, k)`` distinct splits
   instead of a single walk-forward path, giving a *distribution* of
   out-of-sample performance rather than a point estimate.

2. **Purging.** Remove training observations whose label horizon could
   overlap a test group's horizon. For a simple 1-day return label,
   this collapses to purging the ``embargo_pct * n_obs`` observations
   *before* each test group (their label window crosses the boundary).

3. **Embargo.** After each test group, discard a buffer of training
   observations (``embargo_pct * n_obs``) to prevent leakage from
   auto-correlated features.

The number of full out-of-sample paths — sequences of disjoint test
groups that together cover the entire series — is::

    phi(N, k) = (k / N) * C(N, k)

Public API
----------
``CPCVSplit``    — dataclass holding one (train_idx, test_idx) pair.
``PurgedKFold``  — splitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class CPCVSplit:
    """One (train, test) split produced by :class:`PurgedKFold`.

    Attributes
    ----------
    train_idx : np.ndarray
        Sorted unique integer indices into the underlying series,
        post purging + embargo.
    test_idx : np.ndarray
        Sorted unique integer indices of the k-combined test groups.
    test_group_ids : tuple[int, ...]
        Which of the ``n_groups`` were selected as test in this split.
    """

    train_idx: np.ndarray
    test_idx: np.ndarray
    test_group_ids: tuple[int, ...]


class PurgedKFold:
    """CPCV splitter for time-ordered series with single-horizon labels.

    Parameters
    ----------
    n_groups : int
        Total partitions of the time series. Must be >= 2.
    test_groups : int
        ``k`` — how many groups form the test side per split.
        Must satisfy 1 <= k < n_groups.
    embargo_pct : float
        Fraction of total observations to purge *before* each test
        group (label-horizon purge) and to embargo *after* each test
        group (serial-correlation buffer). Common values 0.01–0.02.

    Examples
    --------
    With ``n_groups=8, test_groups=2`` you get
    ``C(8, 2) = 28`` distinct splits. Each test_idx concatenates two
    groups (not necessarily adjacent) so every split exercises a
    different slice of the series.
    """

    def __init__(
        self,
        n_groups: int = 8,
        test_groups: int = 2,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_groups < 2:
            raise ValueError(f"n_groups must be >= 2, got {n_groups}")
        if test_groups < 1 or test_groups >= n_groups:
            raise ValueError(
                f"test_groups must be in [1, {n_groups - 1}], got {test_groups}"
            )
        if not (0.0 <= embargo_pct <= 0.5):
            raise ValueError(f"embargo_pct must be in [0, 0.5], got {embargo_pct}")
        self.n_groups = int(n_groups)
        self.test_groups = int(test_groups)
        self.embargo_pct = float(embargo_pct)

    @property
    def n_splits(self) -> int:
        """Number of (train, test) pairs this splitter yields — ``C(N, k)``."""
        return comb(self.n_groups, self.test_groups)

    @property
    def n_paths(self) -> int:
        """Number of disjoint OOS paths that tile the full series — ``(k/N) * C(N, k)``."""
        return int(self.test_groups * comb(self.n_groups, self.test_groups) / self.n_groups)

    # ------------------------------------------------------------ core split
    def split(self, n_obs: int) -> Iterator[CPCVSplit]:
        """Yield every CPCV split over a series of length ``n_obs``.

        Parameters
        ----------
        n_obs : int
            Length of the time-ordered series to split.

        Yields
        ------
        CPCVSplit
            ``n_splits`` times, one for each ``C(N, k)`` combination.

        Raises
        ------
        ValueError
            If ``n_obs`` is too small to hold at least 2 observations
            per group.
        """
        if n_obs < 2 * self.n_groups:
            raise ValueError(
                f"Need at least {2 * self.n_groups} observations to form "
                f"{self.n_groups} groups with >=2 obs each, got {n_obs}"
            )

        # Contiguous group boundaries: equal-ish sized bins.
        bounds = np.linspace(0, n_obs, self.n_groups + 1, dtype=np.int64)
        groups: list[np.ndarray] = [
            np.arange(bounds[g], bounds[g + 1], dtype=np.int64)
            for g in range(self.n_groups)
        ]

        embargo_len = int(np.ceil(self.embargo_pct * n_obs))

        for test_group_ids in combinations(range(self.n_groups), self.test_groups):
            test_idx = np.concatenate([groups[g] for g in test_group_ids])
            test_idx.sort()

            # Build the set of forbidden training indices:
            #   - every test index itself
            #   - `embargo_len` indices immediately BEFORE each test
            #     group (purges label horizons that would cross the
            #     boundary into test)
            #   - `embargo_len` indices immediately AFTER each test
            #     group (buffer against serial-correlation leakage)
            forbidden: set[int] = set(test_idx.tolist())
            for g in test_group_ids:
                start = int(groups[g][0])
                stop = int(groups[g][-1])
                forbidden.update(range(max(0, start - embargo_len), start))
                forbidden.update(range(stop + 1, min(n_obs, stop + 1 + embargo_len)))

            train_idx = np.fromiter(
                (i for i in range(n_obs) if i not in forbidden),
                dtype=np.int64,
            )

            yield CPCVSplit(
                train_idx=train_idx,
                test_idx=test_idx,
                test_group_ids=test_group_ids,
            )
