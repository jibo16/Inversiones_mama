"""Unit tests for Combinatorial Purged Cross-Validation."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from inversiones_mama.simulation.cpcv import CPCVSplit, PurgedKFold


# --- Construction / validation ---------------------------------------------


@pytest.mark.parametrize("n_groups", [0, 1, -5])
def test_rejects_bad_n_groups(n_groups):
    with pytest.raises(ValueError, match="n_groups must be >= 2"):
        PurgedKFold(n_groups=n_groups, test_groups=1)


@pytest.mark.parametrize("test_groups", [0, -1, 5, 10])
def test_rejects_bad_test_groups(test_groups):
    with pytest.raises(ValueError, match="test_groups must be in"):
        PurgedKFold(n_groups=5, test_groups=test_groups)


@pytest.mark.parametrize("embargo_pct", [-0.01, 0.51, 1.0])
def test_rejects_bad_embargo(embargo_pct):
    with pytest.raises(ValueError, match="embargo_pct must be in"):
        PurgedKFold(n_groups=5, test_groups=1, embargo_pct=embargo_pct)


# --- Combinatorial counting ------------------------------------------------


@pytest.mark.parametrize(
    ("n", "k", "expected_splits", "expected_paths"),
    [
        (5, 1, 5, 1),
        (6, 2, 15, 5),
        (8, 2, 28, 7),
        (10, 2, 45, 9),
        (10, 3, 120, 36),
    ],
)
def test_n_splits_and_paths_match_combinatorics(n, k, expected_splits, expected_paths):
    pk = PurgedKFold(n_groups=n, test_groups=k)
    assert pk.n_splits == expected_splits
    assert pk.n_paths == expected_paths


def test_n_paths_formula_matches_closed_form():
    # phi(N, k) = (k / N) * C(N, k)
    for n in range(3, 12):
        for k in range(1, n):
            pk = PurgedKFold(n_groups=n, test_groups=k)
            expected = int(k * comb(n, k) / n)
            assert pk.n_paths == expected


# --- Split behavior --------------------------------------------------------


def test_split_rejects_tiny_series():
    pk = PurgedKFold(n_groups=8, test_groups=2)
    with pytest.raises(ValueError, match="Need at least"):
        list(pk.split(n_obs=10))


def test_split_produces_exact_number_of_combinations():
    pk = PurgedKFold(n_groups=8, test_groups=2, embargo_pct=0.0)
    splits = list(pk.split(n_obs=1000))
    assert len(splits) == pk.n_splits == 28


def test_each_split_has_disjoint_train_and_test():
    pk = PurgedKFold(n_groups=8, test_groups=2, embargo_pct=0.02)
    for split in pk.split(n_obs=1000):
        train_set = set(split.train_idx.tolist())
        test_set = set(split.test_idx.tolist())
        assert train_set.isdisjoint(test_set), (
            f"train/test overlap in split {split.test_group_ids}"
        )


def test_test_idx_is_sorted_and_unique():
    pk = PurgedKFold(n_groups=8, test_groups=3, embargo_pct=0.01)
    for split in pk.split(n_obs=800):
        assert np.all(np.diff(split.test_idx) > 0), "test_idx must be sorted & unique"
        assert np.all(np.diff(split.train_idx) > 0), "train_idx must be sorted & unique"


def test_test_group_cardinality_matches_k():
    pk = PurgedKFold(n_groups=10, test_groups=3)
    for split in pk.split(n_obs=1200):
        assert len(split.test_group_ids) == 3


def test_test_group_ids_are_unique_combinations():
    pk = PurgedKFold(n_groups=8, test_groups=2, embargo_pct=0.0)
    seen: set[tuple[int, ...]] = set()
    for split in pk.split(n_obs=800):
        assert split.test_group_ids not in seen
        seen.add(split.test_group_ids)
    assert len(seen) == 28


# --- Purge and embargo -----------------------------------------------------


def test_embargo_removes_buffer_after_test_group():
    # With embargo_pct = 0.01 on n_obs=1000, embargo_len = 10 obs.
    # If group 0 is the test (indices 0..124 for equal groups), the
    # indices [125, 135) must be purged from train.
    pk = PurgedKFold(n_groups=8, test_groups=1, embargo_pct=0.01)
    for split in pk.split(n_obs=1000):
        if split.test_group_ids == (0,):
            # Group 0 spans [0, 125). After embargo, indices 125..134
            # must NOT appear in train.
            train_set = set(split.train_idx.tolist())
            for i in range(125, 135):
                assert i not in train_set, (
                    f"embargo violated: index {i} present in train"
                )
            # Index 135 should be the first allowed train observation.
            assert 135 in train_set
            break
    else:
        pytest.fail("Did not visit test group (0,)")


def test_purge_removes_buffer_before_test_group():
    # Same setup, but test group 4 (indices 500..624).
    # The purge zone is [490, 500) and must not appear in train.
    pk = PurgedKFold(n_groups=8, test_groups=1, embargo_pct=0.01)
    for split in pk.split(n_obs=1000):
        if split.test_group_ids == (4,):
            train_set = set(split.train_idx.tolist())
            for i in range(490, 500):
                assert i not in train_set, (
                    f"purge violated: index {i} present in train"
                )
            # Index 489 should still be in train (outside purge zone)
            assert 489 in train_set
            break
    else:
        pytest.fail("Did not visit test group (4,)")


def test_zero_embargo_still_gives_disjoint_splits():
    """With embargo=0 the purge zone is empty; only the test indices themselves
    are excluded from train, and that's enough to prevent overlap."""
    pk = PurgedKFold(n_groups=5, test_groups=2, embargo_pct=0.0)
    for split in pk.split(n_obs=500):
        # train ∪ test should cover all indices when embargo=0
        all_idx = np.concatenate([split.train_idx, split.test_idx])
        assert sorted(all_idx.tolist()) == list(range(500))


# --- Coverage: union of test_idx across all splits = full series -----------


def test_union_of_test_groups_covers_all_indices():
    """If you take every split, the test_idx union equals {0, ..., n-1}."""
    pk = PurgedKFold(n_groups=10, test_groups=2, embargo_pct=0.0)
    covered: set[int] = set()
    for split in pk.split(n_obs=1000):
        covered.update(split.test_idx.tolist())
    assert covered == set(range(1000))
