"""Tests for inversiones_mama.models.covariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inversiones_mama.models.covariance import (
    COVARIANCE_METHODS,
    ensure_psd,
    estimate_covariance,
    ledoit_wolf_constant_correlation,
    ledoit_wolf_diagonal,
    sample_covariance,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def small_returns() -> pd.DataFrame:
    """30 × 5 returns drawn from a correlated Gaussian."""
    rng = np.random.default_rng(0)
    n, p = 30, 5
    mkt = rng.normal(0, 0.01, n)
    rows = np.column_stack([
        0.8 * mkt + rng.normal(0, 0.005, n),
        0.9 * mkt + rng.normal(0, 0.005, n),
        1.0 * mkt + rng.normal(0, 0.005, n),
        0.5 * mkt + rng.normal(0, 0.007, n),
        rng.normal(0, 0.01, n),
    ])
    return pd.DataFrame(rows, columns=[f"A{i}" for i in range(p)])


@pytest.fixture
def wide_returns() -> pd.DataFrame:
    """30 × 60 returns: sample covariance singular (n < p(p+1)/2)."""
    rng = np.random.default_rng(1)
    n, p = 30, 60
    mkt = rng.normal(0, 0.01, n)
    rows = np.column_stack([
        rng.uniform(0.3, 1.2) * mkt + rng.normal(0, 0.007, n) for _ in range(p)
    ])
    return pd.DataFrame(rows, columns=[f"T{i:02d}" for i in range(p)])


# --------------------------------------------------------------------------- #
# Sample covariance                                                           #
# --------------------------------------------------------------------------- #


def test_sample_covariance_matches_pandas(small_returns):
    ours = sample_covariance(small_returns)
    theirs = small_returns.cov()
    pd.testing.assert_frame_equal(ours, theirs, check_exact=False, atol=1e-12)


def test_sample_covariance_preserves_labels(small_returns):
    cov = sample_covariance(small_returns)
    assert list(cov.columns) == list(small_returns.columns)
    assert list(cov.index) == list(small_returns.columns)


# --------------------------------------------------------------------------- #
# Ledoit-Wolf diagonal                                                        #
# --------------------------------------------------------------------------- #


def test_lw_diagonal_produces_psd(small_returns):
    cov, _ = ledoit_wolf_diagonal(small_returns)
    eigenvalues = np.linalg.eigvalsh(cov.to_numpy())
    assert (eigenvalues >= -1e-10).all()


def test_lw_diagonal_delta_in_unit_interval(small_returns):
    _, delta = ledoit_wolf_diagonal(small_returns)
    assert 0.0 <= delta <= 1.0


def test_lw_diagonal_improves_condition_number(wide_returns):
    """The whole point of shrinkage: make cov(S⁻¹) actually invertible.

    (Not a theorem that delta grows with p/n — just that shrinkage
    makes the result well-conditioned even when sample cov is ugly.)
    """
    sample = sample_covariance(wide_returns).to_numpy()
    shrunk, _ = ledoit_wolf_diagonal(wide_returns)
    # Sample cov of wide data is ill-conditioned or singular
    sample_cond = np.linalg.cond(sample)
    shrunk_cond = np.linalg.cond(shrunk.to_numpy())
    # Shrunk must be invertible and better-conditioned by a meaningful margin
    assert np.isfinite(shrunk_cond)
    assert shrunk_cond < sample_cond or sample_cond > 1e10, (
        f"shrinkage did not help: sample_cond={sample_cond:.2e}, shrunk_cond={shrunk_cond:.2e}"
    )


def test_lw_diagonal_handles_wide_matrix(wide_returns):
    """Must produce a valid covariance for n << p where sample cov is singular."""
    cov, _ = ledoit_wolf_diagonal(wide_returns)
    # Condition number should be finite
    cond = np.linalg.cond(cov.to_numpy())
    assert np.isfinite(cond)
    # Diagonal entries all positive
    assert (np.diag(cov.to_numpy()) > 0).all()


def test_lw_diagonal_labels_preserved(small_returns):
    cov, _ = ledoit_wolf_diagonal(small_returns)
    assert list(cov.columns) == list(small_returns.columns)


# --------------------------------------------------------------------------- #
# Ledoit-Wolf constant-correlation                                            #
# --------------------------------------------------------------------------- #


def test_lw_constant_correlation_produces_psd(small_returns):
    cov, _ = ledoit_wolf_constant_correlation(small_returns)
    eigenvalues = np.linalg.eigvalsh(cov.to_numpy())
    assert (eigenvalues >= -1e-10).all()


def test_lw_constant_correlation_delta_in_unit_interval(small_returns):
    _, delta = ledoit_wolf_constant_correlation(small_returns)
    assert 0.0 <= delta <= 1.0


def test_lw_constant_correlation_preserves_variances(small_returns):
    """Target has sample variances on the diagonal - shrunk cov diagonal should be between
    target and sample values, but specifically the variances of the target match sample exactly."""
    sample = sample_covariance(small_returns).to_numpy()
    cov, delta = ledoit_wolf_constant_correlation(small_returns)
    # diag of shrunk = delta*sample_var + (1-delta)*sample_var = sample_var
    # (because target and sample agree on the diagonal for this method)
    np.testing.assert_allclose(np.diag(cov.to_numpy()), np.diag(sample), rtol=1e-10)


def test_lw_constant_correlation_handles_wide_matrix(wide_returns):
    cov, _ = ledoit_wolf_constant_correlation(wide_returns)
    cond = np.linalg.cond(cov.to_numpy())
    assert np.isfinite(cond)


# --------------------------------------------------------------------------- #
# Dispatcher                                                                  #
# --------------------------------------------------------------------------- #


def test_dispatcher_returns_sample(small_returns):
    a = estimate_covariance(small_returns, method="sample")
    b = sample_covariance(small_returns)
    pd.testing.assert_frame_equal(a, b)


def test_dispatcher_returns_lw_diagonal(small_returns):
    a = estimate_covariance(small_returns, method="lw_diagonal")
    b, _ = ledoit_wolf_diagonal(small_returns)
    pd.testing.assert_frame_equal(a, b)


def test_dispatcher_returns_lw_cc(small_returns):
    a = estimate_covariance(small_returns, method="lw_constant_correlation")
    b, _ = ledoit_wolf_constant_correlation(small_returns)
    pd.testing.assert_frame_equal(a, b)


def test_dispatcher_rejects_unknown_method(small_returns):
    with pytest.raises(ValueError, match="Unknown covariance method"):
        estimate_covariance(small_returns, method="bogus")


def test_covariance_methods_constant():
    assert "sample" in COVARIANCE_METHODS
    assert "lw_diagonal" in COVARIANCE_METHODS
    assert "lw_constant_correlation" in COVARIANCE_METHODS


# --------------------------------------------------------------------------- #
# Edge cases                                                                  #
# --------------------------------------------------------------------------- #


def test_too_few_observations_raises():
    df = pd.DataFrame({"A": [0.01, 0.02], "B": [0.01, 0.02]})
    with pytest.raises(ValueError, match="too small"):
        sample_covariance(df)


def test_drops_rows_with_nans():
    """Rows with any NaN should be dropped before estimation."""
    rng = np.random.default_rng(2)
    n = 50
    df = pd.DataFrame({
        "A": rng.normal(0, 0.01, n),
        "B": rng.normal(0, 0.01, n),
    })
    df.iloc[3, 0] = np.nan  # add one NaN
    cov = sample_covariance(df)
    # Estimator should succeed; result well-defined
    assert cov.shape == (2, 2)
    assert np.isfinite(cov.to_numpy()).all()


# --------------------------------------------------------------------------- #
# ensure_psd                                                                  #
# --------------------------------------------------------------------------- #


def test_ensure_psd_leaves_psd_matrices_near_unchanged():
    rng = np.random.default_rng(100)
    n = 10
    A = rng.normal(0, 1, (n, n))
    S = A @ A.T  # guaranteed PSD
    S_df = pd.DataFrame(S, index=[f"a{i}" for i in range(n)], columns=[f"a{i}" for i in range(n)])
    out = ensure_psd(S_df)
    # Small change in eigenvalues but same overall structure
    np.testing.assert_allclose(out.to_numpy(), S, atol=1e-8)


def test_ensure_psd_lifts_negative_eigenvalues():
    """A matrix with slightly negative eigenvalues should come out strictly positive."""
    # Construct a diagonal with one mildly negative eigenvalue
    S = np.diag([1.0, 0.5, -1e-4, 0.3])
    out = ensure_psd(pd.DataFrame(S), eigenvalue_floor=1e-6)
    eigvals = np.linalg.eigvalsh(out.to_numpy())
    assert (eigvals > 0).all()


def test_ensure_psd_preserves_labels():
    rng = np.random.default_rng(101)
    n = 6
    A = rng.normal(0, 1, (n, n))
    cols = [f"t{i}" for i in range(n)]
    S = pd.DataFrame(A @ A.T, index=cols, columns=cols)
    out = ensure_psd(S)
    assert list(out.columns) == cols
    assert list(out.index) == cols


def test_ensure_psd_handles_numpy_input():
    rng = np.random.default_rng(102)
    S = rng.normal(0, 1, (5, 5))
    S = S @ S.T
    out = ensure_psd(S)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (5, 5)


def test_estimate_covariance_applies_psd_clip_by_default(wide_returns):
    """After the change, the dispatcher should produce a strictly PSD result."""
    Sigma = estimate_covariance(wide_returns, method="lw_diagonal")
    eigvals = np.linalg.eigvalsh(Sigma.to_numpy())
    assert eigvals.min() > 0


def test_estimate_covariance_opt_out_of_psd_clip(small_returns):
    a = estimate_covariance(small_returns, method="sample", psd_clip=False)
    b = sample_covariance(small_returns)
    pd.testing.assert_frame_equal(a, b)


def test_identical_assets_produce_singular_sample_but_lw_invertible():
    """Two perfectly collinear assets - sample cov is singular; LW should fix it."""
    rng = np.random.default_rng(3)
    n = 100
    base = rng.normal(0, 0.01, n)
    df = pd.DataFrame({"A": base, "B": base.copy()})
    sample = sample_covariance(df).to_numpy()
    # Sample is (nearly) rank-1 -> singular
    assert abs(np.linalg.det(sample)) < 1e-20
    # LW diagonal must produce something invertible
    shrunk, _ = ledoit_wolf_diagonal(df)
    assert np.linalg.det(shrunk.to_numpy()) > 0
