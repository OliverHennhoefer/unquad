import numpy as np
import pytest

import nonconform._internal.tuning as tuning
from nonconform._internal.tuning import (
    _compute_cv_log_likelihood,
    _fit_kde,
    _normalise_kernels,
)
from nonconform.enums import Kernel


@pytest.mark.parametrize(
    ("kernel_options", "expected"),
    [
        (Kernel.TRIANGULAR, [Kernel.TRIANGULAR]),
        (
            [Kernel.GAUSSIAN, Kernel.EPANECHNIKOV],
            [Kernel.GAUSSIAN, Kernel.EPANECHNIKOV],
        ),
        (
            [Kernel.BOX, Kernel.GAUSSIAN, Kernel.TRIANGULAR],
            [Kernel.BOX, Kernel.GAUSSIAN, Kernel.TRIANGULAR],
        ),
        ([], [Kernel.GAUSSIAN]),
    ],
    ids=["single-enum", "pair", "order-preserved", "empty-defaults-to-gaussian"],
)
def test_normalise_kernels_returns_nonempty_list(kernel_options, expected):
    result = _normalise_kernels(kernel_options)

    assert isinstance(result, list)
    assert result == expected


@pytest.mark.parametrize(
    "kernel", [Kernel.GAUSSIAN, Kernel.EPANECHNIKOV, Kernel.TRIANGULAR]
)
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_fit_kde_is_invariant_to_input_order(sample_calibration_data, kernel, weighted):
    data = sample_calibration_data(n_samples=100)
    rng = np.random.default_rng(42)
    weights = rng.uniform(0.5, 1.5, len(data)) if weighted else None
    order = rng.permutation(len(data))
    shuffled_weights = None if weights is None else weights[order]

    grid, pdf = _fit_kde(data, 1.0, kernel, weights).evaluate()
    shuffled_grid, shuffled_pdf = _fit_kde(
        data[order], 1.0, kernel, shuffled_weights
    ).evaluate()

    np.testing.assert_allclose(shuffled_grid, grid)
    np.testing.assert_allclose(shuffled_pdf, pdf)


@pytest.mark.parametrize(
    ("n_samples", "cv_folds", "weighted"),
    [(100, 5, False), (100, 10, False), (50, -1, False), (100, 5, True)],
    ids=["5-fold", "10-fold", "leave-one-out", "weighted"],
)
def test_cv_log_likelihood_is_finite_and_reproducible(
    sample_calibration_data, n_samples, cv_folds, weighted
):
    data = sample_calibration_data(n_samples=n_samples)
    weights = np.ones(n_samples) if weighted else None

    first = _compute_cv_log_likelihood(
        data,
        Kernel.GAUSSIAN,
        bandwidth=1.0,
        cv_folds=cv_folds,
        weights=weights,
        seed=42,
    )
    second = _compute_cv_log_likelihood(
        data,
        Kernel.GAUSSIAN,
        bandwidth=1.0,
        cv_folds=cv_folds,
        weights=weights,
        seed=42,
    )

    assert isinstance(first, float)
    assert np.isfinite(first)
    assert first == second


@pytest.mark.parametrize("error", [ValueError, np.linalg.LinAlgError])
def test_cv_log_likelihood_is_negative_infinity_when_kde_fit_fails(
    monkeypatch, sample_calibration_data, error
):
    def failing_fit(*args, **kwargs):
        raise error("singular data")

    monkeypatch.setattr(tuning, "_fit_kde", failing_fit)

    result = _compute_cv_log_likelihood(
        sample_calibration_data(n_samples=20), Kernel.GAUSSIAN, 1.0, cv_folds=2
    )

    assert result == -np.inf
