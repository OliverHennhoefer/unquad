"""Tests for Optuna-based KDE hyperparameter tuning."""

import numpy as np
import pytest

from nonconform._internal.tuning import tune_kde_hyperparameters
from nonconform.enums import Kernel

MULTIPLE_KERNELS = [Kernel.GAUSSIAN, Kernel.EPANECHNIKOV, Kernel.TRIANGULAR]


@pytest.mark.parametrize(
    ("kernel_options", "n_trials", "allowed_kernels"),
    [
        ([Kernel.GAUSSIAN], 5, {Kernel.GAUSSIAN}),
        (Kernel.GAUSSIAN, 5, {Kernel.GAUSSIAN}),
        ([], 5, {Kernel.GAUSSIAN}),
        (MULTIPLE_KERNELS, 10, set(MULTIPLE_KERNELS)),
    ],
    ids=["single-kernel-list", "kernel-enum", "empty-list", "multiple-kernels"],
)
def test_tuning_returns_study_backed_result(
    sample_calibration_data, kernel_options, n_trials, allowed_kernels
):
    result = tune_kde_hyperparameters(
        sample_calibration_data(n_samples=100), kernel_options, n_trials=n_trials
    )

    assert set(result) == {"bandwidth", "kernel", "best_score", "study"}
    assert result["bandwidth"] > 0
    assert isinstance(result["kernel"], Kernel)
    assert result["kernel"] in allowed_kernels
    assert isinstance(result["best_score"], int | float)
    assert result["best_score"] == result["study"].best_value
    assert len(result["study"].trials) == n_trials


@pytest.mark.parametrize("n_trials", [0, -1])
@pytest.mark.parametrize(
    "data",
    [
        np.random.default_rng(42).standard_normal(100),
        np.full(100, 5.0),
        np.array([3.14]),
    ],
    ids=["normal", "constant", "single-point"],
)
def test_heuristic_mode_returns_default_bandwidth_without_study(data, n_trials):
    result = tune_kde_hyperparameters(data, [Kernel.GAUSSIAN], n_trials=n_trials)

    assert result["bandwidth"] > 0
    assert result["kernel"] is Kernel.GAUSSIAN
    assert result["best_score"] is None
    assert result["study"] is None


@pytest.mark.parametrize(
    ("distribution", "n_samples", "weights", "cv_folds"),
    [
        ("normal", 50, None, -1),
        ("normal", 100, None, 5),
        ("normal", 100, "uniform", -1),
        ("normal", 100, "random", -1),
        ("uniform", 100, None, -1),
        ("bimodal", 100, None, -1),
        ("skewed", 100, None, -1),
    ],
    ids=[
        "leave-one-out",
        "k-fold",
        "uniform-weights",
        "random-weights",
        "uniform-distribution",
        "bimodal-distribution",
        "skewed-distribution",
    ],
)
def test_tuned_bandwidth_is_positive_across_data_and_cv_modes(
    sample_calibration_data, sample_weights, distribution, n_samples, weights, cv_folds
):
    data = sample_calibration_data(distribution=distribution, n_samples=n_samples)
    calibration_weights = (
        None
        if weights is None
        else sample_weights(n_samples=n_samples, pattern=weights)
    )

    result = tune_kde_hyperparameters(
        data,
        [Kernel.GAUSSIAN],
        n_trials=5,
        cv_folds=cv_folds,
        weights=calibration_weights,
    )

    assert result["bandwidth"] > 0


@pytest.mark.parametrize(
    ("data", "cv_folds"),
    [
        (np.array([1.0, 2.0]), 2),
        (np.array([1.0, 2.0]), -1),
        (np.full(100, 5.0), -1),
        (np.random.default_rng(42).standard_normal(50) * 1e10, -1),
        (np.random.default_rng(42).standard_normal(50) * 1e-10, -1),
    ],
    ids=["two-points-2-fold", "two-points", "constant", "large-values", "small-values"],
)
def test_tuned_bandwidth_is_finite_for_degenerate_scales(data, cv_folds):
    result = tune_kde_hyperparameters(
        data, [Kernel.GAUSSIAN], n_trials=5, cv_folds=cv_folds
    )

    assert np.isfinite(result["bandwidth"])
    assert result["bandwidth"] > 0


def test_same_seed_reproduces_bandwidth_and_kernel(sample_calibration_data):
    data = sample_calibration_data(n_samples=100)

    first = tune_kde_hyperparameters(data, MULTIPLE_KERNELS, n_trials=10, seed=42)
    second = tune_kde_hyperparameters(data, MULTIPLE_KERNELS, n_trials=10, seed=42)

    assert first["bandwidth"] == second["bandwidth"]
    assert first["kernel"] == second["kernel"]


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_tuning_is_invariant_to_input_order(
    sample_calibration_data, sample_weights, weighted
):
    data = sample_calibration_data(n_samples=100)
    weights = sample_weights(n_samples=100, pattern="random") if weighted else None
    order = np.random.default_rng(42).permutation(len(data))
    shuffled_weights = None if weights is None else weights[order]

    result = tune_kde_hyperparameters(
        data, [Kernel.GAUSSIAN], n_trials=5, weights=weights, seed=42
    )
    shuffled = tune_kde_hyperparameters(
        data[order], [Kernel.GAUSSIAN], n_trials=5, weights=shuffled_weights, seed=42
    )

    assert shuffled["bandwidth"] == result["bandwidth"]
