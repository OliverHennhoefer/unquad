"""Shared fixtures for integration tests."""

from functools import partial

import numpy as np
import pytest


def _make_dataset(
    *,
    n_train=200,
    n_test=100,
    n_features=5,
    shift_mean=0.0,
    shift_scale=1.0,
    anomaly_rate_test=0.1,
    seed=42,
):
    """Generate (X_train, X_test, y_test) with an optional test covariate shift."""
    rng = np.random.default_rng(seed)

    # Training data: standard normal
    x_train = rng.standard_normal((n_train, n_features)).astype(np.float32)

    # Test data: (shifted) normal + anomalies
    n_anomalies = int(n_test * anomaly_rate_test)
    n_normal = n_test - n_anomalies

    x_test_normal = (
        rng.standard_normal((n_normal, n_features)) * shift_scale + shift_mean
    ).astype(np.float32)

    # Anomalies sit 3 units beyond the (shifted) normal mean
    x_test_anomaly = (
        rng.standard_normal((n_anomalies, n_features)) + shift_mean + 3
    ).astype(np.float32)

    x_test = np.vstack([x_test_normal, x_test_anomaly])
    y_test = np.array([0] * n_normal + [1] * n_anomalies)

    # Shuffle test data
    shuffle_idx = rng.permutation(n_test)
    x_test = x_test[shuffle_idx]
    y_test = y_test[shuffle_idx]

    return x_train, x_test, y_test


@pytest.fixture
def simple_dataset():
    """Create a simple synthetic dataset for basic integration testing."""
    return _make_dataset


@pytest.fixture
def shifted_dataset():
    """Create dataset with covariate shift between train and test.

    Returns:
        Factory function that generates (X_train, X_test, y_test) tuple
        where test distribution is shifted.
    """
    return partial(_make_dataset, shift_mean=1.0, shift_scale=0.5)
