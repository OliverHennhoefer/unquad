import numpy as np
import pytest


@pytest.fixture
def sample_calibration_data():
    """Create sample calibration data with various distributions."""

    def _create(distribution="normal", n_samples=100, seed=42):
        rng = np.random.default_rng(seed)

        if distribution == "normal":
            return rng.standard_normal(n_samples).astype(np.float64)
        elif distribution == "uniform":
            return rng.uniform(-3, 3, n_samples).astype(np.float64)
        elif distribution == "bimodal":
            mode1 = rng.standard_normal(n_samples // 2) - 2
            mode2 = rng.standard_normal(n_samples // 2) + 2
            return np.concatenate([mode1, mode2]).astype(np.float64)
        elif distribution == "skewed":
            return rng.exponential(2.0, n_samples).astype(np.float64)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    return _create


@pytest.fixture
def sample_weights():
    """Create sample weight arrays for weighted KDE."""

    def _create(n_samples=100, pattern="uniform", seed=42):
        rng = np.random.default_rng(seed)

        if pattern == "uniform":
            return np.ones(n_samples, dtype=np.float64)
        elif pattern == "random":
            return rng.uniform(0.5, 1.5, n_samples).astype(np.float64)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

    return _create
