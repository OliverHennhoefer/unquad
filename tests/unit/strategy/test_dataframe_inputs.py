"""Tests for strategy support of pandas DataFrame inputs."""

import numpy as np
import pandas as pd
import pytest

from nonconform import CrossValidation, JackknifeBootstrap, Split
from tests.conftest import MockDetector


@pytest.mark.parametrize(
    ("strategy", "n_models", "n_calibration"),
    [
        (Split(n_calib=0.25), 1, 3),
        (CrossValidation(k=3, mode="plus", shuffle=True), 3, 12),
        (JackknifeBootstrap(n_bootstraps=3, mode="plus"), 3, 12),
    ],
    ids=["split", "cross-validation", "jackknife-bootstrap"],
)
def test_strategy_accepts_dataframe(strategy, n_models, n_calibration):
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.standard_normal((12, 3)))

    detector_set, calib_scores = strategy.fit_calibrate(
        x=df, detector=MockDetector(), seed=42
    )

    assert len(detector_set) == n_models
    assert len(calib_scores) == n_calibration
