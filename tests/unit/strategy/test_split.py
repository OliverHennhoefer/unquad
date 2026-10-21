"""Unit tests for the Split calibration strategy."""

import numpy as np
import pytest

from nonconform import Split
from tests.conftest import MockDetector


@pytest.fixture
def sample_data():
    """Sample training data."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 5))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, 0.1), ({"n_calib": 0.3}, 0.3), ({"n_calib": 100}, 100)],
    ids=["default", "proportional", "absolute"],
)
def test_calib_size_reflects_configuration(kwargs, expected):
    assert Split(**kwargs).calib_size == expected


@pytest.mark.parametrize(
    ("n_samples", "n_calib", "expected_calib"),
    [(100, 0.2, 20), (100, 30, 30), (10, 0.3, 3)],
    ids=["proportional", "absolute", "small-dataset"],
)
def test_fit_calibrate_returns_one_fitted_seeded_model(
    n_samples, n_calib, expected_calib
):
    """fit_calibrate trains one seeded model on the non-calibration rows."""
    x = np.random.default_rng(42).standard_normal((n_samples, 5))

    detector_set, calib_scores = Split(n_calib=n_calib).fit_calibrate(
        x=x, detector=MockDetector(), seed=123
    )

    assert isinstance(detector_set, list)
    assert len(detector_set) == 1
    assert detector_set[0]._fitted
    assert detector_set[0]._training_size == n_samples - expected_calib
    assert detector_set[0].get_params()["random_state"] == 123
    assert isinstance(calib_scores, np.ndarray)
    assert len(calib_scores) == expected_calib


@pytest.mark.parametrize("weighted", [True, False])
def test_calibration_ids_are_stored_only_when_weighted(sample_data, weighted):
    strategy = Split(n_calib=0.2)
    strategy.fit_calibrate(
        x=sample_data, detector=MockDetector(), seed=42, weighted=weighted
    )

    if weighted:
        assert len(strategy.calibration_ids) == 20
        assert set(strategy.calibration_ids) <= set(range(len(sample_data)))
    else:
        assert strategy.calibration_ids is None


@pytest.mark.parametrize(
    ("second_seed", "same_split"), [(42, True), (123, False)], ids=["same", "different"]
)
def test_seed_determines_calibration_split(sample_data, second_seed, same_split):
    calibration_ids = []
    for seed in (42, second_seed):
        strategy = Split(n_calib=0.2)
        strategy.fit_calibrate(
            x=sample_data.copy(), detector=MockDetector(), seed=seed, weighted=True
        )
        calibration_ids.append(strategy.calibration_ids)

    assert (calibration_ids[0] == calibration_ids[1]) is same_split


@pytest.mark.parametrize(
    ("n_calib", "error", "match"),
    [
        (0.0, ValueError, "Proportional n_calib"),
        (1.0, ValueError, "Proportional n_calib"),
        (-0.1, ValueError, "Proportional n_calib"),
        (0, ValueError, "Absolute n_calib"),
        (10, ValueError, "No training data remaining"),
        (11, ValueError, "Calibration size"),
        ("bad", TypeError, "n_calib must be int or float"),
    ],
)
def test_invalid_n_calib_raises(n_calib, error, match):
    x = np.random.default_rng(42).standard_normal((10, 2))
    strategy = Split(n_calib=n_calib)

    with pytest.raises(error, match=match):
        strategy.fit_calibrate(x=x, detector=MockDetector(), seed=42)
