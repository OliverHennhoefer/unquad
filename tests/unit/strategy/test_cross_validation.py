"""Unit tests for strategy/calibration/cross_validation.py."""

import numpy as np
import pytest

from nonconform import CrossValidation
from nonconform.enums import ConformalMode
from tests.conftest import MockDetector


class SetParamsTypeErrorDetector:
    """Detector whose set_params raises TypeError (unsupported random_state)."""

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "SetParamsTypeErrorDetector":
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.arange(len(X), dtype=float)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    def set_params(self, **params: object) -> "SetParamsTypeErrorDetector":
        raise TypeError("set_params is not supported")

    def __copy__(self) -> "SetParamsTypeErrorDetector":
        return SetParamsTypeErrorDetector()

    def __deepcopy__(self, memo: dict) -> "SetParamsTypeErrorDetector":
        return SetParamsTypeErrorDetector()


class TestCrossValidationFitCalibrate:
    """Tests for CrossValidation.fit_calibrate()."""

    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        rng = np.random.default_rng(42)
        return rng.standard_normal((50, 4))

    @pytest.mark.parametrize(
        ("mode", "n_models"),
        [("plus", 5), (ConformalMode.PLUS, 5), ("single_model", 1)],
        ids=["plus", "plus-enum", "single-model"],
    )
    def test_mode_controls_number_of_models(self, sample_data, mode, n_models):
        """Plus mode keeps all fold models; single-model mode refits one."""
        strategy = CrossValidation(k=5, mode=mode, shuffle=True)
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=MockDetector(), seed=123
        )
        assert len(detector_set) == n_models
        assert len(calib_scores) == len(sample_data)

    def test_calibration_ids_cover_all_samples(self, sample_data):
        """Calibration IDs cover all samples exactly once."""
        strategy = CrossValidation(k=5, mode="plus", shuffle=True)
        strategy.fit_calibrate(
            x=sample_data, detector=MockDetector(), seed=123, weighted=True
        )
        calib_ids = strategy.calibration_ids
        assert len(calib_ids) == len(sample_data)
        assert set(calib_ids) == set(range(len(sample_data)))

    def test_jackknife_factory_uses_leave_one_out(self):
        """Jackknife factory uses k=n (leave-one-out)."""
        rng = np.random.default_rng(7)
        sample_data = rng.standard_normal((12, 3))
        strategy = CrossValidation.jackknife(mode="plus")
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=MockDetector(), seed=7
        )
        assert len(detector_set) == len(sample_data)
        assert len(calib_scores) == len(sample_data)

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"mode": "invalid"}, ValueError, "mode must be one of"),
            ({"shuffle": 1}, TypeError, "shuffle must be a boolean value"),
        ],
    )
    def test_invalid_configuration_raises(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            CrossValidation(**({"k": 5, "mode": "plus"} | kwargs))

    @pytest.mark.parametrize(
        ("k", "match"), [(1, "k must be at least 2"), (51, "Not enough samples")]
    )
    def test_invalid_fold_count_raises(self, sample_data, k, match):
        strategy = CrossValidation(k=k, mode="plus", shuffle=True)
        with pytest.raises(ValueError, match=match):
            strategy.fit_calibrate(
                x=sample_data,
                detector=MockDetector(),
                seed=123,
            )

    def test_set_params_type_error_is_ignored(self, sample_data):
        strategy = CrossValidation(k=5, mode="single_model", shuffle=True)
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data,
            detector=SetParamsTypeErrorDetector(),
            seed=123,
        )
        assert len(detector_set) == 1
        assert len(calib_scores) == len(sample_data)

    def test_public_properties_reflect_configuration(self):
        plus = CrossValidation(k=7, mode="plus", shuffle=True)
        assert plus.k == 7
        assert plus.mode == "plus"
        jackknife = CrossValidation.jackknife(mode=ConformalMode.SINGLE_MODEL)
        assert jackknife.k is None
        assert jackknife.mode == "single_model"
