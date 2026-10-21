"""Unit tests for strategy/calibration/jackknife_bootstrap.py."""

import numpy as np
import pytest

from nonconform import JackknifeBootstrap
from tests.conftest import MockDetector


class SetParamsValueErrorDetector:
    """Detector whose set_params raises ValueError (unsupported random_state)."""

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "SetParamsValueErrorDetector":
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.arange(len(X), dtype=float)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    def set_params(self, **params: object) -> "SetParamsValueErrorDetector":
        raise ValueError("set_params rejected random_state")

    def __copy__(self) -> "SetParamsValueErrorDetector":
        return SetParamsValueErrorDetector()

    def __deepcopy__(self, memo: dict) -> "SetParamsValueErrorDetector":
        return SetParamsValueErrorDetector()


class TestJackknifeBootstrapFitCalibrate:
    """Tests for JackknifeBootstrap.fit_calibrate()."""

    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        rng = np.random.default_rng(101)
        return rng.standard_normal((30, 5))

    @pytest.fixture
    def detector(self):
        """Deterministic mock detector."""
        return MockDetector(scores=np.array([0.1, 0.5, 0.9]))

    @pytest.mark.parametrize(
        ("n_bootstraps", "mode", "n_jobs", "n_models"),
        [(5, "plus", None, 5), (5, "single_model", None, 1), (3, "plus", -1, 3)],
        ids=["plus", "single-model", "parallel-plus"],
    )
    def test_fit_calibrate_scores_every_sample_out_of_bag(
        self, sample_data, detector, n_bootstraps, mode, n_jobs, n_models
    ):
        """Every sample gets an OOB calibration score, also with n_jobs=-1."""
        strategy = JackknifeBootstrap(n_bootstraps=n_bootstraps, mode=mode)
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=detector, seed=123, n_jobs=n_jobs
        )
        assert len(detector_set) == n_models
        assert len(calib_scores) == len(sample_data)
        assert not np.any(np.isnan(calib_scores))
        assert strategy.calibration_ids == list(range(len(sample_data)))

    def test_n_jobs_zero_raises(self, sample_data, detector):
        """n_jobs must be None, -1, or positive integer."""
        strategy = JackknifeBootstrap(n_bootstraps=3, mode="plus")
        with pytest.raises(ValueError, match="n_jobs"):
            strategy.fit_calibrate(x=sample_data, detector=detector, seed=123, n_jobs=0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_bootstraps": 3, "mode": "invalid"}, "mode must be one of"),
            ({"n_bootstraps": 1, "mode": "plus"}, "at least 2"),
        ],
    )
    def test_invalid_configuration_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            JackknifeBootstrap(**kwargs)

    def test_single_model_ignores_set_params_value_error(self, sample_data):
        strategy = JackknifeBootstrap(n_bootstraps=3, mode="single_model")
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data,
            detector=SetParamsValueErrorDetector(),
            seed=123,
        )
        assert len(detector_set) == 1
        assert len(calib_scores) == len(sample_data)

    def test_generate_bootstrap_indices_requires_at_least_two_samples(self):
        strategy = JackknifeBootstrap(n_bootstraps=3, mode="plus")
        with pytest.raises(ValueError, match="at least 2 samples"):
            strategy._generate_bootstrap_indices(
                generator=np.random.default_rng(0),
                n_samples=1,
            )

    def test_generate_bootstrap_indices_raises_when_coverage_incomplete(self):
        strategy = JackknifeBootstrap(n_bootstraps=2, mode="plus")
        # Force a degenerate setting to exercise the defensive incomplete-coverage path.
        strategy._n_bootstraps = 1
        with pytest.raises(
            ValueError, match="Failed to generate complete OOB coverage"
        ):
            strategy._generate_bootstrap_indices(
                generator=np.random.default_rng(0),
                n_samples=4,
            )

    def test_aggregate_predictions_handles_empty_and_median(self):
        strategy = JackknifeBootstrap(
            n_bootstraps=3,
            mode="plus",
            aggregation_method="median",
        )
        assert np.isnan(strategy._aggregate_predictions([]))
        assert strategy._aggregate_predictions([1.0, 3.0, 2.0]) == 2.0

    def test_aggregate_predictions_rejects_unsupported_method(self):
        strategy = JackknifeBootstrap(n_bootstraps=3, mode="plus")
        strategy._aggregation_method = "minimum"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unsupported aggregation"):
            strategy._aggregate_predictions([1.0, 2.0, 3.0])

    def test_compute_oob_scores_raises_for_missing_predictions(self):
        strategy = JackknifeBootstrap(n_bootstraps=2, mode="plus")
        x = np.arange(8, dtype=float).reshape(4, 2)
        strategy._bootstrap_models = [
            MockDetector(scores=np.array([0.1, 0.2, 0.3, 0.4])),
            MockDetector(scores=np.array([0.5, 0.6, 0.7, 0.8])),
        ]
        strategy._oob_mask = np.array(
            [
                [True, False, True, False],
                [False, False, False, True],
            ],
            dtype=bool,
        )

        with pytest.raises(ValueError, match="have no OOB predictions"):
            strategy._compute_oob_scores(x)

    def test_property_accessors_expose_configuration(self):
        strategy = JackknifeBootstrap(
            n_bootstraps=7,
            mode="plus",
            aggregation_method="median",
        )
        assert strategy.n_bootstraps == 7
        assert strategy.aggregation_method == "median"
