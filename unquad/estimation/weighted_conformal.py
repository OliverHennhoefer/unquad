import os

from sklearn.linear_model import LogisticRegression

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # noqa: E402

import numpy as np
import pandas as pd

from typing import Union
from pyod.models.base import BaseDetector
from tqdm import tqdm

from unquad.estimation.properties.configuration import DetectorConfig
from unquad.estimation.properties.parameterization import set_params
from unquad.strategy.base import BaseStrategy
from unquad.utils.aggregation import aggregate
from unquad.utils.decorator import ensure_numpy_array
from unquad.utils.multiplicity import multiplicity_correction
from unquad.utils.statistical import get_decision, calculate_weighted_p_val


class WeightedConformalDetector:
    """
    Weighted Conformal anomaly detector using a specified detector and strategy.

    This class provides functionality to fit and predict using a conformal anomaly detection model.
    It uses an underlying detector model and a strategy for calibration, and applies statistical
    methods for anomaly detection, adjusting for multiple hypotheses as needed.

    The implementation is described in the publication
    "Model-free selective inference under covariate shift via weighted conformal p-values" that
    adapts the concepts of "Conformal Prediction Under Covariate Shift" to anomaly detection

    Attributes:
        detector (BaseDetector): The anomaly detection model to be used.
        strategy (BaseStrategy): The strategy used to calibrate the detector.
        config (DetectorConfig): Configuration parameters for the anomaly detection process.
        detector_set (list): A list of trained anomaly detection models used for predictions.
        calibration_set (list): A list of calibration values used to adjust predictions.

    Methods:
        __init__(detector, strategy, config=DetectorConfig()):
            Initializes the ConformalDetector with a detector, strategy, and configuration.

        fit(x):
            Fits the conformal anomaly detector model by calibrating it using the provided data.

            Args:
                x (Union[pd.DataFrame, np.ndarray]): The training data to calibrate the model.

            Returns:
                None

        predict(x, raw=False):
            Predicts anomaly scores and decisions for the given data.

            Args:
                x (Union[pd.DataFrame, np.ndarray]): The data to make predictions on.
                raw (bool): If True, returns raw p-values; otherwise, returns the final anomaly decisions.

            Returns:
                Union[np.ndarray, bool]: Either raw p-values or binary anomaly decisions.
    """

    def __init__(
        self,
        detector: BaseDetector,
        strategy: BaseStrategy,
        config: DetectorConfig = DetectorConfig(),
    ):
        self.detector: BaseDetector = set_params(detector, config.seed)

        self.strategy: BaseStrategy = strategy
        self.config: DetectorConfig = config

        self.detector_set: list[BaseDetector] = []
        self.calibration_set: list[float] = []

        self.calibration_samples: np.ndarray = np.ndarray([])

    @ensure_numpy_array
    def fit(self, x: Union[pd.DataFrame, np.ndarray]) -> None:

        self.detector_set, self.calibration_set = self.strategy.fit_calibrate(
            x=x, detector=self.detector, weighted=True, seed=self.config.seed
        )

        self.calibration_samples = x[self.strategy.calibration_ids]

    @ensure_numpy_array
    def predict(self, x: Union[pd.DataFrame, np.ndarray], raw=False):

        scores_list = [
            model.decision_function(x)
            for model in tqdm(
                self.detector_set,
                total=len(self.detector_set),
                desc="Inference",
                disable=self.config.silent,
            )
        ]

        w_cal, w_x = self._compute_weights(x)
        estimates = aggregate(self.config.aggregation, scores_list)
        p_val = calculate_weighted_p_val(estimates, self.calibration_set, w_x, w_cal)
        p_val_adj = multiplicity_correction(self.config.adjustment, p_val)

        return p_val if raw else get_decision(self.config.alpha, p_val_adj)

    def _compute_weights(self, test_instances: np.ndarray) -> ([float], [float]):
        calib_labeled = np.hstack(
            (self.calibration_samples, np.zeros((self.calibration_samples.shape[0], 1)))
        )
        tests_labeled = np.hstack(
            (test_instances, np.ones((test_instances.shape[0], 1)))
        )

        joint_labeled = np.vstack((calib_labeled, tests_labeled))
        np.random.default_rng(seed=self.config.seed).shuffle(joint_labeled)

        x = joint_labeled[:, :-1]
        y = joint_labeled[:, -1]

        model = LogisticRegression(random_state=self.config.seed)
        model.fit(x, y)

        calib_prob = model.predict_proba(calib_labeled[:, :-1])
        tests_prob = model.predict_proba(tests_labeled[:, :-1])

        w_calib = calib_prob[:, 1] / calib_prob[:, 0]
        w_tests = tests_prob[:, 1] / tests_prob[:, 0]

        clipped_w_calib = np.clip(w_calib, 0.35, 45)
        clipped_w_tests = np.clip(w_tests, 0.35, 45)

        return clipped_w_calib, clipped_w_tests
