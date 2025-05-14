"""Anomaly detection using conformal prediction strategies.

This module provides the `ConformalDetector` class, which integrates a base
anomaly detection model with a conformal prediction strategy to provide
statistically grounded anomaly scores, p-values, and decisions.

It also disables TensorFlow OneDNN optimizations by default via an environment
variable to ensure consistent numerical behavior across different CPU architectures,
which can be important for reproducibility in some anomaly detection models.
"""

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd

from typing import Union, Literal, List  # List imported for type hint clarity
from pyod.models.base import BaseDetector as PyODBaseDetector  # Alias for clarity
from tqdm import tqdm

from unquad.estimation.properties.configuration import DetectorConfig
from unquad.estimation.properties.parameterization import set_params
from unquad.strategy.base import BaseStrategy
from unquad.utils.aggregation import aggregate
from unquad.utils.decorator import ensure_numpy_array
from unquad.utils.multiplicity import multiplicity_correction
from unquad.utils.statistical import calculate_p_val, get_decision


class ConformalDetector:
    """Applies conformal prediction to an anomaly detector.

    This detector uses an underlying anomaly detection model and a specified
    strategy (e.g., split conformal, CV+) to calibrate non-conformity
    scores. It then uses these calibrated scores to make predictions on new
    data, providing options for raw scores, p-values, or binary decisions,
    while accounting for multiple hypothesis testing.

    Attributes:
        detector (PyODBaseDetector): The underlying anomaly detection model,
            initialized with a specific seed.
        strategy (BaseStrategy): The strategy used to fit and calibrate the
            detector (e.g., split conformal, cross-validation).
        config (DetectorConfig): Configuration object containing parameters like
            alpha, seed, and adjustment methods.
        detector_set (List[PyODBaseDetector]): A list of trained anomaly detector
            models. Populated after the `fit` method is called. Depending on
            the strategy, this might contain one or multiple models.
        calibration_set (List[float]): A list of calibration scores
            (non-conformity scores) obtained from the calibration data.
            Populated after the `fit` method is called.
    """

    def __init__(
        self,
        detector: PyODBaseDetector,
        strategy: BaseStrategy,
        config: DetectorConfig = DetectorConfig(),
    ):
        """Initializes the ConformalDetector.

        Args:
            detector (PyODBaseDetector): The base anomaly detection model to be
                used (e.g., an instance of a PyOD detector).
            strategy (BaseStrategy): The conformal strategy to apply for fitting
                and calibration.
            config (DetectorConfig, optional): Configuration settings for the
                detector, including significance level (alpha), random seed,
                and correction methods. Defaults to a standard `DetectorConfig`
                instance.
        """
        self.detector: PyODBaseDetector = set_params(detector, config.seed)
        self.strategy: BaseStrategy = strategy
        self.config: DetectorConfig = config

        self.detector_set: List[PyODBaseDetector] = []
        self.calibration_set: List[float] = []

    @ensure_numpy_array
    def fit(self, x: Union[pd.DataFrame, np.ndarray]) -> None:
        """Fits the detector model(s) and computes calibration scores.

        This method uses the specified strategy to train the base detector(s)
        on parts of the provided data and then calculates non-conformity
        scores on other parts (calibration set) to establish a baseline for
        typical behavior. The resulting trained models and calibration scores
        are stored in `self.detector_set` and `self.calibration_set`.

        Args:
            x (typing.Union[pd.DataFrame, np.ndarray]): The dataset used for
                fitting the model(s) and determining calibration scores.
                The strategy will dictate how this data is split or used.
        """
        self.detector_set, self.calibration_set = self.strategy.fit_calibrate(
            x=x, detector=self.detector, weighted=False, seed=self.config.seed
        )

    @ensure_numpy_array
    def predict(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        output: Literal["decision", "p-value", "score"] = "decision",
    ) -> np.ndarray:
        """Predicts anomaly status, p-values, or scores for new data.

        Based on the fitted models and calibration scores, this method evaluates
        new data points. It can return raw anomaly scores, p-values indicating
        how unusual each point is, or binary anomaly decisions based on the
        configured alpha level.

        Args:
            x (typing.Union[pd.DataFrame, np.ndarray]): The new data instances
                for which to make predictions.
            output (typing.Literal["decision", "p-value", "score"], optional):
                The type of output desired. Defaults to "decision".
                * "decision": Returns binary decisions (0 for normal, 1 for
                  anomaly) based on adjusted p-values and the configured
                  alpha.
                * "p-value": Returns the raw, unadjusted p-values for each
                  data point.
                * "score": Returns the aggregated anomaly scores (non-conformity
                  estimates) from the detector set for each data point.

        Returns:
            np.ndarray: An array containing the predictions. The content of the
            array depends on the `output` argument:
            - If "decision", a binary array of 0s and 1s.
            - If "p-value", an array of p-values (float).
            - If "score", an array of anomaly scores (float).
        """
        scores_list = [
            model.decision_function(x)
            for model in tqdm(
                self.detector_set,
                total=len(self.detector_set),
                desc="Inference",
                disable=self.config.silent,
            )
        ]

        estimates = aggregate(self.config.aggregation, scores_list)
        p_val = calculate_p_val(estimates, self.calibration_set)
        p_val_adj = multiplicity_correction(self.config.adjustment, p_val)

        if output == "score":
            return estimates
        elif output == "p-value":
            return p_val
        else:  # Default case is "decision"
            return get_decision(self.config.alpha, p_val_adj)
