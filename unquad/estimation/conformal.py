import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd

from typing import Union, Literal
from pyod.models.base import BaseDetector
from tqdm import tqdm

from unquad.estimation.properties.configuration import DetectorConfig
from unquad.estimation.properties.parameterization import set_params
from unquad.strategy.base import BaseStrategy
from unquad.utils.aggregation import aggregate
from unquad.utils.decorator import ensure_numpy_array
from unquad.utils.multiplicity import multiplicity_correction
from unquad.utils.statistical import calculate_p_val, get_decision


class ConformalDetector:
    """
    Conformal anomaly detector using a specified detector and strategy.

    Provides functionality to fit and predict using a conformal anomaly detectors.
    It uses an underlying detector model and a strategy for calibration,
    and applies statistical methods for anomaly detection,
    adjusting for multiple hypotheses as needed.

    Attributes:
        detector (BaseDetector): The anomaly detection model to be used.
        strategy (BaseStrategy): The strategy used to calibrate the detector.
        config (DetectorConfig): Configuration parameters.
        detector_set (list): A list of trained anomaly detectors used for predictions.
        calibration_set (list): A list of calibration values used to adjust predictions.

    Methods:
        __init__(detector, strategy, config=DetectorConfig()):
            Initializes the ConformalDetector
            with a detector, strategy, and configuration.

        fit(x):
            Fits the conformal anomaly detector model
            by calibrating it using the provided data.

            Args:
                x (Union[pd.DataFrame, np.ndarray]): Data to calibrate the model.

            Returns:
                None

        predict(x, output="decision"):
            Predicts anomaly status based on the specified output type.

            Args:
                x (Union[pd.DataFrame, np.ndarray]): The data to make predictions on.
                output (Literal["decision", "p-value", "score"]): Output type.
                Defaults to "decision".
                    - "decision": Returns decisions based on adj. p-values and alpha.
                    - "p-value": Returns the raw p-values.
                    - "score": Returns the aggregated anomaly scores (estimates).

            Returns:
                np.ndarray: An array containing the requested output
                (decisions, p-values, or scores).
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

    @ensure_numpy_array
    def fit(self, x: Union[pd.DataFrame, np.ndarray]) -> None:

        self.detector_set, self.calibration_set = self.strategy.fit_calibrate(
            x=x, detector=self.detector, seed=self.config.seed
        )

    @ensure_numpy_array
    def predict(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        output: Literal["decision", "p-value", "score"] = "decision",
    ):

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
