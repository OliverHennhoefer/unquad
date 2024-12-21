import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # noqa: E402

import numpy as np
import pandas as pd

from typing import Union
from pyod.models.base import BaseDetector
from tqdm import tqdm

from unquad.estimator.configuration import EstimatorConfig
from unquad.estimator.parameter import set_params
from unquad.strategy.base import BaseStrategy
from unquad.utils.aggregation import aggregate
from unquad.utils.decorator.performance import ensure_numpy_array
from unquad.utils.multiplicity import multiplicity_correction
from unquad.utils.statistical import calculate_p_val, get_decision


class ConformalDetector:

    def __init__(
        self,
        detector: BaseDetector,
        strategy: BaseStrategy,
        config: EstimatorConfig = EstimatorConfig(),
    ):
        self.detector: BaseDetector = set_params(detector, config.seed)

        self.strategy: BaseStrategy = strategy
        self.config: EstimatorConfig = config

        self.detector_set: list[BaseDetector] = []
        self.calibration_set: list[float] = []

    @ensure_numpy_array
    def fit(self, x: Union[pd.DataFrame, np.ndarray]) -> None:

        self.detector_set, self.calibration_set = self.strategy.fit_calibrate(
            x=x, detector=self.detector, seed=self.config.seed
        )

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

        estimates = aggregate(self.config.aggregation, scores_list)
        p_val = calculate_p_val(estimates, self.calibration_set)
        p_val_adj = multiplicity_correction(self.config.adjustment, p_val)

        return p_val if raw else get_decision(self.config.alpha, p_val_adj)
