import numpy as np
import pandas as pd

from typing import Union
from pyod.models.base import BaseDetector

from unquad.strategy.base import BaseStrategy
from unquad.strategy.cross_val import CrossValidation


class Jackknife(BaseStrategy):
    """
    Jackknife conformal anomaly detection strategy.

    Implements a conformal strategy using the jackknife resampling method.
    Basically `CrossValidationConformal` with k set to the size of the dataset,
    effectively training a separate model for each sample (loo) and calibrating
    the models based on the left-out sample.

    Attributes:
        _plus (bool): A flag indicating whether to append models during calibration.
        Default is False.
        _strategy (BaseStrategy): `CrossValidationConformal` strategy with k=1.

    Methods:
        __init__(plus=False):
            Initializes the JackknifeConformal object with the specified `plus` flag.

        fit_calibrate(x, detector, seed=None):
            Fits and calibrates the detector using the jackknife resampling method.

            Args:
                x (Union[pd.DataFrame, np.ndarray]): Data used to train/calibrate.
                detector (BaseDetector): The base anomaly detection model to be used.
                seed (Optional[int]): An optional random seed for reproducibility.

            Returns:
                tuple: A tuple containing:
                    - list[BaseDetector]: A list of trained anomaly detection models.
                    - list[list]: A list of calibration scores.
    """

    def __init__(self, plus: bool = False):
        super().__init__(plus)
        self._plus: bool = plus
        self._strategy: BaseStrategy = CrossValidation(k=1, plus=plus)
        self._calibration_ids: [int] = None

        self._detector_list: [BaseDetector] = []
        self._calibration_set: [float] = []

    def fit_calibrate(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        detector: BaseDetector,
        weighted: bool = False,
        seed: int = 1,
    ):

        self._strategy._k = len(x)
        self._detector_list, self._calibration_set = self._strategy.fit_calibrate(
            x, detector, weighted, seed
        )
        self._calibration_ids = self._strategy.calibration_ids
        return self._detector_list, self._calibration_set

    @property
    def calibration_ids(self):
        return self._calibration_ids
