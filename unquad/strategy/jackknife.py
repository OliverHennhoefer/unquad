import numpy as np
import pandas as pd

from typing import Union, Optional
from pyod.models.base import BaseDetector

from unquad.strategy.base import BaseStrategy
from unquad.strategy.cross_val import CrossValidation


class Jackknife(BaseStrategy):
    """
    Jackknife conformal anomaly detection strategy.

    This class implements a conformal anomaly detection strategy using the jackknife resampling method.
    It leverages the `CrossValidationConformal` strategy with k set to the size of the dataset,
    effectively training a separate model for each sample (leave-one-out approach) and calibrating
    the models based on the left-out sample.

    Attributes:
        plus (bool): A flag indicating whether to append models during calibration. Default is False.
        strategy (BaseStrategy): An instance of the `CrossValidationConformal` strategy with k=1.

    Methods:
        __init__(plus=False):
            Initializes the JackknifeConformal object with the specified `plus` flag.

        fit_calibrate(x, detector, seed=None):
            Fits and calibrates the anomaly detection model using the jackknife resampling method.

            Args:
                x (Union[pd.DataFrame, np.ndarray]): The data used to train and calibrate the detector.
                detector (BaseDetector): The base anomaly detection model to be used.
                seed (Optional[int]): An optional random seed for reproducibility.

            Returns:
                tuple: A tuple containing:
                    - list[BaseDetector]: A list of trained anomaly detection models.
                    - list[list]: A list of calibration scores.
    """

    def __init__(self, plus: bool = False):
        super().__init__(plus)
        self.plus: bool = plus
        self.strategy: BaseStrategy = CrossValidation(k=1, plus=plus)

    def fit_calibrate(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        detector: BaseDetector,
        seed: Optional[int],
    ):

        self.strategy.k = len(x)
        return self.strategy.fit_calibrate(x, detector, seed)
