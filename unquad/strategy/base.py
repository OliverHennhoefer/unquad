import abc

import numpy as np
import pandas as pd

from typing import Union, Optional
from pyod.models.base import BaseDetector


class BaseStrategy(abc.ABC):
    """
    Abstract base class for calibration strategies in anomaly detection.

    Serves as the base for all calibration strategies used in anomaly detection.
    It provides a framework for fitting and calibrating detectors
    with specific strategies, while allowing customization through subclasses.

    Attributes:
        _plus (bool): A flag to indicate whether a specific adjustment
        (e.g., adding a constant) should be applied during calibration.
        Default is False.

    Methods:
        __init__(plus=False):
            Initializes the base strategy with an optional flag
            to enable specific adjustments during calibration.

        fit_calibrate(x, detector, seed):
            Abstract method for fitting and calibrating the detector.
            This must be implemented by subclasses.

            Args:
                x (Union[pd.DataFrame, np.ndarray]): The data used for calibration.
                detector (BaseDetector): The anomaly detection model to be calibrated.
                seed (Optional[int]): Optional random seed for reproducibility.

            Raises:
                NotImplementedError: This method must be implemented by subclasses.
    """

    def __init__(self, plus: bool = False):
        self._plus: bool = plus
        self._calibration_ids: [int]

    @abc.abstractmethod
    def fit_calibrate(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        detector: BaseDetector,
        weighted: Optional[bool],
        seed: Optional[int],
    ):

        raise NotImplementedError("The _calibrate() method must be implemented.")

    @property
    @abc.abstractmethod
    def calibration_ids(self):
        pass
