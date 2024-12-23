import numpy as np
import pandas as pd

from typing import Union
from pyod.models.base import BaseDetector
from sklearn.model_selection import train_test_split

from unquad.strategy.base import BaseStrategy


class SplitConformal(BaseStrategy):
    """
    Split conformal anomaly detection strategy.

    This class implements a conformal anomaly detection strategy using a split approach.
    It splits the data into a training set and a calibration set, trains an anomaly detection model
    on the training set, and uses the calibration set to calibrate the model.

    Attributes:
        calib_size (float | int): The proportion or absolute size of the data to be used for calibration.
            If a float, it represents the proportion of the dataset to be used for calibration.
            If an integer, it represents the absolute number of samples to be used for calibration.
            Default is 0.1.

    Methods:
        __init__(calib_size=0.1):
            Initializes the SplitConformal object with the specified calibration set size.

        fit_calibrate(x, detector, seed=1):
            Fits and calibrates the anomaly detection model using a train-test split.

            Args:
                x (Union[pd.DataFrame, np.ndarray]): The data used to train and calibrate the detector.
                detector (BaseDetector): The base anomaly detection model to be used.
                seed (int, optional): The random seed for reproducibility. Default is 1.

            Returns:
                tuple: A tuple containing:
                    - list[BaseDetector]: A list containing the trained anomaly detection model.
                    - list[list]: A list of calibration scores for the calibration set.
    """

    def __init__(self, calib_size: float | int = 0.1) -> None:
        super().__init__()
        self.calib_size: float | int = calib_size

    def fit_calibrate(
        self, x: Union[pd.DataFrame, np.ndarray], detector: BaseDetector, seed: int = 1
    ) -> (list[BaseDetector], list[list]):

        train, calib = train_test_split(
            x, test_size=self.calib_size, shuffle=True, random_state=seed
        )
        detector.fit(train)
        calibration_set = detector.decision_function(calib)

        return [detector], calibration_set
