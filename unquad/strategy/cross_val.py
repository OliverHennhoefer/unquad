import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Union
from copy import copy, deepcopy
from pyod.models.base import BaseDetector
from sklearn.model_selection import KFold

from unquad.estimation.properties.parameterization import set_params
from unquad.strategy.base import BaseStrategy


class CrossValidation(BaseStrategy):
    """
    Cross-validation conformal anomaly detection strategy.

    This class implements a conformal anomaly detection strategy using k-fold cross-validation.
    It trains multiple anomaly detection models on different training folds and calibrates them
    using the corresponding calibration set. The strategy supports model calibration with or
    without appending models during the process.

    Attributes:
        k (int): The number of folds in the k-fold cross-validation.
        plus (bool): A flag indicating whether to append models during calibration. Default is False.
        _detector_list (list): A list of trained anomaly detection models.
        _calibration_set (list): A list of calibration scores used for making decisions.

    Methods:
        __init__(k, plus=False):
            Initializes the CrossValidationConformal object with specified parameters.

        fit_calibrate(x, detector, seed=1):
            Fits and calibrates the anomaly detection model using k-fold cross-validation.

            Args:
                x (Union[pd.DataFrame, np.ndarray]): The data used to train and calibrate the detector.
                detector (BaseDetector): The base anomaly detection model to be used.
                seed (int, optional): The random seed for reproducibility. Default is 1.

            Returns:
                tuple: A tuple containing:
                    - list[BaseDetector]: A list of trained anomaly detection models.
                    - list[list]: A list of calibration scores.
    """

    def __init__(self, k: int, plus: bool = False):
        super().__init__(plus)
        self.k = k
        self.plus: bool = plus

        self._detector_list: [BaseDetector] = []
        self._calibration_set: [float] = []

        self.calib_id: [int] = None

    def fit_calibrate(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        detector: BaseDetector,
        weighted: bool = False,
        seed: int = 1,
    ) -> (list[BaseDetector], list[list]):

        _detector = detector

        folds = KFold(
            n_splits=self.k,
            shuffle=True,
            random_state=seed,
        )

        for i, (train_idx, calib_idx) in enumerate(
            tqdm(folds.split(x), total=self.k, desc="Training", disable=False)
        ):

            model = copy(_detector)
            model = set_params(model, seed=seed, random_iteration=True, iteration=i)
            model.fit(x[train_idx, :])

            self._detector_list.append(deepcopy(model)) if self.plus else None
            self._calibration_set.extend(model.decision_function(x[calib_idx, :]))

        if not self.plus:
            model = copy(_detector)
            model = set_params(
                model, seed=seed, random_iteration=True, iteration=(i + 1)
            )
            model.fit(x)
            self._detector_list.append(deepcopy(model))

        return self._detector_list, self._calibration_set

    @property
    def calibration_ids(self):
        return None
