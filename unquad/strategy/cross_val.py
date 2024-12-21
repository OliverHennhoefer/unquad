from copy import copy, deepcopy
from typing import Union

import numpy as np
import pandas as pd
from pyod.models.base import BaseDetector
from sklearn.model_selection import KFold
from tqdm import tqdm

from unquad.estimator.parameter import set_params
from unquad.strategy.base import BaseStrategy


class CrossValidationConformal(BaseStrategy):

    def __init__(self, k: int, plus: bool = False):
        self.k = k
        self.plus: bool = plus

        self._detector_list: [BaseDetector] = []
        self._calibration_set: [float] = []

    def fit_calibrate(
        self, x: Union[pd.DataFrame, np.ndarray], detector: BaseDetector, seed: int = 1
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
