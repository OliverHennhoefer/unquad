import math
from copy import copy, deepcopy
from typing import Union

import numpy as np
import pandas as pd
from pyod.models.base import BaseDetector
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

from unquad.estimator.parameter import set_params
from unquad.strategy.base import BaseStrategy


class BootstrapConformal(BaseStrategy):

    def __init__(
        self,
        resampling_ratio: float = None,
        n_bootstraps: int = None,
        n_calib: int = None,
        plus: bool = False,
    ):
        self.resampling_ratio: None | float = resampling_ratio
        self.n_bootstraps: None | int = n_bootstraps
        self.n_calib: None | int = n_calib
        self.plus: bool = plus

        self._detector_list: [BaseDetector] = []
        self._calibration_set: [float] = []

    def fit_calibrate(
        self, x: Union[pd.DataFrame, np.ndarray], detector: BaseDetector, seed: int = 1
    ) -> (list[BaseDetector], list[list]):
        self._configure(len(x))

        _detector = detector
        _generator = np.random.default_rng(seed)

        folds = ShuffleSplit(
            n_splits=self.n_bootstraps,
            train_size=self.resampling_ratio,
            random_state=seed,
        )

        n_folds = folds.get_n_splits()
        for i, (train_idx, calib_idx) in enumerate(
            tqdm(folds.split(x), total=n_folds, desc="Training", disable=False)
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

        if self.n_calib is not None:
            self._calibration_set = _generator.choice(
                self._calibration_set, size=self.n_calib, replace=False
            )

        return self._detector_list, self._calibration_set

    def _sanity_check(self):
        num_defined: int = sum(
            param is not None
            for param in (self.resampling_ratio, self.n_bootstraps, self.n_calib)
        )
        if num_defined != 2:
            raise ValueError(
                "Exactly 2 parameters (resampling_ratio, n_bootstraps, n_calib) must be defined."
            )

    def _configure(self, n: int) -> (float, int, int):

        def calculate_n_calib(
            n_train: int, n_bootstraps: int, resampling_ratio: float
        ) -> int:
            return math.ceil(n_bootstraps * n_train * (1 - resampling_ratio))

        def calculate_n_bootstraps(
            n_train: int, n_calib: int, resampling_ratio: float
        ) -> int:
            return math.ceil(n_calib / (n_train * (1 - resampling_ratio)))

        def calculate_resampling_ratio(
            n_train: int, n_bootstraps: int, n_calib: int
        ) -> float:
            return 1 - (n_calib / (n_bootstraps * n_train))

        if self.n_bootstraps is not None and self.resampling_ratio is not None:
            # has no effect
            self.n_calib = calculate_n_calib(
                n_train=n,
                n_bootstraps=self.n_bootstraps,
                resampling_ratio=self.resampling_ratio,
            )
        elif self.n_bootstraps is not None and self.n_calib is not None:
            self.resampling_ratio = calculate_resampling_ratio(
                n_train=n,
                n_bootstraps=self.n_bootstraps,
                n_calib=self.n_calib,
            )
        elif self.resampling_ratio is not None and self.n_calib is not None:
            self.n_bootstraps = calculate_n_bootstraps(
                n_train=n, resampling_ratio=self.resampling_ratio, n_calib=self.n_calib
            )
