import math
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Union
from copy import copy, deepcopy
from pyod.models.base import BaseDetector
from sklearn.model_selection import ShuffleSplit

from unquad.estimation.properties.parameterization import set_params
from unquad.strategy.base import BaseStrategy


class Bootstrap(BaseStrategy):
    """
    Bootstrap conformal anomaly detection strategy.

    Implements a conformal strategy based on bootstrap resampling.
    It trains multiple detectors on resampled data and calibrates them using
    a calibration set.
    The strategy allows flexibility in the number of bootstraps,
    resampling ratio, and calibration set size.

    Attributes:
        _resampling_ratio (float, optional): The ratio of the training set size to the
        total dataset size for each bootstrap sample. Default is None.
        _n_bootstraps (int, optional): The number of bootstrap samples (models)
        to train. Default is None.
        _n_calib (int, optional): The number of calibration points to sample
        from the calibration set. Default is None.
        _plus (bool): A flag indicating whether to append models during calibration.
        Default is False.
        _detector_list (list): A list of trained anomaly detection models.
        _calibration_set (list): A list of calibration scores used for making decisions.

    Methods:
        __init__(resampling_ratio=None, n_bootstraps=None, n_calib=None, plus=False):
            Initializes the BootstrapConformal object with specified parameters.

        fit_calibrate(x, detector, seed=1):
            Fits and calibrates the anomaly detection model using bootstrap resampling.

            Args:
                x (Union[pd.DataFrame, np.ndarray]): Data to train/calibrate a detector.
                detector (BaseDetector): The base anomaly detection model to be used.
                seed (int, optional): The random seed for reproducibility. Default is 1.

            Returns:
                tuple: A tuple containing:
                    - list[BaseDetector]: A list of trained anomaly detection models.
                    - list[list]: A list of calibration scores.

        _sanity_check():
            Performs a sanity check to ensure exactly two of the parameters
            (resampling_ratio, n_bootstraps, n_calib) are defined.

        _configure(n):
            Configures the parameters for bootstrap resampling and calibration
            based on the provided data size.

            Args:
                n (int): The number of samples in the dataset.

            Returns:
                tuple: A tuple containing the configured values for resampling_ratio,
                    n_bootstraps, and n_calib.

            Raises:
                ValueError: If not exactly two parameters among resampling_ratio,
                n_bootstraps, and n_calib are defined.
    """

    def __init__(
        self,
        resampling_ratio: float = None,
        n_bootstraps: int = None,
        n_calib: int = None,
        plus: bool = False,
    ):
        super().__init__(plus)
        self._resampling_ratio: None | float = resampling_ratio
        self._n_bootstraps: None | int = n_bootstraps
        self._n_calib: None | int = n_calib
        self._plus: bool = plus

        self._detector_list: [BaseDetector] = []
        self._calibration_set: [float] = []

        self._calibration_ids: [int] = []

    def fit_calibrate(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        detector: BaseDetector,
        weighted: bool = False,
        seed: int = 1,
    ) -> (list[BaseDetector], list[list]):
        self._configure(len(x))

        _detector = detector
        _generator = np.random.default_rng(seed)

        folds = ShuffleSplit(
            n_splits=self._n_bootstraps,
            train_size=self._resampling_ratio,
            random_state=seed,
        )

        n_folds = folds.get_n_splits()
        for i, (train_idx, calib_idx) in enumerate(
            tqdm(folds.split(x), total=n_folds, desc="Training", disable=False)
        ):

            self._calibration_ids.extend(calib_idx.tolist())

            model = copy(_detector)
            model = set_params(model, seed=seed, random_iteration=True, iteration=i)
            model.fit(x[train_idx])

            self._detector_list.append(deepcopy(model)) if self._plus else None
            self._calibration_set.extend(model.decision_function(x[calib_idx]))

        if not self._plus:
            model = copy(_detector)
            model = set_params(
                model, seed=seed, random_iteration=True, iteration=(i + 1)  # noqa
            )
            model.fit(x)
            self._detector_list.append(deepcopy(model))

        if self._n_calib is not None:
            ids = _generator.choice(
                len(self._calibration_set), size=self._n_calib, replace=False
            )
            self._calibration_set = [self._calibration_set[i] for i in ids]
            self._calibration_ids = (
                [self._calibration_ids[i] for i in ids] if weighted else None
            )

        return self._detector_list, self._calibration_set

    def _sanity_check(self):
        num_defined: int = sum(
            param is not None
            for param in (self._resampling_ratio, self._n_bootstraps, self._n_calib)
        )
        if num_defined != 2:
            raise ValueError(
                "Exactly two parameters (resampling_ratio, n_bootstraps, n_calib) "
                "must be defined."
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

        if self._n_bootstraps is not None and self._resampling_ratio is not None:
            # has no effect
            self._n_calib = calculate_n_calib(
                n_train=n,
                n_bootstraps=self._n_bootstraps,
                resampling_ratio=self._resampling_ratio,
            )
        elif self._n_bootstraps is not None and self._n_calib is not None:
            self._resampling_ratio = calculate_resampling_ratio(
                n_train=n,
                n_bootstraps=self._n_bootstraps,
                n_calib=self._n_calib,
            )
        elif self._resampling_ratio is not None and self._n_calib is not None:
            self._n_bootstraps = calculate_n_bootstraps(
                n_train=n,
                resampling_ratio=self._resampling_ratio,
                n_calib=self._n_calib,
            )

    @property
    def calibration_ids(self):
        return self._calibration_ids
