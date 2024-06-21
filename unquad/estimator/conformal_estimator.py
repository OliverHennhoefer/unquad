import os
import warnings

from unquad.enums.aggregation import Aggregation
from unquad.estimator.split_configuration import SplitConfiguration

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # noqa: E402

import sys
import numpy as np
import pandas as pd

from typing import Union
from copy import copy, deepcopy
from scipy import stats
from tqdm import tqdm

from pyod.models.alad import ALAD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.rgraph import RGraph
from pyod.models.sampling import Sampling
from pyod.models.so_gaal import SO_GAAL
from pyod.models.sos import SOS
from pyod.models.vae import VAE
from pyod.models.base import BaseDetector

from sklearn.utils import check_array
from sklearn.model_selection import KFold, ShuffleSplit, train_test_split

from unquad.enums.adjustment import Adjustment
from unquad.enums.method import Method
from unquad.errors.forbidden_model_error import ForbiddenModelError


class ConformalEstimator:
    """
    Wrapper class for 'PyOD' anomaly estimators.
    ConformalEstimator allows to fit a model by applying a conformal calibration scheme.
    Conformal anomaly detection translates anomaly scores into statistical p-values
    by comparing anomaly estimates of test data to a set of calibration scores obtained
    on normal data (see One-Class Classification). Obtained p-values instead of the usual
    anomaly estimates allow for statistical FDR-control by procedures like Benjamini-Hochberg.
    Conformal anomaly detection is based on the principles of conformal prediction.

    Parameters
    ----------
    detector : BaseDetector
        An untrained instantiation of 'PyOD' anomaly estimator.

    method : enum:Method
        The conformal method for model training.

    adjustment: enum:Adjustment (optional, default=Benjamini-Hochberg)
        Statistical multiplicity adjustment.

    aggregation: enum:Aggregation (optional, default=Median)
        Aggregation function for ensemble methods.

    split_config float or integer (optional, fallback depending on 'method' parameter)
        The number of splits to be performed regarding estimator calibration.
        Has no effect for the 'split-conformal' calibration.
        Fallback values are defined, in case no parameter definition was set.
        In case the parameter value is <1.0, the split will be performed based
        on relative proportions (see sklearn::train_test_split())

    alpha: float (optional, default=0.1)
        Nominal FDR level.

    seed: integer (optional, default=None)
        Random state for reproducibility.

    silent: boolean (optional, default=False)
        Whether to indicate training/calibration and inference progress.
    """

    def __init__(
        self,
        detector: BaseDetector,
        method: Method,
        split: SplitConfiguration = None,
        adjustment: Adjustment = Adjustment.BENJAMINI_HOCHBERG,
        aggregation: Aggregation = Aggregation.MEDIAN,
        alpha: float = 0.2,
        seed: int = None,
        silent: bool = False,
    ):
        self._sanity_check(method, split, alpha)

        self.method = method
        self.detector = detector
        self.split = split
        self.adjustment = adjustment
        self.aggregation = aggregation
        self.alpha = alpha
        self.seed = seed
        self.silent = silent

        self.calibration_set = np.array([], dtype=np.float16)
        self.detector_set = []

        self._set_params()
        np.random.seed(self.seed)

    def fit(self, x: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Fits and calibrates an anomaly estimator.
        :param x: Numpy Array or Pandas DataFrame, training data.
        :return: None
        """

        x = self._check_x(x)
        n_calib = None

        if self.method in [Method.NAIVE]:
            self.detector.fit(x)
            self.calibration_set = self.detector.decision_function(x)
            return

        elif self.method in [Method.SPLIT_CONFORMAL]:
            x_train, x_calib = train_test_split(
                x,
                test_size=self.split.n_split,
                shuffle=True,
                random_state=self.seed,
            )

            self.detector.fit(x_train)
            self.calibration_set = self.detector.decision_function(x_calib)
            return

        elif self.method in [Method.JACKKNIFE, Method.JACKKNIFE_PLUS]:
            len_x = np.shape(x)[0]
            folds = KFold(n_splits=len_x, shuffle=True, random_state=self.seed)

        elif self.method in [Method.CV, Method.CV_PLUS]:
            folds = KFold(
                n_splits=self.split.n_split,
                shuffle=True,
                random_state=self.seed,
            )

        elif self.method in [
            Method.JACKKNIFE_AFTER_BOOTSTRAP,
            Method.JACKKNIFE_PLUS_AFTER_BOOTSTRAP,
        ]:
            n_calib = self.split.n_calib
            self.split.configure(n_train=np.shape(x)[0])

            folds = ShuffleSplit(
                n_splits=self.split.n_bootstraps,
                train_size=self.split.n_split,
                random_state=self.seed,
            )
        else:
            raise ValueError("Unknown conformal method.")

        n_folds = folds.get_n_splits()
        i = None
        for i, (train_idx, calib_idx) in enumerate(
            tqdm(folds.split(x), total=n_folds, desc="Training", disable=self.silent)
        ):
            self._set_params(random_iteration=True, iteration=i)

            model = copy(self.detector)
            model.fit(x[train_idx, :])

            if self.method in [
                Method.CV_PLUS,
                Method.JACKKNIFE_PLUS,
                Method.JACKKNIFE_PLUS_AFTER_BOOTSTRAP,
            ]:
                self.detector_set.append(deepcopy(model))

            self.calibration_set = np.append(
                self.calibration_set,
                model.decision_function(x[calib_idx, :]),
            )

        if self.method in [
            Method.CV,
            Method.JACKKNIFE,
            Method.JACKKNIFE_AFTER_BOOTSTRAP,
        ]:
            self._set_params(random_iteration=True, iteration=(i + 1))
            model = copy(self.detector)
            model.fit(x)
            self.detector = deepcopy(model)

        if self.method in [
            Method.JACKKNIFE_AFTER_BOOTSTRAP,
            Method.JACKKNIFE_PLUS_AFTER_BOOTSTRAP,
        ]:
            if n_calib is not None:
                self.calibration_set = np.random.choice(
                    self.calibration_set, size=n_calib, replace=False
                )

        return

    def predict(self, x: Union[pd.DataFrame, np.ndarray], raw=False) -> np.array:
        """
        Performs anomaly estimates with fitted conformal anomaly estimators.
        :param x: Numpy Array or Pandas DataFrame, set of test data for anomaly estimation.
        :param raw: Boolean, whether the raw scores should be return or the anomaly labels.
        :return: Numpy Array, set of anomaly estimates obtained from the conformal anomaly estimators.
        """

        x = self._check_x(x)

        if self.method in [
            Method.CV_PLUS,
            Method.JACKKNIFE_PLUS,
            Method.JACKKNIFE_PLUS_AFTER_BOOTSTRAP,
        ]:
            scores_array = np.stack(
                [
                    model.decision_function(x)
                    for model in tqdm(
                        self.detector_set,
                        total=len(self.detector_set),
                        desc="Inference",
                        disable=self.silent,
                    )
                ],
                axis=0,
            )
            if self.aggregation in [Aggregation.MEDIAN]:
                estimates = np.median(scores_array, axis=0)
            elif self.aggregation in [Aggregation.MEAN]:
                estimates = np.mean(scores_array, axis=0)
            elif self.aggregation in [Aggregation.MINIMUM]:
                estimates = np.min(scores_array, axis=0)
            elif self.aggregation in [Aggregation.MAXIMUM]:
                estimates = np.max(scores_array, axis=0)
            else:
                raise ValueError("No valid aggregation function defined.")
        else:
            estimates = self.detector.decision_function(x)

        p_val = self._calculate_p_val(estimates)

        if raw:
            return self._correct_multiplicity(p_val)
        else:
            return self._correct_multiplicity(p_val) <= self.alpha

    def _calculate_p_val(self, scores: np.array) -> np.array:
        p_val = np.sum(self.calibration_set >= scores[:, np.newaxis], axis=1)
        return (1.0 + p_val) / (1.0 + len(self.calibration_set))

    def _correct_multiplicity(self, p_val: np.array) -> np.array:
        if self.adjustment.value == "bh":
            p_val = stats.false_discovery_control(p_val, method="bh")
        if self.adjustment.value == "by":
            p_val = stats.false_discovery_control(p_val, method="by")

        return p_val

    @staticmethod
    def _check_x(x):
        if isinstance(x, np.ndarray):
            return check_array(x)
        elif isinstance(x, pd.DataFrame):
            return check_array(x.to_numpy())
        else:
            raise TypeError("Expected a pd.DataFrame or np.ndarray.")

    def _set_params(
        self, random_iteration: bool = False, iteration: int = None
    ) -> None:
        if self.detector.__class__ in [
            ALAD,
            CBLOF,
            COF,
            DeepSVDD,
            MO_GAAL,
            RGraph,
            Sampling,
            SO_GAAL,
            SOS,
            VAE,
        ]:
            raise ForbiddenModelError(
                f"{self.detector.__class__.__name__} is not supported."
            )

        if "contamination" in self.detector.get_params().keys():
            self.detector.set_params(
                **{
                    "contamination": sys.float_info.min,  # One-Class Classification
                }
            )

        if "n_jobs" in self.detector.get_params().keys():
            self.detector.set_params(
                **{
                    "n_jobs": -1,
                }
            )

        if "random_state" in self.detector.get_params().keys():
            if random_iteration and iteration is not None:
                self.detector.set_params(
                    **{
                        "random_state": hash((iteration, self.seed)) % 4294967296,
                    }
                )

            else:
                self.detector.set_params(
                    **{
                        "random_state": self.seed,
                    }
                )

    @staticmethod
    def _sanity_check(method: Method, split: SplitConfiguration, alpha: float):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("Parameter 'alpha' should be in range (0, 1].")

        if (
            method in [Method.NAIVE, Method.JACKKNIFE, Method.JACKKNIFE_PLUS]
            and split is not None
        ):
            warnings.warn(
                "Split configuration has no effect for defined conformal method."
            )

        if (
            method
            in [
                Method.SPLIT_CONFORMAL,
                Method.CV,
                Method.CV_PLUS,
                Method.JACKKNIFE_AFTER_BOOTSTRAP,
                Method.JACKKNIFE_PLUS_AFTER_BOOTSTRAP,
            ]
            and split is None
        ):
            raise ValueError(f"Split configuration must be defined for {method}.")

        if (method in [Method.CV, Method.CV_PLUS]) and split.n_params != 1:
            raise ValueError(
                f"Split configuration must only define 'n_split' for {method}."
            )
