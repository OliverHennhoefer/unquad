import sys
from typing import Union

import numpy as np
import pandas as pd

from copy import copy
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
from sklearn.model_selection import KFold, ShuffleSplit, train_test_split

from noncon.enums.adjustment import Adjustment
from noncon.enums.method import Method
from noncon.errors.forbidden_model_error import ForbiddenModelError


class ConformalEstimator:
    """
    Objects of the ConformalEstimator class wrap a 'PyOD' detector to perform conformal anomaly detection (CAD).
    """

    def __init__(
        self,
        detector: BaseDetector,
        method: Method,
        adjustment: Adjustment,
        split: float = None,
        bootstrap: float = 0.2,
        alpha: float = 0.05,
        random_state: int = 0,
        silent: bool = False,
    ):
        self.detector = detector
        self.method = method
        self.procedure = adjustment

        self.split = split
        self.bootstrap = bootstrap
        self.alpha = alpha
        self.random_state = random_state
        self.silent = silent

        self.calibration_set = np.array([], dtype=np.float16)
        self.detector_set = []

        self._set_params()

    def fit(self, x: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Fits the given estimator on (non-anomalous) training data.
        :param x: Numpy Array or Pandas DataFrame, set of (non-anomalous) training data.
        :return: None
        """

        x = self._check_x(x)

        if self.method.value in ["SC"]:
            split = min(1_000.0, len(x) // 3) if self.split is None else self.split
            x_train, x_calib = train_test_split(
                x, test_size=split, shuffle=True, random_state=self.random_state
            )

            self.detector.fit(x_train)
            self.calibration_set = self.detector.decision_function(x_calib)

            return

        if self.method.value in ["CV", "CV+", "J", "J+"]:

            split = self._get_split(x, fallback=10)
            folds = KFold(n_splits=split, shuffle=True, random_state=self.random_state)

        elif self.method.value in ["JaB", "J+aB"]:
            split = self._get_split(x, fallback=30)
            folds = ShuffleSplit(
                n_splits=split,
                test_size=(1 - self.bootstrap),
                random_state=self.random_state,
            )
        else:
            folds = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        n_folds = folds.get_n_splits()
        i = None
        for i, (train_idx, calib_idx) in enumerate(
            tqdm(folds.split(x), total=n_folds, desc="Training", disable=self.silent)
        ):

            self._set_params(random_iteration=True, iteration=i)

            model = copy(self.detector)
            model.fit(x[train_idx, :])

            if self.method.value in ["CV+", "J+", "J+aB"]:
                self.detector_set.append(copy(model))

            self.calibration_set = np.append(
                self.calibration_set,
                model.decision_function(x[calib_idx, :]),
            )

        if self.method.value in ["CV", "J", "JaB"]:
            self._set_params(random_iteration=True, iteration=i)
            model = copy(self.detector)
            model.fit(x)
            self.detector = copy(model)

        return

    def predict(self, x: Union[pd.DataFrame, np.ndarray], raw=False) -> np.array:
        """
        Performs anomaly estimates with fitted conformal anomaly estimators.
        :param x: Numpy Array, set of test data for uncertainty-quantified anomaly estimation.
        :param raw: Boolean, whether the raw scores should be return or the anomaly labels.
        :return: Numpy Array, set of anomaly estimates obtained from the conformal anomaly estimators.
        """

        x = self._check_x(x)

        if self.method.value in ["CV+", "J+", "J+aB"]:
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
            estimates = np.median(scores_array, axis=0)
        else:
            estimates = self.detector.decision_function(x)

        p_val = self.marginal_p_val(estimates)

        if raw:
            return self.correction(p_val)  # scores
        else:
            return self.correction(p_val) <= self.alpha  # labels

    def marginal_p_val(self, scores: np.array) -> np.array:
        """
        Calculates marginal p-values given the test scores on test data.
        The p-values are determined by comparing the obtained set of calibration scores from calling '.fit()'.
        :param scores: Numpy Array, a set of anomaly scores (estimates) from trained estimator(s)
        :return: Numpy Array, Set of marginal p-values
        """

        p_val = np.sum(self.calibration_set >= scores[:, np.newaxis], axis=1)
        return (1.0 + p_val) / (1.0 + len(self.calibration_set))

    def correction(self, p_val: np.array) -> np.array:
        """
        Performs adjustments in regard to multiple testing in order to control the marginal FDR.
        :param p_val: Numpy Array, a set of p-values to be adjusted.
        :return: Numpy Array, a set of adjusted p-values
        """

        if self.procedure.value == "bh":
            p_val = stats.false_discovery_control(p_val, method="bh")
        if self.procedure.value == "by":
            p_val = stats.false_discovery_control(p_val, method="by")

        return p_val

    def _get_split(self, x: np.array, fallback: int) -> int:
        """
        Get number of splits to be performed on training data.
        :param x: Numpy Array, the training data
        :param fallback: Integer, fallback number of splits when undefined
        :return: Integer, number of splits
        """
        split = fallback if self.split is None else self.split
        if self.method.value in ["J", "J+"]:
            split = np.shape(x)[0]
        return split

    @staticmethod
    def _check_x(x):
        return x if isinstance(x, np.ndarray) else x.to_numpy()

    def _set_params(
        self, random_iteration: bool = False, iteration: int = None
    ) -> None:
        """
        Set parameters at run-time, depending on passed model object.
        Filters models unsuitable for one-class classification or otherwise unsupported.
        :param random_iteration: Boolean, whether parameters are set during cross-validation procedure.
        :param iteration: Integer, iteration within cross-validation procedure for seed randomization.
        :return: None
        """

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
                        "random_state": hash((iteration, self.random_state))
                        % 4294967296,
                    }
                )

            else:

                self.detector.set_params(
                    **{
                        "random_state": self.random_state,
                    }
                )
