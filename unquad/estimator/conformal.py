import os
import warnings

from unquad.enums.aggregation import Aggregation
from unquad.estimator.split_config.bootstrap_config import BootstrapConfiguration

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

from scipy.stats import gaussian_kde
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
        A 'PyOD' anomaly estimator as derived from the BaseDetector class.

    method : enum:Method
        The conformal calibration scheme to be applied during training.

    adjustment: enum:Adjustment (optional, default=None)
        Statistical adjustment procedure to account for multiple testing.

    adjustment: enum:Adjustment (optional, default=Median)
        Aggregation function for ensemble methods like CV+

    split: float or integer (optional, fallback depending on 'method' parameter)
        The number of splits to be performed regarding estimator calibration.
        Has no effect for the 'split-conformal' calibration.
        Fallback values are defined, in case no parameter definition was set.
        In case the parameter value is <1.0, the split will be performed based
        on relative proportions (see sklearn::train_test_split())

    bootstrap_config: float (optional, default=30)
        The number of bootstraps to define regarding estimator calibration.
        Only has an effect for calibration procedures based on the 'split_config'.
        Fallback values are defined, in case no parameter definition was set.

    alpha: float (optional, default=0.1)
        Nominal FDR level to be controlled for.

    kde_sampling: integer (optional, default=None)
        For the default setting, only the calibration set as obtained from respective
        calibration methods will be used. For a given integer, the calibration set will be
        modeled by the kernel density function that will subsequently be sampled by the given
        number. This allows to increase the calibration set in order to obtain smaller
        p-values than what would be possible given only the calibration set. With that, higher
        certainty towards particular observations will be possible for the conformal estimator
        to represent, potentially resulting in higher FDR/Power for small calibration sets.
        Experimental.

    random_state: integer (optional, default=None)
        Random state to fix outcomes for determinism.

    silent: boolean (optional, default=False)
        Whether to show the progress regarding model training, calibration and inference.
    """

    def __init__(
        self,
        detector: BaseDetector,
        method: Method,
        adjustment: Adjustment = Adjustment.NONE,
        aggregation: Aggregation = Aggregation.MEDIAN,
        split: float = None,
        bootstrap_config: BootstrapConfiguration = None,
        alpha: float = 0.1,
        kde_sampling: int = None,
        random_state: int = None,
        silent: bool = False,
    ):

        self._sanity_check(method, split, alpha, bootstrap_config)

        self.detector = detector
        self.method = method
        self.adjustment = adjustment
        self.aggregation = aggregation

        self.split = split
        self.bootstrap_config = bootstrap_config
        self.alpha = alpha
        self.kde_sampling = kde_sampling
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
        x = check_array(x)

        c = None  # split_config configuration
        enforce_c = None  # split_config configuration
        if self.method in [Method.NAIVE]:
            self.detector.fit(x)
            self.calibration_set = self.detector.decision_function(x)

            self.sample_kde() if self.kde_sampling is not None else None
            return

        elif self.method in [Method.SPLIT_CONFORMAL]:
            split = min(1_000, len(x) // 3) if self.split is None else self.split
            x_train, x_calib = train_test_split(
                x, test_size=split, shuffle=True, random_state=self.random_state
            )

            self.detector.fit(x_train)
            self.calibration_set = self.detector.decision_function(x_calib)

            self.sample_kde() if self.kde_sampling is not None else None
            return

        elif self.method in [Method.JACKKNIFE, Method.JACKKNIFE_PLUS]:
            len_x = np.shape(x)[0]
            folds = KFold(n_splits=len_x, shuffle=True, random_state=self.random_state)

        elif self.method in [Method.CV, Method.CV_PLUS]:
            n_splits = self.split if self.split is not None else 10
            folds = KFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )

        elif self.method in [
            Method.JACKKNIFE_AFTER_BOOTSTRAP,
            Method.JACKKNIFE_PLUS_AFTER_BOOTSTRAP,
        ]:

            b = self.bootstrap_config.b
            m = self.bootstrap_config.m
            c = self.bootstrap_config.c
            enforce_c = self.bootstrap_config.enforce_c

            folds = ShuffleSplit(
                n_splits=b,
                test_size=m,
                random_state=self.random_state,
            )
        else:
            raise ValueError("Unknown method.")

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
            self._set_params(random_iteration=True, iteration=i + 1)
            model = copy(self.detector)
            model.fit(x)
            self.detector = deepcopy(model)

        if self.method in [
            Method.JACKKNIFE_AFTER_BOOTSTRAP,
            Method.JACKKNIFE_PLUS_AFTER_BOOTSTRAP,
        ]:
            if c is not None and enforce_c is True:
                self.calibration_set = np.random.choice(
                    self.calibration_set, size=c, replace=False
                )

        self.sample_kde() if self.kde_sampling is not None else None
        return

    def sample_kde(self) -> None:
        # Experimental; may help for small calibration sets or large batches when using multiple testing adjustments.
        kde = gaussian_kde(self.calibration_set)
        kde_sample = kde.resample(self.kde_sampling)[0]
        self.calibration_set = np.concatenate(
            (self.calibration_set, kde_sample), axis=0
        )

    def predict(self, x: Union[pd.DataFrame, np.ndarray], raw=False) -> np.array:
        """
        Performs anomaly estimates with fitted conformal anomaly estimators.
        :param x: Numpy Array or Pandas DataFrame, set of test data for anomaly estimation.
        :param raw: Boolean, whether the raw scores should be return or the anomaly labels.
        :return: Numpy Array, set of anomaly estimates obtained from the conformal anomaly estimators.
        """

        x = self._check_x(x)
        x = check_array(x)

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

        if self.adjustment.value == "bh":
            p_val = stats.false_discovery_control(p_val, method="bh")
        if self.adjustment.value == "by":
            p_val = stats.false_discovery_control(p_val, method="by")

        return p_val

    @staticmethod
    def _check_x(x):
        return x if isinstance(x, np.ndarray) else x.to_numpy()

    def _set_params(
        self, random_iteration: bool = False, iteration: int = None
    ) -> None:
        """
        Sets parameters at run-time, depending on passed model object.
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

    @staticmethod
    def _sanity_check(
        method: Method,
        split: Union[int, float],
        alpha: float,
        bootstrap_config: BootstrapConfiguration = None,
    ):

        # check alpha range
        if not (0.0 < alpha <= 1.0):
            raise ValueError("Parameter 'alpha' should be in range (0, 1].")

        if method in [Method.NAIVE] and split is not None:
            warnings.warn("Parameter 'split' has no effect for the naive method.")

        # check split_config configuration for JaB/J+aB
        if (
            method
            in [Method.JACKKNIFE_AFTER_BOOTSTRAP, Method.JACKKNIFE_PLUS_AFTER_BOOTSTRAP]
            and bootstrap_config is None
        ):
            raise ValueError("Parameter 'bootstrap_config' must be set for JaB/J+aB.")

        # check split number for CV/CV+
        if method in [Method.CV, Method.CV_PLUS] and split is None:
            warnings.warn(
                "Parameter 'split' is not defined and will default to a method specific value."
            )
