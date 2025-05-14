import os

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd

from typing import Union, Literal, List, Tuple  # Added List, Tuple
from pyod.models.base import BaseDetector
from tqdm import tqdm

from unquad.estimation.properties.configuration import DetectorConfig
from unquad.estimation.properties.parameterization import set_params
from unquad.strategy.base import BaseStrategy
from unquad.utils.aggregation import aggregate
from unquad.utils.decorator import ensure_numpy_array
from unquad.utils.multiplicity import multiplicity_correction
from unquad.utils.statistical import get_decision, calculate_weighted_p_val


class WeightedConformalDetector:
    """Weighted conformal anomaly detector with covariate shift adaptation.

    This detector implements a conformal prediction framework for anomaly
    detection, incorporating weights to adapt to potential covariate shifts
    between calibration and test data. It leverages an underlying PyOD
    detector, a calibration strategy, and a configuration object.

    The weighting mechanism estimates the density ratio between calibration
    and test instances using a logistic regression model trained to
    distinguish between them. These weights are then used in the calculation
    of p-values.

    The methodology is inspired by concepts for handling covariate shift in
    conformal prediction, adapted for anomaly detection.

    Attributes:
        detector (BaseDetector): The underlying PyOD anomaly detection model,
            initialized with parameters from the config.
        strategy (BaseStrategy): The calibration strategy (e.g., Bootstrap,
            CrossValidation) used to generate calibration scores and identify
            calibration samples.
        config (DetectorConfig): Configuration settings including significance
            level (alpha), p-value adjustment method, aggregation method,
            random seed, and verbosity.
        detector_set (List[BaseDetector]): A list of one or more trained
            detector instances, populated by the `fit` method via the strategy.
        calibration_set (List[float]): A list of non-conformity scores obtained
            from the calibration process, populated by the `fit` method.
        calibration_samples (numpy.ndarray): The actual data instances from the
            input `x` that were used for calibration, identified by the
            strategy. Populated by the `fit` method.
    """

    def __init__(
        self,
        detector: BaseDetector,
        strategy: BaseStrategy,
        config: DetectorConfig = DetectorConfig(),
    ):
        """Initializes the WeightedConformalDetector.

        Args:
            detector (BaseDetector): A PyOD anomaly detector instance. It will
                be configured with the seed from `config`.
            strategy (BaseStrategy): A calibration strategy instance.
            config (DetectorConfig, optional): Configuration for the detector.
                Defaults to ``DetectorConfig()``.
        """
        self.detector: BaseDetector = set_params(detector, config.seed)
        self.strategy: BaseStrategy = strategy
        self.config: DetectorConfig = config

        self.detector_set: List[BaseDetector] = []
        self.calibration_set: List[float] = []
        self.calibration_samples: np.ndarray = np.array([])  # Initialize as empty

    @ensure_numpy_array
    def fit(self, x: Union[pd.DataFrame, np.ndarray]) -> None:
        """Fits the detector and prepares for conformal prediction.

        This method uses the provided strategy to fit the underlying detector(s)
        and generate a set of calibration scores. It also identifies and stores
        the data samples used for calibration. The `weighted` flag is passed
        as ``True`` to the strategy's `fit_calibrate` method, signaling that
        calibration sample identification is required.

        Args:
            x (Union[pandas.DataFrame, numpy.ndarray]): The input data used for
                training/fitting the detector(s) and for calibration. The
                `@ensure_numpy_array` decorator converts `x` to a
                ``numpy.ndarray`` internally.
        """
        self.detector_set, self.calibration_set = self.strategy.fit_calibrate(
            x=x, detector=self.detector, weighted=True, seed=self.config.seed
        )
        if (
            self.strategy.calibration_ids is not None
            and len(self.strategy.calibration_ids) > 0
        ):
            self.calibration_samples = x[self.strategy.calibration_ids]
        else:
            # Handle case where calibration_ids might be empty or None
            # This might happen if the strategy doesn't yield IDs or x is too small
            self.calibration_samples = np.array([])

    @ensure_numpy_array
    def predict(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        output: Literal["decision", "p-value", "score"] = "decision",
    ) -> np.ndarray:
        """Predicts anomaly status, p-values, or scores for new data.

        For each test instance in `x`:
        1. Anomaly scores are obtained from each detector in `detector_set`.
        2. These scores are aggregated using the method specified in `config`.
        3. Importance weights are computed for calibration and test instances
           to account for covariate shift, using `_compute_weights`.
        4. Weighted p-values are calculated using the aggregated scores,
           calibration scores, and computed weights.
        5. P-values are adjusted for multiplicity if specified in `config`.
        6. The final output (decision, p-value, or score) is returned.

        Args:
            x (Union[pandas.DataFrame, numpy.ndarray]): The input data for which
                predictions are to be made. The `@ensure_numpy_array`
                decorator converts `x` to a ``numpy.ndarray`` internally.
            output (Literal["decision", "p-value", "score"], optional):
                Determines the type of output:
                - ``"decision"``: Binary decisions (0 or 1, or ``False``/``True``
                  depending on `get_decision`) based on adjusted p-values
                  and `config.alpha`.
                - ``"p-value"``: Raw (unadjusted) weighted p-values.
                - ``"score"``: Aggregated anomaly scores.
                Defaults to ``"decision"``.

        Returns:
            numpy.ndarray: An array containing the predictions. The data type
            and shape depend on the `output` type.
        """
        scores_list = [
            model.decision_function(x)
            for model in tqdm(
                self.detector_set,
                total=len(self.detector_set),
                desc="Inference",
                disable=self.config.silent,
            )
        ]

        w_cal, w_x = self._compute_weights(x)
        estimates = aggregate(
            self.config.aggregation, np.array(scores_list)
        )  # Ensure scores_list is array for aggregate
        p_val = calculate_weighted_p_val(
            np.array(estimates),
            np.array(self.calibration_set),
            np.array(w_x),
            np.array(w_cal),
        )
        p_val_adj = multiplicity_correction(self.config.adjustment, p_val)

        if output == "score":
            return np.array(estimates)
        elif output == "p-value":
            return np.array(p_val)
        else:
            return get_decision(self.config.alpha, np.array(p_val_adj))

    def _compute_weights(
        self, test_instances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes importance weights for calibration and test instances.

        This method trains a logistic regression classifier to distinguish
        between samples from the calibration distribution and samples from the
        test distribution. The probabilities from this classifier are used
        to estimate the density ratio w(z) = p_test(z) / p_calib(z).

        The weights are clipped to a predefined range [0.35, 45] to prevent
        extreme values.

        Args:
            test_instances (numpy.ndarray): The test data instances for which
                weights need to be computed.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
                A tuple containing:
                - ``clipped_w_calib``: Clipped weights for calibration samples.
                - ``clipped_w_tests``: Clipped weights for test instances.

        Raises:
            ValueError: If `self.calibration_samples` is empty, as weights
                cannot be computed without calibration data.
        """
        if self.calibration_samples.shape[0] == 0:
            raise ValueError(
                "Calibration samples are empty. Weights cannot be computed. "
                "Ensure fit() was called and strategy provided calibration_ids."
            )

        calib_labeled = np.hstack(
            (self.calibration_samples, np.zeros((self.calibration_samples.shape[0], 1)))
        )
        tests_labeled = np.hstack(
            (test_instances, np.ones((test_instances.shape[0], 1)))
        )

        joint_labeled = np.vstack((calib_labeled, tests_labeled))
        rng = np.random.default_rng(seed=self.config.seed)
        rng.shuffle(joint_labeled)

        x_joint = joint_labeled[:, :-1]
        y_joint = joint_labeled[:, -1]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1_000,
                random_state=self.config.seed,
                verbose=0,
                class_weight="balanced",
            ),
        )
        model.fit(x_joint, y_joint)

        calib_prob = model.predict_proba(self.calibration_samples)
        tests_prob = model.predict_proba(test_instances)

        # Density ratio w(z) = p_test(z) / p_calib(z)
        # p_calib(z) = P(label=0 | z) ; p_test(z) = P(label=1 | z)
        # For calibration samples, weight is P(label=1 | z_calib) / P(label=0 | z_calib)
        # For test samples, weight is P(label=1 | z_test) / P(label=0 | z_test)
        # These are likelihood ratios p(z | test) / p(z | calib)
        w_calib = calib_prob[:, 1] / (
            calib_prob[:, 0] + 1e-9
        )  # Add epsilon for stability
        w_tests = tests_prob[:, 1] / (
            tests_prob[:, 0] + 1e-9
        )  # Add epsilon for stability

        clipped_w_calib = np.clip(w_calib, 0.35, 45.0)
        clipped_w_tests = np.clip(w_tests, 0.35, 45.0)

        return clipped_w_calib, clipped_w_tests
