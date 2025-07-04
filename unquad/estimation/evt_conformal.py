"""Extreme Value Theory enhanced conformal anomaly detection.

This module provides the `EVTConformalDetector` class, which extends the base
`ConformalDetector` with Extreme Value Theory capabilities for better modeling
of extreme anomalies. It uses Generalized Pareto Distribution (GPD) to model
the tail of the calibration score distribution.
"""

import os
from typing import Literal, Optional, Tuple, Union, Callable

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from tqdm import tqdm

from pyod.models.base import BaseDetector as PyODBaseDetector
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.base import BaseStrategy
from unquad.utils.stat.aggregation import aggregate
from unquad.utils.func.decorator import ensure_numpy_array
from unquad.utils.func.enums import Aggregation
from unquad.utils.stat.evt import fit_gpd, select_threshold
from unquad.utils.stat.statistical import calculate_evt_p_val, calculate_p_val


class EVTConformalDetector(ConformalDetector):
    """Conformal anomaly detector with Extreme Value Theory enhancement.

    This detector extends the standard conformal prediction framework by
    modeling extreme calibration scores using a Generalized Pareto Distribution.
    This provides better calibration for detecting extreme anomalies while
    maintaining the standard empirical approach for normal values.

    The detector automatically fits a GPD to calibration scores exceeding
    a configurable threshold and uses a hybrid p-value calculation that
    combines empirical and parametric approaches.

    Attributes
    ----------
        detector (PyODBaseDetector): The underlying anomaly detection model.
        strategy (BaseStrategy): The strategy used for fitting and calibration.
        aggregation (Aggregation): Method used for aggregating scores from
            multiple detector models.
        seed (int): Random seed for reproducibility in stochastic processes.
        silent (bool): Whether to suppress progress bars and logs.
        evt_threshold_method (Literal): Method for selecting EVT threshold.
        evt_threshold_value (Union[float, Callable]): Parameter for threshold method.
        evt_min_tail_size (int): Minimum number of exceedances required for GPD fitting.
        detector_set (List[PyODBaseDetector]): Trained detector models.
        calibration_set (List[float]): Calibration scores.
        evt_threshold (float): Threshold separating bulk and tail distributions.
        gpd_params (Tuple[float, float, float]): Fitted GPD parameters
            (shape, loc, scale).
    """

    def __init__(
        self,
        detector: PyODBaseDetector,
        strategy: BaseStrategy,
        aggregation: Aggregation = Aggregation.MEDIAN,
        seed: int = 1,
        silent: bool = True,
        evt_threshold_method: Literal["percentile", "top_k", "mean_excess", "custom"] = "percentile",
        evt_threshold_value: Union[float, Callable[[np.ndarray], float]] = 0.95,
        evt_min_tail_size: int = 10,
    ):
        """Initialize the EVTConformalDetector.

        Args:
            detector (PyODBaseDetector): The base anomaly detection model.
            strategy (BaseStrategy): The conformal strategy for calibration.
            aggregation (Aggregation, optional): Method used for aggregating
                scores from multiple detector models. Defaults to Aggregation.MEDIAN.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            silent (bool, optional): Whether to suppress progress bars and logs.
                Defaults to True.
            evt_threshold_method (Literal, optional): Method for selecting EVT
                threshold. Defaults to "percentile".
            evt_threshold_value (Union[float, Callable], optional): Parameter for
                threshold method. Defaults to 0.95.
            evt_min_tail_size (int, optional): Minimum number of exceedances
                required for GPD fitting. Defaults to 10.
        """
        super().__init__(detector, strategy, aggregation, seed, silent)

        # EVT-specific parameters
        self.evt_threshold_method: Literal["percentile", "top_k", "mean_excess", "custom"] = evt_threshold_method
        self.evt_threshold_value: Union[float, Callable[[np.ndarray], float]] = evt_threshold_value
        self.evt_min_tail_size: int = evt_min_tail_size

        # EVT-specific attributes
        self.evt_threshold: Optional[float] = None
        self.gpd_params: Optional[Tuple[float, float, float]] = None

    @ensure_numpy_array
    def fit(self, x: pd.DataFrame | np.ndarray) -> None:
        """Fits the detector and prepares EVT-enhanced calibration.

        This method extends the parent's fit method by additionally:
        1. Determining the threshold for extreme values
        2. Fitting a GPD to calibration scores exceeding the threshold

        The EVT parameters are only fitted if there are sufficient
        exceedances as specified by evt_min_tail_size.

        Args:
            x (Union[pd.DataFrame, np.ndarray]): Training data.
        """
        # Call parent's fit method
        super().fit(x)

        # Fit EVT components
        if len(self.calibration_set) > 0:
            calibration_array = np.array(self.calibration_set)

            # Select threshold
            self.evt_threshold = select_threshold(
                calibration_array,
                self.evt_threshold_method,
                self.evt_threshold_value,
            )

            # Fit GPD to exceedances
            exceedances = (
                calibration_array[calibration_array > self.evt_threshold]
                - self.evt_threshold
            )

            if len(exceedances) >= self.evt_min_tail_size:
                try:
                    self.gpd_params = fit_gpd(exceedances)
                except Exception as e:
                    # If GPD fitting fails, fall back to standard approach
                    if not self.silent:
                        print(
                            f"Warning: GPD fitting failed: {e}. Using standard approach."
                        )
                    self.gpd_params = None
            else:
                if not self.silent:
                    print(
                        f"Warning: Only {len(exceedances)} exceedances, "
                        f"need at least {self.evt_min_tail_size}. "
                        f"Using standard approach."
                    )
                self.gpd_params = None

    @ensure_numpy_array
    def predict(
        self,
        x: pd.DataFrame | np.ndarray,
        raw: bool = False,
    ) -> np.ndarray:
        """Predicts using EVT-enhanced conformal prediction.

        This method calculates anomaly scores and then computes p-values
        using either the standard empirical approach or the EVT-enhanced
        hybrid approach, depending on whether EVT parameters were successfully
        fitted during training.

        Args:
            x (Union[pd.DataFrame, np.ndarray]): Test data.
            raw (bool, optional): Whether to return raw anomaly scores or
                p-values. Defaults to False.

        Returns:
            np.ndarray: Predictions based on the output type.
        """
        # Calculate anomaly scores
        scores_list = [
            model.decision_function(x)
            for model in tqdm(
                self.detector_set,
                total=len(self.detector_set),
                desc="Inference",
                disable=self.silent,
            )
        ]

        estimates = aggregate(method=self.aggregation, scores=scores_list)

        if raw:
            return estimates

        # Calculate p-values using EVT if available
        if self.gpd_params is not None:
            p_val = calculate_evt_p_val(
                estimates,
                self.calibration_set,
                threshold_method=self.evt_threshold_method,
                threshold_value=self.evt_threshold_value,
                min_tail_size=self.evt_min_tail_size,
                gpd_params=self.gpd_params,
                threshold=self.evt_threshold,
            )
        else:
            # Fall back to standard empirical p-values
            p_val = calculate_p_val(estimates, self.calibration_set)

        return np.array(p_val)