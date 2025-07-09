"""Base abstract class for conformal anomaly detectors.

This module provides the abstract base class that defines the common interface
for all conformal anomaly detection implementations.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from unquad.utils.func.decorator import ensure_numpy_array


class BaseConformalDetector(ABC):
    """Abstract base class for conformal anomaly detectors.

    This class defines the minimal interface that all conformal anomaly detection
    implementations must provide. It requires concrete classes to implement
    the fit and predict methods.
    """

    @ensure_numpy_array
    @abstractmethod
    def fit(self, x: pd.DataFrame | np.ndarray) -> None:
        """Fit the detector model(s) and compute calibration scores.

        Args:
            x (typing.Union[pd.DataFrame, np.ndarray]): The dataset used for
                fitting the model(s) and determining calibration scores.
        """
        pass

    @ensure_numpy_array
    @abstractmethod
    def predict(
        self,
        x: pd.DataFrame | np.ndarray,
        raw: bool = False,
    ) -> np.ndarray:
        """Generate anomaly estimates or p-values for new data.

        Args:
            x (typing.Union[pd.DataFrame, np.ndarray]): The new data instances
                for which to make anomaly estimates.
            raw (bool, optional): Whether to return raw anomaly scores or
                processed anomaly estimates (e.g., p-values). Defaults to False.

        Returns
        -------
            np.ndarray: An array containing the anomaly estimates.
        """
        pass
