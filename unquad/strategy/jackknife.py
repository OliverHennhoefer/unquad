import numpy as np
import pandas as pd

from typing import Union, List, Tuple, Optional  # Added List, Tuple, Optional
from pyod.models.base import BaseDetector

from unquad.strategy.base import BaseStrategy
from unquad.strategy.cross_val import CrossValidation


class Jackknife(BaseStrategy):
    """Jackknife (leave-one-out) conformal anomaly detection strategy.

    This strategy implements conformal prediction using the jackknife method,
    which is a special case of k-fold cross-validation where k equals the
    number of samples in the dataset (leave-one-out). For each sample, a
    model is trained on all other samples, and the left-out sample is used
    for calibration.

    It internally uses a :class:`~unquad.strategy.cross_val.CrossValidation`
    strategy, dynamically setting its `_k` parameter to the dataset size.

    Attributes:
        _plus (bool): If ``True``, each model trained (one for each left-out
            sample) is retained. If ``False``, a single model trained on the
            full dataset (after leave-one-out calibration) is retained. This
            behavior is delegated to the internal `CrossValidation` strategy.
        _strategy (CrossValidation): An instance of the
            :class:`~unquad.strategy.cross_val.CrossValidation` strategy,
            configured for leave-one-out behavior.
        _calibration_ids (Optional[List[int]]): Indices of the samples from
            the input data `x` used for calibration. Populated after
            :meth:`fit_calibrate` and accessible via :attr:`calibration_ids`.
            Initially ``None``.
        _detector_list (List[BaseDetector]): A list of trained detector models,
            populated by :meth:`fit_calibrate` via the internal strategy.
        _calibration_set (List[float]): A list of calibration scores, one for
            each sample, populated by :meth:`fit_calibrate` via the internal
            strategy.
    """

    def __init__(self, plus: bool = False):
        """Initializes the Jackknife strategy.

        Args:
            plus (bool, optional): If ``True``, instructs the internal
                cross-validation strategy to retain all models trained during
                the leave-one-out process. Defaults to ``False``.
        """
        super().__init__(plus)
        self._plus: bool = plus
        self._strategy: CrossValidation = CrossValidation(k=1, plus=plus)
        self._calibration_ids: Optional[List[int]] = None

        self._detector_list: List[BaseDetector] = []
        self._calibration_set: List[float] = []

    def fit_calibrate(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        detector: BaseDetector,
        weighted: bool = False,  # Parameter passed to internal strategy
        seed: int = 1,
    ) -> Tuple[List[BaseDetector], List[float]]:
        """Fits detector(s) and gets calibration scores using jackknife.

        This method configures the internal
        :class:`~unquad.strategy.cross_val.CrossValidation` strategy to
        perform leave-one-out cross-validation by setting its number of
        folds (`_k`) to the total number of samples in `x`. It then delegates
        the fitting and calibration process to this internal strategy.

        The results (trained models and calibration scores) and calibration
        sample IDs are retrieved from the internal strategy.

        Args:
            x (Union[pandas.DataFrame, numpy.ndarray]): The input data.
            detector (BaseDetector): The PyOD base detector instance.
            weighted (bool, optional): Passed to the internal `CrossValidation`
                strategy's `fit_calibrate` method. Its effect depends on the
                `CrossValidation` implementation. Defaults to ``False``.
            seed (int, optional): Random seed, passed to the internal
                `CrossValidation` strategy for reproducibility. Defaults to ``1``.

        Returns:
            Tuple[List[BaseDetector], List[float]]:
                A tuple containing:
                - A list of trained PyOD detector models.
                - A list of calibration scores (one per sample in `x`).
        """
        self._strategy._k = len(x)
        (
            self._detector_list,
            self._calibration_set,
        ) = self._strategy.fit_calibrate(x, detector, weighted, seed)
        self._calibration_ids = self._strategy.calibration_ids
        return self._detector_list, self._calibration_set

    @property
    def calibration_ids(self) -> Optional[List[int]]:
        """Returns indices from `x` used for calibration via jackknife.

        These are the indices of samples used to obtain calibration scores.
        In jackknife (leave-one-out), each sample is used once for
        calibration. The list is populated after `fit_calibrate` is called.

        Returns:
            Optional[List[int]]: A list of integer indices, or ``None`` if
                `fit_calibrate` has not been called.
        """
        return self._calibration_ids
