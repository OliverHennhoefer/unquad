import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Union, List, Tuple  # Added List, Tuple
from copy import copy, deepcopy
from pyod.models.base import BaseDetector
from sklearn.model_selection import KFold

from unquad.estimation.properties.parameterization import set_params
from unquad.strategy.base import BaseStrategy


class CrossValidation(BaseStrategy):
    """Cross-validation based conformal anomaly detection strategy.

    This strategy employs k-fold cross-validation to generate calibration
    scores for conformal prediction. In each fold, a portion of the data
    is used for training a detector, and the remaining part is used for
    calibration (i.e., to obtain non-conformity scores).

    The strategy can either retain all models trained on each fold (if `plus`
    is ``True``) or train a single final model on the entire dataset after
    collecting calibration scores from all folds (if `plus` is ``False``).

    Attributes:
        _k (int): The number of folds to use in the k-fold cross-validation.
        _plus (bool): If ``True``, each model trained on a fold is retained
            in `_detector_list`. If ``False``, only a single model trained on
            the full dataset (after cross-validation for calibration) is
            retained.
        _detector_list (List[BaseDetector]): A list of trained detector models.
            Populated by the :meth:`fit_calibrate` method.
        _calibration_set (List[float]): A list of calibration scores obtained
            from the hold-out sets in each fold. Populated by
            :meth:`fit_calibrate`.
        _calibration_ids (List[int]): Indices of the samples from the input
            data `x` that were used to form the `_calibration_set` (i.e., all
            samples are used for calibration across the folds). Populated by
            :meth:`fit_calibrate` and accessible via the
            :attr:`calibration_ids` property.
    """

    def __init__(self, k: int, plus: bool = False):
        """Initializes the CrossValidation strategy.

        Args:
            k (int): The number of folds for cross-validation. Must be at
                least 2.
            plus (bool, optional): If ``True``, appends each fold-trained model
                to `_detector_list`. If ``False``, `_detector_list` will contain
                one model trained on all data after calibration scores are
                collected. Defaults to ``False``.
        """
        super().__init__(plus)
        self._k: int = k
        self._plus: bool = plus

        self._detector_list: List[BaseDetector] = []
        self._calibration_set: List[float] = []
        self._calibration_ids: List[int] = []

    def fit_calibrate(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        detector: BaseDetector,
        weighted: bool = False,  # This argument is present but not used in current logic
        seed: int = 1,
    ) -> Tuple[List[BaseDetector], List[float]]:
        """Fits detector(s) and generates calibration scores using k-fold CV.

        This method divides the data `x` into `_k` folds. For each fold:
        1. The fold is used as a calibration set, and the remaining `_k-1`
           folds are used as the training set.
        2. A copy of the `detector` is trained on the training set.
        3. If `self._plus` is ``True``, the trained model is stored.
        4. Decision scores from the trained model on the current fold's
           calibration data are collected.
        5. Indices of these calibration samples are stored.

        After all folds are processed, if `self._plus` is ``False``, a final
        model is trained on the entire dataset `x` and stored. The
        calibration set comprises scores from all samples, each obtained when
        its respective fold was used for calibration.

        Args:
            x (Union[pandas.DataFrame, numpy.ndarray]): The input data for
                training and calibration.
            detector (BaseDetector): The PyOD base detector instance to be
                trained.
            weighted (bool, optional): This parameter is accepted but not
                currently utilized in the cross-validation logic.
                Defaults to ``False``.
            seed (int, optional): Random seed for reproducibility of data
                shuffling in k-fold splitting and model parameter setting.
                Defaults to ``1``.

        Returns:
            Tuple[List[BaseDetector], List[float]]:
                A tuple containing:
                - A list of trained PyOD detector models.
                - A list of calibration scores, where each score corresponds to
                  a sample in `x` obtained when it was in a hold-out fold.
        """
        _detector = detector

        folds = KFold(
            n_splits=self._k,
            shuffle=True,
            random_state=seed,
        )

        last_iteration_index = 0
        for i, (train_idx, calib_idx) in enumerate(
            tqdm(folds.split(x), total=self._k, desc="Training", disable=False)
        ):
            last_iteration_index = i
            self._calibration_ids.extend(calib_idx.tolist())

            model = copy(_detector)
            model = set_params(model, seed=seed, random_iteration=True, iteration=i)
            model.fit(x[train_idx])

            if self._plus:
                self._detector_list.append(deepcopy(model))
            self._calibration_set.extend(model.decision_function(x[calib_idx]))

        if not self._plus:
            model = copy(_detector)
            model = set_params(
                model,
                seed=seed,
                random_iteration=True,
                iteration=(last_iteration_index + 1),
            )
            model.fit(x)
            self._detector_list.append(deepcopy(model))

        return self._detector_list, self._calibration_set

    @property
    def calibration_ids(self) -> List[int]:
        """Returns the list of indices from `x` used for calibration.

        In k-fold cross-validation, every sample in the input data `x` is
        used exactly once as part of a calibration set (when its fold is
        the hold-out set). This property returns a list of all these indices,
        typically covering all indices from 0 to len(x)-1, but ordered by
        fold processing.

        Returns:
            List[int]: A list of integer indices.
        """
        return self._calibration_ids
