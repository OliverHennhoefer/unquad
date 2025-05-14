import math
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Union, Optional, List, Tuple
from copy import copy, deepcopy
from pyod.models.base import BaseDetector
from sklearn.model_selection import ShuffleSplit

from unquad.estimation.properties.parameterization import set_params
from unquad.strategy.base import BaseStrategy


class Bootstrap(BaseStrategy):
    """Bootstrap-based conformal anomaly detection strategy.

    This strategy implements conformal prediction using bootstrap resampling.
    It involves training multiple instances of a base anomaly detector on
    different bootstrap samples of the data. A portion of the data not used
    for training in each bootstrap iteration is used to form a calibration set.
    The final calibration scores are derived from these out-of-bag samples.

    The configuration allows specifying two of the following three parameters:
    resampling ratio, number of bootstraps, or desired calibration set size,
    and the third will be calculated.

    Attributes:
        _resampling_ratio (Optional[float]): The proportion of the dataset
            to be used for training in each bootstrap split. If ``None``, it's
            calculated based on `_n_bootstraps` and `_n_calib`.
        _n_bootstraps (Optional[int]): The number of bootstrap iterations,
            which typically corresponds to the number of models trained (if
            `_plus` is ``True``) or contributes to a diverse calibration set.
            If ``None``, it's calculated.
        _n_calib (Optional[int]): The desired size of the final calibration
            set, sampled from all collected out-of-bag scores. If ``None``,
            it's determined by the other parameters. If set, the collected
            calibration scores and potentially `_calibration_ids` are
            subsampled to this size.
        _plus (bool): If ``True``, each model trained on a bootstrap sample is
            retained in `_detector_list`. If ``False``, only a single model
            trained on the full dataset (after bootstrap iterations for
            calibration) is retained.
        _detector_list (List[BaseDetector]): A list of trained detector models.
            Populated by the :meth:`fit_calibrate` method.
        _calibration_set (List[float]): A list of calibration scores obtained
            from out-of-bag samples. Populated by :meth:`fit_calibrate`.
        _calibration_ids (List[int]): Indices of the samples from the input
            data `x` that were used to form the `_calibration_set`.
            Populated by :meth:`fit_calibrate` and accessible via the
            :attr:`calibration_ids` property. This list might be subsampled
            if `_n_calib` is set and `weighted` is ``True``.
    """

    def __init__(
        self,
        resampling_ratio: Optional[float] = None,
        n_bootstraps: Optional[int] = None,
        n_calib: Optional[int] = None,
        plus: bool = False,
    ):
        """Initializes the Bootstrap strategy.

        Exactly two of `resampling_ratio`, `n_bootstraps`, and `n_calib`
        should be provided. The third will be calculated by `_configure`.

        Args:
            resampling_ratio (Optional[float], optional): The proportion of
                data to use for training in each bootstrap. Defaults to ``None``.
            n_bootstraps (Optional[int], optional): The number of bootstrap
                iterations. Defaults to ``None``.
            n_calib (Optional[int], optional): The desired size of the final
                calibration set. If set, collected scores/IDs might be
                subsampled. Defaults to ``None``.
            plus (bool, optional): If ``True``, appends each bootstrapped model
                to `_detector_list`. If ``False``, `_detector_list` will contain
                one model trained on all data after calibration scores are
                collected. Defaults to ``False``.
        """
        super().__init__(plus)
        self._resampling_ratio: Optional[float] = resampling_ratio
        self._n_bootstraps: Optional[int] = n_bootstraps
        self._n_calib: Optional[int] = n_calib
        self._plus: bool = plus

        self._detector_list: List[BaseDetector] = []
        self._calibration_set: List[float] = []
        self._calibration_ids: List[int] = []

    def fit_calibrate(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        detector: BaseDetector,
        weighted: bool = False,
        seed: int = 1,
    ) -> Tuple[List[BaseDetector], List[float]]:
        """Fits detector(s) and generates calibration scores using bootstrap.

        This method first configures bootstrap parameters (resampling ratio,
        number of bootstraps, calibration size) by calling `_configure`.
        It then performs `_n_bootstraps` iterations using `ShuffleSplit`.
        In each iteration:
        1. Data is split into a training set and a calibration (out-of-bag) set.
        2. A copy of the `detector` is trained on the training set.
        3. If `self._plus` is ``True``, the trained model is stored.
        4. Decision scores from the model on the calibration part are collected.
        5. Indices of these calibration samples are stored in `_calibration_ids`.

        After iterations, if `self._plus` is ``False``, a final model is
        trained on the entire dataset `x` and stored.
        If `_n_calib` is set and less than the number of collected calibration
        scores, `_calibration_set` is randomly subsampled. If `weighted` is
        also ``True``, `_calibration_ids` is subsampled consistently.

        Args:
            x (Union[pandas.DataFrame, numpy.ndarray]): The input data for
                training and calibration.
            detector (BaseDetector): The PyOD base detector instance to be
                trained.
            weighted (bool, optional): If ``True`` and `_n_calib` is specified
                for subsampling, `_calibration_ids` will also be subsampled.
                Defaults to ``False``.
            seed (int, optional): Random seed for reproducibility of data
                splitting, model parameter setting, and potential subsampling.
                Defaults to ``1``.

        Returns:
            Tuple[List[BaseDetector], List[float]]:
                A tuple containing:
                - A list of trained PyOD detector models.
                - A list of calibration scores.
        """
        self._configure(len(x))

        _detector = detector
        _generator = np.random.default_rng(seed)

        folds = ShuffleSplit(
            n_splits=self._n_bootstraps,
            train_size=self._resampling_ratio,
            random_state=seed,
        )

        n_folds = folds.get_n_splits()
        last_iteration_index = (
            0  # To ensure unique iteration for final model if not _plus
        )
        for i, (train_idx, calib_idx) in enumerate(
            tqdm(folds.split(x), total=n_folds, desc="Training", disable=False)
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

        if self._n_calib is not None and self._n_calib < len(self._calibration_set):
            ids = _generator.choice(
                len(self._calibration_set), size=self._n_calib, replace=False
            )
            self._calibration_set = [self._calibration_set[i] for i in ids]
            if weighted:
                self._calibration_ids = [self._calibration_ids[i] for i in ids]
            else:
                # If not weighted, _calibration_ids might become inconsistent
                # with a subsampled _calibration_set. Consider if this is intended.
                # For now, it remains the full list of IDs if not weighted.
                pass

        return self._detector_list, self._calibration_set

    def _sanity_check(self) -> None:
        """Ensures that exactly two configuration parameters are provided.

        Verifies that exactly two of `_resampling_ratio`, `_n_bootstraps`,
        and `_n_calib` have been set during initialization. The third
        parameter is derived from these two and the dataset size by
        `_configure`.

        Raises:
            ValueError: If not exactly two of the three parameters
                (resampling_ratio, n_bootstraps, n_calib) are defined.
        """
        num_defined: int = sum(
            param is not None
            for param in (self._resampling_ratio, self._n_bootstraps, self._n_calib)
        )
        if num_defined != 2:
            raise ValueError(
                "Exactly two parameters (resampling_ratio, n_bootstraps, n_calib) "
                "must be defined."
            )

    def _configure(self, n: int) -> None:
        """Configures bootstrap parameters based on two provided settings.

        Calculates and sets the third missing parameter among
        `_resampling_ratio`, `_n_bootstraps`, and `_n_calib` based on the
        two that were provided during initialization and the total number of
        samples `n`. This method modifies the instance attributes in place.

        It calls `_sanity_check` first to ensure valid initial parameters.
        The formulas assume `_n_calib` refers to the total number of unique
        calibration points desired or expected after all bootstrap samples.

        Args:
            n (int): The total number of samples in the dataset.

        Raises:
            ValueError: If `_sanity_check` fails (i.e., not exactly two
                parameters were initially defined), or if calculated
                `resampling_ratio` is not within (0, 1) or `n_bootstraps` < 1.
        """
        self._sanity_check()

        def calculate_n_calib_target(
            n_data: int, num_bootstraps: int, res_ratio: float
        ) -> int:
            if not (0 < res_ratio < 1):
                raise ValueError("Resampling ratio must be between 0 and 1.")
            if num_bootstraps < 1:
                raise ValueError("Number of bootstraps must be at least 1.")
            return math.ceil(num_bootstraps * n_data * (1.0 - res_ratio))

        def calculate_n_bootstraps_target(
            n_data: int, num_calib_target: int, res_ratio: float
        ) -> int:
            if not (0 < res_ratio < 1):
                raise ValueError("Resampling ratio must be between 0 and 1.")
            if n_data * (1.0 - res_ratio) <= 0:
                raise ValueError(
                    "Product n_data * (1 - res_ratio) must be positive for "
                    "calculating n_bootstraps."
                )
            n_b = math.ceil(num_calib_target / (n_data * (1.0 - res_ratio)))
            if n_b < 1:
                raise ValueError("Calculated number of bootstraps is less than 1.")
            return n_b

        def calculate_resampling_ratio_target(
            n_data: int, num_bootstraps: int, num_calib_target: int
        ) -> float:
            if num_bootstraps < 1:
                raise ValueError("Number of bootstraps must be at least 1.")
            if n_data <= 0 or num_bootstraps * n_data == 0:
                raise ValueError("Product n_data * num_bootstraps must be positive.")

            val = 1.0 - (float(num_calib_target) / (num_bootstraps * n_data))
            if not (0 < val < 1):
                raise ValueError(
                    f"Calculated resampling_ratio ({val:.3f}) is not between 0 and 1. "
                    "Check input n_calib, n_bootstraps, and data size."
                )
            return val

        if self._n_bootstraps is not None and self._resampling_ratio is not None:
            self._n_calib = calculate_n_calib_target(
                n_data=n,
                num_bootstraps=self._n_bootstraps,
                res_ratio=self._resampling_ratio,
            )
        elif self._n_bootstraps is not None and self._n_calib is not None:
            self._resampling_ratio = calculate_resampling_ratio_target(
                n_data=n,
                num_bootstraps=self._n_bootstraps,
                num_calib_target=self._n_calib,
            )
        elif self._resampling_ratio is not None and self._n_calib is not None:
            self._n_bootstraps = calculate_n_bootstraps_target(
                n_data=n,
                res_ratio=self._resampling_ratio,
                num_calib_target=self._n_calib,
            )

    @property
    def calibration_ids(self) -> List[int]:
        """Returns the list of indices used for calibration.

        These are indices relative to the original input data `x` provided to
        :meth:`fit_calibrate`. The list contains indices of all out-of-bag
        samples encountered during bootstrap iterations. If `_n_calib` was
        set and `weighted` was ``True`` in `fit_calibrate`, this list might
        be a subsample of all encountered IDs, corresponding to the
        subsampled `_calibration_set`.

        Returns:
            List[int]: A list of integer indices.
        """
        return self._calibration_ids
