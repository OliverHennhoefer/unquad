from typing import Union

import numpy as np
import pandas as pd
from pyod.models.base import BaseDetector
from sklearn.model_selection import train_test_split

from unquad.strategy.base import BaseStrategy


class SplitConformal(BaseStrategy):

    def __init__(self, calib_size: float | int = 0.1) -> None:
        self.calib_size: float | int = calib_size

    def fit_calibrate(
        self, x: Union[pd.DataFrame, np.ndarray], detector: BaseDetector, seed: int = 1
    ) -> (list[BaseDetector], list[list]):

        train, calib = train_test_split(
            x, test_size=self.calib_size, shuffle=True, random_state=seed
        )
        detector.fit(train)
        calibration_set = detector.decision_function(calib)

        return [detector], calibration_set
