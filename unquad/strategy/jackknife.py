from typing import Union, Optional

import numpy as np
import pandas as pd
from pyod.models.base import BaseDetector

from unquad.strategy.base import BaseStrategy
from unquad.strategy.cross_val import CrossValidationConformal


class JackknifeConformal(BaseStrategy):

    def __init__(self, plus: bool = False):
        self.plus: bool = plus
        self.strategy: BaseStrategy = CrossValidationConformal(k=1, plus=plus)

    def fit_calibrate(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        detector: BaseDetector,
        seed: Optional[int],
    ):

        self.strategy.k = len(x)
        return self.strategy.fit_calibrate(x, detector, seed)
