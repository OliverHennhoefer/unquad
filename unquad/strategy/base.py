import abc

import numpy as np
import pandas as pd

from typing import Union, Optional
from pyod.models.base import BaseDetector


class BaseStrategy(abc.ABC):

    def __init__(self, plus: bool = False):
        self.plus: bool = plus

    @abc.abstractmethod
    def fit_calibrate(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        detector: BaseDetector,
        seed: Optional[int],
    ):

        raise NotImplementedError("The _calibrate() method must be implemented.")
