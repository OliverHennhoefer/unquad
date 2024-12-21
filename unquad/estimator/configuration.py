from dataclasses import dataclass

from unquad.utils.enums.adjustment import Adjustment
from unquad.utils.enums.aggregation import Aggregation


@dataclass
class EstimatorConfig:
    alpha: float = 0.2
    adjustment: Adjustment = Adjustment.BENJAMINI_HOCHBERG
    aggregation: Aggregation = Aggregation.MEDIAN
    seed: int = 1
    silent: bool = True
