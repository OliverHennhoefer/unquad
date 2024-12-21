import numpy as np

from unquad.utils.decorator.performance import performance_conversion
from unquad.utils.enums.aggregation import Aggregation


@performance_conversion("scores")
def aggregate(method: Aggregation, scores: np.array) -> list[float]:

    aggregation_methods = {
        Aggregation.MEAN: lambda x: np.mean(x, axis=0),
        Aggregation.MEDIAN: lambda x: np.median(x, axis=0),
        Aggregation.MINIMUM: lambda x: np.min(x, axis=0),
        Aggregation.MAXIMUM: lambda x: np.max(x, axis=0),
    }

    func = aggregation_methods.get(method)
    if not func:
        raise ValueError(f"Unsupported aggregation method: {method}")
    return func(scores)
