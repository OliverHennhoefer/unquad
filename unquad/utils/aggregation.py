import numpy as np

from unquad.utils.decorator.performance import performance_conversion
from unquad.utils.enums.aggregation import Aggregation


@performance_conversion("scores")
def aggregate(method: Aggregation, scores: np.array) -> list[float]:
    """
    Aggregate anomaly scores using the specified aggregation method.

    This function applies the chosen aggregation method to a set of anomaly scores.
    It supports various methods such as mean, median, minimum, and maximum.

    Args:
        method (Aggregation): The aggregation method to use. Must be one of the
                               methods defined in the Aggregation enum (e.g., MEAN, MEDIAN, MINIMUM, MAXIMUM).
        scores (np.array): A 2D NumPy array where each row represents an individual model's anomaly scores.

    Returns:
        list[float]: The aggregated anomaly scores as a list.

    Raises:
        ValueError: If the provided aggregation method is not supported.
    """
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
