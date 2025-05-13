import numpy as np

from scipy.stats import false_discovery_control

from unquad.utils.decorator import performance_conversion
from unquad.utils.enums import Adjustment


@performance_conversion("scores")
def multiplicity_correction(method: Adjustment, scores: np.array) -> float:
    """
    Multiplicity correction to the given scores using the specified adjustment method.

    The function adjusts p-values (or scores) for multiple comparisons using
    either the Benjamini-Hochberg or Benjamini-Yekutieli procedure,
    or it returns the scores unadjusted.

    Args:
        method (Adjustment): The adjustment method to apply. It can be one of
                              Adjustment.NONE, Adjustment.BENJAMINI_HOCHBERG,
                              or Adjustment.BENJAMINI_YEKUTIELI.
        scores (np.array): The array of p-values or scores to adjust.

    Returns:
        float: The adjusted scores based on the specified method.

    Raises:
        ValueError: If an unsupported adjustment method is provided.
    """
    aggregation_methods = {
        Adjustment.NONE: lambda x: x,
        Adjustment.BH: lambda x: false_discovery_control(x, method="bh"),
        Adjustment.BY: lambda x: false_discovery_control(x, method="by"),
    }

    func = aggregation_methods.get(method)
    if not func:
        raise ValueError(f"Unsupported aggregation method: {method}")

    return func(scores)
