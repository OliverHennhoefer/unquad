import numpy as np
from scipy.stats import false_discovery_control

from unquad.utils.decorator.performance import performance_conversion
from unquad.utils.enums.adjustment import Adjustment


@performance_conversion("scores")
def multiplicity_correction(method: Adjustment, scores: np.array) -> float:
    aggregation_methods = {
        Adjustment.NONE: lambda x: x,
        Adjustment.BENJAMINI_HOCHBERG: lambda x: false_discovery_control(
            x, method="bh"
        ),
        Adjustment.BENJAMINI_YEKUTIELI: lambda x: false_discovery_control(
            x, method="by"
        ),
    }

    func = aggregation_methods.get(method)
    if not func:
        raise ValueError(f"Unsupported aggregation method: {method}")

    return func(scores)
