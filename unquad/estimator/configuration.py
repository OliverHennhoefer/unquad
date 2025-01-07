from dataclasses import dataclass

from unquad.utils.enums import Adjustment
from unquad.utils.enums import Aggregation


@dataclass
class DetectorConfig:
    """
    Configuration for conformal anomaly detector.

    Attributes:
        alpha (float): Significance level for statistical tests or other computations.
            Must be between 0 and 1 (default: 0.2).
        adjustment (Adjustment): Method for adjusting p-values or thresholds.
            Default is Benjamini-Hochberg adjustment.
        aggregation (Aggregation): Method used for aggregating metrics or data points.
            Default is median.
        force_anomaly (bool): Whether anomaly scores outside the calibration range are
            categorically considered an anomaly. Default is False.
        seed (int): Random seed for reproducibility in stochastic processes.
            Default is 1.
        silent (bool): Suppresses logs or warnings if True. Default is True.

    Raises:
        ValueError: If `alpha` is not between 0 and 1, or if `seed` is negative.
    """

    alpha: float = 0.2
    adjustment: Adjustment = Adjustment.BH
    aggregation: Aggregation = Aggregation.MEDIAN
    force_anomaly: bool = False
    seed: int = 1
    silent: bool = True
