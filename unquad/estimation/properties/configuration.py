from dataclasses import dataclass

from unquad.utils.enums import Adjustment
from unquad.utils.enums import Aggregation


@dataclass
class DetectorConfig:
    """Configuration for a conformal anomaly detector.

    Attributes:
        alpha (float): Significance level for statistical tests or other
            computations. Must be between 0 and 1. Default is ``0.2``.
        adjustment (Adjustment): Method for adjusting p-values or thresholds.
            Defaults to ``Adjustment.BH`` (Benjamini-Hochberg).
        aggregation (Aggregation): Method used for aggregating metrics or data
            points. Defaults to ``Aggregation.MEDIAN`` (median).
        seed (int): Random seed for reproducibility in stochastic processes.
            Default is ``1``.
        silent (bool): Suppresses logs or warnings if True. Default is ``True``.

    Raises:
        ValueError: If ``alpha`` is not between 0 and 1, or if ``seed`` is
            negative.
    """

    alpha: float = 0.2
    adjustment: Adjustment = Adjustment.BH
    aggregation: Aggregation = Aggregation.MEDIAN
    seed: int = 1
    silent: bool = True

    def __post_init__(self):
        """Validates the configuration after initialization."""
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be between 0 and 1, exclusive.")
        if self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
