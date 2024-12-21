from enum import Enum


class Aggregation(Enum):
    """
    Enumerators of aggregation functions for ensemble methods.
    """

    MEAN: str = "mean"
    MEDIAN: str = "median"
    MINIMUM: str = "minimum"
    MAXIMUM: str = "maximum"
