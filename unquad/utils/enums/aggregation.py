from enum import Enum


class Aggregation(Enum):
    """
    Enumerators for aggregation functions used in ensemble methods.

    This enum defines the aggregation methods used to combine predictions or scores
    from multiple models in ensemble learning methods. The aggregation function
    determines how the final decision or score is calculated from the outputs of
    individual models.

    Attributes:
        MEAN (str): The mean aggregation function.
        MEDIAN (str): The median aggregation function.
        MINIMUM (str): The minimum aggregation function.
        MAXIMUM (str): The maximum aggregation function.
    """

    MEAN: str = "mean"
    MEDIAN: str = "median"
    MINIMUM: str = "minimum"
    MAXIMUM: str = "maximum"
