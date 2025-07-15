from enum import Enum


class Aggregation(Enum):
    """Aggregation functions for combining multiple model outputs or scores.

    This enumeration lists strategies for aggregating data, commonly employed
    in ensemble methods to combine predictions or scores from several models.

    Attributes
    ----------
        MEAN: Represents aggregation by calculating the arithmetic mean.
            The underlying value is typically ``"mean"``.
        MEDIAN: Represents aggregation by calculating the median.
            The underlying value is typically ``"median"``.
        MINIMUM: Represents aggregation by selecting the minimum value.
            The underlying value is typically ``"minimum"``.
        MAXIMUM: Represents aggregation by selecting the maximum value.
            The underlying value is typically ``"maximum"``.
    """

    MEAN = "mean"
    MEDIAN = "median"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
