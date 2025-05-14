from enum import Enum


class Adjustment(Enum):
    """Procedures for False Discovery Rate (FDR) control in multiple tests.

    This enumeration defines methods for adjusting p-values or significance
    levels when conducting multiple statistical hypothesis tests to manage
    the overall error rate.

    Attributes:
        BH: Represents the Benjamini-Hochberg procedure. The underlying value
            is typically ``"bh"``.
        BY: Represents the Benjamini-Yekutieli procedure. The underlying value
            is typically ``"by"``.
        NONE: Represents that no adjustment should be applied. The underlying
            value is ``None``.
    """

    BH: str = "bh"
    BY: str = "by"
    NONE = None


class Aggregation(Enum):
    """Aggregation functions for combining multiple model outputs or scores.

    This enumeration lists strategies for aggregating data, commonly employed
    in ensemble methods to combine predictions or scores from several models.

    Attributes:
        MEAN: Represents aggregation by calculating the arithmetic mean.
            The underlying value is typically ``"mean"``.
        MEDIAN: Represents aggregation by calculating the median.
            The underlying value is typically ``"median"``.
        MINIMUM: Represents aggregation by selecting the minimum value.
            The underlying value is typically ``"minimum"``.
        MAXIMUM: Represents aggregation by selecting the maximum value.
            The underlying value is typically ``"maximum"``.
    """

    MEAN: str = "mean"
    MEDIAN: str = "median"
    MINIMUM: str = "minimum"
    MAXIMUM: str = "maximum"
