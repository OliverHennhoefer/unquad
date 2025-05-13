from enum import Enum


class Adjustment(Enum):
    """
    Enumerates adjustment procedures for False Discovery Rate (FDR) control.

    This enum represents different methods for adjusting p-values
    when performing multiple hypothesis tests.

    Attributes:
        BH (str): The Benjamini-Hochberg procedure.
        BY (str): The Benjamini-Yekutieli procedure.
        NONE: No adjustment.
    """

    BH: str = "bh"
    BY: str = "by"
    NONE = None


class Aggregation(Enum):
    """
    Enumerates aggregation functions for ensemble methods.

    This enum defines different aggregation strategies for combining multiple
    model outputs, typically used in ensemble methods.

    Attributes:
        MEAN (str): The mean aggregation method.
        MEDIAN (str): The median aggregation method.
        MINIMUM (str): The minimum aggregation method.
        MAXIMUM (str): The maximum aggregation method.
    """

    MEAN: str = "mean"
    MEDIAN: str = "median"
    MINIMUM: str = "minimum"
    MAXIMUM: str = "maximum"
