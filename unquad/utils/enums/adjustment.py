from enum import Enum


class Adjustment(Enum):
    """
    Enumerators for adjustment procedures used in False Discovery Rate (FDR) control.

    This enum defines the adjustment methods for controlling the False Discovery Rate
    in multiple hypothesis testing. These methods are used to adjust p-values to account
    for the multiple comparisons problem, ensuring a controlled false positive rate.

    Attributes:
        BENJAMINI_HOCHBERG (str): The Benjamini-Hochberg procedure for FDR control.
        BENJAMINI_YEKUTIELI (str): The Benjamini-Yekutieli procedure for FDR control.
        NONE (None): No adjustment procedure is applied.

    """

    BENJAMINI_HOCHBERG: str = "bh"
    BENJAMINI_YEKUTIELI: str = "by"
    NONE = None
