from enum import Enum


class Adjustment(Enum):
    """
    Enumerators of adjustment procedures for FDR control.
    """

    BENJAMINI_HOCHBERG = "bh"
    BENJAMINI_YEKUTIELI = "by"
    NONE = None
