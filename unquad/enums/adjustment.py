from enum import Enum


class Adjustment(Enum):
    """
    Enumerators of available adjustment procedures for control of the false discovery rate.
    """

    BENJAMINI_HOCHBERG = "bh"
    BENJAMINI_YEKUTIELI = "by"
    NONE = None
