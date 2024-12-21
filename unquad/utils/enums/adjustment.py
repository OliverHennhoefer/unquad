from enum import Enum


class Adjustment(Enum):
    """
    Enumerators of adjustment procedures for FDR control.
    """

    BENJAMINI_HOCHBERG: str = "bh"
    BENJAMINI_YEKUTIELI: str = "by"
    NONE = None
