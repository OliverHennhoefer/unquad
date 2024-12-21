from enum import Enum


class Strategy(Enum):
    """
    Enumerators of conformal strategies for anomaly detection.
    """

    SPLIT: str = "SC"
    CV: str = "CV"
    CV_PLUS: str = "CV+"
    J: str = "J"
    J_PLUS: str = "J+"
    JaB: str = "JaB"
    JaB_PLUS: str = "J+aB"
