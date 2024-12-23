from enum import Enum


class Strategy(Enum):
    """
    Enumerators for conformal strategies used in anomaly detection.

    This enum defines various conformal strategies that can be applied in anomaly
    detection methods. These strategies determine how the model is trained and how
    calibration sets are used for conformal prediction.

    Attributes:
        SPLIT (str): The Split Conformal strategy (SC).
        CV (str): The Cross-Validation Conformal strategy (CV).
        CV_PLUS (str): The Cross-Validation Conformal strategy with additional models (CV+).
        J (str): The Jackknife Conformal strategy (J).
        J_PLUS (str): The Jackknife Conformal strategy with additional models (J+).
        JaB (str): The Jackknife-and-Bootstrap Conformal strategy (JaB).
        JaB_PLUS (str): The Jackknife-and-Bootstrap Conformal strategy with additional models (J+aB).
    """

    SPLIT: str = "SC"
    CV: str = "CV"
    CV_PLUS: str = "CV+"
    J: str = "J"
    J_PLUS: str = "J+"
    JaB: str = "JaB"
    JaB_PLUS: str = "J+aB"
