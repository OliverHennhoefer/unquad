from enum import Enum


class Adjustment(Enum):
    """
    Enumerates adjustment procedures for False Discovery Rate (FDR) control.

    This enum represents different methods for adjusting p-values when performing multiple hypothesis tests.

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

    This enum defines different aggregation strategies for combining multiple model outputs,
    typically used in ensemble methods.

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


class Dataset(Enum):
    """
    Enumerates the available datasets for anomaly detection tasks.

    This enum represents various datasets that can be used for training and testing anomaly detection models.

    Attributes:
        BREAST (str): The breast cancer dataset.
        FRAUD (str): The fraud detection dataset.
        IONOSPHERE (str): The ionosphere dataset.
        MAMMOGRAPHY (str): The mammography dataset.
        MUSK (str): The musk dataset.
        SHUTTLE (str): The shuttle dataset.
        THYROID (str): The thyroid dataset.
        WBC (str): The Wisconsin Breast Cancer (WBC) dataset.
    """

    BREAST: str = "breast"
    FRAUD: str = "fraud"
    IONOSPHERE: str = "ionosphere"
    MAMMOGRAPHY: str = "mammography"
    MUSK: str = "musk"
    SHUTTLE: str = "shuttle"
    THYROID: str = "thyroid"
    WBC: str = "wbc"


class Strategy(Enum):
    """
    Enumerates conformal strategies for anomaly detection.

    This enum defines various conformal strategies for anomaly detection tasks,
    used to control how anomalies are detected and validated.

    Attributes:
        SPLIT (str): Split conformal strategy.
        CV (str): Cross-validation conformal strategy.
        CV_PLUS (str): Cross-validation conformal strategy with an additional model.
        J (str): Jackknife conformal strategy.
        J_PLUS (str): Jackknife conformal strategy with an additional model.
        JaB (str): Jackknife + Bootstrap conformal strategy.
        JaB_PLUS (str): Jackknife + Bootstrap conformal strategy with an additional model.
    """

    SPLIT: str = "SC"
    CV: str = "CV"
    CV_PLUS: str = "CV+"
    J: str = "J"
    J_PLUS: str = "J+"
    JaB: str = "JaB"
    JaB_PLUS: str = "J+aB"
