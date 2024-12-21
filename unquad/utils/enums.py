from enum import Enum


class Adjustment(Enum):
    BH: str = "bh"
    BY: str = "by"
    NONE = None


class Aggregation(Enum):
    MEAN: str = "mean"
    MEDIAN: str = "median"
    MINIMUM: str = "minimum"
    MAXIMUM: str = "maximum"


class Dataset(Enum):
    BREAST: str = "breast"
    FRAUD: str = "fraud"
    IONOSPHERE: str = "ionosphere"
    MAMMOGRAPHY: str = "mammography"
    MUSK: str = "musk"
    SHUTTLE: str = "shuttle"
    THYROID: str = "thyroid"
    WBC: str = "wbc"


class Strategy(Enum):

    SPLIT: str = "SC"
    CV: str = "CV"
    CV_PLUS: str = "CV+"
    J: str = "J"
    J_PLUS: str = "J+"
    JaB: str = "JaB"
    JaB_PLUS: str = "J+aB"
