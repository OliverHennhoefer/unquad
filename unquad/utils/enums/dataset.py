from enum import Enum


class Dataset(Enum):
    BREAST: str = "breast"
    FRAUD: str = "fraud"
    IONOSPHERE: str = "ionosphere"
    MAMMOGRAPHY: str = "mammography"
    MUSK: str = "musk"
    SHUTTLE: str = "shuttle"
    THYROID: str = "thyroid"
    WBC: str = "wbc"
