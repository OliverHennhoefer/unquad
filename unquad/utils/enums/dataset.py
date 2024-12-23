from enum import Enum


class Dataset(Enum):
    """
    Enumerators for available datasets.

    This enum defines the names of datasets that are commonly used in machine learning
    and anomaly detection tasks. These datasets represent a variety of real-world
    problems, including classification and fraud detection.

    Attributes:
        BREAST (str): The breast cancer dataset.
        FRAUD (str): The credit card fraud detection dataset.
        IONOSPHERE (str): The ionosphere dataset, used for detecting abnormalities.
        MAMMOGRAPHY (str): The mammography dataset for detecting abnormalities.
        MUSK (str): The musk dataset, used for detecting abnormalities.
        SHUTTLE (str): The shuttle dataset, typically used for anomaly detection.
        THYROID (str): The thyroid disease detection dataset for detecting abnormalities.
        WBC (str): The white blood cell (WBC) dataset used for detecting abnormalities.
    """

    BREAST: str = "breast"
    FRAUD: str = "fraud"
    IONOSPHERE: str = "ionosphere"
    MAMMOGRAPHY: str = "mammography"
    MUSK: str = "musk"
    SHUTTLE: str = "shuttle"
    THYROID: str = "thyroid"
    WBC: str = "wbc"
