import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from unquad.utils.enums.dataset import Dataset


class DataLoader:
    """
    Data loader for loading and preparing datasets for anomaly detection.

    This class is responsible for loading a dataset in Parquet format and providing
    a setup for anomaly detection tasks, following the setup in the paper
    'Testing for Outliers with Conformal p-Values' (Bates, 2023). It provides methods
    for splitting the data into training and test sets, as well as creating examples
    for training and testing models.

    Attributes:
        _df (pd.DataFrame): The loaded dataset stored as a Pandas DataFrame.

    Methods:
        __init__(dataset):
            Initializes the DataLoader object by loading the dataset specified by the `dataset` argument.

        load_data(dataset):
            Loads the dataset from a Parquet file based on the dataset name.

        get_example_setup(random_state=1):
            Sets up the training and test sets according to the setup in the paper
            'Testing for Outliers with Conformal p-Values' (Bates, 2023).
            It splits the data into inliers and outliers, with a training set of inliers
            and a test set consisting of both inliers and outliers.

            Args:
                random_state (int, optional): The random seed for reproducibility. Default is 1.

            Returns:
                tuple: A tuple containing:
                    - x_train (pd.DataFrame): The features of the training set.
                    - x_test (pd.DataFrame): The features of the test set.
                    - y_test (pd.Series): The labels of the test set.

        df:
            Returns the loaded dataset as a Pandas DataFrame.
    """

    def __init__(self, dataset: Dataset):
        self._df: pd.DataFrame = self.load_data(dataset)

    @staticmethod
    def load_data(dataset: Dataset) -> pd.DataFrame:
        dataset = dataset.value
        path = Path(__file__).parent / "parquet" / f"{dataset}" / f"{dataset}.parquet"
        df = pd.read_parquet(path)
        return df

    def get_example_setup(self, random_state: int = 1):
        """
        Setup following 'Testing for Outliers with Conformal p-Values' (Bates. 2023).
        """

        inliers = self._df.loc[self._df.Class == 0]
        n_train = len(inliers) // 2
        n_test = min(1000, n_train // 3)
        n_test_outlier = n_test // 10
        n_test_inlier = n_test - n_test_outlier
        train_set, test_set = train_test_split(
            inliers,
            train_size=n_train,
            random_state=random_state,
        )
        x_train = train_set.drop(["Class"], axis=1)
        test_set = pd.concat(
            [
                test_set.sample(n=n_test_inlier, random_state=random_state),
                self._df.loc[self._df.Class == 1].sample(
                    n=n_test_outlier, random_state=random_state
                ),
            ],
            ignore_index=True,
        )
        x_test = test_set.drop(["Class"], axis=1)
        y_test = test_set["Class"]
        return x_train, x_test, y_test

    @property
    def df(self):
        return self._df
