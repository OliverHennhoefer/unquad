from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from unquad.enums.dataset import Dataset


class DataLoader:
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
