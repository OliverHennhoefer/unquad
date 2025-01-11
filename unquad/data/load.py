import gzip
from importlib import resources

import pandas as pd

from pathlib import Path
from typing import Union, Tuple

from sklearn.model_selection import train_test_split


def load_breast(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    file = resources.files("unquad.data.datasets").joinpath("breast.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_fraud(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    file = resources.files("unquad.data.datasets").joinpath("fraud.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_ionosphere(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    file = resources.files("unquad.data.datasets").joinpath("ionosphere.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_mammography(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    file = resources.files("unquad.data.datasets").joinpath("mammography.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_musk(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    file = resources.files("unquad.data.datasets").joinpath("musk.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_shuttle(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    file = resources.files("unquad.data.datasets").joinpath("shuttle.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_thyroid(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    file = resources.files("unquad.data.datasets").joinpath("thyroid.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_wbc(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    file = resources.files("unquad.data.datasets").joinpath("wbc.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def _load_dataset(file_path: Path, setup: bool, random_state: int):

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}. ")

    with gzip.open(file_path, "rb") as f:
        df = pd.read_parquet(f)  # noqa

    if setup:
        return _create_setup(df, random_state=random_state)

    return df


def _create_setup(df: pd.DataFrame, random_state: int):
    inliers = df.loc[df.Class == 0]
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
            df.loc[df.Class == 1].sample(n=n_test_outlier, random_state=random_state),
        ],
        ignore_index=True,
    )
    x_test = test_set.drop(["Class"], axis=1)
    y_test = test_set["Class"]
    return x_train, x_test, y_test
