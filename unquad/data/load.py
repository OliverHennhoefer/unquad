import gzip
import pandas as pd

from pathlib import Path
from importlib import resources
from typing import Union, Tuple

from sklearn.model_selection import train_test_split

root = "unquad.data.datasets"


def load_breast(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads the breast cancer dataset.

    Args:
        setup: If True, creates an experimental setup as described in Bates (2023).
        random_state: Seed for random number generation.

    Returns:
        Either the full dataset or a tuple of (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("breast.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_fraud(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads the fraud detection dataset.

    Args:
        setup: If True, creates an experimental setup as described in Bates (2023).
        random_state: Seed for random number generation.

    Returns:
        Either the full dataset or a tuple of (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("fraud.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_ionosphere(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads the ionosphere dataset.

    Args:
        setup: If True, creates an experimental setup as described in Bates (2023).
        random_state: Seed for random number generation.

    Returns:
        Either the full dataset or a tuple of (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("ionosphere.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_mammography(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads the mammography dataset.

    Args:
        setup: If True, creates an experimental setup as described in Bates (2023).
        random_state: Seed for random number generation.

    Returns:
        Either the full dataset or a tuple of (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("mammography.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_musk(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads the musk dataset.

    Args:
        setup: If True, creates an experimental setup as described in Bates (2023).
        random_state: Seed for random number generation.

    Returns:
        Either the full dataset or a tuple of (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("musk.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_shuttle(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads the shuttle dataset.

    Args:
        setup: If True, creates an experimental setup as described in Bates (2023).
        random_state: Seed for random number generation.

    Returns:
        Either the full dataset or a tuple of (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("shuttle.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_thyroid(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads the thyroid dataset.

    Args:
        setup: If True, creates an experimental setup as described in Bates (2023).
        random_state: Seed for random number generation.

    Returns:
        Either the full dataset or a tuple of (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("thyroid.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_wbc(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads the white blood cells (WBC) dataset.

    Args:
        setup: If True, creates an experimental setup as described in Bates (2023).
        random_state: Seed for random number generation.

    Returns:
        Either the full dataset or a tuple of (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("wbc.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def _load_dataset(file_path: Path, setup: bool, random_state: int):
    """Loads a dataset from a gzipped parquet file.

    Args:
        file_path: Path to the dataset file.
        setup: If True, creates an experimental setup as described in Bates (2023).
        random_state: Seed for random number generation.

    Returns:
        Either the full dataset or a tuple of (x_train, x_test, y_test).

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}. ")

    with gzip.open(file_path, "rb") as f:
        df = pd.read_parquet(f)  # noqa

    if setup:
        return _create_setup(df, random_state=random_state)

    return df


def _create_setup(df: pd.DataFrame, random_state: int):
    """Creates an experimental setup as described in Bates (2023).

    Args:
        df: The dataset to process.
        random_state: Seed for random number generation.

    Returns:
        A tuple of (x_train, x_test, y_test), where:
        - x_train: Training data (normal samples only).
        - x_test: Test data (a mix of normal and outlier samples).
        - y_test: Labels for the test data (1 for outliers, 0 for normal).
    """
    normal = df[df["Class"] == 0]
    n_train = len(normal) // 2
    n_test = min(1000, n_train // 3)
    n_test_outlier = n_test // 10
    n_test_normal = n_test - n_test_outlier

    x_train, test_set = train_test_split(
        normal, train_size=n_train, random_state=random_state
    )
    x_train = x_train.drop(columns=["Class"])

    test_normal = test_set.sample(n=n_test_normal, random_state=random_state)
    test_outliers = df[df["Class"] == 1].sample(
        n=n_test_outlier, random_state=random_state
    )
    test_set = pd.concat([test_normal, test_outliers], ignore_index=True)

    x_test = test_set.drop(columns=["Class"])
    y_test = test_set["Class"]

    return x_train, x_test, y_test
