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
    """Loads the Breast Cancer Wisconsin (Diagnostic) dataset.

    This dataset contains features computed from a digitized image of a
    fine needle aspirate (FNA) of a breast mass. They describe
    characteristics of the cell nuclei present in the image. The "Class"
    column typically indicates malignant (1) or benign (0).

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        random_state (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``1``.

    Returns:
        Union[pandas.DataFrame, Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("breast.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_fraud(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads a credit card fraud detection dataset.

    This dataset typically contains transactions made by European cardholders.
    It presents transactions that occurred in two days, where it has features
    that are numerical input variables, the result of a PCA transformation.
    The "Class" column indicates fraudulent (1) or legitimate (0) transactions.

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        random_state (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``1``.

    Returns:
        Union[pandas.DataFrame, Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("fraud.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_ionosphere(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads the Ionosphere dataset.

    This radar data was collected by a system in Goose Bay, Labrador.
    The targets were free electrons in the ionosphere. "Good" radar returns
    are those showing evidence of some type of structure in the ionosphere.
    "Bad" returns are those that do not; their signals pass through the
    ionosphere. The "Class" column indicates "good" (0) or "bad" (1, anomaly).

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        random_state (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``1``.

    Returns:
        Union[pandas.DataFrame, Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("ionosphere.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_mammography(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads the Mammography dataset.

    This dataset is used for detecting breast cancer based on mammographic
    findings. It contains features related to BI-RADS assessment, age,
    shape, margin, and density. The "Class" column usually indicates
    benign (0) or malignant (1, anomaly).

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        random_state (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``1``.

    Returns:
        Union[pandas.DataFrame, Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("mammography.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_musk(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads the Musk (Version 2) dataset.

    This dataset describes a set of 102 molecules of which 39 are judged
    by human experts to be musks and the remaining 63 molecules are
    judged to be non-musks. The 166 features describe the three-dimensional
    conformation of the molecules. The "Class" indicates musk (1, anomaly)
    or non-musk (0).

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        random_state (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``1``.

    Returns:
        Union[pandas.DataFrame, Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("musk.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_shuttle(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads the Shuttle dataset.

    This dataset contains data from a NASA space shuttle mission concerning
    the position of radiators in the shuttle. The "Class" column indicates
    normal (0) or anomalous (1) states. The original dataset has multiple
    anomaly classes; this version typically simplifies it to binary.

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        random_state (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``1``.

    Returns:
        Union[pandas.DataFrame, Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("shuttle.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_thyroid(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads the Thyroid Disease (ann-thyroid) dataset.

    This dataset is used for diagnosing thyroid conditions based on patient
    attributes and test results. The "Class" column indicates normal (0)
    or some form of thyroid disease (1, anomaly).

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        random_state (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``1``.

    Returns:
        Union[pandas.DataFrame, Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("thyroid.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def load_wbc(
    setup: bool = False, random_state: int = 1
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads the Wisconsin Breast Cancer (Original) (WBC) dataset.

    This dataset contains features derived from clinical observations of
    breast cancer, such as clump thickness, cell size uniformity, etc.
    The "Class" column indicates benign (0) or malignant (1, anomaly).
    Note: This is distinct from the "breast" dataset which is often the
    Diagnostic version.

    Args:
        setup (bool, optional): If ``True``, splits the data into training
            and testing sets according to a specific experimental setup,
            returning (x_train, x_test, y_test). `x_train` contains only
            normal samples. Defaults to ``False``, which returns the full
            DataFrame.
        random_state (int, optional): Seed for random number generation used
            in data splitting if `setup` is ``True``. Defaults to ``1``.

    Returns:
        Union[pandas.DataFrame, Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    file = resources.files(root).joinpath("wbc.parquet.gz")
    path = Path(str(file))
    return _load_dataset(path, setup, random_state)


def _load_dataset(
    file_path: Path, setup: bool, random_state: int
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Loads a dataset from a gzipped Parquet file and optionally sets it up.

    This is a helper function used by the specific dataset loaders. It reads
    a Parquet file compressed with gzip. If `setup` is true, it calls
    `_create_setup` to split the data.

    Args:
        file_path (pathlib.Path): The full path to the gzipped Parquet file.
        setup (bool): If ``True``, the data is processed by `_create_setup`
            to produce training and testing sets.
        random_state (int): Seed for random number generation, used if
            `setup` is ``True``.

    Returns:
        Union[pandas.DataFrame, Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).

    Raises:
        FileNotFoundError: If the dataset file at `file_path` does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}.")

    with gzip.open(file_path, "rb") as f:
        df = pd.read_parquet(f)

    if setup:
        return _create_setup(df, random_state=random_state)

    return df


def _create_setup(
    df: pd.DataFrame, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Creates an experimental train/test split from a dataset.

    This setup aims to create a scenario for anomaly detection where:
    - The training set (`x_train`) contains only normal samples (Class 0).
    - The test set (`x_test`, `y_test`) contains a mix of normal samples
      and a smaller proportion of outlier samples (Class 1).

    The sizes are determined as follows:
    - `n_train`: Half of the available normal samples.
    - `n_test`: The minimum of 1000 or one-third of `n_train`.
    - `n_test_outlier`: 10% of `n_test`.
    - `n_test_normal`: The remaining 90% of `n_test`.

    The "Class" column is dropped from `x_train` and `x_test`.

    Args:
        df (pandas.DataFrame): The input DataFrame, expected to have a "Class"
            column where 0 indicates normal and 1 indicates an outlier.
        random_state (int): Seed for random number generation used in
            splitting and sampling.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series]:
            A tuple (x_train, x_test, y_test):
            - `x_train`: DataFrame of training features (normal samples only).
            - `x_test`: DataFrame of test features.
            - `y_test`: Series of test labels (0 for normal, 1 for outlier).
    """
    normal = df[df["Class"] == 0]
    n_train = len(normal) // 2
    n_test = min(1000, n_train // 3)
    n_test_outlier = n_test // 10
    n_test_normal = n_test - n_test_outlier

    x_train_full, test_set_normal_pool = train_test_split(
        normal, train_size=n_train, random_state=random_state
    )
    x_train = x_train_full.drop(columns=["Class"])

    # Ensure enough samples are available for sampling
    # These checks could raise errors or adjust sample sizes if needed
    actual_n_test_normal = min(n_test_normal, len(test_set_normal_pool))
    test_normal = test_set_normal_pool.sample(
        n=actual_n_test_normal, random_state=random_state
    )

    outliers_available = df[df["Class"] == 1]
    actual_n_test_outlier = min(n_test_outlier, len(outliers_available))
    test_outliers = outliers_available.sample(
        n=actual_n_test_outlier, random_state=random_state
    )

    test_set = pd.concat([test_normal, test_outliers], ignore_index=True)

    x_test = test_set.drop(columns=["Class"])
    y_test = test_set["Class"]

    return x_train, x_test, y_test