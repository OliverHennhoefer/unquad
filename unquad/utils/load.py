import gzip
import hashlib
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset configuration
DATASET_VERSION = os.environ.get("UNQUAD_DATASET_VERSION", "v.0.8.1-datasets")  # Update this when releasing new datasets
DATASET_BASE_URL = os.environ.get(
    "UNQUAD_DATASET_URL",
    f"https://github.com/OliverHennhoefer/unquad/releases/download/{DATASET_VERSION}/"
)
CACHE_DIR = Path(os.environ.get("UNQUAD_CACHE_DIR", str(Path.home() / ".unquad" / "datasets")))

# Dataset checksums for verification (optional, update when releasing)
DATASET_CHECKSUMS = {
    "breast.parquet.gz": None,  # Add SHA256 checksums here after release
    "fraud.parquet.gz": None,
    "ionosphere.parquet.gz": None,
    "mammography.parquet.gz": None,
    "musk.parquet.gz": None,
    "shuttle.parquet.gz": None,
    "thyroid.parquet.gz": None,
    "wbc.parquet.gz": None,
}

# Check if pyarrow is available for reading parquet files
try:
    import pyarrow  # noqa: F401
    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False


def load_breast(
    setup: bool = False, random_state: int = 1
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Breast Cancer Wisconsin (Diagnostic) dataset.

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

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path("breast.parquet.gz"), setup, random_state)


def load_fraud(
    setup: bool = False, random_state: int = 1
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load a credit card fraud detection dataset.

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

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path("fraud.parquet.gz"), setup, random_state)


def load_ionosphere(
    setup: bool = False, random_state: int = 1
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Ionosphere dataset.

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

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path("ionosphere.parquet.gz"), setup, random_state)


def load_mammography(
    setup: bool = False, random_state: int = 1
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Mammography dataset.

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

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path("mammography.parquet.gz"), setup, random_state)


def load_musk(
    setup: bool = False, random_state: int = 1
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Musk (Version 2) dataset.

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

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path("musk.parquet.gz"), setup, random_state)


def load_shuttle(
    setup: bool = False, random_state: int = 1
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Shuttle dataset.

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

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path("shuttle.parquet.gz"), setup, random_state)


def load_thyroid(
    setup: bool = False, random_state: int = 1
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Thyroid Disease (ann-thyroid) dataset.

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

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path("thyroid.parquet.gz"), setup, random_state)


def load_wbc(
    setup: bool = False, random_state: int = 1
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the Wisconsin Breast Cancer (Original) (WBC) dataset.

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

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).
    """
    return _load_dataset(Path("wbc.parquet.gz"), setup, random_state)


def _download_dataset(filename: str, show_progress: bool = True) -> Path:
    """Download dataset from GitHub releases and cache locally.
    
    Args:
        filename: Name of the dataset file (e.g., "breast.parquet.gz")
        show_progress: Whether to show download progress
        
    Returns:
        Path to the cached dataset file
        
    Raises:
        URLError: If download fails
        RuntimeError: If checksum verification fails
    """
    cache_path = CACHE_DIR / filename
    
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # If file already cached, verify and return
    if cache_path.exists():
        if DATASET_CHECKSUMS.get(filename) is not None:
            if _verify_checksum(cache_path, DATASET_CHECKSUMS[filename]):
                return cache_path
            else:
                print(f"Checksum mismatch for cached {filename}, re-downloading...")
                cache_path.unlink()
        else:
            # No checksum available, trust cached file
            return cache_path
    
    # Download file
    url = urljoin(DATASET_BASE_URL, filename)
    print(f"Downloading {filename} from {url}...")
    
    try:
        # Add headers to avoid GitHub rate limiting
        req = Request(url, headers={'User-Agent': 'unquad-dataset-loader'})
        
        with urlopen(req) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            
            # Download with progress bar if requested
            if show_progress and total_size > 0:
                try:
                    from tqdm import tqdm
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        with open(cache_path, 'wb') as f:
                            while True:
                                chunk = response.read(8192)
                                if not chunk:
                                    break
                                f.write(chunk)
                                pbar.update(len(chunk))
                except ImportError:
                    # Fallback to simple download without progress bar
                    with open(cache_path, 'wb') as f:
                        f.write(response.read())
            else:
                with open(cache_path, 'wb') as f:
                    f.write(response.read())
                    
    except (URLError, HTTPError) as e:
        if cache_path.exists():
            cache_path.unlink()
        raise URLError(f"Failed to download {filename}: {str(e)}") from e
    
    # Verify checksum if available
    if DATASET_CHECKSUMS.get(filename) is not None:
        if not _verify_checksum(cache_path, DATASET_CHECKSUMS[filename]):
            cache_path.unlink()
            raise RuntimeError(f"Checksum verification failed for {filename}")
    
    print(f"Successfully downloaded {filename}")
    return cache_path


def _verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """Verify SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest() == expected_checksum


def _load_dataset(
    file_path: Path, setup: bool, random_state: int
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load a dataset from a gzipped Parquet file and optionally sets it up.

    This is a helper function used by the specific dataset loaders. It downloads
    the dataset from GitHub releases if not cached locally, then reads the
    Parquet file compressed with gzip. If `setup` is true, it calls
    `_create_setup` to split the data.

    Args:
        file_path (pathlib.Path): The path to the dataset file (used to extract filename).
        setup (bool): If ``True``, the data is processed by `_create_setup`
            to produce training and testing sets.
        random_state (int): Seed for random number generation, used if
            `setup` is ``True``.

    Returns
    -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
            If `setup` is ``False``, returns the complete dataset as a
            DataFrame. If `setup` is ``True``, returns a tuple:
            (x_train, x_test, y_test).

    Raises
    ------
        ImportError: If pyarrow is not available for reading parquet files.
        URLError: If dataset download fails.
    """
    if not _PYARROW_AVAILABLE:
        raise ImportError(
            "The datasets functionality requires pyarrow to read parquet files. "
            "Please install the data dependencies with: pip install unquad[data]"
        )
    
    # Extract filename from the provided path
    filename = file_path.name
    
    # Download dataset if not cached
    cached_path = _download_dataset(filename)
    
    with gzip.open(cached_path, "rb") as f:
        df = pd.read_parquet(f)

    if setup:
        return _create_setup(df, random_state=random_state)

    return df


def _create_setup(
    df: pd.DataFrame, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Create an experimental train/test split from a dataset.

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

    Returns
    -------
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


def clear_cache(dataset: Optional[str] = None) -> None:
    """Clear cached dataset files.
    
    Args:
        dataset: Specific dataset name to clear (e.g., "breast"). 
                If None, clears all cached datasets.
    """
    if dataset is not None:
        # Clear specific dataset
        cache_file = CACHE_DIR / f"{dataset}.parquet.gz"
        if cache_file.exists():
            cache_file.unlink()
            print(f"Cleared cached dataset: {dataset}")
        else:
            print(f"No cached dataset found: {dataset}")
    else:
        # Clear all datasets
        if CACHE_DIR.exists():
            for file in CACHE_DIR.glob("*.parquet.gz"):
                file.unlink()
                print(f"Cleared cached dataset: {file.stem}")
            print("All cached datasets cleared.")
        else:
            print("No cache directory found.")


def list_cached_datasets() -> list[str]:
    """List all cached datasets.
    
    Returns:
        List of cached dataset names (without .parquet.gz extension)
    """
    if not CACHE_DIR.exists():
        return []
    
    # Remove .parquet.gz extension to get just the dataset name
    return [f.name.removesuffix('.parquet.gz') for f in CACHE_DIR.glob("*.parquet.gz")]


def get_cache_info() -> dict:
    """Get information about the dataset cache.
    
    Returns:
        Dictionary with cache information including path, size, and datasets
    """
    info = {
        "cache_dir": str(CACHE_DIR),
        "exists": CACHE_DIR.exists(),
        "datasets": [],
        "total_size_mb": 0
    }
    
    if CACHE_DIR.exists():
        for file in CACHE_DIR.glob("*.parquet.gz"):
            size_mb = file.stat().st_size / (1024 * 1024)
            info["datasets"].append({
                "name": file.name.removesuffix('.parquet.gz'),
                "size_mb": round(size_mb, 2)
            })
            info["total_size_mb"] += size_mb
    
    info["total_size_mb"] = round(info["total_size_mb"], 2)
    return info
