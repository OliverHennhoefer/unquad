"""Batch generator for anomaly detection experiments with configurable anomaly mixing."""

from __future__ import annotations

from typing import Iterator, Optional, Union

import numpy as np
import pandas as pd


class BatchGenerator:
    """Generate batches with configurable anomaly contamination for evaluation.

    This generator creates batches from a test dataset with a specified proportion
    of anomalous instances. It ensures reproducibility through seed control and
    prevents duplicate instances within each batch.

    Parameters
    ----------
    x_normal : pd.DataFrame
        Feature matrix containing only normal instances.
    x_anomaly : pd.DataFrame
        Feature matrix containing only anomalous instances.
    batch_size : int
        Number of instances per batch.
    anomaly_proportion : float or int
        If < 1: percentage of anomalies in each batch (e.g., 0.1 = 10%).
        If >= 1: absolute number of anomalies per batch.
    random_state : Optional[int], default=None
        Seed for random number generator to ensure reproducibility.

    Attributes
    ----------
    n_normal : int
        Total number of normal instances available.
    n_anomaly : int
        Total number of anomalous instances available.
    rng : np.random.Generator
        Random number generator for sampling.

    Examples
    --------
    >>> from unquad.utils.data.load import load_shuttle
    >>> from unquad.utils.data.batch_generator import BatchGenerator
    >>>
    >>> # Load data with setup
    >>> x_train, x_test, y_test = load_shuttle(setup=True)
    >>>
    >>> # Separate test data by class
    >>> x_normal = x_test[y_test == 0]
    >>> x_anomaly = x_test[y_test == 1]
    >>>
    >>> # Create generator with 10% anomalies
    >>> batch_gen = BatchGenerator(
    ...     x_normal=x_normal,
    ...     x_anomaly=x_anomaly,
    ...     batch_size=100,
    ...     anomaly_proportion=0.1,
    ...     random_state=42
    ... )
    >>>
    >>> # Generate batches
    >>> for x_batch, y_batch in batch_gen.generate(n_batches=5):
    ...     print(f"Batch shape: {x_batch.shape}, Anomalies: {y_batch.sum()}")
    """

    def __init__(
        self,
        x_normal: pd.DataFrame,
        x_anomaly: pd.DataFrame,
        batch_size: int,
        anomaly_proportion: Union[float, int],
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize the batch generator."""
        self.x_normal = x_normal
        self.x_anomaly = x_anomaly
        self.batch_size = batch_size
        self.anomaly_proportion = anomaly_proportion
        self.random_state = random_state

        # Store dataset sizes
        self.n_normal = len(x_normal)
        self.n_anomaly = len(x_anomaly)

        # Initialize random number generator
        self.rng = np.random.default_rng(random_state)

        # Calculate anomaly count per batch
        if anomaly_proportion < 1:
            self.n_anomaly_per_batch = int(batch_size * anomaly_proportion)
        else:
            self.n_anomaly_per_batch = int(anomaly_proportion)

        self.n_normal_per_batch = batch_size - self.n_anomaly_per_batch

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the generator configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.anomaly_proportion < 0:
            raise ValueError(
                f"anomaly_proportion must be non-negative, got {self.anomaly_proportion}"
            )

        if self.n_anomaly_per_batch > self.batch_size:
            raise ValueError(
                f"Anomaly count ({self.n_anomaly_per_batch}) exceeds batch_size ({self.batch_size})"
            )

        if self.n_normal_per_batch > self.n_normal:
            raise ValueError(
                f"Not enough normal instances ({self.n_normal}) for requested "
                f"batch configuration (need {self.n_normal_per_batch} per batch)"
            )

        if self.n_anomaly_per_batch > self.n_anomaly:
            raise ValueError(
                f"Not enough anomaly instances ({self.n_anomaly}) for requested "
                f"batch configuration (need {self.n_anomaly_per_batch} per batch)"
            )

    def generate(
        self, n_batches: Optional[int] = None
    ) -> Iterator[tuple[pd.DataFrame, pd.Series]]:
        """Generate batches with mixed normal and anomalous instances.

        Parameters
        ----------
        n_batches : Optional[int], default=None
            Number of batches to generate. If None, generates indefinitely.

        Yields
        ------
        x_batch : pd.DataFrame
            Feature matrix for the batch.
        y_batch : pd.Series
            Labels for the batch (0 = normal, 1 = anomaly).
        """
        batch_count = 0

        while n_batches is None or batch_count < n_batches:
            # Sample indices without replacement for this batch
            normal_idx = self.rng.choice(
                self.n_normal, size=self.n_normal_per_batch, replace=False
            )
            anomaly_idx = self.rng.choice(
                self.n_anomaly, size=self.n_anomaly_per_batch, replace=False
            )

            # Extract samples
            x_batch_normal = self.x_normal.iloc[normal_idx]
            x_batch_anomaly = self.x_anomaly.iloc[anomaly_idx]

            # Combine samples
            x_batch = pd.concat(
                [x_batch_normal, x_batch_anomaly], axis=0, ignore_index=True
            )

            # Create labels
            y_batch = pd.Series(
                [0] * self.n_normal_per_batch + [1] * self.n_anomaly_per_batch,
                dtype=int,
            )

            # Shuffle the batch to mix normal and anomalous instances
            shuffle_idx = self.rng.permutation(self.batch_size)
            x_batch = x_batch.iloc[shuffle_idx].reset_index(drop=True)
            y_batch = y_batch.iloc[shuffle_idx].reset_index(drop=True)

            yield x_batch, y_batch
            batch_count += 1

    def reset(self) -> None:
        """Reset the random number generator to its initial state."""
        self.rng = np.random.default_rng(self.random_state)

    @property
    def max_batches(self) -> int:
        """Maximum number of unique batches (approximate due to random sampling)."""
        # This is an approximation based on combinations
        from math import comb

        max_normal_batches = comb(self.n_normal, self.n_normal_per_batch)
        max_anomaly_batches = comb(self.n_anomaly, self.n_anomaly_per_batch)

        return min(max_normal_batches, max_anomaly_batches)

    def __repr__(self) -> str:
        """String representation of the generator."""
        return (
            f"BatchGenerator("
            f"n_normal={self.n_normal}, "
            f"n_anomaly={self.n_anomaly}, "
            f"batch_size={self.batch_size}, "
            f"anomaly_per_batch={self.n_anomaly_per_batch})"
        )


def create_batch_generator(
    df: pd.DataFrame,
    train_size: float = 0.5,
    batch_size: int = 100,
    anomaly_proportion: Union[float, int] = 0.1,
    random_state: Optional[int] = None,
) -> tuple[pd.DataFrame, BatchGenerator]:
    """Convenience function to create a batch generator from a dataset.

    This function splits the dataset into training data (normal only) and
    creates a batch generator for the remaining data.

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset with a 'Class' column (0=normal, 1=anomaly).
    train_size : float, default=0.5
        Proportion of normal instances to use for training.
    batch_size : int, default=100
        Number of instances per batch.
    anomaly_proportion : float or int, default=0.1
        If < 1: percentage of anomalies in each batch.
        If >= 1: absolute number of anomalies per batch.
    random_state : Optional[int], default=None
        Seed for random number generator.

    Returns
    -------
    x_train : pd.DataFrame
        Training data containing only normal instances (without 'Class' column).
    batch_generator : BatchGenerator
        Configured batch generator for evaluation.

    Examples
    --------
    >>> from unquad.utils.data.load import load_shuttle
    >>> from unquad.utils.data.batch_generator import create_batch_generator
    >>>
    >>> df = load_shuttle()
    >>> x_train, batch_gen = create_batch_generator(
    ...     df,
    ...     train_size=0.5,
    ...     batch_size=200,
    ...     anomaly_proportion=0.15,
    ...     random_state=42
    ... )
    """
    # Separate normal and anomalous instances
    normal_mask = df["Class"] == 0
    df_normal = df[normal_mask]
    df_anomaly = df[~normal_mask]

    # Split normal data into train and test
    n_train = int(len(df_normal) * train_size)

    # Use numpy's random generator for consistency
    rng = np.random.default_rng(random_state)
    train_idx = rng.choice(len(df_normal), size=n_train, replace=False)
    test_normal_mask = np.ones(len(df_normal), dtype=bool)
    test_normal_mask[train_idx] = False

    # Create datasets
    x_train = df_normal.iloc[train_idx].drop(columns=["Class"])
    x_test_normal = df_normal.iloc[test_normal_mask].drop(columns=["Class"])
    x_test_anomaly = df_anomaly.drop(columns=["Class"])

    # Create batch generator
    batch_generator = BatchGenerator(
        x_normal=x_test_normal,
        x_anomaly=x_test_anomaly,
        batch_size=batch_size,
        anomaly_proportion=anomaly_proportion,
        random_state=random_state,
    )

    return x_train, batch_generator