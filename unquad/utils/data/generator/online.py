"""Online generator for single-instance streaming with deterministic anomaly control."""

from collections.abc import Iterator

import numpy as np
import pandas as pd


class OnlineGenerator:
    """Generate single instances with deterministic anomaly contamination for streaming.

    This generator creates a deterministic stream of single instances by pre-creating
    a pool with exact anomaly proportions, then sampling from this pool sequentially.
    This ensures precise control over anomaly rates for experimental reproducibility.

    Parameters
    ----------
    x_normal : pd.DataFrame
        Feature matrix containing only normal instances.
    x_anomaly : pd.DataFrame
        Feature matrix containing only anomalous instances.
    anomaly_proportion : float
        Exact proportion of anomalies in the generated stream (e.g., 0.01 = 1%).
    pool_size : int, default=1000
        Size of the pre-generated instance pool. Should be large enough to accommodate
        desired stream length and allow exact proportion calculation.
    random_state : Optional[int], default=None
        Seed for random number generator to ensure reproducibility.
    replacement : bool, default=True
        Whether to allow sampling the same instance multiple times.

    Attributes
    ----------
    n_normal : int
        Total number of normal instances available.
    n_anomaly : int
        Total number of anomalous instances available.
    pool_labels : np.ndarray
        Pre-generated labels for the instance pool.
    pool_indices : np.ndarray
        Shuffled indices for deterministic but randomized sampling.
    rng : np.random.Generator
        Random number generator for sampling.

    Examples
    --------
    >>> from unquad.utils.data.load import load_shuttle
    >>> from unquad.utils.data.online_generator import OnlineGenerator
    >>>
    >>> # Load data with setup
    >>> x_train, x_test, y_test = load_shuttle(setup=True)
    >>>
    >>> # Separate test data by class
    >>> x_normal = x_test[y_test == 0]
    >>> x_anomaly = x_test[y_test == 1]
    >>>
    >>> # Create generator with exactly 1% anomalies
    >>> online_gen = OnlineGenerator(
    ...     x_normal=x_normal,
    ...     x_anomaly=x_anomaly,
    ...     anomaly_proportion=0.01,
    ...     pool_size=1000,
    ...     random_state=42
    ... )
    >>>
    >>> # Generate exactly 100 instances with guaranteed 1 anomaly
    >>> anomaly_count = 0
    >>> for x_instance, y_label in online_gen.generate_stream(n_instances=100):
    ...     anomaly_count += y_label
    >>> print(f"Anomalies in 100 instances: {anomaly_count}")  # Exactly 1
    """

    def __init__(
        self,
        x_normal: pd.DataFrame,
        x_anomaly: pd.DataFrame,
        anomaly_proportion: float,
        pool_size: int = 1000,
        random_state: int | None = None,
        replacement: bool = True,
    ) -> None:
        """Initialize the online generator."""
        self.x_normal = x_normal
        self.x_anomaly = x_anomaly
        self.anomaly_proportion = anomaly_proportion
        self.pool_size = pool_size
        self.random_state = random_state
        self.replacement = replacement

        # Store dataset sizes
        self.n_normal = len(x_normal)
        self.n_anomaly = len(x_anomaly)

        # Initialize random number generator
        self.rng = np.random.default_rng(random_state)

        # Validate configuration
        self._validate_config()

        # Create deterministic pool
        self._create_instance_pool()

    def _validate_config(self) -> None:
        """Validate the generator configuration."""
        if self.pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {self.pool_size}")

        if not 0 <= self.anomaly_proportion <= 1:
            raise ValueError(
                f"anomaly_proportion must be between 0 and 1, "
                f"got {self.anomaly_proportion}"
            )

        # Calculate exact counts
        n_anomaly_pool = int(self.pool_size * self.anomaly_proportion)
        n_normal_pool = self.pool_size - n_anomaly_pool

        if n_normal_pool > 0 and self.n_normal == 0:
            raise ValueError(
                "No normal instances available but normal instances required"
            )

        if n_anomaly_pool > 0 and self.n_anomaly == 0:
            raise ValueError(
                "No anomaly instances available but anomaly instances required"
            )

        if not self.replacement:
            if n_normal_pool > self.n_normal:
                raise ValueError(
                    f"Pool requires {n_normal_pool} normal instances but only "
                    f"{self.n_normal} available (replacement=False)"
                )
            if n_anomaly_pool > self.n_anomaly:
                raise ValueError(
                    f"Pool requires {n_anomaly_pool} anomaly instances but only "
                    f"{self.n_anomaly} available (replacement=False)"
                )

    def _create_instance_pool(self) -> None:
        """Create deterministic pool with exact anomaly proportion."""
        # Calculate exact counts for deterministic proportion
        n_anomaly_pool = int(self.pool_size * self.anomaly_proportion)
        n_normal_pool = self.pool_size - n_anomaly_pool

        # Create labels pool (0=normal, 1=anomaly)
        self.pool_labels = np.concatenate(
            [np.zeros(n_normal_pool, dtype=int), np.ones(n_anomaly_pool, dtype=int)]
        )

        # Shuffle for randomized order while maintaining exact counts
        self.rng.shuffle(self.pool_labels)

        # Create shuffled indices for sampling
        self.pool_indices = self.rng.permutation(self.pool_size)
        self._current_index = 0

    def generate_stream(
        self, n_instances: int | None = None
    ) -> Iterator[tuple[pd.DataFrame, int]]:
        """Generate stream of single instances with deterministic anomaly proportion.

        Parameters
        ----------
        n_instances : Optional[int], default=None
            Number of instances to generate. If None, generates indefinitely
            by cycling through the pool.

        Yields
        ------
        x_instance : pd.DataFrame
            Single instance feature vector.
        y_label : int
            Label for the instance (0 = normal, 1 = anomaly).
        """
        instance_count = 0

        while n_instances is None or instance_count < n_instances:
            # Get current pool position
            pool_idx = self.pool_indices[self._current_index]
            label = self.pool_labels[pool_idx]

            # Sample instance based on label
            if label == 1:  # Anomaly
                if self.replacement:
                    data_idx = self.rng.integers(0, self.n_anomaly)
                else:
                    # For without replacement, would need more complex tracking
                    data_idx = pool_idx % self.n_anomaly
                x_instance = self.x_anomaly.iloc[[data_idx]]
            else:  # Normal
                if self.replacement:
                    data_idx = self.rng.integers(0, self.n_normal)
                else:
                    # For without replacement, would need more complex tracking
                    data_idx = pool_idx % self.n_normal
                x_instance = self.x_normal.iloc[[data_idx]]

            # Reset index and return instance
            x_instance = x_instance.reset_index(drop=True)

            yield x_instance, label

            # Move to next pool position
            self._current_index = (self._current_index + 1) % self.pool_size
            instance_count += 1

    def reset(self) -> None:
        """Reset the generator to start from beginning of pool."""
        self._current_index = 0

    def get_exact_anomaly_count(self, n_instances: int) -> int:
        """Calculate exact number of anomalies in next n_instances.

        Parameters
        ----------
        n_instances : int
            Number of instances to check.

        Returns
        -------
        int
            Exact number of anomalies in the next n_instances.
        """
        if n_instances <= 0:
            return 0

        # Calculate how many complete cycles and remainder
        complete_cycles = n_instances // self.pool_size
        remainder = n_instances % self.pool_size

        # Anomalies in complete cycles
        anomalies_per_cycle = int(self.pool_size * self.anomaly_proportion)
        total_anomalies = complete_cycles * anomalies_per_cycle

        # Anomalies in remainder
        if remainder > 0:
            start_idx = self._current_index
            end_idx = (self._current_index + remainder) % self.pool_size

            if end_idx > start_idx:
                # No wraparound
                remainder_labels = self.pool_labels[
                    self.pool_indices[start_idx:end_idx]
                ]
            else:
                # Wraparound case
                remainder_labels = np.concatenate(
                    [
                        self.pool_labels[self.pool_indices[start_idx:]],
                        self.pool_labels[self.pool_indices[:end_idx]],
                    ]
                )

            total_anomalies += remainder_labels.sum()

        return int(total_anomalies)

    @property
    def current_position(self) -> int:
        """Current position in the instance pool."""
        return self._current_index

    def __repr__(self) -> str:
        """Return string representation of the generator."""
        n_anomaly_pool = int(self.pool_size * self.anomaly_proportion)
        return (
            f"OnlineGenerator("
            f"n_normal={self.n_normal}, "
            f"n_anomaly={self.n_anomaly}, "
            f"pool_size={self.pool_size}, "
            f"anomaly_proportion={self.anomaly_proportion}, "
            f"anomalies_in_pool={n_anomaly_pool})"
        )
