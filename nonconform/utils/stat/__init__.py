"""Statistical utilities for conformal anomaly detection.

This module provides statistical functions including aggregation methods,
extreme value theory functions, evaluation metrics, and general statistical
operations used in conformal prediction.
"""

from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power

from .aggregation import aggregate
from .extreme import fit_gpd, select_threshold
from .statistical import calculate_evt_p_val, calculate_p_val

__all__ = [
    # Aggregation functions
    "aggregate",
    # Statistical functions
    "calculate_evt_p_val",
    "calculate_p_val",
    # Evaluation metrics
    "false_discovery_rate",
    # Extreme value theory
    "fit_gpd",
    "select_threshold",
    "statistical_power",
]
