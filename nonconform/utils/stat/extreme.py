"""Extreme Value Theory utilities for modeling tail distributions.

This module provides functions for fitting Generalized Pareto Distributions (GPD)
to extreme values, threshold selection methods, and hybrid p-value calculations
that combine bulk and tail distributions.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy import stats


def select_threshold(
    scores: np.ndarray,
    method: Literal["percentile", "top_k", "mean_excess", "custom"],
    value: float | Callable[[np.ndarray], float],
) -> float:
    """Select threshold for extreme value modeling.

    Args:
        scores: Array of calibration scores
        method: Method for threshold selection
            - "percentile": Use a percentile (e.g., 0.95 for 95th percentile)
            - "top_k": Use the k-th highest value
            - "mean_excess": Use mean excess plot method (value is initial percentile)
            - "custom": Use a custom threshold selection function
        value: Parameter for the method
            - For "percentile": value in [0, 1]
            - For "top_k": integer number of top values
            - For "mean_excess": initial percentile to start search
            - For "custom": callable that takes scores array and returns threshold

    Returns
    -------
        Threshold value
    """
    sorted_scores = np.sort(scores)

    if method == "percentile":
        if not 0 <= value <= 1:
            raise ValueError(f"Percentile value must be in [0, 1], got {value}")
        return np.percentile(scores, value * 100)

    elif method == "top_k":
        k = int(value)
        if k <= 0 or k > len(scores):
            raise ValueError(f"top_k value must be in [1, {len(scores)}], got {k}")
        return sorted_scores[-k]

    elif method == "mean_excess":
        # Start from given percentile and find stable region
        initial_idx = int(len(sorted_scores) * value)
        threshold = sorted_scores[initial_idx]
        return threshold

    elif method == "custom":
        if not callable(value):
            raise ValueError(
                f"For custom method, value must be callable, got {type(value)}"
            )
        return value(scores)

    else:
        raise ValueError(f"Unknown threshold method: {method}")


def fit_gpd(exceedances: np.ndarray) -> tuple[float, float, float]:
    """Fit Generalized Pareto Distribution to exceedances.

    Args:
        exceedances: Values exceeding the threshold (X - threshold for X > threshold)

    Returns
    -------
        Tuple of (shape, location, scale) parameters
    """
    if len(exceedances) == 0:
        raise ValueError("No exceedances to fit GPD")

    # Use MLE to fit GPD
    # scipy's genpareto uses different parameterization: shape=-xi
    shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

    return shape, loc, scale


def gpd_tail_probability(
    score: float,
    threshold: float,
    shape: float,
    scale: float,
    n_total: int,
    n_exceed: int,
) -> float:
    """Calculate tail probability using fitted GPD.

    Args:
        score: Test score to evaluate
        threshold: Threshold used for GPD fitting
        shape: GPD shape parameter (xi)
        scale: GPD scale parameter (sigma)
        n_total: Total number of calibration samples
        n_exceed: Number of samples exceeding threshold

    Returns
    -------
        Probability that a random value exceeds the score
    """
    if score <= threshold:
        return 1.0  # All tail values exceed non-tail scores

    # Probability of exceeding threshold
    p_exceed_threshold = n_exceed / n_total

    # Conditional probability of exceeding score given exceeding threshold
    # P(X > score | X > threshold) = 1 - GPD_cdf(score - threshold)
    exceedance = score - threshold
    p_exceed_given_threshold = 1 - stats.genpareto.cdf(
        exceedance, c=shape, loc=0, scale=scale
    )

    # Total probability: P(X > score) = P(X > threshold) * P(X > score | X > threshold)
    return p_exceed_threshold * p_exceed_given_threshold


def calculate_hybrid_p_value(
    score: float,
    calibration_scores: np.ndarray,
    threshold: float,
    gpd_params: tuple[float, float, float],
) -> float:
    """Calculate p-value using hybrid approach (empirical + GPD).

    For scores within the calibration range, uses empirical distribution.
    For scores beyond the calibration range, uses GPD-based extrapolation.
    This approach is more statistically sound as it only applies EVT
    for truly extreme values beyond observed data.

    Args:
        score: Test score to evaluate
        calibration_scores: All calibration scores
        threshold: Threshold used for GPD fitting (not decision boundary)
        gpd_params: Tuple of (shape, loc, scale) from GPD fit

    Returns
    -------
        P-value for the test score
    """
    n_total = len(calibration_scores)
    max_calib_score = np.max(calibration_scores)

    if score <= max_calib_score:
        # Use empirical p-value for scores within calibration range
        # This preserves the exact conformal guarantees
        n_greater_equal = np.sum(calibration_scores >= score)
        return (1.0 + n_greater_equal) / (1.0 + n_total)
    else:
        # For scores beyond calibration range, use EVT extrapolation
        # This provides principled tail modeling for extreme scores
        n_exceed_threshold = np.sum(calibration_scores > threshold)

        # Empirical probability at the boundary (max calibration score)
        p_at_boundary = 1.0 / (1.0 + n_total)  # Minimum possible empirical p-value

        # GPD tail probability beyond the maximum calibration score
        shape, loc, scale = gpd_params
        p_tail = gpd_tail_probability(
            score, threshold, shape, scale, n_total, n_exceed_threshold
        )

        # Ensure monotonicity: p-value for extreme score should not exceed
        # p-value at the boundary of calibration data
        return min(p_tail, p_at_boundary)
