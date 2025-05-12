import numpy as np

from unquad.utils.decorator import performance_conversion


@performance_conversion("y", "y_hat") # Keep decorator if needed
def false_discovery_rate(y: np.array, y_hat: np.array, dec: int = 3) -> float:
    """
    Calculate the False Discovery Rate (FDR) for binary classification.

    The FDR is the proportion of false positives among all positive predictions made by the model.
    FDR = FP / (FP + TP)

    Args:
        y (np.array): The true labels, where 1 indicates positive/anomaly and 0 indicates negative/normal.
        y_hat (np.array): The predicted labels, where 1 indicates positive/anomaly and 0 indicates negative/normal.
        dec (int, optional): The number of decimal places to round the result. Default is 3.

    Returns:
        float: The calculated False Discovery Rate, rounded to the specified decimal places.
    """
    # Ensure boolean or 0/1 int arrays for logical operations
    y_true = (y == 1)
    y_pred = (y_hat == 1)

    true_positives = np.sum(y_pred & y_true)
    false_positives = np.sum(y_pred & ~y_true) # Use logical NOT on boolean array

    total_predicted_positives = true_positives + false_positives

    if total_predicted_positives == 0:
        fdr = 0.0 # Or np.nan, depending on desired behavior when no positives are predicted
    else:
        fdr = false_positives / total_predicted_positives

    return round(fdr, dec)


@performance_conversion("y", "y_hat")
def statistical_power(y: np.array, y_hat: np.array, dec: int = 3) -> float:
    """
    Corrected: Calculate the statistical power of a binary classifier.
    """
    y = y.astype(int)
    y_hat = y_hat.astype(int)

    true_positives = sum((y == 1) & (y_hat == 1))
    false_negatives = sum((y == 1) & (y_hat == 0))
    total_actual_outliers = true_positives + false_negatives

    power = (true_positives / total_actual_outliers) if total_actual_outliers > 0 else 0
    return round(power, dec)