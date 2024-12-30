import numpy as np

from unquad.utils.decorator import performance_conversion


@performance_conversion("y", "y_hat")
def false_discovery_rate(y: np.array, y_hat: np.array, dec: int = 3) -> float:
    """
    Calculate the False Discovery Rate (FDR) for binary classification.

    The FDR is the proportion of false positives among all positive predictions made by the model.

    Args:
        y (np.array): The true labels, where 1 indicates an anomaly and 0 indicates normal.
        y_hat (np.array): The predicted labels, where 1 indicates an anomaly and 0 indicates normal.
        dec (int, optional): The number of decimal places to round the result. Default is 3.

    Returns:
        float: The calculated False Discovery Rate, rounded to the specified decimal places.
    """
    y = y.astype(int)
    y_hat = y_hat.astype(int)

    false_positives = sum(y_hat & ~y)
    true_positives = sum(y_hat & y)
    total_positives = false_positives + true_positives

    fdr = (false_positives / total_positives) if total_positives != 0 else 0
    return round(fdr, dec)


@performance_conversion("y", "y_hat")
def statistical_power(y: np.array, y_hat: np.array, dec: int = 3) -> float:
    """
    Calculate the statistical power of a binary classifier.

    Statistical power is the proportion of actual positives (true anomalies) correctly identified by the model.

    Args:
        y (np.array): The true labels, where 1 indicates an anomaly and 0 indicates normal.
        y_hat (np.array): The predicted labels, where 1 indicates an anomaly and 0 indicates normal.
        dec (int, optional): The number of decimal places to round the result. Default is 3.

    Returns:
        float: The calculated statistical power, rounded to the specified decimal places.
    """
    y = y.astype(int)
    y_hat = y_hat.astype(int)

    true_positives = sum(y & y_hat)
    false_negatives = sum(~y & y_hat)
    total_actual_outliers = true_positives + false_negatives

    power = (true_positives / total_actual_outliers) if total_actual_outliers > 0 else 0
    return round(power, dec)
