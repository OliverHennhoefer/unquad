import numpy as np


def false_discovery_rate(y: np.array, y_hat: np.array, dec: int = 3) -> float:
    """
    Calculates the false discovery rate (FDR).
    :param y: Numpy Array, set of ground truth labels
    :param y_hat: Numpy Array, set of predictions
    :param dec: Integer, number of decimal places
    :return: Float, the false discovery rate
    """

    y = y.astype(int)
    y_hat = y_hat.astype(int)

    false_positives = sum(y_hat & ~y)
    true_positives = sum(y_hat & y)

    fdr = false_positives / (false_positives + true_positives)
    return round(fdr, dec)


def statistical_power(y: np.array, y_hat: np.array, dec: int = 3) -> float:
    """
    Calculates the statistical power (as sensitivity or recall).
    :param y: Numpy Array, set of ground truth labels
    :param y_hat: Numpy Array, set of predictions
    :param dec: Integer, number of decimal places
    :return: Float, the statistical power
    """

    y = y.astype(int)
    y_hat = y_hat.astype(int)

    true_positives = sum(y & y_hat)
    false_negatives = sum(~y & y_hat)
    total_actual_outliers = true_positives + false_negatives

    power = (true_positives / total_actual_outliers) if total_actual_outliers > 0 else 0
    return round(power, dec)
