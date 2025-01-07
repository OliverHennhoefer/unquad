import numpy as np

from unquad.estimator.configuration import DetectorConfig
from unquad.utils.decorator import performance_conversion


@performance_conversion("scores", "calibration_set")
def calculate_p_val(scores: np.array, calibration_set: np.array) -> np.array:
    """
    Calculate p-values based on the given scores and calibration set.

    This function computes the p-value for each score by comparing it against
    the calibration set. The p-value is calculated as the proportion of
    calibration scores greater than or equal to the given scores.

    Args:
        scores (np.array): The array of test scores for which p-values need to be calculated.
        calibration_set (np.array): The array of calibration scores used to compute p-values.

    Returns:
        np.array: The array of p-values corresponding to the input scores.

    Notes:
        The p-value for each score is calculated using the formula:
            p_val = (1 + number of calibration_set values >= score) / (1 + number of calibration_set values)
    """
    p_val = np.sum(calibration_set >= scores[:, np.newaxis], axis=1)
    return (1.0 + p_val) / (1.0 + len(calibration_set))


@performance_conversion("scores")
def get_decision(config: DetectorConfig, scores: np.array) -> [bool]:
    """
    Make a decision for each score based on the given significance level.

    This function compares each score to the specified alpha threshold and
    returns a decision indicating whether the score is less than or equal to alpha.
    If the 'fore_anomaly' parameter is true,

    Args:
        config (DetectorConfig): Detector configuration.
        scores (np.array): The array of test scores to evaluate.

    Returns:
        list[bool]: A list of boolean values indicating whether each score
                    is less than or equal to the alpha threshold.

    Notes:
        The decision for each score is made as:
            True if score <= alpha, otherwise False.
    """
    if config.force_anomaly:
        bound = 1 / (len(scores) + 1)
        scores = np.array([0 if x <= bound else x for x in scores])

    return scores <= config.alpha
