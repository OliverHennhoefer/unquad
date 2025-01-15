import numpy as np

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
    sum_smaller = np.sum(calibration_set >= scores[:, np.newaxis], axis=1)
    return (1.0 + sum_smaller) / (1.0 + len(calibration_set))


@performance_conversion("scores")
def get_decision(alpha: float, scores: np.array) -> [bool]:
    """
    Make a decision for each score based on the given significance level.

    This function compares each score to the specified alpha threshold and
    returns a decision indicating whether the score is less than or equal to alpha.

    Args:
        alpha (float): The significance threshold used to make the decision.
        scores (np.array): The array of test scores to evaluate.

    Returns:
        list[bool]: A list of boolean values indicating whether each score
                    is less than or equal to the alpha threshold.

    Notes:
        The decision for each score is made as:
            True if score <= alpha, otherwise False.
    """
    return scores <= alpha
