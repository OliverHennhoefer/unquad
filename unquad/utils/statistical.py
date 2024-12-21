import numpy as np

from unquad.utils.decorator.performance import performance_conversion


@performance_conversion("scores", "calibration_set")
def calculate_p_val(scores: np.array, calibration_set: np.array) -> np.array:
    p_val = np.sum(calibration_set >= scores[:, np.newaxis], axis=1)
    return (1.0 + p_val) / (1.0 + len(calibration_set))


@performance_conversion("scores")
def get_decision(alpha: float, scores: np.array) -> [bool]:
    return scores <= alpha
