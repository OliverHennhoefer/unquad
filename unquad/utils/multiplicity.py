import numpy as np

from scipy.stats import false_discovery_control

from unquad.utils.decorator import performance_conversion
from unquad.utils.enums import Adjustment


@performance_conversion("scores")
def multiplicity_correction(
    method: Adjustment, scores: np.ndarray
) -> list:  # Adjusted return type based on actual behavior
    """Applies multiplicity correction to scores using a specified method.

    This function processes an array of p-values (or scores from multiple tests)
    to account for the multiplicity of comparisons. It can apply the
    Benjamini-Hochberg (BH) or Benjamini-Yekutieli (BY) procedures,
    which result in boolean values indicating whether each corresponding null
    hypothesis can be rejected while controlling the False Discovery Rate (FDR).
    These procedures are implemented via ``scipy.stats.false_discovery_control``.
    If no adjustment (``Adjustment.NONE``) is specified, the original scores
    are returned.

    The `@performance_conversion("scores")` decorator ensures that the input
    `scores` argument is a ``numpy.ndarray`` (converting from a list if
    necessary) and that the output (if a ``numpy.ndarray``) is converted to a
    Python ``list``.

    Args:
        method (Adjustment): The multiplicity adjustment procedure to apply.
            Must be a member of the :class:`~unquad.utils.enums.Adjustment`
            enum, e.g., ``Adjustment.BH``, ``Adjustment.BY``, or
            ``Adjustment.NONE``.
        scores (numpy.ndarray): A 1D array of p-values (typically floats
            between 0 and 1) or scores from multiple statistical tests that
            require adjustment.

    Returns:
        list: A list representing the outcome of the multiplicity correction.
            - If `method` is ``Adjustment.BH`` or ``Adjustment.BY``, this is a
              list of booleans (``True`` if the null hypothesis for the
              corresponding p-value can be rejected, ``False`` otherwise).
            - If `method` is ``Adjustment.NONE``, this is a list containing
              the original scores.
            The type of elements in the list depends on the method applied.

    Raises:
        ValueError: If an unsupported `method` is provided (i.e., not defined
            in the function's internal mapping for adjustments).
    """
    # Renaming for clarity, as these are adjustment methods
    adjustment_procedures = {
        Adjustment.NONE: lambda x: x,
        Adjustment.BH: lambda x: false_discovery_control(x, method="bh"),
        Adjustment.BY: lambda x: false_discovery_control(x, method="by"),
    }

    # Get the appropriate adjustment function
    adjust_func = adjustment_procedures.get(method)
    if not adjust_func:
        # The original error message uses "aggregation method" due to variable name.
        # Keeping the error message consistent with the code's variable.
        raise ValueError(f"Unsupported aggregation method: {method}")

    # Apply the adjustment function to the scores
    # The @performance_conversion decorator handles np.ndarray -> list conversion
    # for the return value.
    return adjust_func(scores)
