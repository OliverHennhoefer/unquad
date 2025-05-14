"""Manages and configures anomaly detection models from the PyOD library.

This module provides utilities for setting up PyOD detector models,
including handling a list of models that are restricted or unsupported.
It dynamically checks for the availability of TensorFlow-dependent models
and updates the list of forbidden models accordingly.

Attributes:
    forbidden_model_list (list[type[BaseDetector]]): A list of PyOD detector
        classes that are considered unsupported or restricted for use by
        the `set_params` function. This list is initially populated with
        certain non-TensorFlow models and conditionally extended with
        TensorFlow-based models (like ALAD, DeepSVDD, etc.) if
        TensorFlow and those models can be imported.
    tf (bool): A flag indicating whether TensorFlow-dependent PyOD models
        (e.g., ALAD, DeepSVDD) were successfully imported. ``True`` if
        imports succeeded, ``False`` otherwise.
"""

import sys

from pyod.models.base import BaseDetector
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.rgraph import RGraph
from pyod.models.sampling import Sampling
from pyod.models.sos import SOS

forbidden_model_list: list[type[BaseDetector]] = [
    CBLOF,
    COF,
    RGraph,
    Sampling,
    SOS,
]

tf: bool = True
try:
    from pyod.models.alad import ALAD  # noqa
    from pyod.models.deep_svdd import DeepSVDD  # noqa
    from pyod.models.so_gaal import SO_GAAL  # noqa
    from pyod.models.mo_gaal import MO_GAAL  # noqa
    from pyod.models.vae import VAE  # noqa

    forbidden_model_list += [ALAD, DeepSVDD, SO_GAAL, MO_GAAL, VAE]
except ImportError:
    tf: bool = False


def set_params(
    detector: BaseDetector,
    seed: int,
    random_iteration: bool = False,
    iteration: int | None = None,
) -> BaseDetector:
    """Configures a PyOD detector with specific default and dynamic parameters.

    This function modifies the provided PyOD detector instance by setting common
    parameters. It sets 'contamination' to a very small value (effectively
    for one-class classification behavior), 'n_jobs' to utilize all available
    CPU cores, and 'random_state' for reproducibility. The 'random_state'
    can be fixed or varied per iteration.

    The function will raise an error if the detector's class is present in
    the module-level `forbidden_model_list`.

    Args:
        detector (BaseDetector): The PyOD detector instance to configure.
        seed (int): The base random seed for reproducibility. If
            `random_iteration` is ``False`` or `iteration` is ``None``,
            this seed is directly used for `random_state`.
        random_iteration (bool, optional): If ``True`` and `iteration` is
            provided, the `random_state` for the detector will be set to a
            hash of `iteration` and `seed`, allowing for different random
            states across iterations. Defaults to ``False``.
        iteration (int | None, optional): The current iteration number. Used in
            conjunction with `random_iteration` to generate a dynamic
            `random_state`. Defaults to ``None``.

    Returns:
        BaseDetector: The configured detector instance with updated parameters.

    Raises:
        ValueError: If the class of the `detector` is found in the
            `forbidden_model_list`.
    """
    if detector.__class__ in forbidden_model_list:
        raise ValueError(
            f"{detector.__class__.__name__} is not supported by set_params."
        )

    # Set contamination to the smallest possible float for one-class classification
    if "contamination" in detector.get_params():
        detector.set_params(contamination=sys.float_info.min)

    # Utilize all available cores if n_jobs parameter exists
    if "n_jobs" in detector.get_params():
        detector.set_params(n_jobs=-1)

    # Set random_state for reproducibility
    if "random_state" in detector.get_params():
        if random_iteration and iteration is not None:
            # Create a reproducible but varying seed per iteration
            # Ensure the result is within the typical 32-bit unsigned int range
            # for random seeds.
            dynamic_seed = hash((iteration, seed)) % (2**32)
            detector.set_params(random_state=dynamic_seed)
        else:
            detector.set_params(random_state=seed)

    return detector
