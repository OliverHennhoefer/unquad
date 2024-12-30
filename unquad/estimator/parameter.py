import sys

from pyod.models.base import BaseDetector
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.rgraph import RGraph
from pyod.models.sampling import Sampling
from pyod.models.sos import SOS

from unquad.utils.error import ForbiddenModelError

forbidden_model_list = [
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
    iteration: int = None,
) -> BaseDetector:
    """
    Imports and configures conformal anomaly detectors, while managing restricted models.

    Imports:
        - BaseDetector from pyod.models.base: The base class for all detectors.
        - Various specific detectors from pyod.models (CBLOF, COF, RGraph, etc.)
        - ForbiddenModelError from unquad.utils.error.forbidden_model: Custom error raised
          when attempting to use unsupported models.

    Attributes:
        forbidden_model_list (list): List of models that are not supported for use.
        tf (bool): Flag indicating whether specific models (ALAD, DeepSVDD, etc.) were successfully imported.

    Functions:
        set_params(detector, seed, random_iteration=False, iteration=None):
            Configures the provided detector by setting parameters such as contamination,
            n_jobs, and random_state. Raises a ForbiddenModelError if the detector is in the
            forbidden_model_list.

        Args:
            detector (BaseDetector): The detector model to configure.
            seed (int): The random seed for reproducibility.
            random_iteration (bool): If True, randomize the `random_state` based on the iteration.
            iteration (int, optional): The current iteration for randomizing the `random_state`.

        Returns:
            BaseDetector: The configured detector with the updated parameters.

        Raises:
            ForbiddenModelError: If the detector is part of the forbidden_model_list.
    """

    if detector.__class__ in forbidden_model_list:
        raise ForbiddenModelError(f"{detector.__class__.__name__} is not supported.")

    if "contamination" in detector.get_params().keys():
        detector.set_params(
            **{
                "contamination": sys.float_info.min,  # One-Class Classification
            }
        )

    if "n_jobs" in detector.get_params().keys():
        detector.set_params(
            **{
                "n_jobs": -1,
            }
        )

    if "random_state" in detector.get_params().keys():
        if random_iteration and iteration is not None:
            detector.set_params(
                **{
                    "random_state": hash((iteration, seed)) % 4294967296,
                }
            )

        else:
            detector.set_params(
                **{
                    "random_state": seed,
                }
            )

    return detector
