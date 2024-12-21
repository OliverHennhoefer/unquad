import sys

from pyod.models.alad import ALAD
from pyod.models.base import BaseDetector
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.rgraph import RGraph
from pyod.models.sampling import Sampling
from pyod.models.so_gaal import SO_GAAL
from pyod.models.sos import SOS
from pyod.models.vae import VAE

from unquad.utils.error.forbidden_model import ForbiddenModelError


def set_params(
    detector: BaseDetector,
    seed: int,
    random_iteration: bool = False,
    iteration: int = None,
) -> BaseDetector:
    if detector.__class__ in [
        ALAD,
        CBLOF,
        COF,
        DeepSVDD,
        MO_GAAL,
        RGraph,
        Sampling,
        SO_GAAL,
        SOS,
        VAE,
    ]:
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
