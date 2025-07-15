"""unquad: Conformal Anomaly Detection with Uncertainty Quantification.

This package provides statistically rigorous anomaly detection with p-values
and error control metrics like False Discovery Rate (FDR) for PyOD detectors.

Main Components:
- Conformal detectors with uncertainty quantification
- Calibration strategies for different data scenarios
- Extreme Value Theory enhanced detection
- Statistical utilities and data handling tools
"""

import warnings

__version__ = "0.8.3"
__author__ = "Oliver Hennhoefer"
__email__ = "oliver.hennhoefer@mail.de"

warnings.warn(
    "The 'unquad' package is deprecated and has been renamed to 'nonconform'. "
    "Please uninstall this package and install 'nonconform' instead.",
    DeprecationWarning,
    stacklevel=2,
)
