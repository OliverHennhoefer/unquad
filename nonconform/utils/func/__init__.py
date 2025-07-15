"""Functional programming utilities for nonconform.

This module provides decorators, enumerations, and parameter utilities
used throughout the nonconform package.
"""

from .decorator import ensure_numpy_array
from .enums import Aggregation
from .params import set_params

__all__ = [
    "Aggregation",
    "ensure_numpy_array",
    "set_params",
]
