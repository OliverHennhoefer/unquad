"""Data utilities for unquad."""

from unquad.utils.data.generator.batch import BatchGenerator, create_batch_generator
from .load import (
    clear_cache,
    get_cache_info,
    get_cache_location,
    list_cached_datasets,
    load_breast,
    load_fraud,
    load_ionosphere,
    load_mammography,
    load_musk,
    load_shuttle,
    load_thyroid,
    load_wbc,
)

__all__ = [
    "BatchGenerator",
    "clear_cache",
    "create_batch_generator",
    "get_cache_info",
    "get_cache_location",
    "list_cached_datasets",
    "load_breast",
    "load_fraud",
    "load_ionosphere",
    "load_mammography",
    "load_musk",
    "load_shuttle",
    "load_thyroid",
    "load_wbc",
]
