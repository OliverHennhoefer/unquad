import numpy as np
import pytest

from nonconform.structures import ConformalResult

ARRAY_FIELDS = (
    "p_values",
    "test_scores",
    "calib_scores",
    "test_weights",
    "calib_weights",
)


def test_constructor_stores_fields_and_defaults_to_empty():
    default = ConformalResult()
    assert all(getattr(default, name) is None for name in ARRAY_FIELDS)
    assert default.metadata == {}

    arrays = {name: np.arange(3.0) + i for i, name in enumerate(ARRAY_FIELDS)}
    metadata = {"key": "value", "number": 42}
    result = ConformalResult(**arrays, metadata=metadata)

    for name, values in arrays.items():
        assert getattr(result, name) is values
    assert result.metadata == metadata


def test_copy_duplicates_arrays_and_deep_copies_metadata(conformal_result):
    result = conformal_result(include_p_values=True, include_weights=True)
    result.metadata["nested"] = [1, 2, 3]

    copied = result.copy()

    assert copied is not result
    for name in ARRAY_FIELDS:
        original, duplicate = getattr(result, name), getattr(copied, name)
        np.testing.assert_array_equal(duplicate, original)
        assert not np.shares_memory(duplicate, original)
    copied.metadata["nested"].append(4)
    assert result.metadata["nested"] == [1, 2, 3]
    empty_copy = ConformalResult().copy()
    assert all(getattr(empty_copy, name) is None for name in ARRAY_FIELDS)


@pytest.mark.parametrize(
    ("result", "expected_parts"),
    [
        (
            ConformalResult(),
            [
                "ConformalResult(",
                "p_values=None",
                "calib_weights=None",
                "metadata_keys=[])",
            ],
        ),
        (
            ConformalResult(
                p_values=np.array([0.1, 0.2]),
                test_scores=np.array([1.0, 2.0]),
                calib_scores=np.array([0.5, 1.5, 2.5]),
                test_weights=np.array([]),
                metadata={"a": 1},
            ),
            [
                "p_values=array(shape=(2,), dtype=float64)",
                "calib_scores=array(shape=(3,), dtype=float64)",
                "test_weights=array(shape=(0,), dtype=float64)",
                "calib_weights=None",
                "metadata_keys=['a'])",
            ],
        ),
        (
            ConformalResult(metadata=dict.fromkeys("gfedcba")),
            ["metadata_keys=['a', 'b', 'c', 'd', 'e']... (total=7))"],
        ),
    ],
    ids=["empty", "arrays", "truncated-metadata"],
)
def test_repr_summarizes_arrays_and_metadata_keys(result, expected_parts):
    text = repr(result)

    for part in expected_parts:
        assert part in text
