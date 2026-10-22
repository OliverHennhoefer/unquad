import numpy as np
import pandas as pd
import pytest

from nonconform._internal.log_utils import ensure_numpy_array


@ensure_numpy_array
def echo(self, x, *args, **kwargs):
    """Return the received data and extra arguments."""
    return x, args, kwargs


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame(np.arange(30, dtype=np.float32).reshape(10, 3)),
        pd.DataFrame(np.arange(10, dtype=np.float32).reshape(10, 1)),
        pd.DataFrame([[1, 2], [3, 4]], dtype=np.int32),
        pd.DataFrame(),
        pd.Series(np.arange(25, dtype=np.float32), name="values"),
        pd.Series([1.0, 2.0, 3.0], dtype=np.float64),
        pd.Series([], dtype=np.float32),
    ],
    ids=[
        "frame",
        "single-column-frame",
        "int32-frame",
        "empty-frame",
        "series",
        "float64-series",
        "empty-series",
    ],
)
def test_pandas_input_is_converted_with_shape_dtype_and_values(data):
    result, _, _ = echo(None, data)

    assert isinstance(result, np.ndarray)
    assert result.shape == data.shape
    assert result.dtype == data.to_numpy().dtype
    np.testing.assert_array_equal(result, data.to_numpy())


@pytest.mark.parametrize("shape", [(0,), (15,), (10, 5), (5, 4, 3)])
def test_numpy_input_passes_through_unchanged(shape):
    array = np.random.default_rng(42).standard_normal(shape)

    result, _, _ = echo(None, array)

    assert result is array


def test_extra_arguments_and_function_metadata_are_preserved():
    frame = pd.DataFrame(np.ones((2, 2)))

    result, args, kwargs = echo(None, frame, "test", 42, key1="value1", key2=123)

    assert isinstance(result, np.ndarray)
    assert args == ("test", 42)
    assert kwargs == {"key1": "value1", "key2": 123}
    assert echo.__name__ == "echo"
    assert echo.__doc__ == "Return the received data and extra arguments."
